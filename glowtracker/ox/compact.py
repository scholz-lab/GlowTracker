"""Conversation compactor — summarise stale mid-conversation messages.

When prompt tokens exceed a threshold, the middle portion is summarised
with a cheap model and replaced in-place. Keeps the context window lean
without losing important earlier decisions.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import structlog

from ox.types import (
    AssistantMessage,
    Message,
    Ok,
    Request,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from ox.provider import Provider

log = structlog.get_logger()

_COMPACTOR_MODEL = os.getenv("OX_COMPACTOR_MODEL", "google/gemini-2.0-flash-001")
_TOKEN_THRESHOLD = int(os.getenv("OX_COMPACT_TOKEN_THRESHOLD", "80000"))
_KEEP_HEAD = int(os.getenv("OX_COMPACT_KEEP_HEAD", "3"))
_KEEP_TAIL = int(os.getenv("OX_COMPACT_KEEP_TAIL", "10"))
_MAX_MID_CHARS = 12_000
_MAX_TOOL_CHARS = 300

_SUMMARY_PROMPT = """\
Summarise this agent conversation excerpt concisely.

Preserve:
- Decisions made and their rationale
- Actions taken (tool calls, file edits, commands run) and outcomes
- Key facts, entities, numbers, file paths mentioned
- Commitments, constraints, or instructions the agent must still follow
- Any unresolved problems or pending work

Skip: greetings, filler, raw tool output, internal reasoning steps.

Return a single coherent summary paragraph (no bullets, no headers).
Keep it under 400 words.

Conversation excerpt:
"""


def needs_compaction(prompt_tokens: int) -> bool:
    return prompt_tokens > _TOKEN_THRESHOLD


def _split(
    messages: list[Message],
) -> tuple[list[Message], list[Message], list[Message]]:
    sys_end = 0
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            sys_end = i + 1
        else:
            break

    rest = messages[sys_end:]
    if len(rest) <= _KEEP_HEAD + _KEEP_TAIL:
        return messages, [], []

    head = messages[:sys_end] + rest[:_KEEP_HEAD]
    mid = rest[_KEEP_HEAD:-_KEEP_TAIL]
    tail = rest[-_KEEP_TAIL:]
    return head, mid, tail


def _format_mid(messages: list[Message]) -> str:
    lines: list[str] = []
    total = 0

    for msg in messages:
        match msg:
            case UserMessage(content=str(c)):
                line = f"[user] {c}"
            case UserMessage(content=list(parts)):
                text = "".join(getattr(p, "text", "") for p in parts)
                line = f"[user] {text}"
            case AssistantMessage(content=str(c)):
                line = f"[assistant] {c}"
                if msg.tool_calls:
                    names = ", ".join(
                        tc.function.name for tc in msg.tool_calls if tc.function
                    )
                    line += f"\n  -> called: {names}"
            case AssistantMessage(content=list(parts)):
                text = "".join(getattr(p, "text", "") for p in parts)
                line = f"[assistant] {text}"
            case ToolMessage(content=str(c)):
                truncated = c[:_MAX_TOOL_CHARS]
                if len(c) > _MAX_TOOL_CHARS:
                    truncated += f"...({len(c)} chars)"
                line = f"[tool_result] {truncated}"
            case _:
                continue

        if total + len(line) > _MAX_MID_CHARS:
            lines.append(f"... ({len(messages) - len(lines)} more messages omitted)")
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)


def _extract_files_read(messages: list[Message]) -> list[str]:
    _FILE_TOOLS = frozenset({"read", "write", "edit", "bash"})
    seen: dict[str, None] = {}
    for msg in messages:
        if not isinstance(msg, AssistantMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if not tc.function or tc.function.name not in _FILE_TOOLS:
                continue
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                continue
            path = args.get("path") or args.get("file_path") or args.get("filename") or ""
            if path and path not in seen:
                seen[path] = None
    return list(seen)


async def compact(
    messages: list[Message],
    provider: Provider,
    model: str | None = None,
) -> list[Message] | None:
    head, mid, tail = _split(messages)
    if not mid:
        return None

    transcript = _format_mid(mid)
    if len(transcript) < 100:
        return None

    log.info("compacting", total=len(messages), mid=len(mid), tail=len(tail))

    summary_text = await _summarise(transcript, provider, model or _COMPACTOR_MODEL)
    if not summary_text:
        summary_text = transcript[:2000] + f"\n\n... ({len(mid)} messages truncated)"

    files_read = _extract_files_read(mid)
    files_note = ""
    if files_read:
        files_note = "\n\nFiles previously accessed (re-read if needed):\n"
        files_note += "\n".join(f"- {p}" for p in files_read)

    summary_msg = SystemMessage(
        content=f"[Conversation summary — {len(mid)} earlier messages compacted]\n\n{summary_text}{files_note}",
    )

    compacted = head + [summary_msg] + tail
    log.info("compacted", before=len(messages), after=len(compacted), mid_summarised=len(mid))
    return compacted


async def _summarise(transcript: str, provider: Provider, model: str) -> str:
    request = Request(
        model=model,
        messages=[
            SystemMessage(content="You are a precise conversation summariser."),
            UserMessage(content=_SUMMARY_PROMPT + transcript),
        ],
        max_tokens=800,
    )

    match await provider.chat(request):
        case Ok(response):
            msg = response.choices[0].message if response.choices else None
            if msg and isinstance(msg.content, str):
                return msg.content.strip()
            if msg and isinstance(msg.content, list):
                return "".join(getattr(p, "text", "") for p in msg.content).strip()
            return ""
        case _:
            return ""
