"""Claude Code subprocess provider.

Uses the `claude` CLI as the LLM backend via `--print` mode with
`--resume` for session continuity. No API key needed — uses Claude
Code's existing OAuth authentication.

Output formats:
  --output-format json        → single result JSON
  --output-format stream-json → NDJSON stream (requires --verbose)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import structlog

from ox.types import (
    AssistantMessage,
    ChatChoice,
    ChatResponse,
    Error,
    ErrorResponse,
    Ok,
    Request,
    Result,
    StreamingResponse,
    SystemMessage,
    ToolMessage,
    UserMessage,
    UsageInfo,
)

log = structlog.get_logger()


def _format_messages(request: Request) -> str:
    """Serialize the latest user/tool messages into text for claude -p."""
    parts: list[str] = []
    for msg in request.messages:
        match msg:
            case ToolMessage(tool_call_id=tid, content=c):
                parts.append(f"[tool_result {tid}] {c}")
            case AssistantMessage(tool_calls=tcs) if tcs:
                pass
            case UserMessage(content=str(c)) if c:
                parts.append(c)
            case AssistantMessage(content=str(c)) if c:
                parts.append(c)
    return "\n\n".join(parts) if parts else ""


def _extract_system(request: Request) -> str | None:
    """Extract system message text."""
    for msg in request.messages:
        if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
            return msg.content
    return None


def _parse_usage(data: dict) -> UsageInfo:
    usage = data.get("usage", {})
    input_t = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    output_t = usage.get("output_tokens", 0)
    return UsageInfo(
        prompt_tokens=input_t,
        completion_tokens=output_t,
        total_tokens=input_t + output_t,
        cost=data.get("total_cost_usd"),
    )


class ClaudeCodeProvider:
    """Provider that shells out to `claude` CLI with --resume for continuity."""

    __slots__ = ("_session_id", "_model")

    def __init__(self, model: str | None = None) -> None:
        self._session_id: str | None = None
        self._model = model

    async def chat(self, request: Request) -> Result[ChatResponse, ErrorResponse]:
        prompt = _format_messages(request)
        if not prompt:
            return Error(ErrorResponse(code=400, message="Empty prompt"))

        cmd = ["claude", "-p", "--output-format", "json"]
        if self._session_id:
            cmd += ["--resume", self._session_id]

        system = _extract_system(request)
        if system and not self._session_id:
            cmd += ["--system-prompt", system]

        if self._model:
            cmd += ["--model", self._model]

        cmd.append(prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except TimeoutError:
            return Error(ErrorResponse(code=408, message="claude CLI timed out"))
        except Exception as e:
            return Error(ErrorResponse(code=0, message=str(e)))

        if proc.returncode != 0:
            err_text = stderr.decode(errors="replace")[:500]
            return Error(ErrorResponse(code=proc.returncode, message=err_text))

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            return Error(ErrorResponse(code=502, message=f"Invalid JSON from claude: {e}"))

        if data.get("is_error"):
            return Error(ErrorResponse(
                code=500,
                message=data.get("result", "Unknown error from claude CLI"),
            ))

        # Persist session for --resume
        sid = data.get("session_id")
        if sid:
            self._session_id = sid

        usage = _parse_usage(data)
        message = AssistantMessage(content=data.get("result", ""))
        model_name = data.get("model", request.model)

        return Ok(ChatResponse(
            id=data.get("uuid", ""),
            choices=[ChatChoice(index=0, message=message, finish_reason=data.get("stop_reason", "end_turn"))],
            model=model_name,
            usage=usage,
        ))

    async def chat_stream(
        self, request: Request,
    ) -> AsyncIterator[Result[StreamingResponse, ErrorResponse]]:
        prompt = _format_messages(request)
        if not prompt:
            yield Error(ErrorResponse(code=400, message="Empty prompt"))
            return

        cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose"]
        if self._session_id:
            cmd += ["--resume", self._session_id]

        system = _extract_system(request)
        if system and not self._session_id:
            cmd += ["--system-prompt", system]

        if self._model:
            cmd += ["--model", self._model]

        cmd.append(prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            yield Error(ErrorResponse(code=0, message=str(e)))
            return

        buf = b""
        async for chunk in proc.stdout:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "assistant":
                    # Extract text from assistant message content
                    inner = data.get("message", {})
                    content_parts = inner.get("content", [])
                    text = ""
                    for part in content_parts:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text += part.get("text", "")
                    if text:
                        delta = AssistantMessage(content=text)
                        sid = data.get("session_id")
                        if sid:
                            self._session_id = sid
                        yield Ok(StreamingResponse(
                            id=inner.get("id", ""),
                            choices=[ChatChoice(index=0, delta=delta)],
                            model=inner.get("model", request.model),
                        ))

                elif msg_type == "result":
                    sid = data.get("session_id")
                    if sid:
                        self._session_id = sid
                    usage = _parse_usage(data)
                    yield Ok(StreamingResponse(
                        id=data.get("uuid", ""),
                        choices=[ChatChoice(
                            index=0,
                            delta=AssistantMessage(),
                            finish_reason=data.get("stop_reason", "end_turn"),
                        )],
                        model=data.get("model", request.model),
                        usage=usage,
                    ))

        await proc.wait()

    async def close(self) -> None:
        pass
