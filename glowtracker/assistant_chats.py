"""Earlier conversations with the AI assistant, read back from their transcripts (the .jsonl files
llm_assist.Transcript writes): the list to choose from, and what the model needs to carry on. No
Kivy import.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import assistant_documents
from ox.types import AssistantMessage, FunctionCall, ToolCallWire, ToolMessage, UserMessage

TITLE_CHARS = 90
STOPPED = '(Stopped by the user before finishing.)'
NO_ANSWER = '(No answer was given.)'


@dataclass
class ChatInfo:
    path: str
    title: str
    model: str
    when: str               # last activity


def title_of(record: dict) -> str:
    """A user message as a one-line title."""
    text = ' '.join((record.get('content') or '').split())
    names = [a.get('name', '') for a in record.get('attachments') or []]
    text = text or ', '.join(names) or '(empty)'
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 2].rstrip() + '..'


def _summary(path: str) -> ChatInfo | None:
    """Reads only as far as the first user message."""
    model = ''
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if record.get('type') == 'start':
                    model = record.get('model', '')
                elif record.get('type') == 'user':
                    when = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))
                    return ChatInfo(path, title_of(record), model, when)
    except (OSError, ValueError):
        logging.warning('assistant: skipping unreadable transcript %s', path)
    return None


def list_chats(folder: str) -> list[ChatInfo]:
    """The saved conversations that have a message in them, most recently used first."""
    try:
        names = [n for n in os.listdir(folder) if n.endswith('.jsonl')]
    except OSError:
        return []
    paths = sorted((os.path.join(folder, n) for n in names), key=os.path.getmtime, reverse=True)
    return [chat for chat in map(_summary, paths) if chat is not None]


def read(path: str) -> list[dict]:
    """All records of a transcript. A line cut off by a crash ends it."""
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except ValueError:
                break
    return records


def usage(records: list[dict]) -> dict:
    """Tokens used so far, as ChatSession.usage."""
    total = {'prompt': 0, 'completion': 0, 'cost': None}
    for record in records:
        if record.get('type') == 'usage':
            total['prompt'] += int(record.get('prompt_tokens') or 0)
            total['completion'] += int(record.get('completion_tokens') or 0)
            if record.get('cost') is not None:
                total['cost'] = (total['cost'] or 0.0) + float(record['cost'])
    return total


def history(records: list[dict]) -> list:
    """The conversation as model messages, without the system prompt. It stays valid for the API:
    tool calls that never got a result (the window was closed, the turn was stopped) are left
    out, and every user message is answered, since some models refuse two in a row."""
    messages: list = []
    lastUser = -1

    def answered() -> None:
        if messages and not (isinstance(messages[-1], AssistantMessage) and not messages[-1].tool_calls):
            messages.append(AssistantMessage(content=NO_ANSWER))

    for record in records:
        kind = record.get('type')
        if kind == 'user':
            answered()
            content = assistant_documents.user_content(record.get('content') or '', record.get('attachments'))
            lastUser = len(messages)
            messages.append(UserMessage(content=content))
        elif kind == 'assistant':
            messages.append(AssistantMessage(content=record.get('content') or ''))
        elif kind == 'tool_call':
            call = ToolCallWire(id=record.get('call_id', ''), type='function', function=FunctionCall(
                name=record.get('name', ''), arguments=record.get('arguments') or '{}'))
            last = messages[-1] if messages else None
            if isinstance(last, AssistantMessage):
                messages[-1] = AssistantMessage(content=last.content, tool_calls=(last.tool_calls or []) + [call])
            else:
                messages.append(AssistantMessage(content='', tool_calls=[call]))
        elif kind == 'tool_result':
            messages.append(ToolMessage(tool_call_id=record.get('call_id', ''), content=record.get('content') or ''))
        elif kind == 'stopped' and lastUser >= 0:
            del messages[lastUser + 1:]         # as the running chat did: the turn's work is dropped
            messages.append(AssistantMessage(content=STOPPED))
    messages = _withoutUnanswered(messages)
    answered()
    return messages


def _withoutUnanswered(messages: list) -> list:
    results = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    kept = []
    for message in messages:
        if isinstance(message, AssistantMessage) and message.tool_calls:
            calls = [c for c in message.tool_calls if c.id in results]
            if not calls and not message.content:
                continue
            message = AssistantMessage(content=message.content, tool_calls=calls or None)
        kept.append(message)
    return kept
