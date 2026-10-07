"""Minimal types for the ox agent runtime.

Message types use msgspec tagged unions for fast serialization and
pattern matching. Request/Response types follow the OpenAI-compatible
chat completion format used by most LLM providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar, Union

import msgspec


# ── Result monad ─────────────────────────────────────────────────────────────

T = TypeVar("T")
E = TypeVar("E")


@dataclass(slots=True)
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass(slots=True)
class Error(Generic[E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> Any:
        raise ValueError(f"Called unwrap on Error: {self.error}")


Result = Union[Ok[T], Error[E]]


# ── Content parts ────────────────────────────────────────────────────────────

class TextPart(msgspec.Struct, frozen=True, tag="text"):
    text: str


class ImagePart(msgspec.Struct, frozen=True, tag="image_url"):
    image_url: dict[str, str]


ContentPart = Union[TextPart, ImagePart]


# ── Messages ─────────────────────────────────────────────────────────────────

class FunctionCall(msgspec.Struct, frozen=True):
    # Defaults: streamed deltas carry the name only in the first chunk.
    name: str = ""
    arguments: str = ""  # JSON string


class ToolCallWire(msgspec.Struct, frozen=True, omit_defaults=True):
    id: str = ""
    type: str = "function"
    function: FunctionCall | None = None
    index: int | None = None


class SystemMessage(msgspec.Struct, frozen=True, tag_field="role", tag="system"):
    content: str | list[TextPart]


class UserMessage(msgspec.Struct, frozen=True, tag_field="role", tag="user"):
    content: str | list[ContentPart]


class AssistantMessage(msgspec.Struct, frozen=True, omit_defaults=True, tag_field="role", tag="assistant"):
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCallWire] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None  # vLLM / DeepSeek name for `reasoning`; decode only


class ToolMessage(msgspec.Struct, frozen=True, tag_field="role", tag="tool"):
    tool_call_id: str
    content: str


# OpenAI wire format: the "role" field is the union tag.
Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# ── Tool definitions (for LLM) ──────────────────────────────────────────────

class FunctionDescription(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    parameters: dict[str, object]
    description: str | None = None


class Tool(msgspec.Struct, frozen=True):
    type: str = "function"
    function: FunctionDescription | None = None


# ── Tool spec (from --tools CLIs) ───────────────────────────────────────────

class ToolSpec(msgspec.Struct, frozen=True):
    """A tool provided by an external CLI."""
    name: str
    description: str
    command: list[str]
    input_schema: dict[str, object] = {}
    stdin_json: bool = False
    timeout: float = 120.0

    def to_tool(self) -> Tool:
        """Convert to LLM-facing Tool definition."""
        return Tool(function=FunctionDescription(
            name=self.name,
            parameters=self.input_schema or {"type": "object", "properties": {}},
            description=self.description,
        ))


# ── Tool result ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ToolResult:
    content: str
    is_error: bool = False


def tool_ok(text: str) -> ToolResult:
    return ToolResult(content=text)


def tool_error(msg: str) -> ToolResult:
    return ToolResult(content=msg, is_error=True)


# ── Request / Response ───────────────────────────────────────────────────────

class CacheControl(msgspec.Struct, frozen=True):
    type: str = "ephemeral"


class Reasoning(msgspec.Struct, frozen=True, omit_defaults=True):
    effort: str | None = None
    max_tokens: int | None = None
    enabled: bool | None = None


class Request(msgspec.Struct, frozen=True, omit_defaults=True):
    model: str
    messages: list[Message]
    tools: list[Tool] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool | None = None
    stream_options: dict[str, object] | None = None
    reasoning: Reasoning | None = None
    cache_control: CacheControl | None = None
    provider: dict[str, object] | None = None
    user: str | None = None


class UsageInfo(msgspec.Struct, frozen=True, omit_defaults=True):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None


class ChatChoice(msgspec.Struct, frozen=True):
    index: int = 0
    message: AssistantMessage = AssistantMessage()
    finish_reason: str | None = None
    delta: AssistantMessage | None = None


class ChatResponse(msgspec.Struct, frozen=True, tag="chat.completion"):
    id: str = ""
    choices: list[ChatChoice] = []
    created: int = 0
    model: str = ""
    usage: UsageInfo | None = None


class StreamingResponse(msgspec.Struct, frozen=True, tag="chat.completion.chunk"):
    id: str = ""
    choices: list[ChatChoice] = []
    created: int = 0
    model: str = ""
    usage: UsageInfo | None = None


class ErrorKind(Enum):
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    INVALID_REQUEST = "invalid_request"
    AUTH = "auth"
    TIMEOUT = "timeout"
    SERVER = "server"
    UNKNOWN = "unknown"


class ErrorResponse(msgspec.Struct, frozen=True):
    code: int | str = 0
    message: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def kind(self) -> ErrorKind:
        msg = self.message.lower()
        code = self.code if isinstance(self.code, int) else 0
        if code == 429 or "rate" in msg:
            return ErrorKind.RATE_LIMIT
        if code == 413 or "context" in msg or "too long" in msg or "too large" in msg:
            return ErrorKind.CONTEXT_OVERFLOW
        if code == 401 or code == 403:
            return ErrorKind.AUTH
        if code == 408 or "timeout" in msg:
            return ErrorKind.TIMEOUT
        if code == 400:
            return ErrorKind.INVALID_REQUEST
        if isinstance(self.code, int) and self.code >= 500:
            return ErrorKind.SERVER
        return ErrorKind.UNKNOWN

    @property
    def retryable(self) -> bool:
        return self.kind in (ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.SERVER)


# ── Model spec ───────────────────────────────────────────────────────────────

class ModelSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    id: str
    name: str = ""
    reasoning: Reasoning | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    providers: dict[str, object] | None = None
    verbosity: str | None = None
