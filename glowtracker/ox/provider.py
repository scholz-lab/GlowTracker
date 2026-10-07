"""LLM provider abstraction.

Any LLM backend (Anthropic, OpenAI, OpenRouter, local) implements this
protocol. The agent loop only depends on this interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ox.types import (
    ChatResponse,
    ErrorResponse,
    Request,
    Result,
    StreamingResponse,
)


class Provider(Protocol):
    async def chat(self, request: Request) -> Result[ChatResponse, ErrorResponse]: ...
    def chat_stream(self, request: Request) -> AsyncIterator[Result[StreamingResponse, ErrorResponse]]: ...
    async def close(self) -> None: ...
