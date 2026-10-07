"""HTTP provider for OpenAI-compatible LLM endpoints.

Uses aiohttp + msgspec for fast async requests with zero-copy
serialization. Works with OpenRouter, OpenAI, Anthropic (via proxy),
and any OpenAI-compatible local server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiohttp
import msgspec
import structlog

from ox.types import (
    ChatResponse,
    Error,
    ErrorResponse,
    Ok,
    Request,
    Result,
    StreamingResponse,
)

log = structlog.get_logger()


class HttpProvider:
    __slots__ = ("_base_url", "_api_key", "_session", "_enc", "_dec", "_dec_stream")

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None
        self._enc = msgspec.json.Encoder()
        self._dec = msgspec.json.Decoder(ChatResponse)
        self._dec_stream = msgspec.json.Decoder(StreamingResponse)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=300),
            )
        return self._session

    async def chat(self, request: Request) -> Result[ChatResponse, ErrorResponse]:
        url = f"{self._base_url}/chat/completions"
        body = self._enc.encode(request)

        try:
            session = self._get_session()
            async with session.post(url, data=body) as resp:
                raw = await resp.read()
                if resp.status != 200:
                    return Error(_map_error(resp.status, raw))
                return Ok(self._dec.decode(raw))

        except aiohttp.ClientError as e:
            log.warning("http_error", error=str(e))
            return Error(ErrorResponse(code=0, message=str(e)))
        except TimeoutError:
            return Error(ErrorResponse(code=408, message="Request timed out"))
        except msgspec.DecodeError as e:
            log.warning("decode_error", error=str(e), body=raw[:500])
            return Error(ErrorResponse(code=502, message=f"Failed to decode response: {e}"))

    async def chat_stream(
        self, request: Request,
    ) -> AsyncIterator[Result[StreamingResponse, ErrorResponse]]:
        url = f"{self._base_url}/chat/completions"
        body = self._enc.encode(request)

        try:
            session = self._get_session()
            async with session.post(url, data=body) as resp:
                if resp.status != 200:
                    raw = await resp.read()
                    yield Error(_map_error(resp.status, raw))
                    return

                buf = b""
                async for chunk in resp.content.iter_any():
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line or line == b":":
                            continue
                        if not line.startswith(b"data: "):
                            continue
                        payload = line[6:]
                        if payload == b"[DONE]":
                            return
                        try:
                            yield Ok(self._dec_stream.decode(payload))
                        except msgspec.DecodeError as e:
                            log.debug("stream_decode_skip", error=str(e))

        except aiohttp.ClientError as e:
            log.warning("stream_http_error", error=str(e))
            yield Error(ErrorResponse(code=0, message=str(e)))
        except TimeoutError:
            yield Error(ErrorResponse(code=408, message="Stream timed out"))

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


def _map_error(status: int, body: bytes) -> ErrorResponse:
    """Parse OpenAI-format error or fall back to raw text."""
    try:
        obj = msgspec.json.decode(body)
        if isinstance(obj, dict):
            err = obj.get("error", obj)
            if isinstance(err, dict):
                return ErrorResponse(code=status, message=err.get("message", str(err)))
            return ErrorResponse(code=status, message=str(err))
    except Exception:
        pass
    text = body[:500].decode(errors="replace")
    return ErrorResponse(code=status, message=text)
