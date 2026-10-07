"""Minimal async message bus for the ox runtime.

Port — async send/recv/close protocol
Msg — serializable message envelope
Envelope — internal wrapper with reply port
AsyncPort — in-process queue implementation
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

import msgspec

from ox.types import Message

T = TypeVar("T")


class Port(Protocol[T]):
    async def send(self, msg: T) -> None: ...
    async def recv(self) -> T: ...
    async def close(self) -> None: ...


class Msg(msgspec.Struct, array_like=True, frozen=True, gc=False):
    session_id: int
    data: Sequence[Message]
    metadata: dict[str, object] | None = None


class Envelope(msgspec.Struct, Generic[T], frozen=True):
    msg: Msg
    reply: Port[T] | None = None


class AsyncPort(Generic[T]):
    """In-process asyncio queue implementing the Port protocol."""

    class _Closed:
        __slots__ = ()

    _CLOSED = _Closed()

    __slots__ = ("_q",)

    def __init__(self, depth: int = 64) -> None:
        self._q: asyncio.Queue[T | AsyncPort._Closed] = asyncio.Queue(depth)

    async def send(self, msg: T) -> None:
        await self._q.put(msg)

    async def recv(self) -> T:
        item = await self._q.get()
        if isinstance(item, AsyncPort._Closed):
            self._q.put_nowait(self._CLOSED)
            raise EOFError("AsyncPort is closed")
        return item

    async def close(self) -> None:
        await self._q.put(self._CLOSED)

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        item = await self._q.get()
        if isinstance(item, AsyncPort._Closed):
            raise StopAsyncIteration
        return item  # type: ignore[return-value]
