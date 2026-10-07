"""Tickless priority-queue scheduler.

Sleeps exactly until the next event is due — no polling, no fixed intervals.
Efficient for sub-second precision across many agents.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(Enum):
    CRON = "cron"
    TASK_CHECK = "task_check"
    SESSION_TIMEOUT = "session_timeout"


@dataclass(order=True)
class ScheduledEvent:
    deadline: float
    kind: EventKind = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class Scheduler:
    __slots__ = ("_heap", "_event", "_closed")

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent] = []
        self._event = asyncio.Event()
        self._closed = False

    def push(self, event: ScheduledEvent) -> None:
        heapq.heappush(self._heap, event)
        self._event.set()

    def push_at(self, deadline: float, kind: EventKind, payload: dict[str, Any] | None = None) -> None:
        self.push(ScheduledEvent(deadline=deadline, kind=kind, payload=payload or {}))

    def push_after(self, delay: float, kind: EventKind, payload: dict[str, Any] | None = None) -> None:
        self.push_at(time.monotonic() + delay, kind, payload)

    @property
    def pending(self) -> int:
        return len(self._heap)

    def next_deadline(self) -> float | None:
        return self._heap[0].deadline if self._heap else None

    def drain_due(self) -> list[ScheduledEvent]:
        now = time.monotonic()
        due: list[ScheduledEvent] = []
        while self._heap and self._heap[0].deadline <= now:
            due.append(heapq.heappop(self._heap))
        return due

    def close(self) -> None:
        self._closed = True
        self._event.set()

    def __aiter__(self):
        return self

    async def __anext__(self) -> ScheduledEvent:
        while True:
            if self._closed:
                raise StopAsyncIteration

            now = time.monotonic()
            if self._heap and self._heap[0].deadline <= now:
                return heapq.heappop(self._heap)

            self._event.clear()
            if self._heap:
                wait = max(0, self._heap[0].deadline - now)
                try:
                    await asyncio.wait_for(self._event.wait(), timeout=wait)
                except TimeoutError:
                    pass
            else:
                await self._event.wait()
