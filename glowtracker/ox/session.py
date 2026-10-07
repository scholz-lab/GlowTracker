"""Agent session state."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ox.loop_guard import LoopGuard

if TYPE_CHECKING:
    from ox.bus import Envelope, Port
    from ox.types import Message, ModelSpec

_DEFAULT_MAX_TURNS = 90


class IterationBudget:
    __slots__ = ("_remaining",)

    def __init__(self, total: int) -> None:
        self._remaining = total

    @property
    def remaining(self) -> int:
        return self._remaining

    def consume(self, n: int = 1) -> bool:
        if self._remaining <= 0:
            return False
        self._remaining = max(0, self._remaining - n)
        return True

    @property
    def exhausted(self) -> bool:
        return self._remaining <= 0


class Session:
    __slots__ = (
        "model", "messages", "index", "max_turns", "turn_count",
        "tool_iterations", "_queue", "_worker", "_ended", "_reply",
        "prompt_tokens", "iteration_budget", "loop_guard",
        "_current", "_turn_start", "_turn_input",
    )

    def __init__(self, index: int, model: ModelSpec, max_turns: int = 0) -> None:
        self.model = model
        self.messages: list[Message] = []
        self.index = index
        self.max_turns = max_turns or _DEFAULT_MAX_TURNS
        self.turn_count = 0
        self.tool_iterations = 0
        self._queue: asyncio.Queue[Envelope] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._ended: bool = False
        self._reply: Port | None = None
        self.prompt_tokens: int = 0
        self.iteration_budget = IterationBudget(self.max_turns)
        self.loop_guard = LoopGuard()
        self._current: asyncio.Task[None] | None = None  # the running _process step
        self._turn_start = 0                             # len(messages) before this turn
        self._turn_input: list[Message] = []             # what started this turn
