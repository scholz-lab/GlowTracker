"""Loop guard — detect and break tool-calling loops.

Detects:
1. Repeated identical tool calls (same name + args, order-invariant)
2. Repeated identical outcomes (same call producing same result)
3. Ping-pong patterns of length 2..5 (A->B->A->B or A->B->C->A->B->C)
4. Circuit-breaker for runaway sessions

Tiered response: Allow -> Warn -> Block -> CircuitBreak.
Poll tools (status/tail/ps) get relaxed thresholds.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from enum import Enum

WARN_THRESHOLD = 3
BLOCK_THRESHOLD = 5
CIRCUIT_BREAKER = 30
POLL_MULTIPLIER = 3
OUTCOME_WARN = 2
OUTCOME_BLOCK = 3
PING_PONG_MIN_REPEATS = 3
MAX_WARNINGS_PER_CALL = 3
WINDOW_SIZE = 20
HISTORY_SIZE = 30
MAX_PATTERN_LEN = 5

_POLL_KEYWORDS = frozenset({
    "status", "poll", "wait", "watch", "tail", "ps",
    "jobs", "pgrep", "docker ps", "kubectl get",
})
_POLL_TOOLS = frozenset({"bash"})


class Verdict(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    CIRCUIT_BREAK = "circuit_break"


class CheckResult:
    __slots__ = ("verdict", "message")

    def __init__(self, verdict: Verdict, message: str = "") -> None:
        self.verdict = verdict
        self.message = message

    @property
    def allowed(self) -> bool:
        return self.verdict in (Verdict.ALLOW, Verdict.WARN)


def _normalize(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def _call_hash(tool_name: str, args: dict | str) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            pass
    normalized = json.dumps(_normalize(args), sort_keys=True, separators=(",", ":"))
    raw = f"{tool_name}|{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _outcome_hash(call_h: str, result: str) -> str:
    raw = f"{call_h}|{result}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_poll_call(tool_name: str, args: dict | str) -> bool:
    if tool_name not in _POLL_TOOLS:
        return False
    args_str = args if isinstance(args, str) else json.dumps(args)
    return any(kw in args_str.lower() for kw in _POLL_KEYWORDS)


class LoopGuard:
    __slots__ = (
        "_turn", "_call_history", "_outcome_counts",
        "_recent_calls", "_blocked_outcomes", "_warnings_emitted",
    )

    def __init__(self) -> None:
        self._turn: int = 0
        self._call_history: dict[str, deque[int]] = {}
        self._outcome_counts: dict[str, int] = {}
        self._blocked_outcomes: set[str] = set()
        self._recent_calls: deque[str] = deque(maxlen=HISTORY_SIZE)
        self._warnings_emitted: dict[str, int] = {}

    def check(self, tool_name: str, args: dict | str) -> CheckResult:
        self._turn += 1
        ch = _call_hash(tool_name, args)
        self._recent_calls.append(ch)

        if ch in self._blocked_outcomes:
            return CheckResult(
                Verdict.BLOCK,
                f"Blocked: {tool_name} with these args has produced the same result "
                f"{OUTCOME_BLOCK} times. Try a different approach.",
            )

        turns = self._call_history.setdefault(ch, deque())
        window_start = self._turn - WINDOW_SIZE
        while turns and turns[0] <= window_start:
            turns.popleft()

        is_repeat = len(turns) > 0
        turns.append(self._turn)
        windowed_count = len(turns)

        if is_repeat:
            total_repeats = sum(
                len(t) - 1 for t in self._call_history.values() if len(t) > 1
            )
            if total_repeats >= CIRCUIT_BREAKER:
                return CheckResult(
                    Verdict.CIRCUIT_BREAK,
                    f"Circuit breaker: {total_repeats} repeated tool calls in session. "
                    "The agent appears stuck in a loop. Stopping tool execution.",
                )

        is_poll = _is_poll_call(tool_name, args)
        mult = POLL_MULTIPLIER if is_poll else 1
        effective_warn = WARN_THRESHOLD * mult
        effective_block = BLOCK_THRESHOLD * mult

        if windowed_count >= effective_block:
            return CheckResult(
                Verdict.BLOCK,
                f"Blocked: {tool_name} called {windowed_count} times in {WINDOW_SIZE} turns "
                f"with identical arguments. Try a different approach.",
            )
        if windowed_count >= effective_warn:
            warned = self._warnings_emitted.get(ch, 0)
            if warned >= MAX_WARNINGS_PER_CALL:
                return CheckResult(
                    Verdict.BLOCK,
                    f"Blocked: {tool_name} called {windowed_count} times with identical arguments "
                    f"(warned {warned} times already). Try a different approach.",
                )
            self._warnings_emitted[ch] = warned + 1
            return CheckResult(
                Verdict.WARN,
                f"Warning: {tool_name} called {windowed_count} times in {WINDOW_SIZE} turns "
                f"with identical arguments. Consider trying a different approach.",
            )

        pp = self._detect_ping_pong()
        if pp is not None:
            desc, repeats = pp
            if repeats >= PING_PONG_MIN_REPEATS:
                return CheckResult(
                    Verdict.BLOCK,
                    f"Blocked: detected repeating pattern ({desc}) "
                    f"repeated {repeats} times. Break the cycle.",
                )
            elif repeats >= 2:
                return CheckResult(
                    Verdict.WARN,
                    f"Warning: possible repeating pattern ({desc}) "
                    f"detected {repeats} times.",
                )

        return CheckResult(Verdict.ALLOW)

    def record_outcome(self, tool_name: str, args: dict | str, result: str) -> str | None:
        ch = _call_hash(tool_name, args)
        oh = _outcome_hash(ch, result)
        count = self._outcome_counts.get(oh, 0) + 1
        self._outcome_counts[oh] = count

        if count >= OUTCOME_BLOCK:
            self._blocked_outcomes.add(ch)
            return (
                f"WARNING: {tool_name} has produced the exact same result {count} times. "
                "This call will be blocked on the next attempt. Try a different approach."
            )
        elif count >= OUTCOME_WARN:
            return (
                f"NOTE: {tool_name} has produced the same result {count} times with these arguments. "
                "Consider changing your approach."
            )
        return None

    def _detect_ping_pong(self) -> tuple[str, int] | None:
        calls = list(self._recent_calls)
        n = len(calls)
        for pattern_len in range(2, MAX_PATTERN_LEN + 1):
            if n < pattern_len * 2:
                continue
            pattern = calls[n - pattern_len:]
            if len(set(pattern)) < 2:
                continue
            repeats = 1
            pos = n - pattern_len * 2
            while pos >= 0:
                segment = calls[pos: pos + pattern_len]
                if segment == pattern:
                    repeats += 1
                    pos -= pattern_len
                else:
                    break
            if repeats >= 2:
                desc = "->".join(h[:6] for h in pattern)
                return desc, repeats
        return None

    def reset(self) -> None:
        self._turn = 0
        self._call_history.clear()
        self._outcome_counts.clear()
        self._blocked_outcomes.clear()
        self._recent_calls.clear()
        self._warnings_emitted.clear()
