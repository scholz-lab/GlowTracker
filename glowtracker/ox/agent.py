"""Async multi-agent LLM runtime.

Runs N agent sessions concurrently. Each session is an independent LLM
conversation loop with tool execution.

By default a session ends after its final reply (one task per session). With
persistent=True a session survives between user turns, so later messages with
the same session_id continue the conversation (chat mode); the end of each
turn is signalled with a reply whose metadata contains {"turn_done": True}.
A running turn can be stopped with cancel(session_id).

Streaming replies carry metadata "stream_delta" (text in content) or
"reasoning_delta" (text in reasoning), then "stream_done" with the full message
(and "usage" when the provider reports it).

Tools registered with confirm=True pause the turn: the reply port gets a message
with metadata {"approval_request": {"call_id", "name", "arguments", "details"}},
and the turn continues when the app calls answer_approval(session_id, call_id, Approval).
"""

from __future__ import annotations

import asyncio
import os
import re
import traceback

import msgspec
import structlog

from ox.bus import AsyncPort, Envelope, Msg, Port
from ox.compact import compact, needs_compaction
from ox.loop_guard import Verdict
from ox.session import IterationBudget, Session
from ox.tool_output import format_tool_output
from ox.tools import ToolResolver
from ox.provider import Provider
from ox.scheduler import Scheduler, ScheduledEvent
from ox.types import (
    Approval,
    AssistantMessage,
    CacheControl,
    Error,
    ErrorKind,
    FunctionCall,
    ModelSpec,
    SystemMessage,
    Ok,
    Request,
    ToolCallWire,
    ToolMessage,
    UserMessage,
)

_MAX_TOOL_ITERATIONS = 20
_HINT_RE = re.compile(r"\nhint\[\d+\]:\n(?:  .+\n?)+\Z")


class AgentServer:
    __slots__ = (
        "_port", "_provider", "_tools", "_log",
        "_sessions", "_model", "_workspace", "_scheduler",
        "_provider_retries", "_persistent", "_compactor_model", "_keep_reasoning",
        "_approvals",
    )

    def __init__(
        self,
        port: Port,
        provider: Provider,
        model: ModelSpec,
        tools: ToolResolver | None = None,
        workspace: str | None = None,
        scheduler: Scheduler | None = None,
        persistent: bool = False,
        compactor_model: str | None = None,
        keep_reasoning: bool = True,
    ) -> None:
        self._port = port
        self._provider = provider
        self._model = model
        self._tools = tools or ToolResolver()
        self._log = structlog.get_logger()
        self._sessions: dict[int, Session] = {}
        self._workspace = workspace or os.getcwd()
        self._scheduler = scheduler
        self._provider_retries: dict[int, int] = {}
        self._persistent = persistent
        self._compactor_model = compactor_model
        self._keep_reasoning = keep_reasoning  # send reasoning back in the history
        self._approvals: dict[tuple[int, str], asyncio.Future[Approval]] = {}

    # ── Run loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._log.info("ox_starting")
        if self._scheduler is None:
            await self._run_simple()
            return

        scheduler_task: asyncio.Task | None = None

        async def _next_scheduled() -> ScheduledEvent:
            assert self._scheduler is not None
            return await self._scheduler.__anext__()

        while True:
            inbox_task = asyncio.ensure_future(self._port.recv())
            if scheduler_task is None or scheduler_task.done():
                scheduler_task = asyncio.ensure_future(_next_scheduled())

            done, _ = await asyncio.wait(
                {inbox_task, scheduler_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                if task is inbox_task:
                    await self._dispatch(task.result())
                elif task is scheduler_task:
                    await self._handle_scheduled(task.result())
                    scheduler_task = None

            if not inbox_task.done():
                inbox_task.cancel()

    async def _handle_scheduled(self, event: ScheduledEvent) -> None:
        self._log.info("scheduled_event", kind=event.kind.value)
        sid = event.payload.get("session_id")
        prompt = event.payload.get("prompt", "")
        if sid is not None and prompt:
            msg = Msg(session_id=sid, data=[UserMessage(content=prompt)])
            await self._dispatch(Envelope(msg=msg))

    async def _run_simple(self) -> None:
        while True:
            envelope = await self._port.recv()
            await self._dispatch(envelope)

    # ── Session management ──────────────────────────────────────────────────

    async def _dispatch(self, envelope: Envelope) -> None:
        sid = envelope.msg.session_id
        if sid not in self._sessions:
            max_turns = 0
            if envelope.msg.metadata:
                max_turns = int(envelope.msg.metadata.get("max_turns", 0) or 0)
            session = Session(sid, self._model, max_turns=max_turns)
            self._sessions[sid] = session
            self._log.info("session_created", session_id=sid, model=self._model.id)

        session = self._sessions[sid]
        await session._queue.put(envelope)
        if session._worker is None:
            session._worker = asyncio.create_task(
                self._session_worker(session), name=f"session-{sid}"
            )

    async def _end_session(self, session: Session) -> None:
        session._ended = True
        if session._reply is not None:
            try:
                await session._reply.close()
            except Exception:
                pass
            session._reply = None
        self._provider_retries.pop(session.index, None)
        self._sessions.pop(session.index, None)
        session._worker = None
        self._log.info("session_ended", session_id=session.index)

    async def _session_worker(self, session: Session) -> None:
        try:
            while not session._ended:
                envelope = await session._queue.get()
                session._reply = envelope.reply
                if any(isinstance(m, (UserMessage, SystemMessage)) for m in envelope.msg.data):
                    session._turn_start = len(session.messages)
                    session._turn_input = list(envelope.msg.data)
                session._current = asyncio.ensure_future(self._process(session, envelope))
                try:
                    await session._current
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise  # the worker itself is being shut down
                    await self._after_cancel(session, envelope)
                except Exception as e:
                    self._log.error("process_error", session_id=session.index, error=str(e))
                    traceback.print_exc()
                    if not self._persistent:
                        break
                    error_msg = AssistantMessage(content=f"[Internal error]: {type(e).__name__}: {e}")
                    await self._send_reply(envelope, Msg(session_id=session.index, data=[error_msg]))
                    await self._finish_turn(session, envelope)
                finally:
                    session._current = None
        finally:
            await self._end_session(session)

    async def answer_approval(self, session_id: int, call_id: str, approval: Approval) -> bool:
        """The user's answer to an approval request. False if nothing was waiting for it."""
        fut = self._approvals.get((session_id, call_id))
        if fut is None or fut.done():
            return False
        fut.set_result(approval)
        return True

    async def _request_approval(self, session: Session, envelope: Envelope, call_id: str,
                                name: str, arguments: dict, details: str) -> Approval:
        fut: asyncio.Future[Approval] = asyncio.get_running_loop().create_future()
        key = (session.index, call_id)
        self._approvals[key] = fut
        self._log.info("approval_requested", session_id=session.index, tool=name)
        await self._send_reply(envelope, Msg(
            session_id=session.index, data=[],
            metadata={"approval_request": {"call_id": call_id, "name": name,
                                           "arguments": arguments, "details": details}},
        ))
        try:
            return await fut
        finally:
            self._approvals.pop(key, None)

    async def cancel(self, session_id: int) -> bool:
        """Stop the running turn of a session. Returns False if nothing was running."""
        session = self._sessions.get(session_id)
        if session is None or session._current is None or session._current.done():
            return False
        session._current.cancel()
        return True

    async def _after_cancel(self, session: Session, envelope: Envelope) -> None:
        """Drop the stopped turn's partial work, keeping what the user said, so the
        history stays valid (no tool calls without results)."""
        while not session._queue.empty():
            session._queue.get_nowait()  # continuations of the stopped turn
        session.messages = (
            session.messages[:session._turn_start]
            + session._turn_input
            + [AssistantMessage(content="(Stopped by the user before finishing.)")]
        )
        self._log.info("turn_cancelled", session_id=session.index)
        if self._persistent:
            session.tool_iterations = 0
            session.turn_count = 0
            session.iteration_budget = IterationBudget(session.max_turns)
            session.loop_guard.reset()
            await self._send_reply(envelope, Msg(
                session_id=session.index, data=[],
                metadata={"turn_done": True, "cancelled": True},
            ))
        else:
            session._ended = True

    # ── LLM turn ────────────────────────────────────────────────────────────

    async def _process(self, session: Session, envelope: Envelope) -> None:
        session.messages += list(envelope.msg.data)

        session.turn_count += 1
        session.iteration_budget.consume()
        at_limit = (
            session.turn_count >= session.max_turns
            or session.iteration_budget.exhausted
        )

        if needs_compaction(session.prompt_tokens):
            compacted = await compact(session.messages, self._provider, model=self._compactor_model)
            if compacted is not None:
                session.messages = compacted

        tools = None if at_limit else (self._tools.definitions() or None)
        stream = bool((envelope.msg.metadata or {}).get("stream"))
        is_anthropic = self._model.id.startswith("anthropic/")

        request = Request(
            model=session.model.id,
            messages=session.messages,
            cache_control=CacheControl() if is_anthropic else None,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            tools=tools,
            reasoning=session.model.reasoning,
            temperature=session.model.temperature,
            top_p=session.model.top_p,
            top_k=session.model.top_k,
            max_tokens=session.model.max_tokens or 16_000,
            provider=session.model.providers,
            user=str(session.index),
        )

        if request.stream:
            await self._process_stream(session, envelope, request)
        else:
            await self._process_batch(session, envelope, request)

    async def _process_batch(self, session: Session, envelope: Envelope, request: Request) -> None:
        match await self._provider.chat(request):
            case Ok(response):
                message = response.choices[0].message if response.choices else AssistantMessage()
                if message.reasoning_content or (message.reasoning and not self._keep_reasoning):
                    # reasoning_content must not be sent back; keep `reasoning` only if asked to.
                    message = msgspec.structs.replace(
                        message, reasoning_content=None,
                        reasoning=(message.reasoning or message.reasoning_content) if self._keep_reasoning else None,
                    )
                if response.usage:
                    session.prompt_tokens = response.usage.prompt_tokens
                self._log_reply(message, response.usage, response.model)
                session.messages.append(message)

                msg = Msg(
                    session_id=envelope.msg.session_id,
                    data=[message],
                    metadata={"usage": response.usage} if response.usage else None,
                )
                await self._send_reply(envelope, msg)

                match message:
                    case AssistantMessage(tool_calls=tool_calls) if tool_calls:
                        await self._handle_tools(session, envelope, tool_calls)
                    case _:
                        await self._finish_turn(session, envelope)

            case Error(err):
                await self._handle_error(session, envelope, err)

    async def _process_stream(self, session: Session, envelope: Envelope, request: Request) -> None:
        text_chunks: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        usage = None
        response_model = ""
        reasoning_chunks: list[str] = []

        async for result in self._provider.chat_stream(request):
            match result:
                case Ok(chunk):
                    usage = chunk.usage or usage
                    response_model = chunk.model or response_model
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if isinstance(delta, AssistantMessage):
                        delta_text = ""
                        if isinstance(delta.content, str) and delta.content:
                            delta_text = delta.content
                        elif isinstance(delta.content, list):
                            delta_text = "".join(getattr(p, "text", "") for p in delta.content)
                        reasoning = delta.reasoning or delta.reasoning_content
                        if reasoning:
                            reasoning_chunks.append(reasoning)
                            await self._send_reply(envelope, Msg(
                                session_id=envelope.msg.session_id,
                                data=[AssistantMessage(reasoning=reasoning)],
                                metadata={"reasoning_delta": True},
                            ))
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index if tc.index is not None else 0
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                                entry = tool_calls_map[idx]
                                if tc.id:
                                    entry["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        entry["name"] += tc.function.name
                                    if tc.function.arguments:
                                        entry["arguments"] += tc.function.arguments
                        if delta_text:
                            text_chunks.append(delta_text)
                            delta_msg = Msg(
                                session_id=envelope.msg.session_id,
                                data=[AssistantMessage(content=delta_text)],
                                metadata={"stream_delta": True},
                            )
                            await self._send_reply(envelope, delta_msg)

                case Error(err):
                    await self._handle_error(session, envelope, err)
                    return

        full_text = "".join(text_chunks)
        final_tool_calls = None
        if tool_calls_map:
            final_tool_calls = [
                ToolCallWire(
                    id=entry["id"], type="function",
                    # a call without arguments is replayed as "{}": chat templates parse this as JSON
                    function=FunctionCall(name=entry["name"], arguments=entry["arguments"] or "{}"),
                )
                for _idx, entry in sorted(tool_calls_map.items())
            ]

        message = AssistantMessage(
            content=full_text or None,
            tool_calls=final_tool_calls or None,
            reasoning=("".join(reasoning_chunks) or None) if self._keep_reasoning else None,
        )

        if usage:
            session.prompt_tokens = usage.prompt_tokens
        self._log_reply(message, usage, response_model)
        session.messages.append(message)

        meta: dict[str, object] = {"stream_done": True}
        if usage:
            meta["usage"] = usage
        await self._send_reply(envelope, Msg(
            session_id=envelope.msg.session_id,
            data=[message],
            metadata=meta,
        ))

        match message:
            case AssistantMessage(tool_calls=tool_calls) if tool_calls:
                await self._handle_tools(session, envelope, tool_calls)
            case _:
                await self._finish_turn(session, envelope)

    # ── Error handling ──────────────────────────────────────────────────────

    async def _handle_error(self, session: Session, envelope: Envelope, err) -> None:
        self._log.error("provider_error", code=err.code, kind=err.kind.value, msg=err.message)

        if err.kind is ErrorKind.CONTEXT_OVERFLOW and self._try_truncate(session):
            self._log.info("context_overflow_recovery", session_id=session.index)
            await session._queue.put(self._retry_envelope(envelope))
            return

        retries = self._provider_retries.get(session.index, 0)
        if err.retryable and retries < 3:
            self._provider_retries[session.index] = retries + 1
            delay = 2 ** retries
            self._log.info("provider_retry", attempt=retries + 1, delay=delay)
            await asyncio.sleep(delay)
            await session._queue.put(self._retry_envelope(envelope))
            return

        if session.messages and envelope.msg.data:
            session.messages.pop()
        error_msg = AssistantMessage(content=f"[LLM Error {err.code}]: {err.message}")
        await self._send_reply(envelope, Msg(session_id=envelope.msg.session_id, data=[error_msg]))
        await self._finish_turn(session, envelope)

    async def _finish_turn(self, session: Session, envelope: Envelope) -> None:
        """The model gave its final answer (or failed): end the session, or in
        persistent mode reset the per-turn budgets and wait for the next message."""
        session.tool_iterations = 0
        self._provider_retries.pop(session.index, None)
        if not self._persistent:
            session._ended = True
            return
        session.turn_count = 0
        session.iteration_budget = IterationBudget(session.max_turns)
        session.loop_guard.reset()
        await self._send_reply(envelope, Msg(
            session_id=envelope.msg.session_id, data=[], metadata={"turn_done": True},
        ))

    def _retry_envelope(self, envelope: Envelope) -> Envelope:
        return Envelope(
            msg=Msg(session_id=envelope.msg.session_id, data=[], metadata=envelope.msg.metadata),
            reply=envelope.reply,
        )

    def _try_truncate(self, session: Session, max_content: int = 2000) -> bool:
        replaced = False
        for i in range(len(session.messages) - 1, -1, -1):
            msg = session.messages[i]
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > max_content:
                session.messages[i] = ToolMessage(
                    tool_call_id=msg.tool_call_id,
                    content=f"ERROR: Result too large ({len(msg.content)} chars). Use bash with head/tail/jq to extract smaller portions.",
                )
                replaced = True
        return replaced

    # ── Tool execution ──────────────────────────────────────────────────────

    async def _handle_tools(self, session: Session, envelope: Envelope, tool_calls) -> None:
        session.tool_iterations += 1
        if session.tool_iterations > _MAX_TOOL_ITERATIONS:
            self._log.warning("tool_iteration_limit", session_id=session.index)
            session.messages.append(UserMessage(
                content=f"SYSTEM: Tool iteration limit ({_MAX_TOOL_ITERATIONS}) reached. "
                "Do NOT call any more tools. Summarize what you have accomplished.",
            ))
            session.turn_count = session.max_turns
            await session._queue.put(self._retry_envelope(envelope))
            return

        guard = session.loop_guard
        _dedup: dict[tuple[str, str], asyncio.Future[ToolMessage]] = {}

        async def _exec_one(tc: ToolCallWire) -> ToolMessage:
            name = tc.function.name if tc.function else "?"
            args = tc.function.arguments if tc.function else "{}"
            dedup_key = (name, args)

            if dedup_key in _dedup:
                original = await _dedup[dedup_key]
                return ToolMessage(tool_call_id=tc.id, content=original.content)

            check = guard.check(name, args)
            if check.verdict in (Verdict.CIRCUIT_BREAK, Verdict.BLOCK):
                return ToolMessage(tool_call_id=tc.id, content=f"ERROR: {check.message}")

            fut: asyncio.Future[ToolMessage] = asyncio.get_running_loop().create_future()
            _dedup[dedup_key] = fut

            result = await self._tools.execute(
                name, args,
                approve=lambda n, a, details: self._request_approval(session, envelope, tc.id, n, a, details),
            )
            outcome_warn = guard.record_outcome(name, args, result.content)

            if result.is_error:
                content = f"ERROR\n{result.content}"
            else:
                content = format_tool_output(result.content, tool_name=name)
            if check.verdict is Verdict.WARN:
                content = f"{content}\n\n⚠ {check.message}"
            if outcome_warn:
                content = f"{content}\n\n⚠ {outcome_warn}"

            tool_msg = ToolMessage(tool_call_id=tc.id, content=content)
            fut.set_result(tool_msg)
            return tool_msg

        tool_messages = list(await asyncio.gather(*(_exec_one(tc) for tc in tool_calls)))

        # Deduplicate hints: keep only on the last result per tool type
        if len(tool_messages) > 1:
            last_of_type: dict[str, int] = {}
            for i, tc in enumerate(tool_calls):
                last_of_type[tc.function.name if tc.function else ""] = i
            keep_hint = set(last_of_type.values())
            tool_messages = [
                ToolMessage(tool_call_id=tm.tool_call_id, content=_HINT_RE.sub("", tm.content))
                if i not in keep_hint and isinstance(tm.content, str) else tm
                for i, tm in enumerate(tool_messages)
            ]

        # Budget pressure warnings
        effective = min(
            session.max_turns - session.turn_count,
            session.iteration_budget.remaining,
        )
        if tool_messages and effective > 0 and session.max_turns:
            pct_used = 1.0 - (effective / session.max_turns)
            warning = ""
            if pct_used >= 0.9:
                warning = f"\n\n⚠ URGENT: {effective} turns remaining. Finish and provide your final response NOW."
            elif pct_used >= 0.7:
                warning = f"\n\n⚠ CAUTION: {effective} turns remaining. Start wrapping up."
            if warning:
                last = tool_messages[-1]
                if isinstance(last.content, str):
                    tool_messages[-1] = ToolMessage(tool_call_id=last.tool_call_id, content=last.content + warning)

        msg = Msg(session_id=envelope.msg.session_id, data=tool_messages, metadata=envelope.msg.metadata)
        await self._send_reply(envelope, msg)
        await session._queue.put(Envelope(msg=msg, reply=envelope.reply))

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _log_reply(self, message, usage, model) -> None:
        if isinstance(message, AssistantMessage):
            text = ""
            if isinstance(message.content, str):
                text = message.content[:500]
            elif isinstance(message.content, list):
                text = "".join(getattr(p, "text", "") for p in message.content)[:500]
            self._log.info("llm_reply", text=text)
            if message.tool_calls:
                for tc in message.tool_calls:
                    self._log.info("llm_tool_call", tool=tc.function.name if tc.function else "?")
        self._log.info(
            "llm_usage",
            input=usage.prompt_tokens if usage else 0,
            output=usage.completion_tokens if usage else 0,
            model=model,
        )

    async def _send_reply(self, envelope: Envelope, reply_msg: Msg) -> None:
        if envelope.reply is not None:
            await envelope.reply.send(reply_msg)
