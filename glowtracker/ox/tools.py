"""Tool execution — builtins, CLI tools, in-process functions, resolver, and discovery.

Builtins: bash, read, write, edit (can be disabled).
CLI tools: external commands discovered via --tools flag.
Functions: Python callables registered by an embedding application.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from typing import Any

import msgspec
import structlog

from ox.types import (
    FunctionDescription,
    Tool,
    ToolResult,
    ToolSpec,
    tool_error,
    tool_ok,
)

log = structlog.get_logger()


# ── Subprocess helpers ──────────────────────────────────────────────────────

async def _run_shell(command: str, timeout: float) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr). Kills on timeout."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace") if stderr else ""


async def _run_exec(cmd: list[str], stdin_data: bytes | None, timeout: float) -> tuple[int, str, str]:
    """Run an exec command, return (returncode, stdout, stderr). Kills on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_data else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin_data), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace") if stderr else ""


def _format_process_result(returncode: int, out: str, err: str) -> ToolResult:
    """Shared formatting for subprocess results."""
    if returncode != 0:
        parts = [f"exit_code: {returncode}"]
        if out.strip():
            parts.append(f"stdout:\n{out}")
        if err.strip():
            parts.append(f"stderr:\n{err}")
        return tool_error("\n".join(parts))

    if not out.strip() and not err.strip():
        return tool_ok("(no output)")

    output = out
    if err.strip():
        output += f"\nstderr:\n{err}"
    return tool_ok(output)


# ── Builtins ────────────────────────────────────────────────────────────────

async def _exec_bash(args: dict) -> ToolResult:
    command = args.get("command", "")
    timeout = args.get("timeout", 120)
    try:
        rc, out, err = await _run_shell(command, timeout)
        return _format_process_result(rc, out, err)
    except TimeoutError:
        return tool_error(f"error: command timed out after {timeout}s")
    except Exception as e:
        return tool_error(f"error: {e}")


async def _exec_read(args: dict) -> ToolResult:
    path = args.get("path", args.get("file_path", ""))
    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)
    grep = args.get("grep", "")
    try:
        with open(path) as f:
            all_lines = f.readlines()
        total = len(all_lines)
        if total == 0:
            return tool_ok(f"(empty file) {path}")

        if grep:
            try:
                pattern = re.compile(grep, re.IGNORECASE)
            except re.error:
                pattern = re.compile(re.escape(grep), re.IGNORECASE)
            matched = [(i, line) for i, line in enumerate(all_lines) if pattern.search(line)]
            if not matched:
                return tool_ok(f"(no matches for {grep!r} in {path}, {total} lines)")
            selected = matched[:limit]
            header = f"lines: {total}, matches: {len(matched)}, showing: {len(selected)}\n"
            text = "".join(f"{i + 1}\t{line}" for i, line in selected)
            if len(matched) > limit:
                text += f"\n... ({len(matched)} matches total — use limit to see more)"
        else:
            selected = all_lines[offset : offset + limit]
            header = f"lines: {total}, showing: {offset + 1}-{min(offset + len(selected), total)}\n"
            text = "".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(selected))
            if total > offset + limit:
                text += f"\n... ({total} lines total — use offset/limit for more)"
        return tool_ok(header + text)
    except Exception as e:
        return tool_error(f"error: {e}")


async def _exec_write(args: dict) -> ToolResult:
    path = args.get("path", args.get("file_path", ""))
    content = args.get("content", "")
    try:
        with open(path, "w") as f:
            f.write(content)
        return tool_ok(f"Wrote {len(content)} bytes to {path}")
    except Exception as e:
        return tool_error(f"error: {e}")


async def _exec_edit(args: dict) -> ToolResult:
    path = args.get("path", args.get("file_path", ""))
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    try:
        if not old:
            return tool_error("old_string must not be empty")
        with open(path) as f:
            content = f.read()
        if old not in content:
            return tool_error(f"old_string not found in {path}")
        pos = content.index(old)
        content = content[:pos] + new + content[pos + len(old):]
        with open(path, "w") as f:
            f.write(content)
        # Return changed region with ±3 lines context
        lines = content.splitlines(keepends=True)
        change_start = content[:pos].count("\n")
        change_end = change_start + new.count("\n")
        ctx_start = max(0, change_start - 3)
        ctx_end = min(len(lines), change_end + 4)
        snippet = "".join(
            f"{ctx_start + i + 1}\t{lines[ctx_start + i]}"
            for i in range(ctx_end - ctx_start)
            if ctx_start + i < len(lines)
        )
        return tool_ok(f"Edited {path}\n{snippet}")
    except Exception as e:
        return tool_error(f"error: {e}")


_BUILTINS: dict[str, Any] = {
    "bash": _exec_bash,
    "read": _exec_read,
    "write": _exec_write,
    "edit": _exec_edit,
}

_BUILTIN_DEFS: list[Tool] = [
    Tool(function=FunctionDescription(
        name="bash",
        description="Run a shell command",
        parameters={"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        }, "required": ["command"]},
    )),
    Tool(function=FunctionDescription(
        name="read",
        description="Read a file. Returns numbered lines. Use grep to filter lines by pattern. Use offset/limit to paginate.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "Start line (0-based, default 0)"},
            "limit": {"type": "integer", "description": "Max lines to return (default 2000)"},
            "grep": {"type": "string", "description": "Regex pattern to filter matching lines"},
        }, "required": ["path"]},
    )),
    Tool(function=FunctionDescription(
        name="write",
        description="Write content to a file",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    )),
    Tool(function=FunctionDescription(
        name="edit",
        description="Replace a string in a file",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        }, "required": ["path", "old_string", "new_string"]},
    )),
]


# ── CLI tool execution ──────────────────────────────────────────────────────

async def _exec_cli_tool(spec: ToolSpec, args: dict) -> ToolResult:
    """Execute a CLI tool via subprocess."""
    try:
        cmd = list(spec.command)
        stdin_data = None

        if spec.stdin_json:
            stdin_data = json.dumps(args).encode()
        else:
            for key, val in args.items():
                if isinstance(val, bool):
                    if val:
                        cmd.append(f"--{key}")
                elif val is not None:
                    cmd.append(f"--{key}")
                    cmd.append(str(val))

        rc, out, err = await _run_exec(cmd, stdin_data, spec.timeout)
        return _format_process_result(rc, out, err)
    except TimeoutError:
        return tool_error(f"error: {spec.name} timed out after {spec.timeout}s")
    except Exception as e:
        return tool_error(f"error: {spec.name} failed: {e}")


# ── Resolver ────────────────────────────────────────────────────────────────

class FunctionTool:
    """A tool backed by a Python callable (sync or async).

    The callable receives the decoded arguments as keyword arguments and returns
    a str or a ToolResult. Exceptions become tool errors.
    """

    __slots__ = ("name", "description", "parameters", "fn")

    def __init__(self, name: str, description: str, fn: Callable[..., Any],
                 parameters: dict[str, object] | None = None) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        self.fn = fn

    def to_tool(self) -> Tool:
        return Tool(function=FunctionDescription(
            name=self.name, parameters=self.parameters, description=self.description,
        ))

    async def execute(self, args: dict) -> ToolResult:
        try:
            result = self.fn(**args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            return tool_error(f"error: {type(e).__name__}: {e}")
        if isinstance(result, ToolResult):
            return result
        return tool_ok("(no output)" if result is None or result == "" else str(result))


class ToolResolver:
    """Resolves tool names to builtins, in-process functions or CLI specs."""

    __slots__ = ("_cli_tools", "_functions", "_builtins")

    def __init__(
        self,
        cli_tools: dict[str, ToolSpec] | None = None,
        builtins: bool = True,
    ) -> None:
        self._cli_tools = cli_tools or {}
        self._functions: dict[str, FunctionTool] = {}
        self._builtins = builtins

    def add(self, spec: ToolSpec) -> None:
        self._cli_tools[spec.name] = spec

    def add_function(self, name: str, description: str, fn: Callable[..., Any],
                     parameters: dict[str, object] | None = None) -> None:
        self._functions[name] = FunctionTool(name, description, fn, parameters)

    def definitions(self) -> list[Tool]:
        builtins = list(_BUILTIN_DEFS) if self._builtins else []
        return (builtins
                + [f.to_tool() for f in self._functions.values()]
                + [s.to_tool() for s in self._cli_tools.values()])

    async def execute(self, name: str, args_raw: str | dict) -> ToolResult:
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except (json.JSONDecodeError, TypeError):
                args = {"command": args_raw} if name == "bash" else {}
        else:
            args = args_raw
        if not isinstance(args, dict):
            return tool_error(f"error: arguments for {name} must be a JSON object")

        if args.get("help") is True:
            return self._tool_help(name)

        if self._builtins and name in _BUILTINS:
            return await _BUILTINS[name](args)

        fn = self._functions.get(name)
        if fn is not None:
            return await fn.execute(args)

        spec = self._cli_tools.get(name)
        if spec is None:
            return tool_error(f"Unknown tool: {name}")

        return await _exec_cli_tool(spec, args)

    def _tool_help(self, name: str) -> ToolResult:
        defs = {d.function.name: d.function for d in self.definitions() if d.function}
        fn = defs.get(name)
        if fn is None:
            return tool_error(f"Unknown tool: {name}")
        params = fn.parameters.get("properties", {})
        required = set(fn.parameters.get("required", []))
        lines = [f"tool: {fn.name}"]
        if fn.description:
            lines.append(f"description: {fn.description}")
        if params:
            lines.append("params:")
            for k, v in params.items():
                req = " (required)" if k in required else ""
                desc = v.get("description", v.get("type", ""))
                lines.append(f"  {k}: {desc}{req}")
        return tool_ok("\n".join(lines))


# ── Discovery ───────────────────────────────────────────────────────────────

async def discover_tools(commands: list[str]) -> dict[str, ToolSpec]:
    """Run --tools on each command and collect tool specs."""
    tools: dict[str, ToolSpec] = {}
    for cmd in commands:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            specs = msgspec.json.decode(stdout, type=list[ToolSpec])
            for spec in specs:
                tools[spec.name] = spec
        except Exception as e:
            log.warning("tool_discovery_failed", command=cmd, error=str(e))
    return tools
