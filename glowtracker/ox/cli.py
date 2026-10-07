"""CLI entrypoint for the ox agent runtime.

Wires config, provider, tools, and agent server together.
Agent output goes to stdout; logs go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import structlog

from ox import config
from ox.agent import AgentServer
from ox.tools import ToolResolver, discover_tools
from ox.bus import AsyncPort, Envelope, Msg
from ox.types import AssistantMessage, ModelSpec, Reasoning, UserMessage


def _show_dashboard(cfg, tool_cmds: list[str], cli_tools: dict) -> None:
    """Content-first: show live state when invoked with no prompt."""
    import shutil
    bin_path = shutil.which("ox") or sys.argv[0]
    home = str(Path.home())
    if bin_path.startswith(home):
        bin_path = "~" + bin_path[len(home):]

    cwd = os.getcwd()
    model = cfg.model or "(not configured)"
    provider_url = cfg.base_url

    lines = [
        f"bin: {bin_path}",
        "description: Minimal async multi-agent LLM runtime with tool use",
        f"cwd: {cwd}",
        f"model: {model}",
        f"provider: {provider_url}",
        f"tools: {len(cli_tools)} external" + (f" ({', '.join(sorted(cli_tools))})" if cli_tools else ""),
        "builtins: bash, read, write, edit",
    ]

    config_path = Path.home() / ".ox" / "config.json"
    if config_path.exists():
        lines.append(f"config: {config_path}")

    lines.append("")
    lines.append("help[3]:")
    lines.append('  Run `ox "your prompt here"` to start an agent session')
    lines.append('  Run `ox -m <model> "prompt"` to override model')
    lines.append('  Run `ox --help` for all flags')

    print("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ox", description="Minimal async LLM agent")
    p.add_argument("prompt", nargs="?", default=None, help="Prompt text")
    p.add_argument("-m", "--model", default=None, help="Model ID")
    p.add_argument("-t", "--tools", action="append", default=[], help="Tool discovery command (repeatable)")
    p.add_argument("-c", "--config", default=None, help="Config file path")
    p.add_argument("-s", "--stream", action="store_true", help="Enable streaming")
    p.add_argument("--max-turns", type=int, default=0, help="Max turns")
    p.add_argument("--provider", choices=["http", "claude"], default="http",
                   help="LLM provider: 'http' (OpenAI-compatible API) or 'claude' (Claude Code CLI)")
    return p.parse_args()


async def _print_replies(reply_port: AsyncPort, stream: bool) -> None:
    try:
        async for msg in reply_port:
            for m in msg.data:
                if not isinstance(m, AssistantMessage):
                    continue
                meta = msg.metadata or {}
                if meta.get("stream_delta") and stream:
                    text = m.content if isinstance(m.content, str) else ""
                    print(text, end="", flush=True, file=sys.stdout)
                elif meta.get("stream_done") and stream:
                    print(file=sys.stdout)
                elif not meta.get("stream_delta"):
                    text = m.content if isinstance(m.content, str) else ""
                    if text:
                        print(text, file=sys.stdout)
    except (EOFError, StopAsyncIteration):
        pass


async def _async_main() -> None:
    args = _parse_args()
    cfg = config.load(args.config)
    model_id = args.model or cfg.model

    # Tools (needed for dashboard and agent)
    tool_cmds = cfg.tools + args.tools
    cli_tools = await discover_tools(tool_cmds) if tool_cmds else {}

    # Prompt — check early so dashboard doesn't require provider setup
    prompt = args.prompt
    if prompt is None:
        if sys.stdin.isatty():
            _show_dashboard(cfg, tool_cmds, cli_tools)
            sys.exit(0)
        prompt = sys.stdin.read().strip()
    if not prompt:
        sys.exit(0)

    # Model
    reasoning = Reasoning(effort=cfg.reasoning_effort) if cfg.reasoning_effort else None
    model = ModelSpec(
        id=model_id or "claude",
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        reasoning=reasoning,
    )

    # Provider
    if args.provider == "claude":
        from ox.provider_claude import ClaudeCodeProvider
        provider = ClaudeCodeProvider(model=model_id or None)
    else:
        if not cfg.api_key:
            print("No API key. Set OX_API_KEY or api_key in config.", file=sys.stderr)
            sys.exit(1)
        if not model_id:
            print("No model specified. Use -m or set model in config.", file=sys.stderr)
            sys.exit(1)
        from ox.provider_http import HttpProvider
        provider = HttpProvider(base_url=cfg.base_url, api_key=cfg.api_key)

    tools = ToolResolver(cli_tools=cli_tools)

    # Bus
    port: AsyncPort[Envelope] = AsyncPort()
    reply_port: AsyncPort[Msg] = AsyncPort()

    # Server
    server = AgentServer(
        port=port, provider=provider, model=model, tools=tools,
    )

    max_turns = args.max_turns or cfg.max_turns
    envelope = Envelope(
        msg=Msg(
            session_id=0,
            data=[UserMessage(content=prompt)],
            metadata={"stream": args.stream, "max_turns": max_turns} if (args.stream or max_turns) else None,
        ),
        reply=reply_port,
    )

    server_task = asyncio.create_task(server.run())
    await port.send(envelope)

    try:
        await _print_replies(reply_port, args.stream)
    finally:
        server_task.cancel()
        await provider.close()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass
