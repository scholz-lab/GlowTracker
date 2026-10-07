"""Configuration loader for the ox runtime.

Reads from a JSON file (~/.ox/config.json or --config path) with
environment variable overrides. No new dependencies — uses msgspec
for decoding.
"""

from __future__ import annotations

import os
from pathlib import Path

import msgspec


class Config(msgspec.Struct, frozen=True, omit_defaults=True):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    tools: list[str] = []
    max_turns: int = 0


_DEFAULT_PATH = Path.home() / ".ox" / "config.json"


def load(path: str | None = None) -> Config:
    """Load config from file + env overlay."""
    cfg = Config()

    # File load
    file_path = Path(path) if path else _DEFAULT_PATH
    if file_path.exists():
        raw = file_path.read_bytes()
        cfg = msgspec.json.decode(raw, type=Config)
    elif path is not None:
        raise SystemExit(f"Config file not found: {path}")

    # Env overlay (env wins over file defaults, file explicit values win)
    base_url = os.environ.get("OX_BASE_URL") or cfg.base_url
    api_key = os.environ.get("OX_API_KEY") or cfg.api_key
    model = os.environ.get("OX_MODEL") or cfg.model

    if base_url != cfg.base_url or api_key != cfg.api_key or model != cfg.model:
        cfg = Config(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
            tools=cfg.tools,
            max_turns=cfg.max_turns,
        )

    return cfg
