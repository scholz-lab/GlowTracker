"""Tool output formatting — TOON conversion and proactive truncation.

Detects JSON in tool results and converts to TOON (Token-Oriented Object
Notation) for ~40% token savings.  Large outputs are truncated with a size
hint so the agent can request more if needed.
"""

from __future__ import annotations

import json
import os

from toon_format import encode as toon_encode

_MAX_TOOL_OUTPUT = int(os.getenv("OX_MAX_TOOL_OUTPUT", "12000"))
_TOON_MIN_ITEMS = 2  # only convert arrays with ≥2 items (scalar JSON not worth it)


def _try_json_to_toon(text: str) -> str | None:
    """If *text* is a JSON array of objects, return TOON; else None."""
    stripped = text.strip()
    if not (stripped.startswith("[") or stripped.startswith("{")):
        return None
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None

    try:
        # Arrays of objects → tabular TOON (biggest win)
        if isinstance(data, list) and len(data) >= _TOON_MIN_ITEMS:
            if all(isinstance(item, dict) for item in data):
                return toon_encode(data)

        # Top-level object with at least one array-of-objects value
        if isinstance(data, dict):
            has_table = any(
                isinstance(v, list)
                and len(v) >= _TOON_MIN_ITEMS
                and all(isinstance(i, dict) for i in v)
                for v in data.values()
            )
            if has_table:
                return toon_encode(data)
    except Exception:
        return None  # fall back to original text if TOON encoding fails

    return None


def _truncate(text: str, limit: int) -> str:
    """Truncate *text* to *limit* chars, appending a size hint."""
    if len(text) <= limit:
        return text
    total = len(text)
    suffix = (f"\n\n... (truncated, {total} chars total"
              " — re-run with head/tail for the rest)")
    cut = max(0, limit - len(suffix))
    return text[:cut] + suffix


_HINTS: dict[str, list[str]] = {
    "read": [
        "Use `edit` to modify, or `read` with grep to filter",
    ],
    "bash": [
        "Pipe through `head`/`tail`/`jq` to narrow results",
    ],
}


def format_tool_output(content: str, tool_name: str = "") -> str:
    """Format a tool result for the LLM: TOON-convert, truncate, add hints."""
    # 1. Try TOON conversion (only for structured JSON)
    toon = _try_json_to_toon(content)
    if toon is not None:
        content = toon

    # 2. Build hint suffix (so truncation can reserve space for it)
    hint_suffix = ""
    hints = _HINTS.get(tool_name)
    if hints:
        hint_block = "\n".join(f"  {h}" for h in hints)
        hint_suffix = f"\nhint[{len(hints)}]:\n{hint_block}"

    # 3. Proactive truncation — reserve room for hints
    content = _truncate(content, _MAX_TOOL_OUTPUT - len(hint_suffix))

    return content + hint_suffix
