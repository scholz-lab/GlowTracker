"""Named stage positions the user marked, e.g. "worm 2". XY only: going back to one must not drive
the focus. Kept in <data root>/stage_markers.json. No Kivy import.
"""
from __future__ import annotations

import json
import math
import os

FILE = 'stage_markers.json'


def _path(root: str) -> str:
    return os.path.join(root, FILE)


def load(root: str) -> list[dict]:
    """The markers, in the order they were made. A broken file counts as none."""
    try:
        with open(_path(root), encoding='utf-8') as f:
            markers = json.load(f)
        return [m for m in markers if isinstance(m, dict) and {'name', 'x', 'y'} <= set(m)]
    except (OSError, ValueError, TypeError) as e:
        if not isinstance(e, FileNotFoundError):
            print(f'Reading the stage markers failed: {e}')
        return []


def _save(root: str, markers: list[dict]) -> None:
    os.makedirs(root, exist_ok=True)
    with open(_path(root), 'w', encoding='utf-8') as f:
        json.dump(markers, f, indent=1)


def add(root: str, name: str, x: float, y: float) -> list[dict]:
    markers = load(root) + [{'name': name.strip(), 'x': round(float(x), 4), 'y': round(float(y), 4)}]
    _save(root, markers)
    return markers


def remove(root: str, marker: dict) -> list[dict]:
    markers = [m for m in load(root) if m != marker]
    _save(root, markers)
    return markers


def next_name(markers: list[dict]) -> str:
    return f'Marker {len(markers) + 1}'


def nearest(markers: list[dict], x: float, y: float, within_mm: float) -> dict | None:
    """The marker closest to a position, if one is that near."""
    if not markers:
        return None
    best = min(markers, key=lambda m: math.hypot(m['x'] - x, m['y'] - y))
    return best if math.hypot(best['x'] - x, best['y'] - y) <= within_mm else None
