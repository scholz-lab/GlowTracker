"""Saved plate runs ("run presets"): the whole plate list with each plate's settings, plus the
run options, under a name. No Kivy import.

The user's presets live in a JSON file next to the plate presets; presets bundled with the app
(settings/scan_runs.json, e.g. a demo run) are listed too, and a user preset with the same name
replaces the bundled one.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

from plate_plan import validate_plate

BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings', 'scan_runs.json')


def _read(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load_all(user_path: str, bundled_path: str = BUNDLED) -> dict:
    """Every preset by name: the bundled ones, then the user's (which win on a name clash)."""
    presets = _read(bundled_path)
    presets.update(_read(user_path))
    return presets


def names(user_path: str, bundled_path: str = BUNDLED) -> list[str]:
    return sorted(load_all(user_path, bundled_path), key=str.lower)


def get(user_path: str, name: str, bundled_path: str = BUNDLED) -> dict:
    """A preset ready to use: validated plates, reset to Ready. Raises KeyError or ValueError."""
    entry = load_all(user_path, bundled_path).get(name)
    if entry is None:
        raise KeyError(f'There is no saved run called {name!r}')
    plates = []
    for plate in entry.get('plates', []):
        plate = validate_plate(plate)
        plate['status'] = 'Ready'
        plates.append(plate)
    if not plates:
        raise ValueError(f'The saved run {name!r} has no plates')
    return {'plates': plates, 'repeat_run': bool(entry.get('repeat_run', False))}


def save(user_path: str, name: str, plates: list, repeat_run: bool) -> None:
    """Store the run under `name` (replacing one with that name). Raises ValueError."""
    name = name.strip()
    if not name:
        raise ValueError('Give the saved run a name')
    if not plates:
        raise ValueError('Add a plate before saving the run')
    stored = []
    for plate in plates:
        plate = validate_plate(plate)
        plate.pop('status', None)
        stored.append(plate)
    data = _read(user_path)
    data[name] = {'plates': stored, 'repeat_run': bool(repeat_run),
                  'saved': time.strftime('%Y-%m-%d %H:%M')}
    folder = os.path.dirname(user_path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.scan_runs_', dir=folder, text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, user_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def summary(user_path: str, name: str, bundled_path: str = BUNDLED) -> str:
    """'4 plates · saved 2026-10-07 21:30' (or 'built in') for the preset list."""
    entry = load_all(user_path, bundled_path).get(name, {})
    count = len(entry.get('plates', []))
    when = entry.get('saved')
    return f'{count} plate{"s" if count != 1 else ""} · ' + (f'saved {when}' if when else 'built in')
