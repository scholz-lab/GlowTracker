"""Camera presets: the user's own named sets of imaging settings, e.g. one for brightfield and one
for fluorescence. One JSON file each in <data root>/camera_presets, so they can be copied or
shared. None are predefined. No Kivy import.

A preset holds what changes between imaging modes: the camera values, and how tracking finds the
animal in such an image (bright on dark or dark on bright, and the threshold). Image size is left
out on purpose: tracking crops the image itself, and the calibration depends on it.
"""
from __future__ import annotations

import json
import math
import os

import run_records

FOLDER = 'camera_presets'
TOLERANCE = 0.01        # the camera rounds what it is given, e.g. to whole line times


def presets_dir(root: str) -> str:
    return os.path.join(root, FOLDER)


def _path(root: str, name: str) -> str:
    return os.path.join(presets_dir(root), run_records.slug(name, 'preset') + '.json')


def list_presets(root: str) -> list[dict]:
    """All presets, by name. A file that cannot be read is left out."""
    presets = []
    try:
        names = os.listdir(presets_dir(root))
    except OSError:
        return []
    for name in names:
        if name.endswith('.json'):
            try:
                with open(os.path.join(presets_dir(root), name), encoding='utf-8') as f:
                    preset = json.load(f)
                if preset.get('name') and isinstance(preset.get('camera'), dict):
                    presets.append(preset)
            except (OSError, ValueError, AttributeError) as e:
                print(f'Skipping camera preset {name}: {e}')
    return sorted(presets, key=lambda p: p['name'].lower())


def find(root: str, name: str) -> dict | None:
    return next((p for p in list_presets(root) if p['name'] == name), None)


def exists(root: str, name: str) -> bool:
    return os.path.exists(_path(root, name))


def save(root: str, name: str, camera: dict, tracking: dict) -> dict:
    preset = {'name': name.strip(), 'camera': camera, 'tracking': tracking}
    os.makedirs(presets_dir(root), exist_ok=True)
    with open(_path(root, name), 'w', encoding='utf-8') as f:
        json.dump(preset, f, indent=1)
    return preset


def delete(root: str, name: str) -> None:
    os.remove(_path(root, name))


def matches(preset: dict, camera: dict, tracking: dict) -> bool:
    """Whether the camera and tracking are set as the preset says."""
    for key, wanted in preset['camera'].items():
        current = camera.get(key)
        if isinstance(wanted, str):
            if current != wanted:
                return False
        elif current is None or not math.isclose(current, wanted, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            return False
    return all(tracking.get(key) == wanted for key, wanted in (preset.get('tracking') or {}).items())
