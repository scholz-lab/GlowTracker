"""Plate visit settings and scheduling, independent of microscope hardware."""

from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
import re


# key: (label, default, minimum, maximum, integer)
FIELDS = {
    'scan_mode': ('Scan mode', 'Sequential', None, None, False),
    'scan_z': ('Starting Z (mm)', 140, 0, None, False),
    'scan_exposure': ('Scan exposure (us)', 100000, 1, None, False),
    'scan_gain': ('Scan gain', 30, 0, None, False),
    'track_exposure': ('Tracking exposure (us)', 5000, 1, None, False),
    'track_gain': ('Tracking gain', 22, 0, None, False),
    'track_framerate': ('Tracking FPS', 30, 0.1, None, False),
    'track_interval': ('Track per visit (s)', 120, 1, None, False),
    'focus_settle_seconds': ('Focus before/after ramp (s)', 3, 1, None, False),
    'exposure_ramp_seconds': ('Minimum exposure ramp (s)', 20, 1, None, False),
    'search_seconds': ('Search limit (s)', 60, 1, None, False),
    'search_passes': ('Search passes', 1, 1, 100, True),
    'scan_settle': ('Settling time (s)', 0.01, 0, None, False),
    'scan_threshold': ('Brightness threshold', 150, 0, None, False),
    'scan_min_pixels': ('Minimum bright pixels', 50, 1, None, True),
    'scan_overlap_w': ('Tile overlap width (%)', 10, 0, 95, False),
    'scan_overlap_h': ('Tile overlap height (%)', 10, 0, 95, False),
    'scan_z_range': ('Focus search range (mm)', 1, 0, None, False),
    'scan_z_frames': ('Focus search images', 30, 2, 1000, True),
}


def parse_setting(key, text):
    if key == 'scan_mode':
        if text not in ('Sequential', 'Continuous'):
            raise ValueError('Scan mode: choose Sequential or Continuous')
        return text
    label, _, low, high, integer = FIELDS[key]
    try:
        value = float(text)
    except (ValueError, TypeError):
        raise ValueError(f'{label}: enter a number') from None
    if not math.isfinite(value) or value < low or (high is not None and value > high):
        limit = f'{low:g} to {high:g}' if high is not None else f'at least {low:g}'
        raise ValueError(f'{label}: use {limit}')
    if integer and not value.is_integer():
        raise ValueError(f'{label}: enter a whole number')
    return int(value) if integer else value


def validate_plate(plate):
    result = deepcopy(plate)
    if not str(result.get('name', '')).strip():
        raise ValueError('Give the plate a name')
    center = result.get('center', [])
    radius = result.get('radius', 0)
    if len(center) != 2 or not all(math.isfinite(float(v)) for v in center):
        raise ValueError('Define the plate using three points around its rim')
    if not math.isfinite(float(radius)) or float(radius) <= 0:
        raise ValueError('The plate radius must be positive')
    result['center'] = [float(v) for v in center]
    result['radius'] = float(radius)
    settings = result.get('settings', {})
    result['settings'] = {
        key: parse_setting(key, settings.get(key, spec[1]))
        for key, spec in FIELDS.items()
    }
    return result


def visits(plates, repeat=False):
    """Visit enabled plates in order, with a stable snapshot for the whole run."""
    enabled = [validate_plate(p) for p in plates if p.get('enabled', True)]
    if not enabled:
        raise ValueError('Enable at least one plate')
    cycle = 1
    while True:
        for plate in enabled:
            yield cycle, deepcopy(plate)
        if not repeat:
            return
        cycle += 1


def brightness_steps(exposure, gain, target_exposure, target_gain):
    """Limit each exposure decrease to 10% and each gain change to 0.5."""
    count = max(math.ceil(abs(math.log(target_exposure / exposure)) / math.log(1 / 0.9)),
                math.ceil(abs(target_gain - gain) / 0.5))
    for index in range(1, count + 1):
        if index == count:
            yield target_exposure, target_gain
        else:
            fraction = index / count
            yield exposure * (target_exposure / exposure) ** fraction, gain + (target_gain - gain) * fraction


def safe_name(name):
    return re.sub(r'[^\w.-]+', '_', name.strip()).strip('._')[:80] or 'plate'


def create_run_directory(destination, name):
    base = Path(destination).expanduser()
    if not base.is_dir():
        raise ValueError('Choose an existing recording folder')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run = base / f'{safe_name(name)}_{stamp}'
    run.mkdir(exist_ok=False)
    return run


def create_visit_directory(run, plate, cycle):
    path = Path(run) / f'{safe_name(plate["name"])}_{plate["id"]}' / f'visit_{cycle:04d}'
    path.mkdir(parents=True, exist_ok=False)
    return path
