"""Remember where the GlowTracker window was and reopen it there.

Why: on Windows, Kivy converts the layout's sizes to pixels once, at the scale of the display the
window is on when the interface is built. Dragging the window to a display with another scale
setting (laptop 150 %, external monitor 100 %) leaves those sizes at the old scale and chops the
layout. Reopening on the display it was last used on builds the layout at that display's scale.

The position must be applied before Kivy creates its window (importing kivy.core.window creates
it), so `apply_before_window_created` is called before that import. Kivy-free: Config is passed in.
"""
from __future__ import annotations

import json
import os
import sys

import platformdirs

DEFAULT_PATH = os.path.join(platformdirs.user_config_dir(appname='GlowTracker', appauthor='Monika Scholz'),
                            'window_geometry.json')
MIN_SIZE = (640, 400)          # logical pixels; smaller saved sizes are ignored
MAX_SIZE = (8000, 5000)        # anything larger is a broken reading, never a real window
MAX_COORD = 30000              # |left|, |top| beyond this is not a real desktop position


def plausible(geometry: dict) -> bool:
    """A real window geometry, not a value read from a window that was already gone."""
    return (abs(geometry['left']) <= MAX_COORD and abs(geometry['top']) <= MAX_COORD
            and 0 <= geometry['width'] <= MAX_SIZE[0] and 0 <= geometry['height'] <= MAX_SIZE[1])


def load(path: str = DEFAULT_PATH) -> dict | None:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        geometry = {'left': int(data['left']), 'top': int(data['top']),
                    'width': int(data.get('width', 0)), 'height': int(data.get('height', 0)),
                    'maximized': bool(data.get('maximized', False))}
    except (KeyError, TypeError, ValueError):
        return None
    return geometry if plausible(geometry) else None


def save(geometry: dict, path: str = DEFAULT_PATH) -> None:
    """Write the geometry, unless it is implausible (then the previous file is kept)."""
    if not plausible(geometry):
        print(f'Not saving an implausible window geometry: {geometry}')
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(geometry, f, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        print(f'Saving the window position failed: {e}')


def on_a_display(left: int, top: int) -> bool:
    """Whether the point lies on a connected display. On Windows this asks the system, so a
    position on a monitor that has since been unplugged is not reused; elsewhere it is trusted."""
    if sys.platform != 'win32':
        return True
    try:
        import ctypes
        from ctypes import wintypes
        MONITOR_DEFAULTTONULL = 0
        user32 = ctypes.windll.user32
        user32.MonitorFromPoint.restype = wintypes.HMONITOR
        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        # a point just inside the title bar, so a window whose corner sits on the border counts
        return bool(user32.MonitorFromPoint(wintypes.POINT(left + 40, top + 10), MONITOR_DEFAULTTONULL))
    except Exception:
        return True


def apply_before_window_created(config, path: str = DEFAULT_PATH) -> dict | None:
    """Put the window where it was last time (Kivy graphics Config), if that is still on a display.
    Returns the geometry that was applied, or None."""
    geometry = load(path)
    if geometry is None or not on_a_display(geometry['left'], geometry['top']):
        return None
    config.set('graphics', 'position', 'custom')
    config.set('graphics', 'left', str(geometry['left']))
    config.set('graphics', 'top', str(geometry['top']))
    return geometry


def logical_size(geometry: dict | None, default: tuple[int, int]) -> tuple[int, int]:
    """The size to give Window.size (logical pixels): the saved one if sensible, else the default."""
    if geometry and MIN_SIZE[0] <= geometry['width'] <= MAX_SIZE[0] \
            and MIN_SIZE[1] <= geometry['height'] <= MAX_SIZE[1]:
        return geometry['width'], geometry['height']
    return default


def capture(window, maximized: bool) -> dict:
    """Current geometry of a Kivy window. Window.size is in physical pixels on Windows while
    assigning Window.size takes logical ones, so the size is stored divided by the display scale."""
    density = float(getattr(window, '_density', 1.0) or 1.0) if sys.platform == 'win32' else 1.0
    width, height = window.size
    return {'left': int(window.left), 'top': int(window.top),
            'width': int(round(width / density)), 'height': int(round(height / density)),
            'maximized': bool(maximized)}
