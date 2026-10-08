"""Remembering the window position (window_geometry); no Kivy window needed."""
import json
from types import SimpleNamespace

import window_geometry as wg


class FakeConfig:
    def __init__(self):
        self.values = {}

    def set(self, section, key, value):
        self.values[(section, key)] = value


def test_round_trip_and_apply(tmp_path, monkeypatch):
    path = str(tmp_path / 'geo.json')
    wg.save({'left': 2100, 'top': 80, 'width': 1400, 'height': 900, 'maximized': False}, path)
    monkeypatch.setattr(wg, 'on_a_display', lambda left, top: True)
    config = FakeConfig()
    applied = wg.apply_before_window_created(config, path)
    assert applied['left'] == 2100
    assert config.values == {('graphics', 'position'): 'custom', ('graphics', 'left'): '2100',
                             ('graphics', 'top'): '80'}
    assert wg.logical_size(applied, (1280, 800)) == (1400, 900)


def test_position_on_an_unplugged_monitor_is_not_used(tmp_path, monkeypatch):
    path = str(tmp_path / 'geo.json')
    wg.save({'left': 5000, 'top': 80, 'width': 1400, 'height': 900, 'maximized': False}, path)
    monkeypatch.setattr(wg, 'on_a_display', lambda left, top: False)
    config = FakeConfig()
    assert wg.apply_before_window_created(config, path) is None
    assert config.values == {}


def test_missing_or_broken_file_and_tiny_sizes_fall_back(tmp_path):
    assert wg.load(str(tmp_path / 'none.json')) is None
    broken = tmp_path / 'broken.json'
    broken.write_text('{not json')
    assert wg.load(str(broken)) is None
    assert wg.logical_size(None, (1280, 800)) == (1280, 800)
    assert wg.logical_size({'width': 100, 'height': 50}, (1280, 800)) == (1280, 800)


def test_capture_stores_logical_size_on_windows(monkeypatch):
    monkeypatch.setattr(wg.sys, 'platform', 'win32')
    window = SimpleNamespace(size=(2880, 1620), left=100, top=60, _density=1.5)
    assert wg.capture(window, maximized=False) == {'left': 100, 'top': 60, 'width': 1920,
                                                   'height': 1080, 'maximized': False}


def test_current_display_check_runs():
    # On Windows this asks the system; the primary display always contains (0, 0).
    assert wg.on_a_display(0, 0) is True


def test_garbage_size_from_a_closed_window_is_never_saved_or_used(tmp_path):
    """The size read from a window that was already destroyed (seen on Windows) crashed the next start."""
    path = str(tmp_path / 'geo.json')
    good = {'left': 0, 'top': 0, 'width': 1280, 'height': 800, 'maximized': False}
    wg.save(good, path)
    wg.save({'left': 0, 'top': 0, 'width': 1611400528, 'height': 1188976096, 'maximized': False}, path)
    assert wg.load(path) == good                       # the bad one was refused, the good one kept
    with open(path, 'w') as f:                         # a bad file written by the earlier version
        json.dump({'left': 0, 'top': 0, 'width': 1611400528, 'height': 1188976096}, f)
    assert wg.load(path) is None
    assert wg.logical_size({'width': 1611400528, 'height': 1188976096}, (1280, 800)) == (1280, 800)
