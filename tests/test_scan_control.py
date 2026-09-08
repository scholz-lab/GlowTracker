from types import SimpleNamespace
from kivy.config import ConfigParser
import numpy as np

from scan import CenterRadiusFromThreePoints
import scan


def test_failed_z_sweep_does_not_start_tile_scan(monkeypatch):
    app = SimpleNamespace(
        camera=object(),
        stage=object(),
        _hardware_teardown=False,
    )
    monkeypatch.setattr(
        scan.App,
        'get_running_app',
        staticmethod(lambda: app),
    )
    panel = CenterRadiusFromThreePoints.__new__(CenterRadiusFromThreePoints)
    panel._scan_thread = None
    panel._plates_thread = None
    panel._teardown_requested = False
    panel._run_generation = 0
    tile_scans = []
    camera_restore = []
    panel._begin_scan_camera = lambda: True
    panel._find_scan_z = lambda: None
    panel._scan = lambda z=None: tile_scans.append(z) or True
    panel._end_scan_camera = lambda found: camera_restore.append(found)
    panel.scan_area()
    panel._scan_thread.join(1)
    assert not panel._scan_thread.is_alive()
    assert tile_scans == []
    assert camera_restore == [False]


def test_empty_search_stops_at_pass_limit(monkeypatch):
    config = ConfigParser()
    config.setdefaults('Stage', {
        'stage_limits': '152,152,152', 'speed_unit': 'mm/s',
        'acceleration_unit': 'mm/s^2', 'precise_speed': '15',
        'precise_acceleration': '200', 'scan_speed': '20', 'scan_acceleration': '100',
    })
    targets = []
    stage = SimpleNamespace(
        is_safe=lambda *args: True,
        set_motion=lambda *args: None,
        move_abs=lambda target, *args, **kwargs: targets.append(target) or True,
        get_position=lambda **kwargs: targets[-1],
    )
    app = SimpleNamespace(stage=stage, camera=SimpleNamespace(singleTake=lambda: (True, np.zeros((4, 4)))),
                          plateCenter=[30, 100], plateRadius=5, config=config,
                          get_fov_mm=lambda: (1, 1))
    monkeypatch.setattr(scan.App, 'get_running_app', staticmethod(lambda: app))
    monkeypatch.setattr(scan.macro, 'generate_scan_tiles', lambda *args, **kwargs: [(30, 100), (31, 100)])
    monkeypatch.setattr(scan.macro, 'detect_worm', lambda *args: (False, (0, 0)))
    monkeypatch.setattr(scan.Clock, 'schedule_once', lambda *args, **kwargs: None)
    panel = CenterRadiusFromThreePoints.__new__(CenterRadiusFromThreePoints)
    panel._stop_all = panel._stop_scan = False
    panel._active_plate_name = 'Test'
    panel.search_passes = 2
    panel._wait_or_stop = lambda duration: False
    assert panel._scan() is False
    assert len(targets) == 4


def test_search_deadline_prevents_another_tile(monkeypatch):
    # A slow move uses up the search budget; no additional tile is started.
    config = ConfigParser()
    config.setdefaults('Stage', {
        'stage_limits': '152,152,152', 'speed_unit': 'mm/s',
        'acceleration_unit': 'mm/s^2', 'precise_speed': '15',
        'precise_acceleration': '200', 'scan_speed': '20', 'scan_acceleration': '100',
    })
    now = [0.0]
    targets = []
    def move(target, *args, **kwargs):
        targets.append(target)
        now[0] += 2
        return True
    stage = SimpleNamespace(is_safe=lambda *args: True, set_motion=lambda *args: None,
                            move_abs=move, get_position=lambda **kwargs: targets[-1])
    app = SimpleNamespace(stage=stage, camera=SimpleNamespace(singleTake=lambda: (True, np.zeros((4, 4)))),
                          plateCenter=[30, 100], plateRadius=5, config=config, get_fov_mm=lambda: (1, 1))
    monkeypatch.setattr(scan.App, 'get_running_app', staticmethod(lambda: app))
    monkeypatch.setattr(scan.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(scan.macro, 'generate_scan_tiles', lambda *args, **kwargs: [(30, 100), (31, 100)])
    monkeypatch.setattr(scan.macro, 'detect_worm', lambda *args: (False, (0, 0)))
    monkeypatch.setattr(scan.Clock, 'schedule_once', lambda *args, **kwargs: None)
    panel = CenterRadiusFromThreePoints.__new__(CenterRadiusFromThreePoints)
    panel._stop_all = panel._stop_scan = False
    panel._active_plate_name = 'Test'
    panel.search_seconds = 1
    panel.search_passes = 10
    panel._wait_or_stop = lambda duration: False
    assert panel._scan() is False
    assert len(targets) == 1
