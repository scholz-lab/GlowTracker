from types import SimpleNamespace

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
