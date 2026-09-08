from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

import continuous_scan as module
from continuous_scan import ContinuousScanMixin, scan_rows
from plate_plan import validate_plate
from plate_run import PlateRunController


class Stage:
    def __init__(self):
        self.position = [0, 50, 140]
        self.target_x = None
        self.moves = []
        self.motions = []
        self.stops = 0

    def get_position(self, **kwargs):
        return list(self.position)

    def move_abs(self, target, *args, **kwargs):
        self.moves.append(('absolute', tuple(target)))
        self.position = list(target)
        return True

    def move_x(self, delta, **kwargs):
        assert kwargs['wait_until_idle'] is False
        self.moves.append(('sweep', delta))
        self.target_x = self.position[0] + delta
        return True

    def advance(self):
        if self.target_x is not None:
            step = np.clip(self.target_x - self.position[0], -1, 1)
            self.position[0] += step
            if self.position[0] == self.target_x:
                self.target_x = None

    def is_busy(self):
        return self.target_x is not None

    def emergency_stop(self):
        self.target_x = None
        self.stops += 1
        return True

    def set_motion(self, *args):
        self.motions.append(args)

    def is_safe(self, *args):
        return True


class Camera:
    def __init__(self, stage, frames=()):
        self.stage = stage
        self.frames = deque(frames)
        self.grabbing = False
        self.single = False
        self.events = []
        self.on_frame = lambda: None
        for name in ('ExposureTime', 'Gain', 'AcquisitionFrameRateEnable', 'AcquisitionFrameRate'):
            setattr(self, name, SimpleNamespace(Value=None))

    def StartGrabbing(self, strategy):
        self.events.append('stream')
        self.single = False
        self.grabbing = True

    def StartGrabbingMax(self, count):
        assert not self.stage.is_busy(), 'Confirmation must be stationary'
        self.events.append('snapshot')
        self.single = True
        self.grabbing = True

    def StopGrabbing(self):
        self.grabbing = False
        self.events.append('stop')

    def IsGrabbing(self):
        return self.grabbing

    def retrieveGrabbingResult(self, timeout_ms):
        assert timeout_ms == 50
        self.stage.advance()
        self.on_frame()
        value = self.frames.popleft() if self.frames else 0
        if isinstance(value, Exception):
            raise value
        if self.single:
            self.grabbing = False
        return True, np.full((10, 10), value, dtype=np.uint8), 0, 0


@pytest.fixture
def rig(monkeypatch):
    panel = ContinuousScanMixin()
    values = dict(_stop_scan=False, _stop_all=False, _teardown_requested=False,
                  scan_z=140, scan_exposure=100000, scan_overlap_w=10, scan_overlap_h=10,
                  scan_settle=0, scan_threshold=150, scan_min_pixels=50,
                  scan_center_tol=15, scan_recenter_iters=3, search_seconds=60,
                  search_passes=1, track_exposure=5000, track_gain=22, track_framerate=30,
                  _active_plate_name='Test plate')
    for key, value in values.items():
        setattr(panel, key, value)
    panel._wait_or_stop = lambda duration: panel._stop_scan or panel._stop_all
    stage = Stage()
    camera = Camera(stage)
    settings = dict(stage_limits='160,160,155', speed_unit='mm/s',
                    acceleration_unit='mm/s^2', scan_speed='26', scan_acceleration='500',
                    precise_speed='5', precise_acceleration='100')
    app = SimpleNamespace(stage=stage, camera=camera, plateCenter=(1.5, 50), plateRadius=10,
                          config=SimpleNamespace(get=lambda section, key: settings[key]),
                          get_fov_mm=lambda: (1, 1), update_coordinates=lambda **kw: None)
    monkeypatch.setattr(module.App, 'get_running_app', staticmethod(lambda: app))
    monkeypatch.setattr(module.Clock, 'schedule_once', lambda callback: callback(0))
    monkeypatch.setattr(module.macro, 'generate_scan_tiles', lambda *a, **kw:
                        [(0, 50), (1, 50), (2, 50), (3, 50)])
    return panel, app


def test_rows_preserve_reverse_direction_and_single_tile():
    assert scan_rows([(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)]) == [
        ((0, 0), (1, 0)), ((1, 1), (0, 1)), ((0, 2), (0, 2))]


def test_scan_moves_once_per_row_and_captures_while_moving(rig):
    panel, app = rig
    assert panel._scan_continuous() is False
    assert app.stage.moves == [('absolute', (0, 50, 140)), ('sweep', 3)]
    assert app.camera.events.count('stream') == 1
    assert panel._continuous_frames == 5  # start, three moving frames, endpoint
    assert not app.camera.IsGrabbing()
    assert not app.stage.is_busy()
    assert app.stage.motions[-1] == (5, 100, 'mm/s', 'mm/s^2')


def test_row_finishing_between_position_and_busy_reads_is_not_an_error(rig):
    panel, app = rig
    original = app.stage.is_busy
    def busy():
        if app.stage.target_x is not None:
            # The earlier position read was mid-row; by this read the move has ended.
            app.stage.position[0] = app.stage.target_x
            app.stage.target_x = None
        return original()
    app.stage.is_busy = busy
    assert panel._scan_continuous() is False
    assert app.stage.position[0] == 3


def test_real_early_stop_reports_actual_and_target_position(rig):
    panel, app = rig
    def early_stop():
        if app.stage.target_x is not None:
            app.stage.target_x = None
        return False
    app.stage.is_busy = early_stop
    with pytest.raises(RuntimeError, match=r'actual X=1\.000 mm, target X=3\.000 mm'):
        panel._scan_continuous()
    assert not app.camera.IsGrabbing()


def test_candidate_requires_stationary_confirmation_before_tracking(rig):
    panel, app = rig
    app.camera.frames.extend([0, 255, 255])
    assert panel._scan_continuous() is True
    assert app.camera.events.count('snapshot') == 2
    # Brightness changes only after focusing on the detected worm.
    assert app.camera.ExposureTime.Value is None
    assert app.camera.AcquisitionFrameRate.Value is None
    assert not app.stage.is_busy()
    assert not app.camera.IsGrabbing()


def test_false_candidate_resumes_sweep_without_starting_tracking(rig):
    panel, app = rig
    app.camera.frames.extend([0, 255, 0, 0, 0])
    assert panel._scan_continuous() is False
    assert app.camera.events.count('stream') == 2
    assert app.camera.ExposureTime.Value is None
    assert app.stage.position == [3, 50, 140]


def test_candidate_can_be_recovered_behind_stopping_position(rig):
    panel, app = rig
    app.camera.frames.extend([0, 255, 0, 255])
    assert panel._scan_continuous() is True
    assert ('absolute', (0.5, 50, 140)) in app.stage.moves


@pytest.mark.parametrize('flag', ['_stop_scan', '_stop_all', '_teardown_requested'])
def test_stop_during_capture_cleans_up_and_does_not_start_tracking(rig, flag):
    panel, app = rig
    def stop_during_motion():
        if not app.camera.single:
            setattr(panel, flag, True)
    app.camera.on_frame = stop_during_motion
    assert panel._scan_continuous() is False
    assert not app.camera.IsGrabbing()
    assert not app.stage.is_busy()
    assert app.camera.ExposureTime.Value is None
    assert app.stage.motions[-1][:2] == (5, 100)


def test_camera_exception_stops_motion_and_restores_speed(rig):
    panel, app = rig
    app.camera.frames.extend([0, RuntimeError('disconnected')])
    with pytest.raises(RuntimeError, match='disconnected'):
        panel._scan_continuous()
    assert not app.stage.is_busy()
    assert not app.camera.IsGrabbing()
    assert app.stage.motions[-1][:2] == (5, 100)


def test_search_deadline_stops_scan(rig):
    panel, app = rig
    panel.search_seconds = 0
    assert panel._scan_continuous() is False
    assert app.stage.moves == []
    assert not app.camera.IsGrabbing()


def test_search_deadline_during_motion_stops_stage(rig, monkeypatch):
    panel, app = rig
    now = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    def expire_during_motion():
        if not app.camera.single:
            now[0] = 61.0
    app.camera.on_frame = expire_during_motion
    assert panel._scan_continuous() is False
    assert not app.camera.IsGrabbing()
    assert not app.stage.is_busy()
    assert app.camera.ExposureTime.Value is None


@pytest.mark.parametrize('flag', ['_stop_scan', '_stop_all', '_teardown_requested'])
def test_stop_during_final_cleanup_prevents_tracking_handoff(rig, flag, capsys):
    panel, app = rig
    app.camera.frames.extend([255])
    original = app.stage.emergency_stop
    def stop():
        setattr(panel, flag, True)
        return original()
    app.stage.emergency_stop = stop
    assert panel._scan_continuous() is False
    assert app.camera.ExposureTime.Value is None
    assert 'found=False' in capsys.readouterr().out


def test_confirmed_worm_survives_search_deadline_during_cleanup(rig, monkeypatch, capsys):
    panel, app = rig
    now = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    app.camera.frames.extend([255])
    original = app.stage.emergency_stop
    def stop():
        now[0] = 61.0  # confirmation succeeded, but final cleanup crosses the search limit
        return original()
    app.stage.emergency_stop = stop
    assert panel._scan_continuous() is True
    assert 'found=True' in capsys.readouterr().out
    assert not app.camera.IsGrabbing() and not app.stage.is_busy()


def test_stop_during_coordinate_update_prevents_tracking_handoff(rig):
    panel, app = rig
    app.camera.frames.extend([255])
    app.update_coordinates = lambda **kwargs: setattr(panel, '_stop_all', True)
    assert panel._scan_continuous() is False


def test_plate_run_enters_tracking_handoff_when_cleanup_crosses_search_limit(rig, monkeypatch):
    panel, app = rig
    now, events = [0.0], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    app.camera.frames.extend([255])
    original = app.stage.emergency_stop
    def stop():
        now[0] = 61.0
        return original()
    app.stage.emergency_stop = stop
    app.bind_keys = lambda: None
    panel._profile = lambda: {}
    panel._ui = lambda callback, **kwargs: callback()
    panel._load_plate = lambda plate: None
    panel._status = lambda text, *args: events.append(text)
    panel._begin_scan_camera = lambda: True
    panel._end_scan_camera = lambda found: None
    panel._find_scan_z = lambda: 140
    panel._scan = panel._scan_continuous
    panel._track_visit = lambda plate, folder: events.append('handoff entered') or 'Visit complete'
    panel._quiesce_visit = lambda: None
    panel.selected_plate = -1
    panel.pause_requested = False
    plan = [validate_plate(dict(id='A', name='A', center=[1.5, 50], radius=10, enabled=True,
                                settings={'scan_mode': 'Continuous'}))]
    PlateRunController._execute_plan(panel, plan, None, False)
    assert events.count('handoff entered') == 1
    assert any('Worm found' in text for text in events)
    assert not any('Nothing found' in text for text in events)
    assert panel.run_status == 'Run complete'


def test_pass_limit_repeats_rows(rig):
    panel, app = rig
    panel.search_passes = 2
    assert panel._scan_continuous() is False
    assert app.camera.events.count('stream') == 2


def test_unsafe_rows_do_not_start_motion(rig):
    panel, app = rig
    app.stage.is_safe = lambda *args: False
    with pytest.raises(RuntimeError, match='No safe rows'):
        panel._scan_continuous()
    assert app.stage.moves == []


def test_old_presets_default_to_sequential_and_new_mode_is_preserved():
    plate = dict(name='Test', center=[1, 2], radius=3)
    assert validate_plate(plate)['settings']['scan_mode'] == 'Sequential'
    plate['settings'] = {'scan_mode': 'Continuous'}
    assert validate_plate(plate)['settings']['scan_mode'] == 'Continuous'
    plate['settings']['scan_mode'] = 'invalid'
    with pytest.raises(ValueError, match='Scan mode'):
        validate_plate(plate)
