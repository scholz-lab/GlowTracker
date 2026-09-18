import textwrap
import threading
import time

import numpy as np
import pytest

import DAQ_control as DAQ
from script_api import BrightnessStats, PluginHost, WormState, trail_velocity, worm_position_mm


class FakeDaq:
    def __init__(self):
        self.feedback = []

    def voltageToDACBits(self, volts, dacNumber, is16Bits):
        return dacNumber, volts

    def getFeedback(self, *commands):
        self.feedback.append(commands)


@pytest.fixture(autouse=True)
def fake_dac_commands(monkeypatch):
    monkeypatch.setattr(DAQ.u3, 'DAC0_8', lambda value: ('DAC0', value))
    monkeypatch.setattr(DAQ.u3, 'DAC1_8', lambda value: ('DAC1', value))


def make_state(**overrides):
    values = dict(
        frame=1, time_s=0.0, wall_time=0.0, stage_xy=(1.0, 2.0), worm_xy=(1.0, 2.0),
        cms_offset_px=(0.0, 0.0), trail=np.empty((0, 2)), velocity=(0.0, 0.0),
        is_reversing=False, is_tracking=True, is_recording=False, voltage=0.0,
        image_shape=(4, 4),
    )
    values.update(overrides)
    return WormState(**values)


def write_plugin(tmp_path, body, name='plugin.py'):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return str(path)


def make_host(daq, calls=None, **kwargs):
    calls = calls if calls is not None else []
    options = dict(
        state_provider=lambda: make_state(frame=len(calls)),
        frame_provider=lambda: np.zeros((4, 4), dtype=np.uint8),
        daq_getter=lambda: daq,
    )
    options.update(kwargs)
    return PluginHost(**options)


def pump(host, n=3, wait=0.5):
    """Deliver n frames and wait until the plugin processed them."""
    for _ in range(n):
        start = host.frames_processed
        host.notify_frame()
        deadline = time.monotonic() + wait
        while host.frames_processed == start and host.is_running() and time.monotonic() < deadline:
            time.sleep(0.005)


# --- DAQControl.set_voltage --------------------------------------------------------------------

def test_set_voltage_clamps_and_tracks_current_voltage():
    control = DAQ.DAQControl()
    control.daq = FakeDaq()
    assert control.set_voltage(9.0) == pytest.approx(DAQ.MAX_VOLTAGE)
    assert control.currentVoltage == pytest.approx(DAQ.MAX_VOLTAGE)
    assert control.set_voltage(-1.0) == 0.0
    assert control.currentVoltage == 0.0
    # both channels were written for the 'on' command
    assert control.daq.feedback[0] == (('DAC0', (0, DAQ.MAX_VOLTAGE)), ('DAC1', (1, DAQ.MAX_VOLTAGE)))


def test_set_voltage_skips_resend_and_dry_runs_without_hardware():
    control = DAQ.DAQControl()
    control.daq = FakeDaq()
    control.set_voltage(2.0)
    control.set_voltage(2.0)
    assert len(control.daq.feedback) == 1

    dry = DAQ.DAQControl()
    assert dry.set_voltage(3.0) == 3.0
    assert dry.currentVoltage == 3.0


def test_update_is_noop_in_plugin_mode():
    control = DAQ.DAQControl()
    control.daq = FakeDaq()
    control.daqMode = DAQ.DAQMode.Plugin
    control.update(frameNum=1, frameTime=0.1, stagePosition=[0, 0, 0], posHist=np.zeros((5, 3)))
    assert control.daq.feedback == []


# --- pure helpers -----------------------------------------------------------------------------

def test_worm_position_adds_converted_centroid_offset():
    # identity matrix maps [offset_y, offset_x] -> [dy, dx] in um
    mat = np.eye(2)
    worm = worm_position_mm((10.0, 20.0), (100.0, -50.0), mat, 0.001)
    assert worm == pytest.approx((10.1, 19.95))
    assert worm_position_mm((10.0, 20.0), None, mat, 0.001) == (10.0, 20.0)


def test_trail_velocity_uses_last_step():
    assert trail_velocity(np.array([[0.0, 0.0], [1.0, 2.0], [1.5, 2.0]])) == (0.5, 0.0)
    assert trail_velocity(np.zeros((1, 2))) == (0.0, 0.0)


# --- PluginHost -------------------------------------------------------------------------------

def test_controller_lifecycle_and_voltage(tmp_path):
    path = write_plugin(tmp_path, '''
        class Controller:
            def __init__(self):
                self.events = []
            def setup(self, scope):
                self.events.append('setup')
            def update(self, state, scope):
                self.events.append(('update', state.frame))
                scope.set_voltage(1.5)
            def teardown(self, scope):
                self.events.append('teardown')
    ''')
    daq = DAQ.DAQControl()
    daq.daq = FakeDaq()
    calls = []
    host = make_host(daq, calls, state_provider=lambda: make_state(frame=7))
    host.load(path)
    assert host.status == 'loaded'
    assert host.start()
    pump(host, 2)
    assert host.frames_processed == 2
    assert daq.currentVoltage == pytest.approx(1.5)
    assert host.stop()
    assert host.status == 'stopped'
    assert host.controller.events == ['setup', ('update', 7), ('update', 7), 'teardown']
    # stopping switches the light off
    assert daq.currentVoltage == 0


def test_module_level_update_function_is_accepted(tmp_path):
    path = write_plugin(tmp_path, '''
        seen = []
        def update(state, scope):
            seen.append(state.worm_xy)
    ''')
    host = make_host(DAQ.DAQControl())
    host.load(path)
    host.start()
    pump(host, 1)
    host.stop()
    assert host.module.seen == [(1.0, 2.0)]


def test_invalid_plugin_is_rejected(tmp_path):
    path = write_plugin(tmp_path, 'x = 1\n')
    host = make_host(DAQ.DAQControl())
    with pytest.raises(ValueError):
        host.load(path)
    with pytest.raises(FileNotFoundError):
        host.load(str(tmp_path / 'missing.py'))


def test_exception_in_update_switches_light_off_and_reports(tmp_path):
    path = write_plugin(tmp_path, '''
        def update(state, scope):
            scope.set_voltage(4.0)
            raise RuntimeError('boom')
    ''')
    daq = DAQ.DAQControl()
    daq.daq = FakeDaq()
    host = make_host(daq)
    host.load(path)
    host.start()
    host.notify_frame()
    deadline = time.monotonic() + 2.0
    while host.is_running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not host.is_running()
    assert host.status == 'error'
    assert 'boom' in host.last_error
    assert daq.currentVoltage == 0


def test_reload_picks_up_edits_and_keeps_running(tmp_path):
    path = write_plugin(tmp_path, '''
        def update(state, scope):
            scope.set_voltage(1.0)
    ''')
    daq = DAQ.DAQControl()
    host = make_host(daq)
    host.load(path)
    host.start()
    pump(host, 1)
    assert daq.currentVoltage == 1.0

    write_plugin(tmp_path, '''
        def update(state, scope):
            scope.set_voltage(2.0)
    ''')
    host.reload()
    assert host.is_running()
    pump(host, 1)
    host.stop()
    assert daq.currentVoltage == 0  # stop() switched it off
    assert host.module.update is not None
    assert host.frames_processed == 1


def test_stage_moves_are_refused_while_blocked(tmp_path):
    class FakeStage:
        def __init__(self):
            self.moves = []
        def move_rel(self, steps, unit, wait_until_idle):
            self.moves.append(('rel', steps, unit))
            return True
        def move_abs(self, position, unit, wait_until_idle):
            self.moves.append(('abs', position, unit))
            return True
        def get_cached_position(self, unit='mm', max_age=None):
            return [1.0, 2.0, 3.0]

    path = write_plugin(tmp_path, '''
        results = []
        def update(state, scope):
            results.append(scope.move_rel(0.1, 0.0))
            results.append(scope.move_abs(5.0, 6.0))
    ''')
    stage = FakeStage()
    blocked = {'reason': 'tracking is active'}
    host = make_host(DAQ.DAQControl(), stage_getter=lambda: stage,
                     moves_blocked_reason=lambda: blocked['reason'])
    host.load(path)
    host.start()
    pump(host, 1)
    assert host.module.results == [False, False]
    assert stage.moves == []
    assert 'refused' in host.message

    blocked['reason'] = None
    pump(host, 1)
    host.stop()
    assert host.module.results[2:] == [True, True]
    assert stage.moves == [('rel', (0.1, 0.0, 0.0), 'mm'), ('abs', (5.0, 6.0, 3.0), 'mm')]


def test_recording_control_and_log(tmp_path):
    path = write_plugin(tmp_path, '''
        def update(state, scope):
            scope.start_recording()
            scope.log(frame=state.frame, worm=state.worm_xy, arr=state.trail)
            scope.stop_recording()
    ''')
    requests = []
    host = make_host(DAQ.DAQControl(), recording_control=requests.append,
                     log_dir_getter=lambda: str(tmp_path / 'rec'))
    host.load(path)
    host.start()
    pump(host, 1)
    host.stop()
    assert requests == [True, False]
    assert host.log_path is not None and host.log_path.startswith(str(tmp_path / 'rec'))
    lines = open(host.log_path, encoding='utf-8').read().splitlines()
    assert len(lines) == 1 and '"worm": [1.0, 2.0]' in lines[0]


def test_wait_for_recording_blocks_until_state_changes(tmp_path):
    path = write_plugin(tmp_path, '''
        results = []
        def setup(scope):
            results.append(('before', scope.is_recording))
            results.append(scope.wait_for_recording(timeout=2.0))
            results.append(('after', scope.is_recording))
        class Controller:
            def setup(self, scope):
                setup(scope)
            def update(self, state, scope):
                pass
    ''')
    recording = {'on': False}
    host = make_host(DAQ.DAQControl(), recording_state=lambda: recording['on'])
    host.load(path)
    host.start()
    time.sleep(0.1)
    assert host.module.results == [('before', False)]
    recording['on'] = True
    deadline = time.monotonic() + 1.0
    while len(host.module.results) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    host.stop()
    assert host.module.results == [('before', False), True, ('after', True)]


def test_wait_for_recording_returns_false_on_stop_and_timeout(tmp_path):
    path = write_plugin(tmp_path, '''
        results = []
        class Controller:
            def setup(self, scope):
                results.append(scope.wait_for_recording(timeout=0.05))
                results.append(scope.wait_for_recording())
            def update(self, state, scope):
                pass
    ''')
    host = make_host(DAQ.DAQControl())
    host.load(path)
    host.start()
    time.sleep(0.15)
    assert host.module.results == [False]
    assert host.stop()
    assert host.module.results == [False, False]


def test_fps_is_measured_and_exposed(tmp_path):
    path = write_plugin(tmp_path, '''
        seen = []
        def update(state, scope):
            seen.append((state.fps, scope.fps, state.frames_for(0.1)))
    ''')
    host = make_host(DAQ.DAQControl())
    host.load(path)
    host.start()
    for _ in range(6):
        host.notify_frame()
        time.sleep(0.02)
    time.sleep(0.05)
    host.stop()
    state_fps, scope_fps, frames = host.module.seen[-1]
    assert 25 < state_fps < 75
    assert scope_fps == pytest.approx(host.fps)
    assert frames == max(1, round(0.1 * state_fps))


def test_fps_measurement_restarts_after_pause():
    host = make_host(DAQ.DAQControl())
    host.notify_frame()
    host._frame_times[-1] -= 5.0          # pretend the last frame was 5 s ago
    host.notify_frame()
    assert len(host._frame_times) == 1     # stale sample dropped, measurement restarts
    assert host.fps == 0.0


def test_analysis_stats_are_optional_and_passed_through(tmp_path):
    path = write_plugin(tmp_path, '''
        seen = []
        def update(state, scope):
            seen.append(None if state.analysis is None else state.analysis.mean)
    ''')
    stats = BrightnessStats(min=1, max=200, mean=42.5, median=40, skewness=0.1,
                            percentile_5=5, percentile_95=150)
    states = iter([make_state(), make_state(analysis=stats)])
    host = make_host(DAQ.DAQControl(), state_provider=lambda: next(states))
    host.load(path)
    host.start()
    pump(host, 2)
    host.stop()
    assert host.module.seen == [None, 42.5]


def test_frames_are_coalesced_when_update_is_slow(tmp_path):
    path = write_plugin(tmp_path, '''
        import time
        def update(state, scope):
            time.sleep(0.05)
    ''')
    host = make_host(DAQ.DAQControl(), update_budget_s=0.01)
    host.load(path)
    host.start()
    for _ in range(10):
        host.notify_frame()
    time.sleep(0.3)
    host.stop()
    assert host.frames_processed < 10
    assert 'skipped' in host.message
