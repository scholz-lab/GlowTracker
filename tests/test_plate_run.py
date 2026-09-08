from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest
import numpy as np
from kivy.config import ConfigParser

import plate_run
from plate_plan import FIELDS, SearchReport, validate_plate
from plate_run import PlateRunController, RunCancelled


def plate(name):
    return validate_plate({'id': name, 'name': name, 'center': [40, 100],
                           'radius': 10, 'settings': {}, 'enabled': True})


class SimulatedRun(PlateRunController):
    def __init__(self):
        for key, spec in FIELDS.items():
            setattr(self, key, spec[1])
        self._stop_all = self._stop_scan = self._teardown_requested = False
        self.pause_requested = self.paused = False
        self._resume_run = Event()
        self.running = True
        self.plates = [plate('A'), plate('B')]
        self.selected_plate = -1
        self.events = []
        self.visited = []

    def _ui(self, callback, cleanup=False):
        if self._stop_all and not cleanup:
            raise RunCancelled()
        return callback()

    def _load_plate(self, value):
        self.current = value['name']
        self.visited.append(self.current)

    def _status(self, text, *args):
        self.events.append(text)

    def _begin_scan_camera(self):
        self.events.append('prepare')
        return True

    def _end_scan_camera(self, found):
        self.events.append(('restore', found))

    def _find_scan_z(self):
        return 140

    def _scan(self, z):
        return self.current == 'B'

    def _track_visit(self, p, folder):
        self.events.append(('track', p['name'], folder))
        return 'Visit complete'

    def _quiesce_visit(self):
        self.events.append('quiesced')


@pytest.fixture
def app(monkeypatch):
    result = SimpleNamespace(stage=SimpleNamespace(emergency_stop=lambda: None),
                             bind_keys=lambda: None)
    monkeypatch.setattr(plate_run.App, 'get_running_app', staticmethod(lambda: result))
    monkeypatch.setattr(plate_run.Clock, 'schedule_once', lambda callback: callback(0))
    return result


def test_empty_plate_advances_and_final_cleanup_runs(app):
    run = SimulatedRun()
    run._execute_plan(run.plates, None, False)
    assert run.visited == ['A', 'B']
    assert [('track', 'B', None)] == [event for event in run.events if isinstance(event, tuple) and event[0] == 'track']
    assert any('No worm found' in event for event in run.events if isinstance(event, str))
    assert run.events[-1] == 'quiesced'
    assert run.run_status == 'Run finished — 1 tracked, 1 no worm found'
    assert not run.running


def test_incomplete_search_is_preserved_in_final_status_and_other_plates_still_run(app):
    run = SimulatedRun()
    def scan(z):
        if run.current == 'A':
            run._search_report = SearchReport(2, 20, 'rows')
            return False
        return True
    run._scan = scan
    run._execute_plan(run.plates, None, False)
    assert run.visited == ['A', 'B']
    assert run.run_status.startswith('Run incomplete — 1 tracked')
    assert 'A: Search incomplete — 2/20 rows checked' in run.run_status
    assert any('Search incomplete' in event for event in run.events if isinstance(event, str))
    assert not run.running


def test_all_successful_tracking_visits_report_complete(app):
    run = SimulatedRun()
    run._scan = lambda z: True
    run._execute_plan(run.plates, None, False)
    assert run.run_status == 'Run complete'


def test_stop_during_search_does_not_start_tracking_or_next_plate(app):
    run = SimulatedRun()
    def scan(z):
        run._stop_all = True
        return True
    run._scan = scan
    run._execute_plan(run.plates, None, True)
    assert run.visited == ['A']
    assert not any(isinstance(event, tuple) and event[0] == 'track' for event in run.events)
    assert run.run_status == 'Run stopped'


def test_focus_failure_restores_camera_and_stops_run(app):
    run = SimulatedRun()
    run._find_scan_z = lambda: None
    run._execute_plan(run.plates, None, False)
    assert run.visited == ['A']
    assert ('restore', False) in run.events
    assert 'focus search failed' in run.run_status


def test_display_restore_failure_still_releases_run_controls(app):
    run = SimulatedRun()
    run.selected_plate = 0
    restored = []
    load = run._load_plate
    def load_plate(p):
        if run.events and run.events[-1] == 'quiesced':
            raise ValueError('invalid plate display')
        load(p)
    run._load_plate = load_plate
    run._scan = lambda z: (_ for _ in ()).throw(RuntimeError('row failed'))
    app.bind_keys = lambda: restored.append(True)
    run._execute_plan(run.plates, None, False)
    assert not run.running and not app._plate_run_active
    assert 'row failed' in run.run_status
    assert 'could not restore plate display' in run.run_status
    assert restored == [True]


def test_repeat_uses_new_visit_folders(app, tmp_path):
    run = SimulatedRun()
    folders = []
    def track(p, folder):
        folders.append(folder)
        if len(folders) == 2:
            run._stop_all = True
        return 'Visit complete'
    run._track_visit = track
    run._execute_plan(run.plates, tmp_path, True)
    assert run.visited == ['A', 'B', 'A', 'B']
    assert folders[0].name == 'visit_0001'
    assert folders[1].name == 'visit_0002'
    assert folders[0] != folders[1]


def test_pause_occurs_between_plates_and_resume_continues(app):
    run = SimulatedRun()
    run.pause_requested = True
    class ResumeAtBoundary:
        def clear(self):
            assert run.visited == ['A']
        def wait(self, timeout):
            assert run.paused
            run.toggle_pause()
            return True
        def set(self):
            pass
    run._resume_run = ResumeAtBoundary()
    run._execute_plan(run.plates, None, False)
    assert run.visited == ['A', 'B']
    assert run.run_status == 'Run finished — 1 tracked, 1 no worm found'


def test_recording_failure_cleans_up_and_restores_settings(app, tmp_path):
    config = ConfigParser()
    config.filename = str(tmp_path / 'settings.ini')
    config.setdefaults('Experiment', {'iscontinuous': '0', 'extension': 'png',
                                      'duration': '5', 'nframes': '150'})
    config.setdefaults('DualColor', {'dualcolormode': 'false'})
    record = SimpleNamespace(state='normal', saveHandoffError='Disk full', acquisitionError=None)
    manager = SimpleNamespace(liveviewbutton=SimpleNamespace(state='normal'), recordbutton=record)
    controls = SimpleNamespace(track_done=Event(), isTracking=False,
        trackingcheckbox=SimpleNamespace(state='normal'), livefocuscheckbox=SimpleNamespace(state='normal'),
        ids=SimpleNamespace(imageacquisitionmanager=manager))
    controls.startTracking = lambda *args: setattr(controls, 'isTracking', True)
    left = SimpleNamespace(savefile=str(tmp_path), ids=SimpleNamespace(camprops=SimpleNamespace(framerate=30)))
    app.root = SimpleNamespace(ids=SimpleNamespace(leftcolumn=left,
                       middlecolumn=SimpleNamespace(ids=SimpleNamespace(runtimecontrols=controls))))
    app.config = config
    app.camera = SimpleNamespace(IsGrabbing=lambda: True)
    app.image = np.zeros((10, 10))
    run = SimulatedRun()
    run.record_format = 'tiff'
    run._prepare_tracking = lambda p: None
    # Exercise the real visit method, while representing hardware shutdown by an event.
    with pytest.raises(RuntimeError, match='Disk full'):
        PlateRunController._track_visit(run, plate('A'), tmp_path / 'visit')
    assert 'quiesced' in run.events
    assert left.savefile == str(tmp_path)
    assert config.get('Experiment', 'iscontinuous') == '0'
    assert config.get('Experiment', 'extension') == 'png'
    assert config.get('Experiment', 'duration') == '5'


@pytest.fixture
def handoff(app, monkeypatch):
    run = SimulatedRun()
    now = [0.0]
    events = []
    clock = SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0],
                            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(plate_run, 'time', clock)

    class Node:
        def __init__(self, name, value):
            self.name, self.value = name, value
        @property
        def Value(self):
            return self.value
        @Value.setter
        def Value(self, value):
            self.value = value
            events.append((self.name, value, now[0]))

    class Manager:
        liveviewbutton = SimpleNamespace(state='normal')
        recordbutton = SimpleNamespace(state='normal')
        @property
        def imageRetrieveTimeStamp(self):
            return now[0]

    manager = Manager()
    controls = SimpleNamespace(track_done=Event(), isTracking=False,
        trackingcheckbox=SimpleNamespace(state='normal'), livefocuscheckbox=Node('focus', 'normal'),
        ids=SimpleNamespace(imageacquisitionmanager=manager), focus_batches=0,
        _focus_brightness_epoch=0, _focus_applied_epoch=0, _focus_fresh_after=0)
    # Record focus transitions using the same property shape as a Kivy button.
    class Focus:
        state = property(lambda self: controls.livefocuscheckbox_node.Value,
                         lambda self, value: setattr(controls.livefocuscheckbox_node, 'Value', value))
    controls.livefocuscheckbox_node = controls.livefocuscheckbox
    controls.livefocuscheckbox = Focus()
    controls.liveFocusThread = SimpleNamespace(
        is_alive=lambda: controls.livefocuscheckbox.state == 'down', join=lambda timeout: None)
    def start_tracking(*args):
        events.append(('tracking', True, now[0]))
        controls.isTracking = True
    controls.startTracking = start_tracking
    app.root = SimpleNamespace(ids=SimpleNamespace(leftcolumn=SimpleNamespace(),
        middlecolumn=SimpleNamespace(ids=SimpleNamespace(runtimecontrols=controls))))
    app.image = np.zeros((10, 10))
    app.config = SimpleNamespace(getboolean=lambda *args: False, getfloat=lambda *args: 15,
                                 getint=lambda *args: 5)
    app.camera = SimpleNamespace(IsGrabbing=lambda: manager.liveviewbutton.state == 'down',
        ResultingFrameRate=lambda: 15,
        ExposureTime=Node('exposure', 100000), Gain=Node('gain', 30),
        AcquisitionFrameRateEnable=Node('fps_enabled', False), AcquisitionFrameRate=Node('fps', 30))
    batch_elapsed = [0.0]
    def brightness(exposure, gain):
        assert controls.isTracking and controls.livefocuscheckbox.state == 'down'
        app.camera.ExposureTime.Value = exposure
        app.camera.Gain.Value = gain
        controls._focus_brightness_epoch += 1
        controls._focus_fresh_after = now[0] + 0.1
        batch_elapsed[0] = 0
    controls.set_tracking_brightness = brightness
    def sleep(seconds):
        now[0] += seconds
        if controls.livefocuscheckbox.state == 'down' and now[0] > controls._focus_fresh_after:
            controls._focus_applied_epoch = controls._focus_brightness_epoch
            batch_elapsed[0] += seconds
            if batch_elapsed[0] >= 1 / 3:
                controls.focus_batches += 1
                batch_elapsed[0] -= 1 / 3
    clock.sleep = sleep
    def quiesce():
        run.events.append('quiesced')
        controls.isTracking = False
        controls.livefocuscheckbox.state = 'normal'
    run._quiesce_visit = quiesce
    p = plate('A')
    p['settings']['track_interval'] = 1
    return run, p, controls, events, now, clock


def test_handoff_keeps_tracking_and_focus_active_through_four_steps_and_pauses(handoff):
    run, p, controls, events, now, clock = handoff
    assert PlateRunController._track_visit(run, p, None) == 'Visit complete'
    exposures = [e for e in events if e[0] == 'exposure']
    started = next(e for e in events if e[0] == 'tracking')
    focus_starts = [e for e in events if e[:2] == ('focus', 'down')]
    assert len(focus_starts) == 1
    assert started[2] < focus_starts[0][2] < exposures[0][2]
    assert exposures[0][2] - focus_starts[0][2] >= 3
    assert [e[1] for e in exposures] == [76250, 52500, 28750, 5000]
    assert [e[1] for e in events if e[0] == 'gain'] == [28, 26, 24, 22]
    pause = p['settings']['exposure_ramp_seconds'] / 4
    for previous, following in zip(exposures, exposures[1:]):
        assert following[2] - previous[2] >= pause
    # The fourth step also gets its pause, before final settling and the timed visit.
    assert now[0] - exposures[-1][2] >= pause + p['settings']['focus_settle_seconds'] + 1
    assert [e for e in events if e[:2] == ('focus', 'normal')][0][2] > exposures[-1][2]


@pytest.mark.parametrize('cancel_at', [0.01, 1.0, 4.0, 20.0])
def test_stop_during_initial_focus_or_ramp_cleans_up(handoff, cancel_at):
    run, p, controls, events, now, clock = handoff
    advance = clock.sleep
    def sleep(seconds):
        advance(seconds)
        if now[0] >= cancel_at:
            run._stop_all = run._stop_scan = True
    clock.sleep = sleep
    with pytest.raises(RunCancelled):
        PlateRunController._track_visit(run, p, None)
    assert run.events[-1] == 'quiesced'
    assert not controls.isTracking
    assert controls.livefocuscheckbox.state == 'normal'
    assert all(e[2] < now[0] for e in events if e[0] == 'exposure')
    if cancel_at < 0.05:
        assert not any(e[0] == 'tracking' for e in events)


def test_ramp_stops_if_autofocus_stops_producing_batches(handoff):
    run, p, controls, events, now, clock = handoff
    clock.sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
    with pytest.raises(RuntimeError, match='enough fresh frames'):
        PlateRunController._track_visit(run, p, None)
    assert not any(e[0] == 'exposure' for e in events)
    assert not controls.isTracking


def test_recording_waits_for_target_exposure_and_final_focus_settling(handoff, app, tmp_path):
    run, p, controls, events, now, clock = handoff
    config = ConfigParser()
    config.filename = str(tmp_path / 'settings.ini')
    config.setdefaults('Autofocus', {'focusfps': '15', 'buffer_n': '5'})
    config.setdefaults('DualColor', {'dualcolormode': 'false'})
    config.setdefaults('Experiment', {'iscontinuous': '0', 'extension': 'png',
                                     'duration': '5', 'nframes': '150'})
    app.config = config
    left = app.root.ids.leftcolumn
    left.savefile = str(tmp_path)
    left.ids = SimpleNamespace(camprops=SimpleNamespace(framerate=30))
    recorded = []
    class Record:
        _state = 'normal'
        @property
        def state(self):
            return self._state
        @state.setter
        def state(self, value):
            self._state = value
            if value == 'down':
                assert controls.isTracking and controls.livefocuscheckbox.state == 'down'
                assert app.camera.ExposureTime.Value == 5000 and app.camera.Gain.Value == 22
                last_change = max(e[2] for e in events if e[0] == 'exposure')
                assert now[0] - last_change >= 3
                recorded.append(now[0])
    controls.ids.imageacquisitionmanager.recordbutton = Record()
    run.record_format = 'tiff'
    assert PlateRunController._track_visit(run, p, tmp_path / 'visit') == 'Visit complete'
    assert len(recorded) == 1
    assert now[0] - recorded[0] >= p['settings']['track_interval']
    assert left.savefile == str(tmp_path)


def test_missing_fresh_frames_prevents_focus_and_tracking(handoff, monkeypatch):
    run, p, controls, events, now, clock = handoff
    manager = controls.ids.imageacquisitionmanager
    monkeypatch.setattr(type(manager), 'imageRetrieveTimeStamp', property(lambda self: 0))
    with pytest.raises(RuntimeError, match='fresh frame'):
        PlateRunController._track_visit(run, p, None)
    assert not any(e[0] in ('tracking', 'exposure') or e[:2] == ('focus', 'down') for e in events)
    assert run.events[-1] == 'quiesced'


def test_tracking_ending_early_is_reported_as_run_failure(handoff, app):
    run, p, controls, events, now, clock = handoff
    prepare = run._prepare_tracking
    def tracking_ends(p):
        prepare(p)
        controls.track_done.set()
    run._prepare_tracking = tracking_ends
    run._scan = lambda z: True
    run._track_visit = lambda p, folder: PlateRunController._track_visit(run, p, folder)
    run._execute_plan([p], None, False)
    assert run.run_status == 'Run stopped: Tracking ended before the visit finished'
    assert not run.running
