from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest
import numpy as np
from kivy.config import ConfigParser

import plate_run
from plate_plan import FIELDS, validate_plate
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
    assert any('Nothing found' in event for event in run.events if isinstance(event, str))
    assert run.events[-1] == 'quiesced'
    assert run.run_status == 'Run complete'
    assert not run.running


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
    assert run.run_status == 'Run complete'


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
    # Exercise the real visit method, while representing hardware shutdown by an event.
    with pytest.raises(RuntimeError, match='Disk full'):
        PlateRunController._track_visit(run, plate('A'), tmp_path / 'visit')
    assert 'quiesced' in run.events
    assert left.savefile == str(tmp_path)
    assert config.get('Experiment', 'iscontinuous') == '0'
    assert config.get('Experiment', 'extension') == 'png'
    assert config.get('Experiment', 'duration') == '5'
