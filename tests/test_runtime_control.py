from threading import Event, Lock, Thread
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import time
from plate_run import PlateRunController
import plate_run

from runtime_control import (
    ManagedStageMove,
    append_new_focus_values,
    controller_velocity,
)


@pytest.mark.parametrize('grabbing', [False, True])
def test_tracking_cleanup_restores_roi_without_restarting_stopped_camera(grabbing):
    source = ast.parse((Path(__file__).parents[1] / 'glowtracker/GlowTracker.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'RuntimeControls')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'stopTracking')
    calls = []
    camera = SimpleNamespace(IsGrabbing=lambda: grabbing, setIsOnHold=lambda value: calls.append(('hold', value)),
        AcquisitionStop=SimpleNamespace(Execute=lambda: calls.append('stop')),
        AcquisitionStart=SimpleNamespace(Execute=lambda: calls.append('start')),
        AcquisitionFrameRate=lambda: 30)
    values = {'Width': 1024, 'Height': 768, 'OffsetX': 0, 'OffsetY': 0, 'CenterX': 0, 'CenterY': 0}
    for name in (*values, 'TLParamsLocked'):
        setattr(camera, name, SimpleNamespace(Value=None))
    app = SimpleNamespace(camera=camera, stage=None, config=SimpleNamespace(getboolean=lambda *args: False),
        root=SimpleNamespace(ids=SimpleNamespace(leftcolumn=SimpleNamespace(cameraConfig=values),
            middlecolumn=SimpleNamespace(ids=SimpleNamespace(imageoverlay=SimpleNamespace(updateOverlay=lambda: None))))))
    namespace = {'App': SimpleNamespace(get_running_app=lambda: app), 'time': SimpleNamespace(sleep=lambda _: None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<stopTracking>', 'exec'), namespace)
    controls = SimpleNamespace(track_done=Event(), coord_updateevent=None)
    namespace['stopTracking'](controls)
    assert camera.Width.Value == 1024 and camera.Height.Value == 768
    assert ('start' in calls) is grabbing
    assert ('stop' in calls) is grabbing
    assert calls[-1] == ('hold', False)


def test_plate_stop_joins_tracker_even_when_no_new_camera_frames_arrive(monkeypatch):
    source = ast.parse((Path(__file__).parents[1] / 'glowtracker/GlowTracker.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'RuntimeControls')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_trackingLoop')
    waiting = Event()
    stops = []
    def grabbing():
        waiting.set()
        return True
    manager = SimpleNamespace(imageRetrieveTimeStamp=0, liveviewbutton=SimpleNamespace(state='down'),
                              recordbutton=SimpleNamespace(state='normal'))
    controls = SimpleNamespace(isTracking=True, trackingcheckbox=SimpleNamespace(state='down'),
        livefocuscheckbox=SimpleNamespace(state='normal'), track_done=Event(),
        imageacquisitionmanager=manager, ids=SimpleNamespace(imageacquisitionmanager=manager))
    app = SimpleNamespace(camera=SimpleNamespace(IsGrabbing=grabbing),
        config=SimpleNamespace(getboolean=lambda *args: False),
        stage=SimpleNamespace(emergency_stop=lambda: stops.append('stop')),
        root=SimpleNamespace(ids=SimpleNamespace(middlecolumn=SimpleNamespace(
            ids=SimpleNamespace(runtimecontrols=controls)))))
    namespace = {'App': SimpleNamespace(get_running_app=lambda: app), 'time': time,
                 'List': list, 'Vec3': tuple}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<trackingLoop>', 'exec'), namespace)
    errors = []
    def track():
        try:
            namespace['_trackingLoop'](controls, 1, 'um', 10, 1, False, 20, 100, 'CMS', 0, 255, [])
        except Exception as error:
            errors.append(error)
    controls.trackthread = Thread(target=track, daemon=True)
    controls.trackthread.start()
    assert waiting.wait(1)
    monkeypatch.setattr(plate_run.App, 'get_running_app', staticmethod(lambda: app))
    run = PlateRunController()
    run._teardown_requested = False
    run._ui = lambda callback, **kwargs: callback()
    try:
        run._quiesce_visit()
        assert not controls.trackthread.is_alive()
        assert not errors
        assert controls.track_done.is_set()
        assert controls.trackingcheckbox.state == 'normal'
        assert manager.liveviewbutton.state == 'normal'
        assert stops == ['stop']
    finally:
        controls.isTracking = False
        controls.trackingcheckbox.state = 'normal'
        controls.trackthread.join(1)


class BlockingStage:
    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.moves = []
        self.stops = 0

    def move_abs(self, target, unit, wait_until_idle):
        self.moves.append((target, unit, wait_until_idle))
        self.started.set()
        self.release.wait(1)
        return True

    def emergency_stop(self):
        self.stops += 1
        self.release.set()


def test_stage_move_is_single_flight_and_cancellable():
    stage = BlockingStage()
    completed = []
    worker = ManagedStageMove(blocked=True)
    assert not worker.start(stage, [1, 2, 3])
    worker.allow()
    assert worker.start(stage, [1, 2, 3], on_success=lambda: completed.append(1))
    assert stage.started.wait(1)
    assert not worker.start(stage, [4, 5, 6])
    worker.request_stop(block_new=True)
    assert worker.wait(1)
    assert stage.stops == 1
    assert len(stage.moves) == 1
    assert completed == []
    assert not worker.start(stage, [4, 5, 6])


def test_stage_move_runs_success_callback_after_completion():
    stage = BlockingStage()
    completed = []
    worker = ManagedStageMove(blocked=False)
    assert worker.start(stage, [1, 2, 3], on_success=lambda: completed.append(1))
    assert stage.started.wait(1)
    stage.release.set()
    assert worker.wait(1)
    assert completed == [1]


def test_stage_move_does_not_start_during_teardown():
    stage = BlockingStage()
    worker = ManagedStageMove(blocked=False)
    assert not worker.start(stage, [1, 2, 3], teardown_requested=lambda: True)
    assert stage.moves == []


def test_controller_velocity_preserves_direction_and_deadband():
    assert controller_velocity(-32767, 20, 0.5) == -20
    assert controller_velocity(32767, 20, 0.5) == 20
    assert controller_velocity(0, 20, 0.5) is None


def test_focus_graph_only_appends_available_values_once():
    graph_x = []
    graph_y = []
    lock = Lock()
    assert append_new_focus_values([], graph_x, graph_y, lock) == 0
    assert append_new_focus_values([10, 20], graph_x, graph_y, lock) == 2
    assert append_new_focus_values([10, 20], graph_x, graph_y, lock) == 0
    assert append_new_focus_values([10, 20, 30], graph_x, graph_y, lock) == 1
    assert graph_x == [0, 1, 2]
    assert graph_y == [10, 20, 30]
