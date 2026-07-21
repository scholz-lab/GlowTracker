import ast
from pathlib import Path
import threading
import time

import Zaber_control as zaber


class FakeAxis:
    def __init__(self, position):
        self.position = position
        self.read_threads = []
        self.velocities = []
        self.stops = 0

    def get_position(self, unit):
        self.read_threads.append(threading.current_thread().name)
        return self.position

    def move_velocity(self, velocity, unit):
        self.velocities.append(velocity)

    def stop(self, wait_until_idle=False):
        self.stops += 1


def make_stage(monkeypatch):
    axes = [FakeAxis(10.0), FakeAxis(20.0), FakeAxis(30.0)]
    connection = object()

    monkeypatch.setattr(
        zaber.Stage,
        'connect_stage',
        lambda self, port: connection,
    )

    def assign_axes(self):
        self.axis_x, self.axis_y, self.axis_z = axes
        self.no_axes = 3

    monkeypatch.setattr(zaber.Stage, 'assign_axes', assign_axes)
    monkeypatch.setattr(zaber.Stage, 'set_maxspeed', lambda self, value, unit: value)
    monkeypatch.setattr(zaber.Stage, 'set_accel', lambda self, value, unit: value)
    return zaber.Stage('test'), axes


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_position_poller_populates_cache_off_the_main_thread(monkeypatch):
    monkeypatch.setattr(zaber, 'POSITION_POLL_INTERVAL', 0.01)
    stage, axes = make_stage(monkeypatch)
    try:
        assert stage.start_position_poller()
        assert wait_until(lambda: stage.get_cached_position() == [10.0, 20.0, 30.0])
        assert all(axis.read_threads for axis in axes)
        assert {
            thread_name
            for axis in axes
            for thread_name in axis.read_threads
        } == {'StagePositionPoller'}
    finally:
        assert stage.stop_position_poller()


def test_failed_y_jog_position_poll_stops_stage(monkeypatch):
    monkeypatch.setattr(zaber, 'JOG_SAFETY_POLL_INTERVAL', 0.01)
    stage, _ = make_stage(monkeypatch)
    stopped = threading.Event()
    stage.state.isMoving_y = True
    stage._jog_velocity[1] = -1.0
    stage.get_position = lambda unit='mm', isAsync=False: None

    def emergency_stop():
        stage.state = zaber.StageState()
        stopped.set()
        return True

    stage.emergency_stop = emergency_stop
    try:
        assert stage.start_position_poller()
        assert stopped.wait(1.0)
    finally:
        assert stage.stop_position_poller()


def test_y_jog_safety_uses_polled_coordinates(monkeypatch):
    monkeypatch.setattr(zaber, 'JOG_SAFETY_POLL_INTERVAL', 0.01)
    stage, axes = make_stage(monkeypatch)
    axes[1].position = 50.0
    axes[2].position = 130.0
    stage.state.isMoving_y = True
    stage._jog_velocity[1] = -1.0
    try:
        assert stage.start_position_poller()
        assert wait_until(lambda: axes[1].stops == 1)
        assert not stage.state.isMoving_y
    finally:
        assert stage.stop_position_poller()


def test_x_jog_does_not_start_collision_poller(monkeypatch):
    stage, axes = make_stage(monkeypatch)
    starts = []
    stage.start_position_poller = lambda: starts.append(True) or True

    assert stage.start_move((1.0, 0.0, 0.0), 'mm/s')
    assert starts == []
    assert axes[0].velocities == [1.0]

    stage.stop(zaber.AxisEnum.X)
    assert stage.start_move((0.0, -1.0, 0.0), 'mm/s')
    assert starts == [True]
    assert axes[1].velocities == [-1.0]


def test_ui_coordinate_paths_only_read_the_stage_cache():
    source = (
        Path(__file__).resolve().parents[1]
        / 'glowtracker'
        / 'GlowTracker.py'
    ).read_text()
    tree = ast.parse(source)
    app_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'GlowTrackerApp'
    )

    for method_name in ('stage_stop', '_keyup', 'update_coordinates'):
        method = next(
            node
            for node in app_class.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        called_attributes = {
            node.func.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert 'get_position' not in called_attributes
        assert 'get_cached_position' in called_attributes
