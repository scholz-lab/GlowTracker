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
        self.no_response_commands = []

    def get_position(self, unit):
        self.read_threads.append(threading.current_thread().name)
        return self.position

    def move_velocity(self, velocity, unit):
        self.velocities.append(velocity)

    def stop(self, wait_until_idle=False):
        self.stops += 1

    def generic_command_no_response(self, command):
        self.no_response_commands.append(command)


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


def test_two_axis_stage_exposes_zero_z_in_reads_and_cache(monkeypatch):
    monkeypatch.setattr(zaber, 'POSITION_POLL_INTERVAL', 0.01)
    stage, axes = make_stage(monkeypatch)
    stage.axis_z = None
    stage.no_axes = 2

    assert stage.get_position(unit='mm', isAsync=False) == [10.0, 20.0, 0.0]
    axes[0].read_threads.clear()
    axes[1].read_threads.clear()
    axes[2].read_threads.clear()

    try:
        assert stage.start_position_poller()
        assert wait_until(lambda: bool(axes[0].read_threads and axes[1].read_threads))
        assert stage.get_cached_position() == [10.0, 20.0, 0.0]
        assert axes[2].read_threads == []
    finally:
        assert stage.stop_position_poller()


def test_failed_y_jog_position_poll_stops_stage(monkeypatch):
    monkeypatch.setattr(zaber, 'JOG_SAFETY_POLL_INTERVAL', 0.01)
    stage, axes = make_stage(monkeypatch)
    stopped = threading.Event()
    stage.state.isMoving_y = True
    stage._jog_velocity[1] = -1.0

    def fail_position(unit):
        raise RuntimeError('position read failed')

    axes[0].get_position = fail_position

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
        assert axes[1].no_response_commands == []
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


def test_interactive_stop_and_next_start_do_not_block_caller(monkeypatch):
    stage, axes = make_stage(monkeypatch)
    read_started = threading.Event()
    release_read = threading.Event()
    events = []

    def slow_position(unit):
        events.append('x-read')
        read_started.set()
        release_read.wait(1.0)
        return 10.0

    def y_position(unit):
        events.append('y-read')
        return 20.0

    def no_response(command):
        events.append(command)
        axes[0].no_response_commands.append(command)

    axes[0].get_position = slow_position
    axes[0].generic_command_no_response = no_response
    axes[1].get_position = y_position
    stage.state.isMoving_x = True

    try:
        assert stage.start_position_poller()
        assert read_started.wait(1.0)
        assert stage.request_stop(zaber.AxisEnum.X)
        release_read.set()
        assert wait_until(lambda: axes[0].no_response_commands == ['stop'])
        assert wait_until(lambda: 'y-read' in events)
        assert events.index('stop') < events.index('y-read')
        assert stage.request_start_move((-1.0, 0.0, 0.0), 'mm/s')
        assert stage.request_stop(zaber.AxisEnum.X)
        assert wait_until(lambda: axes[0].velocities == [-1.0])
        assert wait_until(lambda: axes[0].no_response_commands == ['stop', 'stop'])
        assert axes[0].stops == 0
        assert not stage.state.isMoving_x
    finally:
        release_read.set()
        assert stage.stop_position_poller()


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

    key_up = next(
        node
        for node in app_class.body
        if isinstance(node, ast.FunctionDef) and node.name == '_keyup'
    )
    key_up_calls = {
        node.func.attr
        for node in ast.walk(key_up)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert 'stop' not in key_up_calls
    assert 'request_stage_stop' in key_up_calls

    key_down = next(
        node
        for node in app_class.body
        if isinstance(node, ast.FunctionDef) and node.name == '_keydown'
    )
    key_down_calls = {
        node.func.attr
        for node in ast.walk(key_down)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert 'start_move' not in key_down_calls
    assert 'set_accel' not in key_down_calls
    assert 'convert_units' not in key_down_calls
    assert 'request_jog' in key_down_calls

    key_down_assignments = {
        target.attr
        for node in ast.walk(key_down)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == 'self'
    }
    assert 'coords' not in key_down_assignments

    layout = (
        Path(__file__).resolve().parents[1]
        / 'glowtracker'
        / 'layout.kv'
    ).read_text()
    assert 'app.stage.stop()' not in layout


# --- move_xy_rel (tracking) ---------------------------------------------------------------

class AsyncAxis(FakeAxis):
    """Answers each move after `latency` s, like a stage reply over the serial line."""
    latency = 0.05

    def __init__(self, position):
        super().__init__(position)
        self.moves = []

    events = []                                         # shared: ('sent'|'replied', step) in order

    async def move_relative_async(self, step, unit, wait_until_idle=False):
        import asyncio
        self.moves.append(step)
        AsyncAxis.events.append(('sent', step))
        await asyncio.sleep(self.latency)
        AsyncAxis.events.append(('replied', step))


def make_xy_stage(monkeypatch, cached=None):
    stage, _ = make_stage(monkeypatch)
    stage.axis_x, stage.axis_y, stage.axis_z = AsyncAxis(0), AsyncAxis(0), AsyncAxis(0)
    reads = []
    monkeypatch.setattr(stage, '_safe_position_mm', lambda max_age=0.3: reads.append(1) or list(cached or [0, 0, 0]))
    if cached is not None:
        with stage._position_cache_condition:
            stage._last_pos, stage._last_pos_time = list(cached), time.monotonic()
    return stage, reads


def test_xy_move_sends_both_axes_without_waiting_for_each_other(monkeypatch):
    stage, _ = make_xy_stage(monkeypatch, cached=[50.0, 100.0, 20.0])
    AsyncAxis.events.clear()
    assert stage.move_xy_rel(0.2, -0.3, unit='mm')
    assert stage.axis_x.moves == [0.2] and stage.axis_y.moves == [-0.3]
    # both commands go out before either reply comes back: one round trip, not two
    assert [kind for kind, _ in AsyncAxis.events] == ['sent', 'sent', 'replied', 'replied']


def test_xy_move_skips_the_safety_read_far_from_keepout_and_checks_near_it(monkeypatch):
    far, reads = make_xy_stage(monkeypatch, cached=[50.0, 100.0, 140.0])   # high Z, but Y far away
    assert far.move_xy_rel(0, -0.5, unit='mm') and reads == []
    y_edge = zaber.Stage.KEEPOUT_Y + zaber.Stage.KEEPOUT_MARGIN
    near, reads = make_xy_stage(monkeypatch, cached=[50.0, y_edge + 0.2, 140.0])
    assert not near.move_xy_rel(0, -1.0, unit='mm')                       # would enter the zone
    assert reads and near.axis_y.moves == []
    assert near.move_xy_rel(0, +1.0, unit='mm')                           # moving away is fine


def test_xy_move_zero_steps_and_unknown_position(monkeypatch):
    stage, reads = make_xy_stage(monkeypatch, cached=None)
    assert stage.move_xy_rel(0, 0, unit='mm') and stage.axis_x.moves == [] == stage.axis_y.moves
    assert stage.move_xy_rel(0.1, 0, unit='mm') and reads == []           # X only: no keep-out check


def test_xy_move_works_from_a_worker_thread(monkeypatch):
    stage, _ = make_xy_stage(monkeypatch, cached=[50.0, 100.0, 20.0])
    results = []
    worker = threading.Thread(target=lambda: results.extend(
        [stage.move_xy_rel(0.1, 0.1, unit='mm'), stage.move_xy_rel(-0.1, -0.1, unit='mm')]))
    worker.start(); worker.join(5)
    assert results == [True, True] and stage.axis_x.moves == [0.1, -0.1]
