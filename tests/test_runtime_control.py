from threading import Event, Lock

from runtime_control import (
    ManagedStageMove,
    append_new_focus_values,
    controller_velocity,
)


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
