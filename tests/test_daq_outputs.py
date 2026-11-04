"""The stepped trace of the DAQ outputs plot."""
import pytest

from daq_outputs import step_trace


def test_a_pulse_is_drawn_as_steps():
    history = [(0.0, 0.0, 0.0), (70.0, 3.0, 0.0), (70.01, 0.0, 0.0)]
    points = step_trace(history, now=100.0, seconds=60.0, channel=0)
    assert points[0] == (0.0, 0.0)                  # the level when the window starts
    assert [v for _, v in points] == [0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
    assert points[-1] == (1.0, 0.0)
    assert points[1][0] == pytest.approx(0.5)


def test_changes_before_the_window_only_set_the_start_level():
    history = [(0.0, 0.0, 0.0), (10.0, 0.0, 2.0)]
    assert step_trace(history, now=100.0, seconds=60.0, channel=1) == [(0.0, 2.0), (1.0, 2.0)]


def test_a_short_history_starts_where_it_starts():
    points = step_trace([(70.0, 1.0, 0.0)], now=100.0, seconds=60.0, channel=0)
    assert points == [(pytest.approx(0.5), 1.0), (1.0, 1.0)]
