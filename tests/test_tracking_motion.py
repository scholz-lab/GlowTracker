"""The stage motion model used by continuous tracking."""
import math

import numpy as np
import pytest

from tracking_motion import AxisTrajectory, StageModel


def simulate(axis_vmax, axis_accel, x0, commands, t_end, dt=1e-4):
    """Brute-force reference: a controller that, every dt, accelerates at most `accel` towards the
    current target and brakes in time to stop on it, never exceeding vmax."""
    x, v, target, out = x0, 0.0, x0, []
    cmds = sorted(commands)
    t = 0.0
    while t <= t_end + 1e-12:
        while cmds and cmds[0][0] <= t + 1e-12:
            target = cmds.pop(0)[1]
        d = target - x
        stop = v * v / (2 * axis_accel)
        v_old = v
        if abs(d) < 1e-9 and abs(v) < axis_accel * dt:
            v = 0.0
        elif v * d < 0 or stop >= abs(d):                       # brake
            v -= math.copysign(min(abs(v), axis_accel * dt), v)
        else:                                                   # speed up towards the target
            v += math.copysign(axis_accel * dt, d)
            v = max(-axis_vmax, min(axis_vmax, v))
        x += 0.5 * (v_old + v) * dt                             # exact for constant acceleration
        out.append((t, x))
        t += dt
    return out


def test_short_move_matches_the_triangle_profile_and_estimate():
    axis = AxisTrajectory(10.0, vmax=20.0, accel=200.0, t=0.0)
    axis.retarget(0.0, 10.05)                                   # a 50 um tracking correction
    t_end = axis.finish_time()
    assert t_end == pytest.approx(2 * math.sqrt(0.05 / 200.0))  # same formula as estimateTravelTime
    assert axis.position(t_end / 2) == pytest.approx(10.025)    # halfway in time = halfway in space
    assert axis.position(t_end + 1) == pytest.approx(10.05)
    assert not axis.moving(t_end + 1e-6)


def test_long_move_cruises_at_vmax():
    axis = AxisTrajectory(0.0, vmax=20.0, accel=200.0, t=0.0)
    axis.retarget(0.0, 10.0)
    assert axis.finish_time() == pytest.approx(2 * 0.1 + (10.0 - 2.0) / 20.0)
    assert axis.state(0.3)[1] == pytest.approx(20.0)


@pytest.mark.parametrize('commands', [
    [(0.0, 0.2), (0.02, 0.25)],                     # same direction, extended mid-move
    [(0.0, 0.2), (0.02, -0.1)],                     # reversed mid-move
    [(0.0, 0.3), (0.015, 0.05)],                    # new target closer than the stopping distance
    [(0.0, 5.0), (0.1, 5.5), (0.2, 4.0)],           # several retargets while cruising
])
def test_retargeting_matches_a_brute_force_controller(commands):
    vmax, accel = 20.0, 200.0
    axis = AxisTrajectory(0.0, vmax, accel, t=0.0)
    for t, target in commands:
        axis.retarget(t, target)
    t_end = axis.finish_time() + 0.05
    ref = simulate(vmax, accel, 0.0, commands, t_end)
    err = max(abs(axis.position(t) - x) for t, x in ref[::50])
    # within 3 um over the whole move; the reference itself is only exact to one time step
    # (dt x vmax = 0.1 ms x 20 mm/s = 2 um)
    assert err < 3e-3


def test_stage_model_moves_x_and_y_independently():
    model = StageModel(1.0, 2.0, vmax=20.0, accel=200.0, t=0.0)
    model.retarget(0.0, x=1.1)
    assert model.position(1.0) == pytest.approx((1.1, 2.0))
    assert model.target == (1.1, 2.0)
