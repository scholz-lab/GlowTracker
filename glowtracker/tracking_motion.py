"""Predict where a stage axis is while it moves, from the targets it was sent.

Used by continuous tracking (Settings > Tracking > Continuous tracking): instead of waiting for
the stage to stop before each image, every frame is used, and the stage position at that frame's
exposure is taken from this model. Each axis follows the usual trapezoidal profile of a Zaber
controller: constant acceleration `accel` up to `vmax`, cruise, constant deceleration. A new
target mid-move is planned from the axis's current position and velocity, as the controller does.

Pure Python, no hardware: positions in mm, times in seconds (time.perf_counter()).
"""
from __future__ import annotations

import math
from collections import deque

HISTORY = 64            # earlier plans kept, so positions in the recent past can be looked up


class AxisTrajectory:
    """One axis: piecewise-constant acceleration phases starting at `t0` from (`x0`, `v0`)."""

    def __init__(self, position: float, vmax: float, accel: float, t: float = 0.0):
        if vmax <= 0 or accel <= 0:
            raise ValueError('vmax and accel must be positive')
        self.vmax = float(vmax)
        self.accel = float(accel)
        self.target = float(position)
        self._t0, self._x0, self._v0 = float(t), float(position), 0.0
        self._phases: list[tuple[float, float]] = []      # (duration, acceleration)
        # (t0, x0, v0, phases) of earlier plans; a frame's exposure is often a little before the
        # latest retarget, so its position must come from the plan that was running then.
        self._history: deque = deque(maxlen=HISTORY)

    # --- state at a time ---------------------------------------------------------------------
    def state(self, t: float) -> tuple[float, float]:
        """(position, velocity) at time t, from the plan that was running at t (before the
        oldest kept plan, the axis is taken to be where that plan started)."""
        t0, x, v, phases = self._t0, self._x0, self._v0, self._phases
        if t < t0:
            for h_t0, h_x0, h_v0, h_phases in reversed(self._history):
                t0, x, v, phases = h_t0, h_x0, h_v0, h_phases
                if t >= h_t0:
                    break
        dt = max(0.0, t - t0)
        for duration, a in phases:
            step = min(dt, duration)
            x += v * step + 0.5 * a * step * step
            v += a * step
            dt -= step
            if dt <= 0:
                return x, v
        return x + v * dt, v            # after the last phase v is 0 (at the target)

    def position(self, t: float) -> float:
        return self.state(t)[0]

    def finish_time(self) -> float:
        return self._t0 + sum(d for d, _ in self._phases)

    def moving(self, t: float) -> bool:
        return t < self.finish_time()

    # --- planning ----------------------------------------------------------------------------
    def retarget(self, t: float, target: float) -> None:
        """Send a new target at time t; the axis continues from where the model says it is."""
        x, v = self.state(t)
        self._history.append((self._t0, self._x0, self._v0, self._phases))
        self._t0, self._x0, self._v0 = float(t), x, v
        self.target = float(target)
        self._phases = self._plan(x, v, float(target))

    def _plan(self, x: float, v: float, target: float) -> list[tuple[float, float]]:
        a, vmax = self.accel, self.vmax
        phases: list[tuple[float, float]] = []
        d = target - x
        if abs(d) < 1e-12 and abs(v) < 1e-12:
            return phases
        s = math.copysign(1.0, d) if d != 0 else -math.copysign(1.0, v)
        stop_dist = v * v / (2 * a)
        # Moving away from the target, or too fast to stop before it: brake to rest first.
        if v * s < 0 or stop_dist > abs(d) + 1e-12:
            t_stop = abs(v) / a
            phases.append((t_stop, -math.copysign(a, v)))
            x += math.copysign(stop_dist, v)
            return phases + self._plan(x, 0.0, target)
        u = abs(v)                                          # speed towards the target
        if u > vmax:                                        # (only if vmax was lowered mid-move)
            phases.append(((u - vmax) / a, -s * a))
            dist = (u * u - vmax * vmax) / (2 * a)
            return phases + self._plan(x + s * dist, s * vmax, target)
        dist = abs(d)
        peak = math.sqrt((2 * a * dist + u * u) / 2)       # triangle: accelerate to peak, then brake
        if peak <= vmax:
            phases.append(((peak - u) / a, s * a))
            phases.append((peak / a, -s * a))
        else:                                               # trapezoid: reach vmax, cruise, brake
            accel_dist = (vmax * vmax - u * u) / (2 * a)
            brake_dist = vmax * vmax / (2 * a)
            phases.append(((vmax - u) / a, s * a))
            phases.append(((dist - accel_dist - brake_dist) / vmax, 0.0))
            phases.append((vmax / a, -s * a))
        return [(d_, acc) for d_, acc in phases if d_ > 0]


class StageModel:
    """X and Y axes together."""

    def __init__(self, x: float, y: float, vmax: float, accel: float, t: float):
        self.x = AxisTrajectory(x, vmax, accel, t)
        self.y = AxisTrajectory(y, vmax, accel, t)

    def position(self, t: float) -> tuple[float, float]:
        return self.x.position(t), self.y.position(t)

    def retarget(self, t: float, x: float | None = None, y: float | None = None) -> None:
        if x is not None:
            self.x.retarget(t, x)
        if y is not None:
            self.y.retarget(t, y)

    @property
    def target(self) -> tuple[float, float]:
        return self.x.target, self.y.target
