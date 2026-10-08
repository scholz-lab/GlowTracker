"""Buzz the worm with a sinusoidal drive on DAC0 while recording.

DAC0 drives the buzzer: 0 V = full buzz, 4.5 V = quiet. The voltage follows a
sine wave between v_min (loudest) and v_max (quietest), so the buzz swells and
fades `frequency_hz` times per second. DAC1 (590 nm LED) stays off.

Assumes the buzzer loudness follows the voltage between 0 V and 4.5 V. If the
buzzer is on/off only, it will simply switch at its threshold voltage.

idle_voltage = (4.5, 0.0): when the plugin stops or fails, the buzzer is held
quiet (DAC0 = 4.5 V) and the LED off (DAC1 = 0 V).
"""

import math

frequency_hz = 1.0   # buzz swells this many times per second
v_min = 0.0          # loudest buzz (buzzer is active at 0 V)
v_max = 4.5          # quietest (buzzer is quiet at 4.5 V)

# Buzzer (DAC0) quiet at 4.5 V, LED (DAC1) off at 0 V: hold this when stopped.
idle_voltage = (4.5, 0.0)

start = None


def setup(scope):
    scope.set_voltage(4.5, channel=0)   # buzzer quiet
    scope.set_voltage(0.0, channel=1)   # LED off
    scope.print("sinusoidal buzzer started; waiting for Record")


def update(state, scope):
    global start
    if not state.is_recording:
        start = None
        scope.set_voltage(4.5, channel=0)   # buzzer quiet
        scope.set_voltage(0.0, channel=1)   # LED off
        return
    if start is None:
        start = state.wall_time
        scope.log(event="recording_started")
    t = state.wall_time - start
    # Sine between v_min and v_max: v_min when sin = +1 (loud), v_max when sin = -1 (quiet).
    v = (v_min + v_max) / 2.0 - (v_max - v_min) / 2.0 * math.sin(2.0 * math.pi * frequency_hz * t)
    scope.set_voltage(v, channel=0)
    scope.set_voltage(0.0, channel=1)       # LED stays off


def teardown(scope):
    scope.set_voltage(4.5, channel=0)   # buzzer quiet
    scope.set_voltage(0.0, channel=1)   # LED off
    scope.print("sinusoidal buzzer stopped")
