"""Five 1 s pulses of 4.5 V on both DACs, every 20 s, starting 10 s after Record.

4.5 V = 590 nm LED on and buzzer quiet; 0 V = LED off (the buzzer buzzes, because it
is active at 0 V). idle_voltage = (4.5, 0.0) so that when the plugin stops or fails,
the buzzer is held quiet (DAC0 = 4.5 V) and the LED stays off (DAC1 = 0 V).
"""

pulse_start_s = 10.0    # first pulse, seconds after Record
pulse_period_s = 20.0   # start-to-start interval
pulse_count = 5
pulse_length_s = 1.0

ON = 4.5    # LED on, buzzer quiet
OFF = 0.0   # LED off

# Buzzer (DAC0) is quiet at 4.5 V, LED (DAC1) is off at 0 V: hold this when stopped.
idle_voltage = (4.5, 0.0)

start = None
was_on = False


def setup(scope):
    scope.set_voltage(OFF)
    scope.print("pulse plugin started; waiting for Record")


def update(state, scope):
    global start, was_on
    if not state.is_recording:
        start = None
        was_on = False
        scope.set_voltage(OFF)
        return
    if start is None:
        start = state.wall_time
        scope.log(event="recording_started")
    t = state.wall_time - start
    on = False
    for i in range(pulse_count):
        t0 = pulse_start_s + i * pulse_period_s
        if t0 <= t < t0 + pulse_length_s:
            on = True
            break
    scope.set_voltage(ON if on else OFF)
    if on != was_on:
        scope.log(event="pulse_on" if on else "pulse_off", t=round(t, 3))
    was_on = on


def teardown(scope):
    scope.set_voltage(4.5, channel=0)   # buzzer quiet
    scope.set_voltage(0.0, channel=1)   # LED off
    scope.print("pulse plugin stopped")
