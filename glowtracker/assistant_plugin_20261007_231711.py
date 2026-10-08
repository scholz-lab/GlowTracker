# Five 1 s pulses of 4.5 V on the 590 nm LED (DAC1), every 20 s,
# starting 10 s after Record. The buzzer on DAC0 is held quiet (4.5 V)
# for the whole run.

pulse_starts_s = [10, 30, 50, 70, 90]
pulse_s = 1.0

LED_ON = 4.5
LED_OFF = 0.0
BUZZER_QUIET = 4.5

# Buzzer is active at 0 V, so when the plugin stops the app holds
# DAC0 at 4.5 V (quiet) and DAC1 at 0 V (LED off).
idle_voltage = (BUZZER_QUIET, LED_OFF)

start = None


def setup(scope):
    scope.set_voltage(BUZZER_QUIET, channel=0)   # buzzer quiet
    scope.set_voltage(LED_OFF, channel=1)        # LED off


def update(state, scope):
    global start
    if not state.is_recording:
        start = None
        scope.set_voltage(BUZZER_QUIET, channel=0)
        scope.set_voltage(LED_OFF, channel=1)
        return
    if start is None:
        start = state.wall_time
        scope.log(pulse="start", t=0.0)
    t = state.wall_time - start
    on = any(p <= t < p + pulse_s for p in pulse_starts_s)
    scope.set_voltage(LED_ON if on else LED_OFF, channel=1)
    if on and not getattr(update, "_was_on", False):
        scope.log(pulse="on", t=t)
    elif not on and getattr(update, "_was_on", False):
        scope.log(pulse="off", t=t)
    update._was_on = on


def teardown(scope):
    scope.set_voltage(BUZZER_QUIET, channel=0)
    scope.set_voltage(LED_OFF, channel=1)
