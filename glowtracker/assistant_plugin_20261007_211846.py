"""590 nm light on at 3 V while the worm moves forward.

DAC1 drives the 590 nm LED: 3 V while the worm is tracked, moving and not reversing.
DAC0 drives the buzzer, which buzzes at 0 V and is quiet at 4.5 V: held at 4.5 V always.
"""

# The buzzer on DAC0 is active at 0 V, so the app holds DAC0 at 4.5 V (quiet)
# and DAC1 at 0 V (light off) whenever the plugin is not running.
idle_voltage = (4.5, 0.0)

QUIET = 4.5       # buzzer quiet voltage on DAC0
LIGHT_ON = 3.0    # LED voltage on DAC1 while the worm moves forward
MIN_SPEED = 0.01  # mm per tracking step; below this the worm counts as stationary


class Controller:

    def setup(self, scope):
        self._light_on = False
        scope.set_voltage(QUIET, channel=0)
        scope.set_voltage(0.0, channel=1)
        scope.print('forward-light plugin started')

    def update(self, state, scope):
        # Keep the buzzer quiet no matter what.
        scope.set_voltage(QUIET, channel=0)

        # Forward = tracked, moving, and not flagged as reversing by the app.
        light_on = (state.is_tracking
                    and not state.is_reversing
                    and state.speed > MIN_SPEED)
        scope.set_voltage(LIGHT_ON if light_on else 0.0, channel=1)

        # Log the moment the light switches on or off.
        if light_on != self._light_on:
            self._light_on = light_on
            scope.log(light_on=light_on, speed=state.speed,
                      reversing=state.is_reversing, frame=state.frame)

    def teardown(self, scope):
        scope.set_voltage(QUIET, channel=0)
        scope.set_voltage(0.0, channel=1)
        scope.print('forward-light plugin stopped')
