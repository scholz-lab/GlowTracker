"""Smallest useful plugin: blink the LED on a fixed schedule and log where the animal is.

It shows the three things every plugin does: read `state`, call `scope.set_voltage`, and
`scope.log`. Load it in DAQ > Plugin and press Start.
"""


class Controller:

    on_frames = 10      # frames with the light on
    off_frames = 40     # frames with the light off
    voltage = 2.0

    def setup(self, scope):
        self.counter = 0
        scope.print('blink plugin started')

    def update(self, state, scope):
        period = self.on_frames + self.off_frames
        light_on = (self.counter % period) < self.on_frames
        self.counter += 1

        scope.set_voltage(self.voltage if light_on else 0.0)

        scope.log(frame=state.frame, t=state.time_s, worm=state.worm_xy,
                  tracking=state.is_tracking, reversing=state.is_reversing, light=light_on)

    def teardown(self, scope):
        scope.print(f'blink plugin stopped after {self.counter} frames')
