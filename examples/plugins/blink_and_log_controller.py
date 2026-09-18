"""
It shows the three things every plugin does: read `state`, call `scope.set_voltage`, and
`scope.log`. Load it in DAQ > Plugin and press Start.
"""


class Controller:

    on_seconds = 0.5    # light on for this long
    off_seconds = 2.0   # then off for this long
    voltage = 2.0

    def setup(self, scope):
        self.counter = 0
        scope.print('blink plugin started')

    def update(self, state, scope):
        # Convert the schedule from seconds to frames using the measured frame rate
        on_frames = state.frames_for(self.on_seconds)
        period = on_frames + state.frames_for(self.off_seconds)
        light_on = (self.counter % period) < on_frames
        self.counter += 1

        scope.set_voltage(self.voltage if light_on else 0.0)

        scope.log(frame=state.frame, t=state.time_s, worm=state.worm_xy,
                  tracking=state.is_tracking, reversing=state.is_reversing, light=light_on)

    def teardown(self, scope):
        scope.print(f'blink plugin stopped after {self.counter} frames')
