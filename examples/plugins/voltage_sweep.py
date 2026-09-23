"""Step the DAQ through a series of voltages so the light intensity can be measured at each.

Holds each voltage for `frames_per_step` frames, from `start_v` to `stop_v` in steps of
`step_v`, then switches off and stops. Does not need tracking. Each step is logged
(`event: step`) with its voltage and start frame, and the recording's coordinate file also
carries the voltage per frame in its daqVol column, so the power-meter readings can be lined up
afterwards. Watch the Plugin tab: it prints the current voltage and the frames remaining.
"""


class Controller:

    start_v = 0.5
    stop_v = 4.5             # DAQ maximum is 4.95 V
    step_v = 0.5
    frames_per_step = 300    # ~10 s at 30 fps
    off_frames_between = 0   # set to e.g. 60 for a dark gap between steps
    channel = None           # None = both outputs, 0 = DAC0 only, 1 = DAC1 only

    def setup(self, scope):
        n = int(round((self.stop_v - self.start_v) / self.step_v)) + 1
        self.levels = [round(self.start_v + i * self.step_v, 3) for i in range(n)]
        self.index = 0
        self.frames_left = self.frames_per_step
        self.in_gap = False
        self.done = False
        scope.print(f'voltage sweep: {self.levels} V, {self.frames_per_step} frames each')
        scope.log(event='sweep_start', levels=self.levels, frames_per_step=self.frames_per_step)
        scope.set_voltage(self.levels[0], self.channel)
        scope.log(event='step', voltage=self.levels[0], frame=0)

    def update(self, state, scope):
        if self.done:
            scope.light_off()
            return

        voltage = 0.0 if self.in_gap else self.levels[self.index]
        scope.set_voltage(voltage, self.channel)
        self.frames_left -= 1
        if self.frames_left % 30 == 0:
            scope.print(f'{voltage:.2f} V, {self.frames_left} frames left, step {self.index + 1}/{len(self.levels)}')
        if self.frames_left > 0:
            return

        if not self.in_gap and self.off_frames_between > 0:
            self.in_gap = True
            self.frames_left = self.off_frames_between
            return

        self.in_gap = False
        self.index += 1
        if self.index >= len(self.levels):
            self.done = True
            scope.light_off()
            scope.log(event='sweep_end', frame=state.frame)
            scope.print('voltage sweep finished, light off')
            return
        self.frames_left = self.frames_per_step
        scope.set_voltage(self.levels[self.index], self.channel)
        scope.log(event='step', voltage=self.levels[self.index], frame=state.frame, t=state.time_s)

    def teardown(self, scope):
        scope.light_off()
