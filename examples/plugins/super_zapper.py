import math


class Controller:

    voltage = 2.0
    zap_frames = 10         
    wait_frames = 600       
    heading_window = 10

    def setup(self, scope):
        self.phase = 'wait'
        self.countdown = self.wait_frames
        self.zap_index = 0
        self.origin = None
        self.heading = None
        self.onset_frame = 0
        self.frames_since_onset = 0
        self.max_dist = 0.0
        self.min_along = 0.0
        scope.print(f'will zap in {self.countdown} frames')

    def update(self, state, scope):
        if not state.is_tracking:
            scope.light_off()
            return

        since_onset = self.frames_since_onset
        if self.origin is not None:
            self.frames_since_onset += 1

        if self.phase == 'wait':
            scope.light_off()
            if self.origin is not None:
                self._log_displacement(state, scope, since_onset, light=False)
            self.countdown -= 1
            if self.countdown > 0 and self.countdown % 100 == 0:
                scope.print(f'next zap ({self.zap_frames} frames) in {self.countdown} frames')
            if self.countdown <= 0:
                self._finish_observation(scope)
                self._start_zap(state, scope)
            return

        if self.phase == 'zap':
            scope.set_voltage(self.voltage)
            self._log_displacement(state, scope, since_onset, light=True)
            if since_onset + 1 >= self.zap_frames:
                self.phase = 'wait'
                self.countdown = self.wait_frames
            return

    def teardown(self, scope):
        scope.light_off()

    def _start_zap(self, state, scope):
        self.zap_index += 1
        self.phase = 'zap'
        self.onset_frame = state.frame
        self.frames_since_onset = 1
        self.origin = state.worm_xy
        self.heading = self._heading(state.trail)
        self.max_dist = 0.0
        self.min_along = 0.0
        scope.set_voltage(self.voltage)
        scope.log(event='zap_start', zap=self.zap_index, zap_frames=self.zap_frames,
                  frame=state.frame, t=state.time_s, origin=self.origin, heading=self.heading,
                  fps=state.fps)
        scope.print(f'zap {self.zap_index}: {self.zap_frames} frames at {self.voltage} V')

    def _finish_observation(self, scope):
        if self.origin is None:
            return
        scope.log(event='zap_end', zap=self.zap_index, zap_frames=self.zap_frames,
                  max_dist_mm=self.max_dist, min_along_mm=self.min_along,
                  reversed=self.min_along < -0.05)
        scope.print(f'zap {self.zap_index}: max {self.max_dist * 1000:.0f} um from start, '
                    f'min along-heading {self.min_along * 1000:.0f} um')

    def _log_displacement(self, state, scope, since_onset, light):
        dx = state.worm_xy[0] - self.origin[0]
        dy = state.worm_xy[1] - self.origin[1]
        dist = math.hypot(dx, dy)
        if self.heading is not None:
            hx, hy = self.heading
            along = dx * hx + dy * hy
            across = -dx * hy + dy * hx
        else:
            along = across = None
        self.max_dist = max(self.max_dist, dist)
        if along is not None:
            self.min_along = min(self.min_along, along)
        scope.log(event='track', zap=self.zap_index, zap_frames=self.zap_frames,
                  frames_since_onset=since_onset, frame=state.frame, t=state.time_s,
                  light=light, worm=state.worm_xy, dx=dx, dy=dy, dist=dist,
                  along=along, across=across, reversing=state.is_reversing)

    def _heading(self, trail):
        if len(trail) < 2:
            return None
        recent = trail[-self.heading_window:]
        vx, vy = recent[-1] - recent[0]
        norm = math.hypot(vx, vy)
        if norm < 0.02:          
            return None
        return (vx / norm, vy / norm)
