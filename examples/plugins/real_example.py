import math


class Controller:

    voltage = 2.0        
    zap_frames = 10    
    wait_frames = 600  
    
    def setup(self, scope):
        self.countdown = self.wait_frames
        self.zap_index = 0
        self.origin = None         
        self.frames_since_zap = 0  
        scope.print(f'first zap in {self.countdown} frames')

    def update(self, state, scope):
        if not state.is_tracking:
            scope.light_off()
            return

        self.countdown -= 1
        if self.countdown <= 0:
            self._start_zap(state, scope)

        light_on = self.origin is not None and self.frames_since_zap < self.zap_frames
        scope.set_voltage(self.voltage if light_on else 0.0)

        if self.origin is not None:
            self._log(state, scope, light_on)
            self.frames_since_zap += 1

    def teardown(self, scope):
        scope.light_off()

    def _start_zap(self, state, scope):
        self.zap_index += 1
        self.origin = state.worm_xy
        self.frames_since_zap = 0
        self.countdown = self.wait_frames
        scope.log(event='zap_start', zap=self.zap_index, frame=state.frame, t=state.time_s,
                  origin=self.origin, fps=state.fps)
        scope.print(f'zap {self.zap_index}, {self.voltage} V')

    def _log(self, state, scope, light_on):
        dx = state.worm_xy[0] - self.origin[0]
        dy = state.worm_xy[1] - self.origin[1]
        scope.log(event='track', zap=self.zap_index, frames_since_zap=self.frames_since_zap,
                  frame=state.frame, t=state.time_s, light=light_on, worm=state.worm_xy,
                  dx=dx, dy=dy, dist=math.hypot(dx, dy), reversing=state.is_reversing)
