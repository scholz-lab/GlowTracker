"""Guide the animal to a target point by triggering AWA-driven reversals.

Rule, evaluated every frame while tracking:
    heading  h   = direction of travel from the last `heading_window` trail points
    to_target t  = target - worm position
    angle        = angle between h and t (0 = straight at the target, 180 = straight away)

    if angle > away_angle_deg for `away_frames` frames in a row  -> pulse the light
    after a pulse: wait `refractory_frames` before judging again (let the reversal happen)
    never pulse while the app already sees a reversal
    within `arrive_mm` of the target: light off, done; guidance resumes if the worm
    wanders back out past `leave_mm` (hysteresis)

Margins:
    away_angle_deg  > 90 means sideways travel is tolerated, only clearly-wrong headings fire
    away_frames     debounces a noisy heading estimate
    arrive_mm / leave_mm  hysteresis around the target so arrival does not flicker

Set `target_xy` in stage mm, or leave it None to use the start position + `target_offset`.
"""
import math


class Controller:

    # --- target ---------------------------------------------------------------------------
    target_xy = None                 # (x, y) in stage mm; None -> start position + target_offset
    target_offset = (2.0, 0.0)       # mm, used when target_xy is None
    arrive_mm = 0.5                  # inside this radius the worm counts as arrived
    leave_mm = 0.8                   # guidance resumes only after it wanders back out past this

    # --- decision --------------------------------------------------------------------------
    away_angle_deg = 100.0           # heading must be more than this off the target direction
    away_frames = 5                  # consecutive wrong-heading frames before a pulse
    heading_window = 10              # trail points for the heading estimate
    min_travel_mm = 0.02             # below this much travel in the window the heading is unknown

    # --- stimulus --------------------------------------------------------------------------
    voltage = 4.5                    # DAQ maximum is 4.95 V
    pulse_frames = 30                # light on for this many frames (~1 s at 30 fps)
    refractory_frames = 120          # no new decision for this long after a pulse starts (~4 s)

    # --- re-zap gate: after the refractory period, zap again only once the worm has ---------
    # --- moved rezap_radius_mm from where the last zap fired, OR rezap_frames have passed ---
    rezap_radius_mm = 0.5            # distance from the last zap point that re-arms the zapper
    rezap_frames = 600               # or this many frames since the last zap (~20 s), whichever first

    def setup(self, scope):
        self.target = None
        self.arrived = False
        self.away_count = 0
        self.pulse_left = 0
        self.cooldown = 0
        self.pulses = 0
        self.frames = 0
        self.last_zap_xy = None      # where the last zap fired
        self.last_zap_frame = None   # and on which of our frames
        self.cos_margin = math.cos(math.radians(self.away_angle_deg))
        scope.print('guidance plugin: waiting for tracking')

    def update(self, state, scope):
        if not state.is_tracking:
            scope.light_off()
            return

        if self.target is None:
            self.target = self.target_xy or (state.worm_xy[0] + self.target_offset[0],
                                             state.worm_xy[1] + self.target_offset[1])
            scope.log(event='target', target=self.target, start=state.worm_xy)
            scope.print(f'target ({self.target[0]:.2f}, {self.target[1]:.2f}) mm')

        self.frames += 1
        tx = self.target[0] - state.worm_xy[0]
        ty = self.target[1] - state.worm_xy[1]
        dist = math.hypot(tx, ty)

        # --- arrival with hysteresis ---
        if self.arrived and dist > self.leave_mm:
            self.arrived = False
            scope.print(f'left the target zone ({dist:.2f} mm), guiding again')
        if not self.arrived and dist < self.arrive_mm:
            self.arrived = True
            scope.log(event='arrived', frame=self.frames, t=state.time_s, dist=dist, pulses=self.pulses)
            scope.print(f'arrived: {dist * 1000:.0f} um from target after {self.pulses} pulses')
        if self.arrived:
            scope.light_off()
            self._log(state, scope, dist, None, 'arrived')
            return

        # --- a pulse in progress keeps the light on for its full length ---
        if self.pulse_left > 0:
            self.pulse_left -= 1
            scope.set_voltage(self.voltage)
            self._log(state, scope, dist, None, 'pulse')
            return

        scope.light_off()

        # --- refractory: give the reversal time to happen before judging again ---
        if self.cooldown > 0:
            self.cooldown -= 1
            self._log(state, scope, dist, None, 'cooldown')
            return

        # --- re-zap gate: stay quiet until the worm has moved away from the last zap point
        #     or enough frames have passed, so one stubborn animal is not zapped repeatedly
        #     on the same spot ---
        if self.last_zap_xy is not None:
            moved = math.hypot(state.worm_xy[0] - self.last_zap_xy[0],
                               state.worm_xy[1] - self.last_zap_xy[1])
            waited = self.frames - self.last_zap_frame
            if moved < self.rezap_radius_mm and waited < self.rezap_frames:
                self.away_count = 0
                self._log(state, scope, dist, None, 'holding')
                return

        heading = self._heading(state.trail)
        if heading is None or dist <= 0:
            self.away_count = 0
            self._log(state, scope, dist, None, 'no_heading')
            return

        cos_theta = (heading[0] * tx + heading[1] * ty) / dist
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
        wrong_way = cos_theta < self.cos_margin and not state.is_reversing
        self.away_count = self.away_count + 1 if wrong_way else 0

        if self.away_count >= self.away_frames:
            self.pulses += 1
            self.pulse_left = self.pulse_frames
            self.cooldown = self.refractory_frames
            self.away_count = 0
            self.last_zap_xy = state.worm_xy
            self.last_zap_frame = self.frames
            scope.set_voltage(self.voltage)
            scope.log(event='pulse', pulse=self.pulses, frame=self.frames, t=state.time_s,
                      worm=state.worm_xy, angle_deg=angle, dist=dist)
            scope.print(f'pulse {self.pulses}: heading {angle:.0f} deg off target, {dist:.2f} mm away')
            self._log(state, scope, dist, angle, 'pulse')
        else:
            self._log(state, scope, dist, angle, 'off')

    def teardown(self, scope):
        scope.light_off()

    # --- helpers ----------------------------------------------------------------------------
    def _heading(self, trail):
        if len(trail) < 2:
            return None
        recent = trail[-self.heading_window:]
        vx, vy = recent[-1] - recent[0]
        norm = math.hypot(vx, vy)
        if norm < self.min_travel_mm:
            return None
        return (vx / norm, vy / norm)

    def _log(self, state, scope, dist, angle, action):
        scope.log(event='track', frame=self.frames, t=state.time_s, worm=state.worm_xy,
                  dist=dist, angle_deg=angle, reversing=state.is_reversing,
                  voltage=state.voltage, action=action, pulses=self.pulses)
