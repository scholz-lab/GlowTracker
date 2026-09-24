"""Guide the animal to a target point by triggering AWA-driven reversals. Version 2.

Same idea as super_zapper.py (zap when the worm heads the wrong way), tuned with the EIM00015
results: a 2 s pulse reverses the worm 84 % of the time with 1.3 s latency, the reversal plus omega
takes 8-12 s and leaves a near-random heading, and spontaneous runs are heavy-tailed (the longer a
run has lasted, the less likely it ends on its own).

What is new compared with v1:
    wait before firing   a wrong-way run must be `away_frames` old (default 3 s). Young runs often
                         end by themselves; old ones are committed and worth a zap.
    roaming gate         judge only while the worm is roaming (speed >= min_speed_um_s over 2 s and
                         straightness >= min_straightness over 3 s). Dwelling worms change heading
                         every few seconds anyway.
    longer refractory    12 s after a pulse, so the next decision sees the heading after the omega.
    outcome scoring      12 s after each pulse the plugin logs an `outcome`: heading error to the
                         target, progress, and whether the pulse helped. Running tally in the tab.
    sham trials          every `sham_every`-th pulse is delivered with the light off (same timing,
                         logged with sham=true) so every session carries its own baseline.
                         Set sham_every = 0 to disable.
    rate cap             at most max_pulses_per_min, plus the re-zap gate from v1.

Set `target_xy` in stage mm, or leave it None to use the start position + `target_offset`.
"""
import math
from collections import deque


class Controller:

    # --- target ---------------------------------------------------------------------------
    target_xy = None                 # (x, y) in stage mm; None -> start position + target_offset
    target_offset = (15.0, 0.0)      # mm, used when target_xy is None
    arrive_mm = 0.5                  # inside this radius the worm counts as arrived
    leave_mm = 0.8                   # guidance resumes only after it wanders back out past this

    # --- decision --------------------------------------------------------------------------
    away_angle_deg = 100.0           # heading must be more than this off the target direction
    away_frames = 45                 # wrong way continuously for this long before a pulse (~1.5 s)
    heading_window = 30              # trail points for the heading estimate
    min_travel_mm = 0.02             # below this much travel in the window the heading is unknown

    # --- roaming gate ----------------------------------------------------------------------
    roaming_gate = False             # OFF: judge in every state; turn on once min_speed / min_straightness are known for the animal
    min_speed_um_s = 100.0           # mean speed over the last 2 s
    min_straightness = 0.5           # net displacement / path length over the last 3 s (0.5 s steps)

    # --- stimulus --------------------------------------------------------------------------
    voltage = 4.5                    # DAQ maximum is 4.95 V
    pulse_frames = 60                # light on for this many frames (~2 s at 30 fps)
    refractory_frames = 360          # no new decision for this long after a pulse starts (~12 s)
    rezap_radius_mm = 0.5            # re-arm once the worm moved this far from the last zap point
    rezap_frames = 300               # or after this many frames (~10 s), whichever first
    max_pulses_per_min = 6

    # --- evaluation and controls -----------------------------------------------------------
    eval_frames = 360                # score the outcome this long after light-on (~12 s)
    success_angle_deg = 90.0         # success = heading within this of the target at evaluation
    sham_every = 5                   # every 5th pulse is a sham (light off); 0 = never

    def setup(self, scope):
        self.target = None
        self.arrived = False
        self.away_count = 0
        self.pulse_left = 0
        self.pulse_voltage = 0.0
        self.cooldown = 0
        self.pulses = 0
        self.shams = 0
        self.frames = 0
        self.last_zap_xy = None
        self.last_zap_frame = None
        self.pulse_frames_log = []
        self.pending = []            # outcomes waiting to be scored
        self.tally = {'stim': [0, 0], 'sham': [0, 0]}   # [successes, n]
        self.hist = deque(maxlen=4 * 30 + 5)
        self.cos_margin = math.cos(math.radians(self.away_angle_deg))
        self.cos_release = math.cos(math.radians(self.away_angle_deg - 15.0))   # hysteresis for a run under way
        scope.print('guidance v2: waiting for tracking')

    def update(self, state, scope):
        if not state.is_tracking:
            scope.light_off()
            self.hist.clear()
            return

        if self.target is None:
            self.target = self.target_xy or (state.worm_xy[0] + self.target_offset[0],
                                             state.worm_xy[1] + self.target_offset[1])
            scope.log(event='target', target=self.target, start=state.worm_xy)
            scope.print(f'target ({self.target[0]:.2f}, {self.target[1]:.2f}) mm')

        self.frames += 1
        self.hist.append((self.frames, state.worm_xy[0], state.worm_xy[1]))
        fps = state.fps if state.fps > 0 else 30.0
        tx = self.target[0] - state.worm_xy[0]
        ty = self.target[1] - state.worm_xy[1]
        dist = math.hypot(tx, ty)

        for p in [p for p in self.pending if self.frames - p['frame'] >= self.eval_frames]:
            self.pending.remove(p)
            self._score(scope, state, p, dist, tx, ty, fps)

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

        # --- a pulse (or sham) in progress ---
        if self.pulse_left > 0:
            self.pulse_left -= 1
            scope.set_voltage(self.pulse_voltage)
            self._log(state, scope, dist, None, 'pulse' if self.pulse_voltage > 0 else 'sham')
            return

        scope.light_off()

        if self.cooldown > 0:
            self.cooldown -= 1
            self.away_count = 0
            self._log(state, scope, dist, None, 'cooldown')
            return

        if self.last_zap_xy is not None:
            moved = math.hypot(state.worm_xy[0] - self.last_zap_xy[0], state.worm_xy[1] - self.last_zap_xy[1])
            if moved < self.rezap_radius_mm and self.frames - self.last_zap_frame < self.rezap_frames:
                self.away_count = 0
                self._log(state, scope, dist, None, 'holding')
                return

        recent = [f for f in self.pulse_frames_log if self.frames - f < 60 * fps]
        if len(recent) >= self.max_pulses_per_min:
            self.away_count = 0
            self._log(state, scope, dist, None, 'rate_limited')
            return

        speed, straight = self._state_metrics(fps)
        if self.roaming_gate and (speed < self.min_speed_um_s or straight < self.min_straightness):
            self.away_count = 0
            self._log(state, scope, dist, None, 'dwelling', speed=speed, straightness=straight)
            return

        heading = self._heading(state.trail)
        if heading is None or dist <= 0:
            self.away_count = 0
            self._log(state, scope, dist, None, 'no_heading')
            return

        cos_theta = (heading[0] * tx + heading[1] * ty) / dist
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
        threshold = self.cos_release if self.away_count > 0 else self.cos_margin
        wrong_way = cos_theta < threshold and not state.is_reversing
        self.away_count = self.away_count + 1 if wrong_way else 0

        if self.away_count >= self.away_frames:
            self._fire(scope, state, dist, angle, fps, speed, straight)
        else:
            self._log(state, scope, dist, angle, 'off', age_s=self.away_count / fps, speed=speed, straightness=straight)

    def teardown(self, scope):
        scope.light_off()
        s, n = self.tally['stim']; hs, hn = self.tally['sham']
        scope.print(f'done: {self.pulses} pulses, stim success {s}/{n}, sham success {hs}/{hn}')

    # --- helpers ----------------------------------------------------------------------------
    def _fire(self, scope, state, dist, angle, fps, speed, straight):
        self.pulses += 1
        sham = self.sham_every > 0 and self.pulses % self.sham_every == 0
        self.shams += int(sham)
        self.pulse_voltage = 0.0 if sham else self.voltage
        self.pulse_left = self.pulse_frames
        self.cooldown = self.refractory_frames
        self.last_zap_xy = state.worm_xy
        self.last_zap_frame = self.frames
        self.pulse_frames_log.append(self.frames)
        age_s = self.away_count / fps
        self.away_count = 0
        self.pending.append({'pulse': self.pulses, 'frame': self.frames, 'dist': dist, 'angle': angle, 'sham': sham, 'age_s': age_s})
        scope.set_voltage(self.pulse_voltage)
        scope.log(event='pulse', pulse=self.pulses, sham=sham, frame=self.frames, t=state.time_s, worm=state.worm_xy,
                  angle_deg=angle, dist=dist, run_age_s=age_s, voltage=self.pulse_voltage, pulse_frames=self.pulse_frames,
                  speed_um_s=speed, straightness=straight)
        scope.print(f'{"SHAM" if sham else "pulse"} {self.pulses}: heading {angle:.0f} deg off for {age_s:.1f} s, {dist:.2f} mm away')
        self._log(state, scope, dist, angle, 'sham' if sham else 'pulse')

    def _score(self, scope, state, p, dist, tx, ty, fps):
        progress = p['dist'] - dist
        heading = self._recent_heading(fps, 1.0)
        angle = None
        if heading is not None and dist > 0:
            cos_t = (heading[0] * tx + heading[1] * ty) / dist
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))
        success = angle is not None and angle < self.success_angle_deg
        key = 'sham' if p['sham'] else 'stim'
        self.tally[key][0] += int(success); self.tally[key][1] += 1
        scope.log(event='outcome', pulse=p['pulse'], sham=p['sham'], run_age_s=p['age_s'], angle_at_pulse=p['angle'],
                  heading_angle_deg=angle, progress_mm=progress, success=success, frame=self.frames, t=state.time_s,
                  tally=self.tally)
        s, n = self.tally['stim']; hs, hn = self.tally['sham']
        scope.print(f'{"sham" if p["sham"] else "pulse"} {p["pulse"]} outcome: heading {"?" if angle is None else round(angle)} deg off, '
                    f'{progress * 1000:+.0f} um -> {"helped" if success else "no"}   [stim {s}/{n}, sham {hs}/{hn}]')

    def _heading(self, trail):
        if len(trail) < 2:
            return None
        recent = trail[-self.heading_window:]
        vx, vy = recent[-1] - recent[0]
        norm = math.hypot(vx, vy)
        if norm < self.min_travel_mm:
            return None
        return (vx / norm, vy / norm)

    def _recent_heading(self, fps, seconds):
        pts = [(x, y) for f, x, y in self.hist if self.frames - f <= seconds * fps]
        if len(pts) < 2:
            return None
        vx, vy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        norm = math.hypot(vx, vy)
        if norm < self.min_travel_mm:
            return None
        return (vx / norm, vy / norm)

    def _state_metrics(self, fps):
        """(mean speed um/s over the last 2 s, straightness over the last 3 s), from positions sampled
        every 0.5 s so the tracker's frame-to-frame jitter does not inflate the path length."""
        step = max(1, int(0.5 * fps))
        pts = [(f, x, y) for f, x, y in self.hist if self.frames - f <= 3 * fps]
        if len(pts) < 2 * step + 1:
            return 0.0, 0.0
        coarse = pts[::-1][::step][::-1]                      # newest point kept, then every 0.5 s back
        if len(coarse) < 3:
            return 0.0, 0.0
        seg = [math.hypot(coarse[i][1] - coarse[i - 1][1], coarse[i][2] - coarse[i - 1][2]) for i in range(1, len(coarse))]
        path = sum(seg)
        net = math.hypot(coarse[-1][1] - coarse[0][1], coarse[-1][2] - coarse[0][2])
        straight = net / path if path > 0 else 0.0
        recent = [(f, x, y) for f, x, y in coarse if self.frames - f <= 2 * fps]
        if len(recent) < 2:
            return 0.0, straight
        path2 = sum(math.hypot(recent[i][1] - recent[i - 1][1], recent[i][2] - recent[i - 1][2]) for i in range(1, len(recent)))
        seconds = max((recent[-1][0] - recent[0][0]) / fps, 1e-6)
        return path2 / seconds * 1000, straight

    def _status(self, scope, fps, action, angle=None, age_s=None, speed=None, straight=None, dist=None):
        """Live line in the Plugin tab, once per second."""
        if self.frames % max(1, int(fps)) != 0:
            return
        parts = [action]
        if angle is not None: parts.append(f'{angle:.0f} deg off')
        if age_s is not None: parts.append(f'run {age_s:.1f} s')
        if speed is not None: parts.append(f'{speed:.0f} um/s')
        if straight is not None: parts.append(f'straight {straight:.2f}')
        if dist is not None: parts.append(f'{dist:.2f} mm to go')
        scope.print(' | '.join(parts))

    def _log(self, state, scope, dist, angle, action, **extra):
        fps = state.fps if state.fps > 0 else 30.0
        self._status(scope, fps, action, angle=angle, age_s=extra.get('age_s'), speed=extra.get('speed'), straight=extra.get('straightness'), dist=dist)
        scope.log(event='track', frame=self.frames, t=state.time_s, worm=state.worm_xy, dist=dist, angle_deg=angle,
                  reversing=state.is_reversing, voltage=state.voltage, action=action, pulses=self.pulses, **extra)
