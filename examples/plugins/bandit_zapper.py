"""Guide the animal to a target by triggering AWA-driven reversals, and learn which stimulus works
from the outcomes (contextual bandit lookup table).

Guidance rule, evaluated every frame while tracking:
    heading  h   = direction of travel from the last `heading_window` trail points
    to_target t  = target - worm position
    angle        = angle between h and t (0 = straight at the target, 180 = straight away)

    if angle > away_angle_deg for `away_frames` frames in a row  -> deliver a stimulus
    then wait `refractory_frames` (or the stimulus length, whichever is longer) before judging
    again, and stay quiet until the worm has moved `rezap_radius_mm` from the stimulus point or
    `rezap_frames` have passed; never stimulate while the app already sees a reversal;
    never more than `max_pulses_per_min`.
    within `arrive_mm` of the target: light off, done; resumes past `leave_mm` (hysteresis)

Stimulus arms, each (voltage, on_frames, n_pulses, gap_frames):
    (4.5, 60, 1, 0)     one 2 s pulse
    (4.5, 150, 1, 0)    one 5 s pulse
    (4.5, 300, 1, 0)    one 10 s pulse
    (4.5, 30, 3, 30)    train: 3 x 1 s pulses with 1 s gaps
    (0.0, 60, 1, 0)     SHAM: same timing, light stays off -> the spontaneous baseline
The sham arm is the control: if its success rate matches the light arms, the light does nothing.

Bandit:
    context  = angle bin at the moment of the stimulus (`angle_bins`)
    reward   = progress toward the target measured `eval_frames` after the stimulus ends, in mm;
               success = progress > success_mm
    table[(angle_bin, arm)] = {n, successes, progress_sum}  ->  expected reward = successes / n
    choice   = every arm `min_trials` times per bin first, then epsilon-greedy on success rate
               (ties -> fewer trials). epsilon = 0 and min_trials = 0 with a loaded table = exploit.

The table is logged after every outcome and, if `table_file` is set, saved there as JSON and
loaded again on the next Start so knowledge accumulates across sessions and animals.
"""
import json
import math
import os
import random


class Controller:

    # --- target ---------------------------------------------------------------------------
    target_xy = None                 # (x, y) in stage mm; None -> start position + target_offset
    target_offset = (2.0, 0.0)       # mm, used when target_xy is None
    arrive_mm = 0.5
    leave_mm = 0.8

    # --- decision --------------------------------------------------------------------------
    away_angle_deg = 100.0
    away_frames = 5
    heading_window = 10
    min_travel_mm = 0.02
    refractory_frames = 120          # minimum quiet time after a stimulus starts (~4 s at 30 fps)
    rezap_radius_mm = 0.5            # re-arm once the worm moved this far from the stimulus point
    rezap_frames = 600               # or after this many frames (~20 s), whichever first
    max_pulses_per_min = 4           # hard cap on stimulation rate

    # --- stimulus arms and bandit ----------------------------------------------------------
    arms = [(4.5, 60, 1, 0), (4.5, 150, 1, 0), (4.5, 300, 1, 0), (4.5, 30, 3, 30), (0.0, 60, 1, 0)]
    angle_bins = (100, 180)          # one context bin; use (100, 140, 180) to split by heading error
    eval_frames = 120                # outcome measured this many frames after the stimulus ends
    success_mm = 0.10
    min_trials = 2
    epsilon = 0.2
    table_file = None                # e.g. r'C:\Users\soroka\zap_table.json'

    def setup(self, scope):
        self.target = None
        self.arrived = False
        self.away_count = 0
        self.segments = []           # remaining (frames, voltage) of the stimulus in progress
        self.cooldown = 0
        self.pulses = 0
        self.frames = 0
        self.pending = None
        self.last_zap_xy = None
        self.last_zap_frame = None
        self.pulse_frames_log = []   # frames at which stimuli started, for the rate cap
        self.cos_margin = math.cos(math.radians(self.away_angle_deg))
        self.table = self._load_table()
        scope.print(f'bandit: {len(self.arms)} arms, table has {sum(v["n"] for v in self.table.values())} trials; waiting for tracking')

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

        if self.pending is not None and self.pending.get('end_frame') is not None \
                and self.frames - self.pending['end_frame'] >= self.eval_frames:
            self._score(scope, dist, state)

        if self.arrived and dist > self.leave_mm:
            self.arrived = False
            scope.print(f'left the target zone ({dist:.2f} mm), guiding again')
        if not self.arrived and dist < self.arrive_mm:
            self.arrived = True
            scope.log(event='arrived', frame=self.frames, t=state.time_s, dist=dist, pulses=self.pulses)
            scope.print(f'arrived: {dist * 1000:.0f} um from target after {self.pulses} stimuli')
        if self.arrived:
            scope.light_off()
            self._log(state, scope, dist, None, 'arrived')
            return

        # --- a stimulus in progress: play its segments (on / gap / on ...) ---
        if self.segments:
            frames_left, voltage = self.segments[0]
            scope.set_voltage(voltage)
            self.segments[0] = (frames_left - 1, voltage)
            if self.segments[0][0] <= 0:
                self.segments.pop(0)
                if not self.segments and self.pending is not None:
                    self.pending['end_frame'] = self.frames
                    self.pending['dist_at_end'] = dist
            self._log(state, scope, dist, None, 'stimulus' if voltage > 0 else 'gap')
            return

        scope.light_off()

        if self.cooldown > 0:
            self.cooldown -= 1
            self._log(state, scope, dist, None, 'cooldown')
            return

        if self.last_zap_xy is not None:
            moved = math.hypot(state.worm_xy[0] - self.last_zap_xy[0], state.worm_xy[1] - self.last_zap_xy[1])
            if moved < self.rezap_radius_mm and self.frames - self.last_zap_frame < self.rezap_frames:
                self.away_count = 0
                self._log(state, scope, dist, None, 'holding')
                return

        recent = [f for f in self.pulse_frames_log if self.frames - f < 60 * max(state.fps, 1.0)]
        if len(recent) >= self.max_pulses_per_min:
            self.away_count = 0
            self._log(state, scope, dist, None, 'rate_limited')
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
            self._fire(scope, state, dist, angle)
        else:
            self._log(state, scope, dist, angle, 'off')

    def teardown(self, scope):
        scope.light_off()
        self._save_table()

    # --- bandit ----------------------------------------------------------------------------
    def _fire(self, scope, state, dist, angle):
        bin_label = self._bin(angle)
        arm = self._choose_arm(bin_label)
        voltage, on_frames, n_pulses, gap = arm
        self.segments = []
        for i in range(n_pulses):
            self.segments.append((on_frames, voltage))
            if i < n_pulses - 1 and gap > 0:
                self.segments.append((gap, 0.0))
        total = sum(f for f, _ in self.segments)
        self.pulses += 1
        self.cooldown = max(self.refractory_frames, total)
        self.away_count = 0
        self.last_zap_xy = state.worm_xy
        self.last_zap_frame = self.frames
        self.pulse_frames_log.append(self.frames)
        self.pending = {'pulse': self.pulses, 'frame': self.frames, 'dist': dist, 'angle': angle,
                        'bin': bin_label, 'arm': arm, 'end_frame': None, 'dist_at_end': None}
        scope.set_voltage(voltage)
        scope.log(event='pulse', pulse=self.pulses, frame=self.frames, t=state.time_s, worm=state.worm_xy,
                  angle_deg=angle, dist=dist, bin=bin_label, voltage=voltage, on_frames=on_frames,
                  n_pulses=n_pulses, gap_frames=gap, sham=voltage == 0)
        scope.print(f'stimulus {self.pulses}: {angle:.0f} deg off, {dist:.2f} mm away, arm {self._arm_text(arm)} '
                    f'({self._rate_text(bin_label, arm)})')
        self._log(state, scope, dist, angle, 'stimulus' if voltage > 0 else 'sham')

    def _score(self, scope, dist, state):
        p = self.pending
        self.pending = None
        progress = p['dist_at_end'] - dist
        success = progress > self.success_mm
        entry = self.table.setdefault(self._key(p['bin'], p['arm']), {'n': 0, 'successes': 0, 'progress_sum': 0.0})
        entry['n'] += 1
        entry['successes'] += int(success)
        entry['progress_sum'] += progress
        scope.log(event='outcome', pulse=p['pulse'], bin=p['bin'], voltage=p['arm'][0], on_frames=p['arm'][1],
                  n_pulses=p['arm'][2], gap_frames=p['arm'][3], sham=p['arm'][0] == 0, progress_mm=progress,
                  success=success, frame=self.frames, t=state.time_s, table=self.table)
        scope.print(f'stimulus {p["pulse"]} outcome: {progress * 1000:+.0f} um ({"success" if success else "no effect"}); '
                    f'{self._arm_text(p["arm"])}: {self._rate_text(p["bin"], p["arm"])}')
        self._save_table()

    def _choose_arm(self, bin_label):
        stats = [(arm, self.table.get(self._key(bin_label, arm), {'n': 0, 'successes': 0})) for arm in self.arms]
        untried = [arm for arm, s in stats if s['n'] < self.min_trials]
        if untried:
            return min(untried, key=lambda a: self.table.get(self._key(bin_label, a), {'n': 0})['n'])
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        return max(stats, key=lambda item: (item[1]['successes'] / item[1]['n'] if item[1]['n'] else 0.0, -item[1]['n']))[0]

    def _bin(self, angle):
        edges = self.angle_bins
        for lo, hi in zip(edges[:-1], edges[1:]):
            if angle < hi:
                return f'{lo}-{hi}'
        return f'{edges[-2]}-{edges[-1]}'

    @staticmethod
    def _key(bin_label, arm):
        return f'{bin_label}|{arm[0]}V|{arm[1]}f x{arm[2]} gap{arm[3]}'

    @staticmethod
    def _arm_text(arm):
        v, on, n, gap = arm
        if v == 0:
            return f'SHAM {on} f'
        return f'{v} V x {on} f' + (f' x{n} (gap {gap} f)' if n > 1 else '')

    def _rate_text(self, bin_label, arm):
        s = self.table.get(self._key(bin_label, arm))
        if not s or not s['n']:
            return 'untried'
        return f'{s["successes"]}/{s["n"]} success, mean {s["progress_sum"] / s["n"] * 1000:+.0f} um'

    def _load_table(self):
        if self.table_file and os.path.isfile(self.table_file):
            with open(self.table_file, encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_table(self):
        if self.table_file:
            with open(self.table_file, 'w', encoding='utf-8') as f:
                json.dump(self.table, f, indent=1)

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
        scope.log(event='track', frame=self.frames, t=state.time_s, worm=state.worm_xy, dist=dist,
                  angle_deg=angle, reversing=state.is_reversing, voltage=state.voltage, action=action, pulses=self.pulses)
