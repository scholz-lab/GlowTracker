"""Guide the animal to a target point by triggering AWA-driven reversals, and learn which
stimulus works from the outcomes (contextual bandit lookup table).

Guidance rule, evaluated every frame while tracking:
    heading  h   = direction of travel from the last `heading_window` trail points
    to_target t  = target - worm position
    angle        = angle between h and t (0 = straight at the target, 180 = straight away)

    if angle > away_angle_deg for `away_frames` frames in a row  -> pulse the light
    after a pulse: wait `refractory_frames` before judging again (let the reversal happen)
    never pulse while the app already sees a reversal
    within `arrive_mm` of the target: light off, done; guidance resumes if the worm
    wanders back out past `leave_mm` (hysteresis)

Bandit:
    arms     = the stimulus settings to choose from, (voltage, pulse_frames)
    context  = angle bin at the moment of the pulse (how wrong the heading was)
    reward   = progress toward the target measured `eval_frames` after the pulse started,
               in mm; success = progress > success_mm
    table[(angle_bin, arm)] = {n, successes, progress_sum}  ->  expected reward = successes / n
    choice   = each arm tried `min_trials` times per bin first, then epsilon-greedy on the
               success rate (ties -> fewer trials). Set `epsilon = 0` and `min_trials = 0`
               with a loaded table to exploit only.

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
    target_offset = (5.0, 0.0)       # mm, used when target_xy is None
    arrive_mm = 0.5                  # inside this radius the worm counts as arrived
    leave_mm = 0.8                   # guidance resumes only after it wanders back out past this

    # --- decision --------------------------------------------------------------------------
    away_angle_deg = 100.0           # heading must be more than this off the target direction
    away_frames = 5                  # consecutive wrong-heading frames before a pulse
    heading_window = 10              # trail points for the heading estimate
    min_travel_mm = 0.02             # below this much travel in the window the heading is unknown

    # --- stimulus arms and bandit ----------------------------------------------------------
    arms = [(4.5, 15), (4.5, 30), (4.5, 60)]   # (voltage, pulse_frames) to choose from
    angle_bins = (100, 140, 180)     # context: angle at pulse time falls in [100,140) or [140,180]
    eval_frames = 120                # measure the outcome this many frames after the pulse ends
    success_mm = 0.10                # progress toward the target above this counts as a success
    refractory_frames = 120          # minimum frames after a pulse starts before the next decision
    min_trials = 2                   # try every arm this often per bin before exploiting
    epsilon = 0.2                    # after that, explore with this probability
    table_file = None                # e.g. r'C:\Users\soroka\zap_table.json' to persist the table

    def setup(self, scope):
        self.target = None
        self.arrived = False
        self.away_count = 0
        self.pulse_left = 0
        self.pulse_voltage = 0.0
        self.cooldown = 0
        self.pulses = 0
        self.frames = 0
        self.pending = None          # the pulse whose outcome is still being measured
        self.cos_margin = math.cos(math.radians(self.away_angle_deg))
        self.table = self._load_table()
        scope.print(f'guidance plugin: {len(self.arms)} arms, table has '
                    f'{sum(v["n"] for v in self.table.values())} trials; waiting for tracking')

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

        # --- outcome of the last pulse, eval_frames after the light went off ---
        if self.pending is not None \
                and self.frames - (self.pending['frame'] + self.pending['arm'][1]) >= self.eval_frames:
            self._score(scope, dist, state)

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
            scope.set_voltage(self.pulse_voltage)
            if self.pulse_left == 0 and self.pending is not None:
                self.pending['dist_at_off'] = dist     # progress is measured from here
            self._log(state, scope, dist, None, 'pulse')
            return

        scope.light_off()

        # --- refractory: give the reversal time to happen before judging again ---
        if self.cooldown > 0:
            self.cooldown -= 1
            self._log(state, scope, dist, None, 'cooldown')
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
        voltage, frames = arm
        self.pulses += 1
        self.pulse_voltage = voltage
        self.pulse_left = frames
        self.cooldown = max(self.refractory_frames, frames)
        self.away_count = 0
        self.pending = {'pulse': self.pulses, 'frame': self.frames, 'dist': dist,
                        'angle': angle, 'bin': bin_label, 'arm': arm}
        scope.set_voltage(voltage)
        scope.log(event='pulse', pulse=self.pulses, frame=self.frames, t=state.time_s,
                  worm=state.worm_xy, angle_deg=angle, dist=dist, bin=bin_label,
                  voltage=voltage, pulse_frames=frames)
        scope.print(f'pulse {self.pulses}: {angle:.0f} deg off, {dist:.2f} mm away, '
                    f'arm {voltage} V x {frames} f ({self._rate_text(bin_label, arm)})')
        self._log(state, scope, dist, angle, 'pulse')

    def _score(self, scope, dist, state):
        p = self.pending
        self.pending = None
        progress = p['dist_at_off'] - dist if p.get('dist_at_off') is not None else p['dist'] - dist
        success = progress > self.success_mm
        entry = self.table.setdefault(self._key(p['bin'], p['arm']), {'n': 0, 'successes': 0, 'progress_sum': 0.0})
        entry['n'] += 1
        entry['successes'] += int(success)
        entry['progress_sum'] += progress
        scope.log(event='outcome', pulse=p['pulse'], bin=p['bin'], voltage=p['arm'][0],
                  pulse_frames=p['arm'][1], progress_mm=progress, success=success,
                  frame=self.frames, t=state.time_s, table=self.table)
        scope.print(f'pulse {p["pulse"]} outcome: {progress * 1000:+.0f} um toward target '
                    f'({"success" if success else "no effect"}); {self._rate_text(p["bin"], p["arm"])}')
        self._save_table()

    def _choose_arm(self, bin_label):
        stats = [(arm, self.table.get(self._key(bin_label, arm), {'n': 0, 'successes': 0}))
                 for arm in self.arms]
        untried = [arm for arm, s in stats if s['n'] < self.min_trials]
        if untried:
            return min(untried, key=lambda a: self.table.get(self._key(bin_label, a), {'n': 0})['n'])
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        return max(stats, key=lambda item: (item[1]['successes'] / item[1]['n'] if item[1]['n'] else 0.0,
                                            -item[1]['n']))[0]

    def _bin(self, angle):
        edges = self.angle_bins
        for lo, hi in zip(edges[:-1], edges[1:]):
            if angle < hi:
                return f'{lo}-{hi}'
        return f'{edges[-2]}-{edges[-1]}'

    @staticmethod
    def _key(bin_label, arm):
        return f'{bin_label}|{arm[0]}V|{arm[1]}f'

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
        scope.log(event='track', frame=self.frames, t=state.time_s, worm=state.worm_xy,
                  dist=dist, angle_deg=angle, reversing=state.is_reversing,
                  voltage=state.voltage, action=action, pulses=self.pulses)
