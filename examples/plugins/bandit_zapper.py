"""Guide the animal to a target by triggering AWA-driven reversals, and learn WHEN a stimulus pays off
(contextual bandit lookup table). Built on the EIM00015 results: a 2 s pulse reverses the worm 84 % of the
time with 1.3 s latency, the exit heading after the omega is close to random, and spontaneous run durations
are heavy-tailed (the longer a run, the less likely it ends by itself).

State  = how long the worm has been heading the wrong way (a "wrong-way run"), while roaming.
Action = at each decision age in `decision_ages_s` (default 1, 3 and 8 s into a wrong-way run) choose one arm:
           a light stimulus, the SHAM (same schedule, light off), or WAIT (do nothing, decide again at the
           next age). The table then holds the expected reward of every action in every state, which is
           the heatmap you want: rows = run age, columns = arms.
Reward = scored `eval_frames` after the action: 'heading' -> heading within success_angle_deg of the
         target; 'progress' -> got closer by success_mm. Both are logged.

Every frame while tracking:
    heading = travel direction over the last `heading_window` trail points
    wrong way = angle to target > away_angle_deg and the app sees no reversal
    roaming gate = speed >= min_speed_um_s over 2 s and straightness >= min_straightness over 3 s;
                   dwelling worms change heading anyway, so their runs are not judged
    after a stimulus or sham: refractory_frames of silence (reversal + omega ~12 s), the re-zap gate
    (moved rezap_radius_mm or rezap_frames elapsed) and max_pulses_per_min

Arms, each (voltage, on_frames, n_pulses, gap_frames):
    (4.5, 60, 1, 0)  2 s pulse      (4.5, 30, 3, 30)  train 3 x 1 s      (2.0, 60, 1, 0)  weak 2 s pulse
    (0.0, 60, 1, 0)  SHAM           plus the implicit WAIT arm

Modes (`mode`):  'learn' epsilon-greedy, table updated (default)   'train' round-robin, table updated
                 'test' best arm per state from the loaded table, table NOT updated (clean measurement)
Set `table_file` to keep the table across sessions; 'test' needs one.
"""
import json
import math
import os
import random
from collections import deque

WAIT = 'wait'


class Controller:

    # --- target ---------------------------------------------------------------------------
    target_xy = None                 # (x, y) stage mm; None -> start position + target_offset
    target_offset = (15.0, 0.0)
    arrive_mm = 0.5
    leave_mm = 0.8

    # --- decision --------------------------------------------------------------------------
    away_angle_deg = 100.0
    decision_ages_s = (1.0, 3.0, 8.0)   # decide at these ages of a wrong-way run
    heading_window = 30
    min_travel_mm = 0.02
    refractory_frames = 360          # ~12 s after a stimulus/sham: reversal + omega
    rezap_radius_mm = 0.5
    rezap_frames = 600
    max_pulses_per_min = 4

    # --- state gate (roaming only) ---------------------------------------------------------
    roaming_gate = True
    min_speed_um_s = 150.0
    min_straightness = 0.6

    # --- arms, reward ----------------------------------------------------------------------
    arms = [(4.5, 60, 1, 0), (4.5, 30, 3, 30), (2.0, 60, 1, 0), (0.0, 60, 1, 0)]
    include_wait = True
    reward = 'heading'               # 'heading' or 'progress'
    eval_frames = 300                # scored this many frames after the action ends (~10 s)
    success_angle_deg = 90.0
    success_mm = 0.10

    # --- bandit ----------------------------------------------------------------------------
    mode = 'learn'                   # 'learn' | 'train' | 'test'
    min_trials = 2
    epsilon = 0.2
    table_file = None                # e.g. r'C:\Users\soroka\zap_table.json'

    def setup(self, scope):
        self.target = None
        self.arrived = False
        self.away_age = 0
        self.decided = set()         # decision ages already used in the current wrong-way run
        self.segments = []
        self.cooldown = 0
        self.pulses = 0
        self.decisions = 0
        self.frames = 0
        self.pending = []            # outcomes still to be scored
        self.last_zap_xy = None
        self.last_zap_frame = None
        self.pulse_frames_log = []
        self.round_robin = {}        # per-state counter for train mode
        self.hist = deque(maxlen=4 * 30 + 5)
        self.cos_margin = math.cos(math.radians(self.away_angle_deg))
        self.table = self._load_table()
        if self.mode == 'test' and not self.table:
            scope.print('WARNING: test mode with an empty table; set table_file to a trained table')
        scope.print(f'bandit [{self.mode}]: {len(self.all_arms())} actions x {len(self.decision_ages_s)} states, '
                    f'reward={self.reward}, table has {sum(v["n"] for v in self.table.values())} trials; waiting for tracking')

    def all_arms(self):
        return list(self.arms) + ([WAIT] if self.include_wait else [])

    # ----------------------------------------------------------------------------------------
    def update(self, state, scope):
        if not state.is_tracking:
            scope.light_off()
            self.hist.clear()
            return

        if self.target is None:
            self.target = self.target_xy or (state.worm_xy[0] + self.target_offset[0],
                                             state.worm_xy[1] + self.target_offset[1])
            scope.log(event='target', target=self.target, start=state.worm_xy, mode=self.mode)
            scope.print(f'target ({self.target[0]:.2f}, {self.target[1]:.2f}) mm')

        self.frames += 1
        self.hist.append((self.frames, state.worm_xy[0], state.worm_xy[1]))
        fps = state.fps if state.fps > 0 else 30.0
        tx = self.target[0] - state.worm_xy[0]
        ty = self.target[1] - state.worm_xy[1]
        dist = math.hypot(tx, ty)

        due = [p for p in self.pending if p.get('end_frame') is not None and self.frames - p['end_frame'] >= self.eval_frames]
        for p in due:
            self.pending.remove(p)
            self._score(scope, state, p, dist, tx, ty, fps)

        if self.arrived and dist > self.leave_mm:
            self.arrived = False
            scope.print(f'left the target zone ({dist:.2f} mm), guiding again')
        if not self.arrived and dist < self.arrive_mm:
            self.arrived = True
            scope.log(event='arrived', frame=self.frames, t=state.time_s, dist=dist, pulses=self.pulses, decisions=self.decisions)
            scope.print(f'arrived: {dist * 1000:.0f} um from target after {self.pulses} stimuli / {self.decisions} decisions')
        if self.arrived:
            scope.light_off()
            self._log(state, scope, dist, None, 'arrived')
            return

        if self.segments:                                     # stimulus (or sham) in progress
            frames_left, voltage = self.segments[0]
            scope.set_voltage(voltage)
            self.segments[0] = (frames_left - 1, voltage)
            if self.segments[0][0] <= 0:
                self.segments.pop(0)
                if not self.segments:
                    for p in self.pending:
                        if p['end_frame'] is None and p['arm'] != WAIT:
                            p['end_frame'] = self.frames; p['dist_at_end'] = dist
            self._log(state, scope, dist, None, 'stimulus' if voltage > 0 else 'gap')
            return

        scope.light_off()

        if self.cooldown > 0:
            self.cooldown -= 1
            self._reset_run()
            self._log(state, scope, dist, None, 'cooldown')
            return

        if self.last_zap_xy is not None:
            moved = math.hypot(state.worm_xy[0] - self.last_zap_xy[0], state.worm_xy[1] - self.last_zap_xy[1])
            if moved < self.rezap_radius_mm and self.frames - self.last_zap_frame < self.rezap_frames:
                self._reset_run()
                self._log(state, scope, dist, None, 'holding')
                return

        recent = [f for f in self.pulse_frames_log if self.frames - f < 60 * fps]
        if len(recent) >= self.max_pulses_per_min:
            self._reset_run()
            self._log(state, scope, dist, None, 'rate_limited')
            return

        speed, straight = self._state_metrics(fps)
        if self.roaming_gate and (speed < self.min_speed_um_s or straight < self.min_straightness):
            self._reset_run()
            self._log(state, scope, dist, None, 'dwelling', speed=speed, straightness=straight)
            return

        heading = self._heading(state.trail)
        if heading is None or dist <= 0:
            self._reset_run()
            self._log(state, scope, dist, None, 'no_heading')
            return

        cos_theta = (heading[0] * tx + heading[1] * ty) / dist
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
        wrong_way = cos_theta < self.cos_margin and not state.is_reversing
        if not wrong_way:
            self._reset_run()
            self._log(state, scope, dist, angle, 'off')
            return

        self.away_age += 1
        age_s = self.away_age / fps
        due_age = next((a for a in self.decision_ages_s if age_s >= a and a not in self.decided), None)
        if due_age is None:
            self._log(state, scope, dist, angle, 'wrong_way', age_s=age_s)
            return
        self.decided.add(due_age)
        self._decide(scope, state, dist, angle, speed, straight, due_age, age_s)

    def teardown(self, scope):
        scope.light_off()
        self._save_table()

    def _reset_run(self):
        self.away_age = 0
        self.decided.clear()

    # --- bandit ----------------------------------------------------------------------------
    def _decide(self, scope, state, dist, angle, speed, straight, due_age, age_s):
        bin_label = self._bin(due_age)
        arm = self._choose_arm(bin_label)
        self.decisions += 1
        rec = {'decision': self.decisions, 'frame': self.frames, 'dist': dist, 'angle': angle, 'age_s': age_s,
               'bin': bin_label, 'arm': arm, 'end_frame': None, 'dist_at_end': None}
        if arm == WAIT:
            rec['end_frame'] = self.frames; rec['dist_at_end'] = dist
            self.pending.append(rec)
            scope.log(event='decision', decision=self.decisions, action='wait', frame=self.frames, t=state.time_s,
                      worm=state.worm_xy, angle_deg=angle, dist=dist, run_age_s=age_s, bin=bin_label, mode=self.mode)
            scope.print(f'decision {self.decisions} @ {bin_label}: WAIT ({self._rate_text(bin_label, arm)})')
            self._log(state, scope, dist, angle, 'wait', age_s=age_s)
            return
        voltage, on_frames, n_pulses, gap = arm
        self.segments = []
        for i in range(n_pulses):
            self.segments.append((on_frames, voltage))
            if i < n_pulses - 1 and gap > 0:
                self.segments.append((gap, 0.0))
        total = sum(f for f, _ in self.segments)
        self.pulses += 1
        self.cooldown = max(self.refractory_frames, total)
        self._reset_run()
        self.last_zap_xy = state.worm_xy
        self.last_zap_frame = self.frames
        self.pulse_frames_log.append(self.frames)
        self.pending.append(rec)
        scope.set_voltage(voltage)
        scope.log(event='decision', decision=self.decisions, action='sham' if voltage == 0 else 'stimulus', pulse=self.pulses,
                  frame=self.frames, t=state.time_s, worm=state.worm_xy, angle_deg=angle, dist=dist, run_age_s=age_s, bin=bin_label,
                  voltage=voltage, on_frames=on_frames, n_pulses=n_pulses, gap_frames=gap, speed_um_s=speed, straightness=straight, mode=self.mode)
        scope.print(f'decision {self.decisions} @ {bin_label}: {self._arm_text(arm)}, {angle:.0f} deg off, {dist:.2f} mm away '
                    f'({self._rate_text(bin_label, arm)})')
        self._log(state, scope, dist, angle, 'stimulus' if voltage > 0 else 'sham')

    def _score(self, scope, state, p, dist, tx, ty, fps):
        progress = p['dist_at_end'] - dist
        heading = self._recent_heading(fps, seconds=1.0)
        if heading is not None and dist > 0:
            cos_t = (heading[0] * tx + heading[1] * ty) / dist
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))
        else:
            angle = None
        success = (angle is not None and angle < self.success_angle_deg) if self.reward == 'heading' else progress > self.success_mm
        key = self._key(p['bin'], p['arm'])
        if self.mode != 'test':
            entry = self.table.setdefault(key, {'n': 0, 'successes': 0, 'progress_sum': 0.0})
            entry['n'] += 1
            entry['successes'] += int(success)
            entry['progress_sum'] += progress
            self._save_table()
        arm = p['arm']
        scope.log(event='outcome', decision=p['decision'], bin=p['bin'], run_age_s=p['age_s'], action=self._arm_text(arm),
                  voltage=None if arm == WAIT else arm[0], on_frames=None if arm == WAIT else arm[1],
                  n_pulses=None if arm == WAIT else arm[2], sham=arm != WAIT and arm[0] == 0, wait=arm == WAIT,
                  progress_mm=progress, heading_angle_deg=angle, success=success, reward=self.reward, mode=self.mode,
                  frame=self.frames, t=state.time_s, table=self.table)
        scope.print(f'decision {p["decision"]} outcome: heading {"?" if angle is None else round(angle)} deg off, '
                    f'{progress * 1000:+.0f} um ({"success" if success else "no"}); {self._arm_text(arm)} @ {p["bin"]}: '
                    f'{self._rate_text(p["bin"], arm)}')

    def _choose_arm(self, bin_label):
        arms = self.all_arms()
        if self.mode == 'train':
            k = self.round_robin.get(bin_label, 0)
            self.round_robin[bin_label] = k + 1
            return arms[k % len(arms)]
        stats = [(arm, self.table.get(self._key(bin_label, arm), {'n': 0, 'successes': 0})) for arm in arms]
        if self.mode == 'test':
            tried = [(arm, s) for arm, s in stats if s['n'] > 0 and not (arm != WAIT and arm[0] == 0)]   # never the sham
            if not tried:
                return WAIT if self.include_wait else next(a for a in self.arms if a[0] > 0)
            return max(tried, key=lambda item: (item[1]['successes'] / item[1]['n'], item[1]['n']))[0]
        untried = [arm for arm, s in stats if s['n'] < self.min_trials]
        if untried:
            return min(untried, key=lambda a: self.table.get(self._key(bin_label, a), {'n': 0})['n'])
        if random.random() < self.epsilon:
            return random.choice(arms)
        return max(stats, key=lambda item: (item[1]['successes'] / item[1]['n'] if item[1]['n'] else 0.0, -item[1]['n']))[0]

    @staticmethod
    def _bin(age):
        return f'age {age:g}s'

    def _key(self, bin_label, arm):
        return f'{bin_label}|{self._arm_text(arm)}'

    @staticmethod
    def _arm_text(arm):
        if arm == WAIT:
            return 'WAIT'
        v, on, n, gap = arm
        if v == 0:
            return f'SHAM {on}f'
        return f'{v}V x {on}f' + (f' x{n} gap{gap}' if n > 1 else '')

    def _rate_text(self, bin_label, arm):
        s = self.table.get(self._key(bin_label, arm))
        if not s or not s['n']:
            return 'untried'
        return f'{s["successes"]}/{s["n"]} success, mean progress {s["progress_sum"] / s["n"] * 1000:+.0f} um'

    def _load_table(self):
        if self.table_file and os.path.isfile(self.table_file):
            with open(self.table_file, encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_table(self):
        if self.table_file and self.mode != 'test':
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

    def _recent_heading(self, fps, seconds=1.0):
        pts = [(x, y) for f, x, y in self.hist if self.frames - f <= seconds * fps]
        if len(pts) < 2:
            return None
        vx, vy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        norm = math.hypot(vx, vy)
        if norm < self.min_travel_mm:
            return None
        return (vx / norm, vy / norm)

    def _state_metrics(self, fps):
        pts3 = [(f, x, y) for f, x, y in self.hist if self.frames - f <= 3 * fps]
        if len(pts3) < 3:
            return 0.0, 0.0
        path = sum(math.hypot(pts3[i][1] - pts3[i - 1][1], pts3[i][2] - pts3[i - 1][2]) for i in range(1, len(pts3)))
        net = math.hypot(pts3[-1][1] - pts3[0][1], pts3[-1][2] - pts3[0][2])
        straight = net / path if path > 0 else 0.0
        pts2 = [p for p in pts3 if self.frames - p[0] <= 2 * fps]
        path2 = sum(math.hypot(pts2[i][1] - pts2[i - 1][1], pts2[i][2] - pts2[i - 1][2]) for i in range(1, len(pts2)))
        seconds = max((pts2[-1][0] - pts2[0][0]) / fps, 1e-6)
        return path2 / seconds * 1000, straight

    def _log(self, state, scope, dist, angle, action, **extra):
        scope.log(event='track', frame=self.frames, t=state.time_s, worm=state.worm_xy, dist=dist, angle_deg=angle,
                  reversing=state.is_reversing, voltage=state.voltage, action=action, pulses=self.pulses, **extra)
