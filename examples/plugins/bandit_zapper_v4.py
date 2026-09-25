"""Contextual bandit for optogenetic guidance, version 4: no hand-written gate, the bandit learns it.

bandit_zapper.py / _v2 only ever decide while a fixed rule says the worm is heading away from the target.
Version 4 removes that rule. The state is the discretised direction of travel relative to the target,
and the bandit decides in EVERY state, including "heading toward the target". WAIT is always one of
the arms, so the bandit can (and must) learn by itself that the right action when heading toward the
target is to do nothing, and that a stimulus only pays when heading away.

State (one label per decision):
    'toward'     heading within `angle_bins` 0-60 deg of the target direction
    'side'       60-120 deg
    'away'       120-180 deg
    'still'      heading unknown (moved < `min_travel_mm` over the heading window)
    'reversing'  the app's reversal detector is on (the worm is backing up right now)

Arms: the light stimuli in `arms` (voltage, on frames, pulses, gap frames) plus WAIT.

Decision cadence: one decision at a time. WAIT holds for `wait_hold_s`, a stimulus holds for
`stimulus_hold_s` (light plus refractory). When the hold ends the decision is scored and the next one
is made at once, so the worm is re-judged every few seconds whatever it does.

Reward: progress toward the target during the hold, as a RATE (um/s), so a 4 s WAIT and a 12 s stimulus
are comparable. Heading-at-eval is not used: the exit heading after a reversal is random (see the
EIM00013/16/21 angle analysis), progress is what we actually want.

Modes: 'learn' (epsilon-greedy on mean reward after `min_trials` per arm), 'train' (round-robin per
state, just fills the table), 'test' (always the best arm per state, table read-only).
Table keys are "<state>|<arm>", so plot_bandit_table.py works unchanged. Set `table_file` to keep the
table across runs.
"""
import json
import math
import os
import random
from collections import deque

WAIT = 'wait'


class Controller:

    # --- target ---------------------------------------------------------------------------
    target_xy = None
    target_offset = (15.0, 0.0)
    arrive_mm = 0.5
    leave_mm = 0.8

    # --- state -----------------------------------------------------------------------------
    angle_bins = ((0.0, 60.0, 'toward'), (60.0, 120.0, 'side'), (120.0, 180.1, 'away'))
    heading_window = 30
    min_travel_mm = 0.02

    # --- arms and holds --------------------------------------------------------------------
    arms = [(4.5, 45, 1, 0), (4.5, 30, 3, 30), (2.0, 45, 1, 0)]
    wait_hold_s = 4.0                # a WAIT decision is scored over this long
    stimulus_hold_s = 12.0           # a stimulus is scored over this long (light + refractory)
    max_pulses_per_min = 6

    # --- bandit ----------------------------------------------------------------------------
    mode = 'learn'                   # 'learn' | 'train' | 'test'
    min_trials = 2
    epsilon = 0.15
    table_file = None

    def setup(self, scope):
        self.target = None
        self.arrived = False
        self.segments = []
        self.hold = None             # the decision in progress
        self.pulses = 0
        self.decisions = 0
        self.frames = 0
        self.pulse_frames_log = []
        self.round_robin = {}
        self.hist = deque(maxlen=4 * 30 + 5)
        self.table = self._load_table()
        if self.mode == 'test' and not self.table:
            scope.print('WARNING: test mode with an empty table; set table_file to a trained table')
        scope.print(f'bandit v4 [{self.mode}]: {len(self.all_arms())} actions x 5 states (direction bins), reward = progress rate; '
                    f'table has {sum(v["n"] for v in self.table.values())} trials; waiting for tracking')

    def all_arms(self):
        return list(self.arms) + [WAIT]

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

        if self.arrived and dist > self.leave_mm:
            self.arrived = False
            scope.print(f'left the target zone ({dist:.2f} mm), guiding again')
        if not self.arrived and dist < self.arrive_mm:
            self.arrived = True
            scope.log(event='arrived', frame=self.frames, t=state.time_s, dist=dist, pulses=self.pulses, decisions=self.decisions)
            scope.print(f'arrived: {dist * 1000:.0f} um from target after {self.pulses} stimuli / {self.decisions} decisions')
        if self.arrived:
            scope.light_off()
            self.hold = None
            self.segments = []
            self._log(state, scope, dist, None, 'arrived')
            return

        # --- light pattern of the current stimulus ---
        if self.segments:
            frames_left, voltage = self.segments[0]
            scope.set_voltage(voltage)
            self.segments[0] = (frames_left - 1, voltage)
            if self.segments[0][0] <= 0:
                self.segments.pop(0)
            self._log(state, scope, dist, None, 'stimulus' if voltage > 0 else 'gap')
            return

        scope.light_off()

        # --- a decision in progress: wait out its hold, then score it ---
        if self.hold is not None:
            if self.frames - self.hold['frame'] < self.hold['hold_frames']:
                self._log(state, scope, dist, None, 'hold', bin=self.hold['bin'])
                return
            self._score(scope, state, self.hold, dist, fps)
            self.hold = None

        # --- new decision ---
        angle, bin_label = self._state_bin(state, tx, ty, dist)
        self._decide(scope, state, dist, angle, bin_label, fps)

    def teardown(self, scope):
        scope.light_off()
        self._save_table()

    # --- state -----------------------------------------------------------------------------
    def _state_bin(self, state, tx, ty, dist):
        if state.is_reversing:
            return None, 'reversing'
        heading = self._heading(state.trail)
        if heading is None or dist <= 0:
            return None, 'still'
        cos_theta = (heading[0] * tx + heading[1] * ty) / dist
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
        for lo, hi, label in self.angle_bins:
            if lo <= angle < hi:
                return angle, label
        return angle, self.angle_bins[-1][2]

    # --- bandit ----------------------------------------------------------------------------
    def _decide(self, scope, state, dist, angle, bin_label, fps):
        arm = self._choose_arm(bin_label, fps)
        self.decisions += 1
        rec = {'decision': self.decisions, 'frame': self.frames, 'dist': dist, 'angle': angle, 'bin': bin_label, 'arm': arm}
        if arm == WAIT:
            rec['hold_frames'] = max(1, int(self.wait_hold_s * fps))
            self.hold = rec
            scope.log(event='decision', decision=self.decisions, action='wait', frame=self.frames, t=state.time_s,
                      worm=state.worm_xy, angle_deg=angle, dist=dist, bin=bin_label, mode=self.mode)
            scope.print(f'decision {self.decisions} @ {bin_label}: WAIT ({self._rate_text(bin_label, arm)})')
            self._log(state, scope, dist, angle, 'wait', bin=bin_label)
            return
        voltage, on_frames, n_pulses, gap = arm
        self.segments = []
        for i in range(n_pulses):
            self.segments.append((on_frames, voltage))
            if i < n_pulses - 1 and gap > 0:
                self.segments.append((gap, 0.0))
        total = sum(f for f, _ in self.segments)
        rec['hold_frames'] = max(total, int(self.stimulus_hold_s * fps))
        self.hold = rec
        self.pulses += 1
        self.pulse_frames_log.append(self.frames)
        scope.set_voltage(voltage)
        scope.log(event='decision', decision=self.decisions, action='stimulus', pulse=self.pulses, frame=self.frames, t=state.time_s,
                  worm=state.worm_xy, angle_deg=angle, dist=dist, bin=bin_label, voltage=voltage, on_frames=on_frames,
                  n_pulses=n_pulses, gap_frames=gap, mode=self.mode)
        scope.print(f'decision {self.decisions} @ {bin_label}: {self._arm_text(arm)}, '
                    f'{"?" if angle is None else round(angle)} deg off, {dist:.2f} mm away ({self._rate_text(bin_label, arm)})')
        self._log(state, scope, dist, angle, 'stimulus', bin=bin_label)

    def _score(self, scope, state, p, dist, fps):
        progress = p['dist'] - dist                              # mm toward the target during the hold
        seconds = p['hold_frames'] / fps
        rate = progress / seconds * 1000                         # um/s, the reward
        success = progress > 0
        key = self._key(p['bin'], p['arm'])
        if self.mode != 'test':
            entry = self.table.setdefault(key, {'n': 0, 'successes': 0, 'progress_sum': 0.0, 'reward_sum': 0.0})
            entry['n'] += 1
            entry['successes'] += int(success)
            entry['progress_sum'] += progress
            entry['reward_sum'] = entry.get('reward_sum', 0.0) + rate
            self._save_table()
        arm = p['arm']
        scope.log(event='outcome', decision=p['decision'], bin=p['bin'], action=self._arm_text(arm),
                  voltage=None if arm == WAIT else arm[0], on_frames=None if arm == WAIT else arm[1],
                  n_pulses=None if arm == WAIT else arm[2], sham=False, wait=arm == WAIT,
                  progress_mm=progress, reward_um_s=rate, hold_s=seconds, success=success, reward='progress_rate',
                  mode=self.mode, frame=self.frames, t=state.time_s, table=self.table)
        scope.print(f'decision {p["decision"]} outcome: {progress * 1000:+.0f} um in {seconds:.0f} s ({rate:+.0f} um/s); '
                    f'{self._arm_text(arm)} @ {p["bin"]}: {self._rate_text(p["bin"], arm)}')

    def _mean_reward(self, s):
        return s['reward_sum'] / s['n'] if s.get('n') else 0.0

    def _choose_arm(self, bin_label, fps):
        arms = self.all_arms()
        recent = [f for f in self.pulse_frames_log if self.frames - f < 60 * fps]
        if len(recent) >= self.max_pulses_per_min:
            return WAIT                                          # rate limit: not a bandit choice, not scored as one
        if self.mode == 'train':
            k = self.round_robin.get(bin_label, 0)
            self.round_robin[bin_label] = k + 1
            return arms[k % len(arms)]
        stats = [(arm, self.table.get(self._key(bin_label, arm), {'n': 0, 'reward_sum': 0.0})) for arm in arms]
        if self.mode == 'test':
            tried = [(arm, s) for arm, s in stats if s.get('n')]
            return WAIT if not tried else max(tried, key=lambda item: (self._mean_reward(item[1]), item[1]['n']))[0]
        untried = [arm for arm, s in stats if s.get('n', 0) < self.min_trials]
        if untried:
            return min(untried, key=lambda a: self.table.get(self._key(bin_label, a), {'n': 0})['n'])
        if random.random() < self.epsilon:
            return random.choice(arms)
        return max(stats, key=lambda item: (self._mean_reward(item[1]), -item[1]['n']))[0]

    def _key(self, bin_label, arm):
        return f'{bin_label}|{self._arm_text(arm)}'

    @staticmethod
    def _arm_text(arm):
        if arm == WAIT:
            return 'WAIT'
        v, on, n, gap = arm
        return f'{v}V x {on}f' + (f' x{n} gap{gap}' if n > 1 else '')

    def _rate_text(self, bin_label, arm):
        s = self.table.get(self._key(bin_label, arm))
        if not s or not s.get('n'):
            return 'untried'
        return f'{s["n"]} trials, mean {self._mean_reward(s):+.0f} um/s, {s["successes"]}/{s["n"]} moved closer'

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

    def _status(self, scope, fps, action, angle=None, dist=None, bin_label=None):
        if self.frames % max(1, int(fps)) != 0:
            return
        parts = [action]
        if bin_label is not None: parts.append(f'state {bin_label}')
        if angle is not None: parts.append(f'{angle:.0f} deg off')
        if dist is not None: parts.append(f'{dist:.2f} mm to go')
        parts.append(f'{self.decisions} decisions, {self.pulses} stimuli')
        scope.print(' | '.join(parts))

    log_every = 3                    # routine per-frame log lines: keep 1 in 3 (events are always logged)

    def _log(self, state, scope, dist, angle, action, **extra):
        fps = state.fps if state.fps > 0 else 30.0
        self._status(scope, fps, action, angle=angle, dist=dist, bin_label=extra.get('bin'))
        if self.frames % self.log_every:
            return
        scope.log(event='track', frame=self.frames, t=state.time_s, worm=state.worm_xy, dist=dist, angle_deg=angle,
                  reversing=bool(state.is_reversing), voltage=state.voltage, action=action, pulses=self.pulses, **extra)
