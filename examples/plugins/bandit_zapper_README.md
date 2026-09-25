# bandit_zapper.py

A GlowTracker plugin that steers a worm toward a target point by triggering AWA-driven reversals with
the optogenetic LED, and learns *when* and *how* to stimulate from the outcomes. It is a contextual
bandit: a lookup table of expected reward for every (state, action) pair, filled in as the session runs.

This file is for lab use. It is not part of the hackathon material.

## The idea in one paragraph

Every frame the plugin knows where the worm is, which way it is heading, and where the target is. When
the worm has been heading *away* from the target for a while, it has to decide: fire the LED (which
reversal), or wait. A reversal backs the worm up about a body length and leaves it pointing in a
near-random direction, so each stimulus is a weighted coin flip toward the target. Waiting is free but
the worm keeps going the wrong way. Which choice is better depends on the situation, so the plugin keeps
score per situation and picks the action with the best record.

## State, action, reward

**State** is how long the worm has been heading the wrong way, the *run age*. Decisions are made when a
wrong-way run reaches each age in `decision_ages_s`, by default 1, 3 and 8 s. Each age is a separate row
of the table.

**Actions** are the arms. Each light arm is a tuple `(voltage, on_frames, n_pulses, gap_frames)`:

| arm | meaning |
|---|---|
| `(4.5, 60, 1, 0)` | one 2 s pulse at 4.5 V |
| `(4.5, 30, 3, 30)` | train of three 1 s pulses with 1 s gaps |
| `(2.0, 60, 1, 0)` | one 2 s pulse at 2 V |
| `(0.0, 60, 1, 0)` | **sham**: same schedule, light stays off. The spontaneous baseline. |
| `WAIT` | do nothing now; decide again at the next decision age |

The sham and WAIT rows are the controls. If a light arm cannot beat them, the light is not helping.

**Reward** is scored `eval_frames` (default 300, about 10 s) after the action ends, by which time the
reversal and omega turn are over:

- `reward = 'heading'` (default): success if the worm's heading over the last second is within
  `success_angle_deg` (90°) of the target direction.
- `reward = 'progress'`: success if the worm got closer to the target by more than `success_mm`.

Both numbers are logged whichever is used. The table stores, per `state|arm`: trials, successes, and
summed progress. Expected reward is successes ÷ trials.

## What happens every frame

```
not tracking?                 -> light off, do nothing
target not set?               -> target = start position + target_offset (or target_xy)
outcome due?                  -> score it, update the table
within arrive_mm of target?   -> light off, done (resumes if it wanders past leave_mm)
stimulus playing?             -> keep driving the DAC through its on/gap segments, return
cooldown?                     -> refractory_frames after a stimulus: do nothing
re-zap gate?                  -> until moved rezap_radius_mm from the last zap point or rezap_frames passed
rate cap?                     -> max_pulses_per_min
roaming gate? (off by default)-> only judge when speed >= min_speed_um_s and straightness >= min_straightness
heading unknown?              -> worm has not moved min_travel_mm in the heading window
heading within away_angle_deg of target? -> 'off': worm is doing the right thing, run age resets
else: run age += 1; if a decision age is reached -> choose an arm and act
```

The wrong-way test has hysteresis: once a run is under way it only ends when the heading comes back
inside `away_angle_deg - 15°`, so a heading hovering at the margin does not reset the counter.

A light arm sets `cooldown` and the re-zap gate; WAIT does not, so the run keeps ageing toward the
next decision. Every frame writes a `track` line to the log with the `action` it took (`cooldown`,
`holding`, `rate_limited`, `dwelling`, `no_heading`, `off`, `wrong_way`, `stimulus`, `gap`, `sham`,
`wait`, `arrived`), and the Plugin tab shows the same once a second with angle, run age, speed,
straightness and distance.

## Modes

| `mode` | choice rule | table |
|---|---|---|
| `'learn'` (default) | every arm `min_trials` times per state, then epsilon-greedy (`epsilon`) on success rate | updated |
| `'train'` | arms in round-robin within each state, balanced data | updated |
| `'test'` | best light arm per state from the loaded table; never the sham; WAIT if the state is unknown | **not** updated |

A test run is a clean measurement of the learned policy; its outcomes are still logged with
`"mode": "test"`. A test run with an empty table will WAIT forever, so it needs a `table_file`.

## The table file

`table_file = None` keeps the table in memory only; it is gone at Stop or Reload. Set it to a path,
e.g. `r'C:\Users\soroka\zap_table.json'`, and the table is saved after every outcome and loaded on the
next Start, so counts accumulate across worms and days. Use a different file for a different protocol
(e.g. the dose series) so tables do not mix. Delete it when the light source or strain changes.

If a run was made without a file, the table can be recovered from the log, because every `outcome`
line carries the whole table at that moment:

```
python -c "import json, plot_bandit_table as p; json.dump(p.load_table('plugin_log_....jsonl'), open('zap_table.json','w'), indent=1)"
```

## Tools

- `plot_bandit_table.py zap_table.json` or `... plugin_log_....jsonl`: heatmap of the table, rows =
  run age, columns = arms, cells = success rate with trial count, plus mean progress.
- `plugin_log_summary.py plugin_log_....jsonl`: where the frames went by action, how long wrong-way
  runs got and what ended them, speed/straightness on gated frames, heading-error distribution, pulse
  intervals. Run this first whenever the plugin "does nothing".

## Parameters worth touching

| parameter | default | note |
|---|---|---|
| `target_xy` / `target_offset` | `None` / `(15, 0)` | absolute stage mm, or offset from where tracking starts |
| `away_angle_deg` | 100 | wrong way means more than this off the target direction |
| `decision_ages_s` | `(1, 3, 8)` | the states; one entry gives a plain bandit over arms |
| `arms` | see above | keep the sham; add `(4.5, 30, 1, 0)` for a 1 s pulse |
| `include_wait` | `True` | set False for a dose series |
| `refractory_frames` | 360 | 12 s: reversal plus omega before the next decision |
| `eval_frames` | 300 | scoring delay after the stimulus ends |
| `max_pulses_per_min` | 6 | hard cap; 2 per minute was habituation-free over 23 min |
| `roaming_gate` | `False` | turn on only with thresholds measured on the animal |
| `min_trials`, `epsilon` | 2, 0.2 | exploration in learn mode |

## What the data say so far (23 and 24 Sep 2026)

- A 1 to 2 s pulse at 4.5 V reverses the worm 80 to 85% of the time, latency 1.2 to 1.3 s, depth about
  one body length. Longer light adds nothing. Two worms, no habituation over 50 stimuli.
- Twelve seconds after a stimulus the worm heads within 90° of the target 54% of the time, with a
  median 264 µm of progress. Waiting instead: 44% and −177 µm after a 1 s run, falling to 12% and
  −2 mm after a 10 s run. Zapping beats waiting at every run age, so the expected optimum is "fire as
  soon as a wrong-way run is confirmed".
- The exit heading after the reversal is close to random. Guidance therefore takes several stimuli per
  correction; about 1.2 mm per minute of net progress was achieved on a 15 mm target.

## Pitfalls

- The log lands in the recording folder that was active when the plugin first wrote. Press Record,
  then Start, or the log ends up in the previous recording's folder.
- If the worm is lost, the tracker follows noise and the plugin keeps scoring nonsense. Watch the
  status line: speed near zero for a minute means the worm is gone. Outcomes after that point must be
  removed from the table.
- `state.frame` restarts when a new acquisition starts; the plugin counts its own frames for that
  reason. Do not change that.
- The sham arm delivers no light but still costs a full refractory period. That is intentional.
