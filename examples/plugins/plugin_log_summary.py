"""Why did the plugin (not) zap? Summarise a plugin log: where the frames went, how far wrong-way runs got
before something reset them, pulse timing. Standard library only, so it runs on the microscope PC.

    python plugin_log_summary.py plugin_log_20260924_101500.jsonl
"""
import json, sys, collections

def main(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    track = [r for r in rows if r.get('event') == 'track']
    pulses = [r for r in rows if r.get('event') in ('pulse', 'decision')]
    if not track:
        print('no track lines in this log'); return
    fps = 30.0
    t = [r.get('t', 0) for r in track]
    dur = (t[-1] - t[0]) if len(t) > 1 else 0
    print(f'{len(track)} tracked frames over {dur/60:.1f} min; {len(pulses)} pulses/decisions')
    mix = collections.Counter(r.get('action') for r in track)
    print('\nwhere the frames went:')
    for a, c in mix.most_common():
        print(f'  {a:14s} {c:7d}  {c/len(track):6.1%}')
    # wrong-way runs: consecutive frames with action wrong_way (or 'off' with age>0 in v2), what ended them
    runs, cur, ender = [], 0, collections.Counter()
    for i, r in enumerate(track):
        a = r.get('action'); age = r.get('age_s')
        in_run = a in ('wrong_way',) or (a == 'off' and age is not None and age > 0)
        if in_run:
            cur += 1
        elif cur:
            runs.append(cur); ender[a] += 1; cur = 0
    if runs:
        runs_s = sorted(x / fps for x in runs)
        print(f'\nwrong-way runs: {len(runs)}; length median {runs_s[len(runs_s)//2]:.1f} s, 90th pct {runs_s[int(len(runs_s)*0.9)]:.1f} s, max {runs_s[-1]:.1f} s')
        print('  what ended them:', ', '.join(f'{k} {v}' for k, v in ender.most_common()))
    else:
        print('\nno wrong-way runs at all: the heading was never judged as away from the target (check dwelling / no_heading / cooldown shares above)')
    dw = [r for r in track if r.get('action') == 'dwelling' and r.get('speed') is not None]
    if dw:
        sp = sorted(r['speed'] for r in dw); st = sorted(r.get('straightness', 0) for r in dw)
        print(f'\ndwelling frames: speed median {sp[len(sp)//2]:.0f} um/s (90th pct {sp[int(len(sp)*0.9)]:.0f}); straightness median {st[len(st)//2]:.2f} (90th pct {st[int(len(st)*0.9)]:.2f})')
        print('  -> if these sit just under min_speed_um_s / min_straightness, lower them')
    ang = [r['angle_deg'] for r in track if r.get('angle_deg') is not None]
    if ang:
        ang.sort(); print(f'\nheading error when judged: median {ang[len(ang)//2]:.0f} deg; share > 100 deg {sum(a > 100 for a in ang)/len(ang):.0%}, share > 90 deg {sum(a > 90 for a in ang)/len(ang):.0%}')
    if len(pulses) > 1:
        pt = [r.get('t', 0) for r in pulses]; gaps = sorted(b - a for a, b in zip(pt[:-1], pt[1:]))
        print(f'\npulse intervals: median {gaps[len(gaps)//2]:.0f} s, min {gaps[0]:.0f} s, max {gaps[-1]:.0f} s')

if __name__ == '__main__':
    main(sys.argv[1])
