"""Heatmap of the bandit table: rows = state (run age at decision), columns = action, colour = expected reward.

    python plot_bandit_table.py zap_table.json [out.png]

Reads the JSON written by bandit_zapper.py (table_file) or the `table` field of the last `outcome` line
in a plugin log (pass the .jsonl instead of the .json). Cells show success rate with the trial count,
and the mean progress toward the target in a second panel.
"""
import json, sys, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_table(path):
    if path.endswith('.jsonl'):
        table = {}
        for line in open(path, encoding='utf-8'):
            if '"event": "outcome"' in line:
                table = json.loads(line).get('table', table)
        return table
    return json.load(open(path, encoding='utf-8'))

def age_key(label):
    m = re.search(r'([\d.]+)', label); return float(m.group(1)) if m else 0.0

def main(path, out=None):
    table = load_table(path)
    rows = sorted({k.split('|')[0] for k in table}, key=age_key)
    cols = sorted({k.split('|', 1)[1] for k in table}, key=lambda c: (c == 'WAIT', c.startswith('SHAM'), c))
    n = np.zeros((len(rows), len(cols))); succ = np.full_like(n, np.nan); prog = np.full_like(n, np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = table.get(f'{r}|{c}')
            if v and v['n']:
                n[i, j] = v['n']; succ[i, j] = v['successes'] / v['n']; prog[i, j] = v['progress_sum'] / v['n'] * 1000
    seq = LinearSegmentedColormap.from_list('seq', ['#F6F7F5', '#F2C48D', '#C46A00', '#5A2E00'])
    div = LinearSegmentedColormap.from_list('div', ['#2B6CB0', '#DDE1E6', '#C46A00'])
    fig, axes = plt.subplots(1, 2, figsize=(4.2 + 1.6 * len(cols), 1.6 + 0.9 * len(rows)))
    for ax, M, cmap, title, fmt, vmin, vmax in ((axes[0], succ, seq, 'expected reward: P(success)', '{:.0%}', 0, 1),
                                                (axes[1], prog, div, 'mean progress toward target (um)', '{:+.0f}', -800, 800)):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=25, ha='right', fontsize=9)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
        ax.set_xlabel('action'); ax.set_ylabel('state: run age at decision' if ax is axes[0] else '')
        ax.set_title(title, fontsize=11)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if n[i, j]:
                    val = M[i, j]; dark = (val > 0.55) if M is succ else (abs(val) > 450)
                    ax.text(j, i, fmt.format(val) + f'\nn={int(n[i, j])}', ha='center', va='center', fontsize=8.5, color='white' if dark else '#1C2128')
                else:
                    ax.text(j, i, 'untried', ha='center', va='center', fontsize=8, color='#9AA5B1')
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which='minor', color='white', lw=2); ax.tick_params(which='minor', length=0)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    total = int(n.sum())
    fig.suptitle(f'Bandit table: {os.path.basename(path)}  ({total} scored decisions)', fontsize=11)
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + '_heatmap.png'
    fig.savefig(out, dpi=130, facecolor='white'); print('wrote', out)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
