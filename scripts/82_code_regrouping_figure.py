#!/usr/bin/env python3
"""Plot the completed frozen regrouping predictions and outcomes."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path('runs/swe-diversity-selection/code-regrouping-first-block/analysis.json')
OUT = Path('runs/swe-diversity-selection/paper-program/figures')
rows = json.loads(SOURCE.read_text())['per_task']
assert len(rows) == 34
OUT.mkdir(exist_ok=True, parents=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, eta in zip(axes, [1.0, 10.0]):
    selected = [r for r in rows if r['eta'] == eta]
    x = np.array([r['prediction'] for r in selected])
    y = np.array([r['delta'] for r in selected])
    controls = np.array([r['task_id'] in [624, 738] for r in selected])
    ax.scatter(x[~controls], y[~controls], s=45, color='#2563a8', label='Other source tasks', alpha=.8)
    ax.scatter(x[controls], y[controls], s=85, color='#d76a16', marker='x', linewidths=2,
               label='Identical-objective controls', zorder=4)
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    margin = .08 * (high - low)
    limits = [low - margin, high + margin]
    ax.plot(limits, limits, linestyle='--', color='.6', linewidth=1, label='Exact prediction')
    ax.axhline(0, color='.8', linewidth=.7)
    ax.axvline(0, color='.8', linewidth=.7)
    ax.set(xlim=limits, ylim=limits, title=f'SGD step {eta:g}',
           xlabel='Predicted mixed advantage (NLL/token)', ylabel='Measured mixed advantage (NLL/token)')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.grid(alpha=.15)
fig.suptitle('Regrouping: calibration prediction versus measured update effect', fontsize=13)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=3, frameon=False)
fig.text(.5, .025, '17 source tasks; all outcomes retained. Positive advantage favors mixed batches. Reference loss is not executable accuracy.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .15, 1, .94))
for extension in ['png', 'pdf']:
    fig.savefig(OUT / f'regrouping_prediction.{extension}', dpi=200)
manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                matplotlib=matplotlib.__version__, no_new_fitting=True, api_spend_usd=0)
(OUT / 'regrouping_prediction_data.json').write_text(json.dumps(rows, indent=2) + '\n')
(OUT / 'regrouping_prediction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Saved standalone PNG, PDF, data and manifest.')
