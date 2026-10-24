#!/usr/bin/env python3
"""Majority E-labels, quality-gated E2 rate and judge agreement for the prevalence audit."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data/g1/prevalence_audit')
ap.add_argument('--judges', nargs='+', default=None)
args = ap.parse_args()
D = args.dir
hidden = {x['pair_id']: x for x in json.load(open(D / 'provenance_hidden.json'))['items']}
J = defaultdict(dict)
for f in sorted((D / 'judgments').glob('*.jsonl')):
    for l in open(f):
        r = json.loads(l); J[r['pair_id']][f.stem] = r
judges = args.judges or sorted({j for v in J.values() for j in v})
maj = lambda xs: (lambda c: c[0][0] if c[0][1] * 2 > len(xs) else 'split')(Counter(xs).most_common())


def fleiss(rows, cats):
    m = np.array([[Counter(r)[c] for c in cats] for r in rows], float); n = m.sum(1)[0]
    p = m.sum(0) / m.sum(); P = ((m * (m - 1)).sum(1) / (n * (n - 1))).mean()
    return (P - (p ** 2).sum()) / (1 - (p ** 2).sum())


out = {'judges': judges, 'groups': {}}
for dom in ('appworld', 'tau2_telecom'):
    for proxy in ('E2', 'E0/E1'):
        pids = [p for p, h in hidden.items() if p.startswith({'appworld': 'aw', 'tau2_telecom': 'tc'}[dom]) and h['proxy'] == proxy and all(j in J[p] for j in judges)]
        if not pids: continue
        labels = [maj([J[p][j]['label'] for j in judges]) for p in pids]
        both_hq = [maj([J[p][j]['a_high_quality'] for j in judges]) == 'yes' and maj([J[p][j]['b_high_quality'] for j in judges]) == 'yes' for p in pids]
        g = {'n': len(pids), 'majority_label': dict(Counter(labels)), 'both_high_quality': int(sum(both_hq)),
             'hq_E2': int(sum(l == 'E2' and h for l, h in zip(labels, both_hq))),
             'unanimous': int(sum(len({J[p][j]['label'] for j in judges}) == 1 for p in pids)),
             'fleiss_kappa_label': round(fleiss([[J[p][j]['label'] for j in judges] for p in pids], ['E0', 'E1', 'E2', 'ambiguous']), 3),
             'per_judge': {j: dict(Counter(J[p][j]['label'] for p in pids)) for j in judges},
             'hq_E2_pairs': [{'pair_id': p, 'task_id': hidden[p]['task_id'], 'A': hidden[p]['A'], 'B': hidden[p]['B'],
                              'diff': J[p][judges[0]]['difference_summary']} for p, l, h in zip(pids, labels, both_hq) if l == 'E2' and h]}
        out['groups'][f'{dom}|proxy={proxy}'] = g
        print(f"{dom:13s} proxy={proxy:6s} n={g['n']:2d} majority={g['majority_label']} both_HQ={g['both_high_quality']} "
              f"HQ_E2={g['hq_E2']} unanimous={g['unanimous']} kappa={g['fleiss_kappa_label']} per_judge={g['per_judge']}")
json.dump(out, open(D / f"summary_{'_'.join(judges)}.json", 'w'), indent=1, ensure_ascii=False)
