#!/usr/bin/env python3
"""Pick pairs for human review (all HQ-E2 candidates + label-split pairs) and write blind page data."""
import json, random, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data/g1/prevalence_audit'
J = defaultdict(dict)
for j in ['gpt56', 'kimi', 'opus']:
    for l in open(D / 'judgments' / f'{j}.jsonl'):
        r = json.loads(l); J[r['pair_id']][j] = r
s = json.load(open(D / 'summary_gpt56_kimi_opus.json'))
cand = [x['pair_id'] for g in s['groups'].values() for x in g['hq_E2_pairs']]
split = [p for p in sorted(J) if len({J[p][j]['label'] for j in J[p]}) > 1 and p not in cand]
rng = random.Random(0)
pick = cand + rng.sample([p for p in split if p.startswith('aw')], 6) + rng.sample([p for p in split if p.startswith('tc')], 4)
rng.shuffle(pick)
packets = {json.loads(l)['pair_id']: json.loads(l) for l in open(D / 'judge_packets.jsonl')}
strip = lambda xs: [re.sub(r'^\d+\. ', '', x) for x in xs]
items = [{'pair_id': p, 'domain': packets[p]['domain'], 'task': packets[p]['task'], 'note': packets[p]['note'],
          'A': strip(packets[p]['A']), 'B': strip(packets[p]['B'])} for p in pick]
(ROOT / 'annotation/prevalence_review_data.js').write_text('window.PREV_REVIEW_DATA = ' + json.dumps(items, ensure_ascii=False) + ';\n')
json.dump({'pairs': pick, 'hq_e2_candidates': cand, 'seed': 0}, open(D / 'human_review_selection.json', 'w'), indent=1)
print(len(items), 'pairs', pick)
