#!/usr/bin/env python3
"""Success rate, cost and structural E2 candidates for AppWorld targeted rollouts."""
import argparse, itertools, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICE = {'gpt56': (4.0, 20.0), 'haiku': (1.0, 5.0), 'kimi': (None, None)}  # $/1M in, out (OpenAI / Anthropic list prices)
ap = argparse.ArgumentParser()
ap.add_argument('--run', default='pilot')
args = ap.parse_args()
run = ROOT / 'outputs/appworld_targeted' / args.run
plans = json.load(open(run / 'plans.json'))
BOOLS = {(a['method'].lower(), re.sub(r'\{[^}]+\}', '{id}', a['path'])): {q['name']: q['default'] for q in a['parameters'] if q['type'] == 'boolean'}
         for f in (ROOT / 'envs/appworld_root/data/api_docs/standard').glob('*.json') for a in json.load(open(f)).values()}


def writes(rec):
    f = ROOT / 'envs/appworld_root/experiments/outputs' / rec['appworld_experiment'] / 'tasks' / rec['task_id'] / 'logs/api_calls.jsonl'
    calls = [json.loads(l) for l in open(f)]
    # effective bool args are included: gn-08 (retain_dates True vs default False) was human-confirmed E2
    # with identical endpoints; defaults are filled in so passing a default explicitly is not a difference
    out = set()
    for c in calls:
        tmpl = re.sub(r'/\d+', '/{id}', c['url'])
        if c['method'] in ('post', 'put', 'patch', 'delete') and not c['url'].startswith(('/supervisor', '/api_docs')) \
                and not re.search(r'/(auth|login|token)', c['url']):
            out.add((c['method'], tmpl, tuple(sorted((BOOLS.get((c['method'], tmpl), {}) | {k: v for k, v in c['data'].items() if isinstance(v, bool)}).items()))))
    return frozenset(out)


recs = [json.load(open(f)) for f in sorted(run.glob('*/*/rollout.json'))]
by = defaultdict(list)
for r in recs:
    r['cond'] = r['plan']['id'] if r['plan'] else 'noplan'
    r['writes'] = writes(r)
    by[r['task_id']].append(r)

print(f'{len(recs)} rollouts')
for m in sorted({r['model'] for r in recs}):
    rs = [r for r in recs if r['model'] == m]
    cost = sum(r['usage']['in'] * PRICE[m][0] + r['usage']['out'] * PRICE[m][1] for r in rs) / 1e6 if PRICE[m][0] else float('nan')
    for c in ('noplan', 'plan'):
        sub = [r for r in rs if (r['cond'] == 'noplan') == (c == 'noplan')]
        if sub: print(f'  {m:6s} {c:7s} success {sum(r["success"] for r in sub)}/{len(sub)}  mean turns {sum(r["turns"] for r in sub)/len(sub):.1f}')
    print(f'  {m:6s} cost ≈ ${cost:.2f} (list price)')

# structural proxy among successful pairs within a task: E2 if the write-endpoint sets differ
lab, cross = Counter(), Counter()
rows = []
for tid, rs in by.items():
    ok = [r for r in rs if r['success']]
    for a, b in itertools.combinations(ok, 2):
        e2 = a['writes'] != b['writes']
        kind = ('noplan-vs-noplan' if a['cond'] == b['cond'] == 'noplan' else
                'same-plan' if a['cond'] == b['cond'] else 'plan-vs-other')
        lab[(kind, e2)] += 1
        if e2: rows.append((tid, f"{a['model']}:{a['cond']}", f"{b['model']}:{b['cond']}", sorted(a['writes'] - b['writes'])[:2], sorted(b['writes'] - a['writes'])[:2]))
for kind in ('noplan-vs-noplan', 'same-plan', 'plan-vs-other'):
    n = lab[(kind, True)] + lab[(kind, False)]
    if n: print(f'  {kind:17s} successful pairs {n:3d}  proxy-E2 {lab[(kind, True)]:3d} ({lab[(kind, True)] / n:.0%})')
print('tasks with >=1 proxy-E2 successful pair:', len({r[0] for r in rows}), '/', len(by))
for r in rows[:12]: print('   ', r)
json.dump({'n': len(recs), 'proxy_e2_pairs': rows}, open(run / 'summary.json', 'w'), indent=1, default=list)
