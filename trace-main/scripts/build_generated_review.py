#!/usr/bin/env python3
"""Pick successful pairs from the plan-conditioned AppWorld pilot for blind human + judge review.

One pair per distinct write-signature pair per task (proxy E2), plus pairs from different
plans with identical writes (information-source plans, proxy E1) and no-plan GPT-vs-Haiku
controls. Appends to the human review page; writes judge packets + hidden provenance.
"""
import argparse, itertools, json, random, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--run', default='pilot')
ap.add_argument('--out', default='generated_audit')
ap.add_argument('--prefix', default='gn')
ap.add_argument('--page', action='store_true', help='also append to annotation/prevalence_review_data.js')
args = ap.parse_args()
RUN = ROOT / 'outputs/appworld_targeted' / args.run
OUT = ROOT / 'data/g1' / args.out
rng = random.Random(0)
BOOLS = {(a['method'].lower(), re.sub(r'\{[^}]+\}', '{id}', a['path'])): {q['name']: q['default'] for q in a['parameters'] if q['type'] == 'boolean'}
         for f in (ROOT / 'envs/appworld_root/data/api_docs/standard').glob('*.json') for a in json.load(open(f)).values()}


def writes(rec):
    f = ROOT / 'envs/appworld_root/experiments/outputs' / rec['appworld_experiment'] / 'tasks' / rec['task_id'] / 'logs/api_calls.jsonl'
    # same proxy as summarize_appworld_targeted.py: endpoint template + effective bool args
    return frozenset((c['method'], t, tuple(sorted((BOOLS.get((c['method'], t), {}) | {k: v for k, v in c['data'].items() if isinstance(v, bool)}).items())))
                     for c in map(json.loads, open(f)) for t in [re.sub(r'/\d+', '/{id}', c['url'])]
                     if c['method'] in ('post', 'put', 'patch', 'delete') and not c['url'].startswith(('/supervisor', '/api_docs'))
                     and not re.search(r'/(auth|login|token)', c['url']))


def render(rec):
    steps, k = [], 0
    for m in rec['messages'][2:]:
        if m['role'] == 'assistant':
            code = re.search(r'```(?:python)?\n(.*?)```', m['content'], re.S)
            k += 1
            steps.append(f'turn {k} CODE:\n' + (code.group(1).strip() if code else m['content'][:600]))
        else:
            out = m['content'].removeprefix('Output:\n')
            steps.append(f'turn {k} OUTPUT:\n' + (out if len(out) <= 1500 else out[:1500] + f'...[+{len(out) - 1500} chars]'))
    return steps


by = defaultdict(list)
for f in sorted(RUN.glob('*/*/rollout.json')):
    r = json.load(open(f))
    if r['success']:
        r['cond'] = r['plan']['id'] if r['plan'] else 'noplan'
        r['w'] = writes(r)
        by[r['task_id']].append(r)

pairs = []
for tid, rs in sorted(by.items()):
    sig = defaultdict(list)
    for r in rs: sig[r['w']].append(r)
    for a, b in itertools.combinations(sorted(sig, key=sorted), 2):
        pairs.append((rng.choice(sig[a]), rng.choice(sig[b]), 'E2'))
info = [(a, b) for rs in by.values() for a, b in itertools.combinations(rs, 2)
        if a['w'] == b['w'] and a['cond'] != b['cond'] and 'noplan' not in (a['cond'], b['cond']) and a['model'] == b['model']]
pairs += [(a, b, 'E1-diffplan') for a, b in rng.sample(info, min(6, len(info)))]
ctl = [(a, b) for rs in by.values() for a, b in itertools.combinations(rs, 2)
       if a['cond'] == b['cond'] == 'noplan' and a['model'] != b['model']]
pairs += [(a, b, 'noplan-ctl') for a, b in rng.sample(ctl, min(2, len(ctl)))]
rng.shuffle(pairs)

packets, hidden = [], []
for i, (a, b, proxy) in enumerate(pairs):
    a, b = rng.sample([a, b], 2)
    pid = f'{args.prefix}-{i:02d}'
    spec = json.load(open(ROOT / 'envs/appworld_root/data/tasks' / a['task_id'] / 'specs.json'))
    packets.append({'pair_id': pid, 'domain': 'appworld_generated', 'task': spec['instruction'],
                    'note': 'Both trajectories passed AppWorld state-based unit tests (incl. collateral-damage checks). Each step is the agent code and its printed output.',
                    'A': render(a), 'B': render(b)})
    hidden.append({'pair_id': pid, 'task_id': a['task_id'], 'proxy': proxy,
                   'A': f"{a['model']}:{a['cond']}", 'B': f"{b['model']}:{b['cond']}",
                   'A_plan': a['plan'], 'B_plan': b['plan']})
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'judge_packets.jsonl', 'w') as f:
    for x in packets: f.write(json.dumps(x, ensure_ascii=False) + '\n')
json.dump({'seed': 0, 'run': str(RUN.relative_to(ROOT)), 'items': hidden}, open(OUT / 'provenance_hidden.json', 'w'), indent=1, ensure_ascii=False)

if not args.page:
    print(len(packets), 'pairs ->', OUT, {k: sum(h['proxy'] == k for h in hidden) for k in ('E2', 'E1-diffplan', 'noplan-ctl')}); raise SystemExit
page = ROOT / 'annotation/prevalence_review_data.js'
items = json.loads(page.read_text().removeprefix('window.PREV_REVIEW_DATA = ').rstrip().rstrip(';'))
items = [x for x in items if not x['pair_id'].startswith('gn-')] + [
    {k: p[k] for k in ('pair_id', 'domain', 'task', 'note', 'A', 'B')} for p in packets]
page.write_text('window.PREV_REVIEW_DATA = ' + json.dumps(items, ensure_ascii=False) + ';\n')
print(len(packets), 'generated pairs;', {k: sum(h['proxy'] == k for h in hidden) for k in ('E2', 'E1-diffplan', 'noplan-ctl')}, '| page now', len(items), 'pairs')
