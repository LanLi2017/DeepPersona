#!/usr/bin/env python3
"""Sample successful same-task trajectory pairs from public AppWorld / tau2-telecom baseline
outputs and build blind pairwise judge packets for the E0/E1/E2 prevalence audit.

Sources are test-split baseline outputs: used for prevalence only, never for G1 calibration.
Per domain: n_e2 pairs the structural proxy flags as E2 + n_ctl proxy-E0/E1 controls,
at most one pair per task.
  envs/appworld/bin/python scripts/build_prevalence_audit.py --replay  (AppWorld replay needs that env)
"""
import argparse, glob, itertools, json, random, re, subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AW = ROOT / 'envs/appworld_root/experiments/outputs'
T2 = ROOT / 'envs/tau2-bench-src/data/tau2'
OUT = ROOT / 'data/g1/prevalence_audit'
p = argparse.ArgumentParser()
p.add_argument('--n-e2', type=int, default=24)
p.add_argument('--n-ctl', type=int, default=6)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--smoke', action='store_true')
args = p.parse_args()
rng = random.Random(args.seed)
if args.smoke:
    args.n_e2, args.n_ctl, OUT = 1, 0, OUT.with_name('prevalence_audit_smoke')


def aw_writes(calls):
    return frozenset((c['method'], re.sub(r'/\d+', '/{id}', c['url'])) for c in calls
                     if c['method'] in ('post', 'put', 'patch', 'delete') and not c['url'].startswith(('/supervisor', '/api_docs'))
                     and not re.search(r'/(auth|login|token)', c['url']))


aw_succ = defaultdict(dict)
for d in sorted(x for x in AW.iterdir() if x.name.endswith(('test_normal', 'test_challenge'))):
    split = 'test_challenge' if d.name.endswith('test_challenge') else 'test_normal'
    for tid, r in json.load(open(d / 'evaluations' / f'{split}.json'))['individual'].items():
        f = d / 'tasks' / tid / 'logs' / 'api_calls.jsonl'
        if r['success'] and f.exists():
            aw_succ[tid][d.name] = [json.loads(l) for l in open(f)]

READ = ('get_', 'check_', 'can_', 'run_speed')
SOFT = {'reboot_device', 'toggle_airplane_mode'}
t2_succ = defaultdict(list)
for f in sorted(glob.glob(str(T2 / 'results/final/*_telecom_default_*.json')) + glob.glob(str(T2 / 'results/final/*_telecom_base_*.json'))):
    for s in json.load(open(f))['simulations']:
        if s['reward_info'] and s['reward_info']['reward'] == 1:
            t2_succ[s['task_id']].append((Path(f).name.split('_telecom')[0], s))
t2_writes = lambda s: frozenset((c['name'], json.dumps(c['arguments'], sort_keys=True)) for m in s['messages']
                                for c in (m.get('tool_calls') or []) if not c['name'].startswith(READ) and c['name'] not in SOFT)


def sample(succ, writes):
    e2, ctl = [], []
    for tid in sorted(succ):
        pairs = list(itertools.combinations(sorted(succ[tid].items()) if isinstance(succ[tid], dict) else succ[tid], 2))
        rng.shuffle(pairs)
        flagged = [pq for pq in pairs if writes(pq[0][1]) != writes(pq[1][1])]
        same = [pq for pq in pairs if writes(pq[0][1]) == writes(pq[1][1])]
        if flagged: e2.append((tid, flagged[0]))
        if same: ctl.append((tid, same[0]))
    rng.shuffle(e2); rng.shuffle(ctl)
    return [(t, pq, 'E2') for t, pq in e2[:args.n_e2]] + [(t, pq, 'E0/E1') for t, pq in ctl[:args.n_ctl]]


def short(x, n=400):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + f'...[+{len(s) - n} chars]'


packets, hidden = [], []
t2_tasks = {t['id']: t for t in json.load(open(T2 / 'domains/telecom/tasks.json'))}
telecom_policy = (T2 / 'domains/telecom/main_policy.md').read_text() + '\n\n' + (T2 / 'domains/telecom/tech_support_manual.md').read_text()

for i, (tid, ((ca, _), (cb, _)), proxy) in enumerate(sample(aw_succ, aw_writes)):
    trajs = []
    for cfg in (ca, cb):
        out = OUT / 'appworld_replay' / cfg / f'{tid}.json'
        if not out.exists():
            subprocess.run([str(ROOT / 'envs/appworld/bin/python'), str(ROOT / 'scripts/replay_appworld.py'), str(AW / cfg), tid, str(out)],
                           check=True, env={'APPWORLD_ROOT': str(ROOT / 'envs/appworld_root'), 'PATH': '/usr/bin:/bin'}, capture_output=True)
        r = json.load(open(out))
        st = [s for s in r['steps'] if not s['url'].startswith('/api_docs')]
        reads = [k for k, s in enumerate(st) if s['method'] == 'get']
        # Long traces (up to ~1.2k calls) overflow judge context: keep every write, first/last 30 reads.
        keep = set(range(len(st))) if len(st) <= 150 else {k for k, s in enumerate(st) if s['method'] != 'get'} | set(reads[:30] + reads[-30:])
        steps, gap = [], 0
        for k, s in enumerate(st, 1):
            if k - 1 not in keep:
                gap += 1; continue
            if gap: steps.append(f'... [{gap} read-only steps omitted]'); gap = 0
            steps.append(f"{k}. {s['method'].upper()} {s['url']} {short(s['data'], 200)} -> {short(s['response'])}")
        trajs.append((cfg, r['replay_success'], steps, r['instruction']))
    order = rng.sample([0, 1], 2)
    pid = f'aw-{i:02d}'
    packets.append({'pair_id': pid, 'domain': 'appworld', 'task': trajs[0][3],
                    'note': 'Both trajectories passed AppWorld state-based unit tests (incl. collateral-damage checks). Steps are API calls with replayed responses.',
                    'A': trajs[order[0]][2], 'B': trajs[order[1]][2]})
    hidden.append({'pair_id': pid, 'task_id': tid, 'A': trajs[order[0]][0], 'B': trajs[order[1]][0], 'proxy': proxy,
                   'replay_success': [trajs[order[0]][1], trajs[order[1]][1]]})

for i, (tid, ((ma, sa), (mb, sb)), proxy) in enumerate(sample({t: v for t, v in t2_succ.items()}, t2_writes)):
    task = t2_tasks[tid]
    render = lambda s: [f"{k}. {m['role']}: " + (short(m['content'], 300) if m.get('content') else '')
                        + ''.join(f" CALL {c['name']}({short(c['arguments'], 150)})" for c in (m.get('tool_calls') or []))
                        for k, m in enumerate(s['messages'], 1)]
    order = rng.sample([(ma, sa), (mb, sb)], 2)
    pid = f'tc-{i:02d}'
    packets.append({'pair_id': pid, 'domain': 'tau2_telecom', 'task': task['ticket'],
                    'note': 'Both trajectories received reward 1 (final device/account state assertions only; the path is not checked). Judge quality against the policy.',
                    'policy': telecom_policy, 'A': render(order[0][1]), 'B': render(order[1][1])})
    hidden.append({'pair_id': pid, 'task_id': tid, 'A': order[0][0], 'B': order[1][0], 'proxy': proxy})

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'judge_packets.jsonl', 'w') as f:
    for x in packets: f.write(json.dumps(x, ensure_ascii=False) + '\n')
json.dump({'seed': args.seed, 'n_e2': args.n_e2, 'n_ctl': args.n_ctl, 'split_note': 'public test-split baseline outputs; prevalence only',
           'items': hidden}, open(OUT / 'provenance_hidden.json', 'w'), indent=1)
print(f'{len(packets)} packets -> {OUT}', {d: sum(x['domain'] == d for x in packets) for d in ('appworld', 'tau2_telecom')})
