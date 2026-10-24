#!/usr/bin/env python3
"""Consensus LLM judge + Chinese digests for human spot-check review.

Final label/quality = majority of the three blind judges (gpt56, kimi, opus). GPT-5.6 separately
writes a neutral Chinese digest of each trajectory and the key differences; it gives no label,
so the digest cannot anchor the reviewer on a verdict. Writes annotation/judge_review_data.js
for a stratified sample.
  ../.venv/bin/python scripts/build_judge_digest.py [--n-sample 12]
"""
import argparse, json, os, random, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / '.env')
SETS = {'prevalence': ROOT / 'data/g1/prevalence_audit', 'generated': ROOT / 'data/g1/generated_audit'}
JUDGES = ['gpt56', 'kimi', 'opus']
OUT = ROOT / 'data/g1/review_digest'
DIGEST_SYS = """你在帮研究人员快速读懂两条 agent 轨迹。两条轨迹解决的是同一个任务，并且都通过了 benchmark 的成功检查。
用中文写，API 名、文件名、ID 保持原样。只描述，不要给 E0/E1/E2 标签，也不要评价谁更好。

对 A 和 B，各写 3-6 条要点，说明 agent 实际做了什么：关键决策、每个改变状态的操作、遇到的错误和补救方式。
每条要点注明对应的步骤号。再写最多 3 条 key_differences：两条轨迹在"做了什么改动、依据什么信息做决定、出错后怎么补救"
上的实质差异。只是多查了信息、措辞不同或重试次数不同，不算实质差异。如果没有实质差异，key_differences 写 ["无实质差异"]。

只返回 JSON：{"A": ["..."], "B": ["..."], "key_differences": ["..."]}"""

ap = argparse.ArgumentParser()
ap.add_argument('--n-sample', type=int, default=12)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)
maj = lambda xs: (lambda c: c[0][0] if c[0][1] * 2 > len(xs) else 'split')(Counter(xs).most_common())

pairs = {}
review = json.load(open(SETS['prevalence'] / 'human_review_selection.json'))['pairs']
for name, d in SETS.items():
    packets = {json.loads(l)['pair_id']: json.loads(l) for l in open(d / 'judge_packets.jsonl')}
    hidden = {x['pair_id']: x for x in json.load(open(d / 'provenance_hidden.json'))['items']}
    J = {j: {json.loads(l)['pair_id']: json.loads(l) for l in open(d / 'judgments' / f'{j}.jsonl')} for j in JUDGES}
    for pid, pk in packets.items():
        if name == 'prevalence' and pid not in review:
            continue
        js = {j: J[j][pid] for j in JUDGES}
        pairs[pid] = {'set': name, 'packet': pk, 'hidden': hidden[pid], 'judges': js,
                      'label': maj([v['label'] for v in js.values()]),
                      'votes': dict(Counter(v['label'] for v in js.values())),
                      'a_hq': maj([v['a_high_quality'] for v in js.values()]),
                      'b_hq': maj([v['b_high_quality'] for v in js.values()])}


def digest(pid):
    pk = pairs[pid]['packet']
    user = json.dumps({k: pk[k] for k in ('task', 'note', 'A', 'B')}, ensure_ascii=False)
    key = os.environ.get('OPENAI_API_KEY') or os.environ['openai_api_key']
    for attempt in range(3):
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions', timeout=900, headers={'Authorization': f'Bearer {key}'},
                              json={'model': 'gpt-5.6-sol', 'max_completion_tokens': 8000,
                                    'messages': [{'role': 'system', 'content': DIGEST_SYS}, {'role': 'user', 'content': user}]})
            r.raise_for_status(); j = r.json(); t = j['choices'][0]['message']['content']
            d = json.loads(t[t.find('{'): t.rfind('}') + 1])
            assert all(isinstance(d[k], list) for k in ('A', 'B', 'key_differences'))
            return {'pair_id': pid, **d, 'usage': j['usage']}
        except Exception as e:
            err = e; time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'digest failed {pid}: {err}')


OUT.mkdir(parents=True, exist_ok=True)
cache = OUT / 'digests.jsonl'
done = {json.loads(l)['pair_id']: json.loads(l) for l in open(cache)} if cache.exists() else {}
todo = [p for p in pairs if p not in done]
with ThreadPoolExecutor(4) as ex, open(cache, 'a') as fh:
    for d in ex.map(digest, todo):
        fh.write(json.dumps(d, ensure_ascii=False) + '\n'); fh.flush(); done[d['pair_id']] = d
        print('digest', d['pair_id'], d['usage']['prompt_tokens'], d['usage']['completion_tokens'], flush=True)

hq = lambda p: pairs[p]['a_hq'] == 'yes' and pairs[p]['b_hq'] == 'yes'
strata = {'E2_both_HQ': [p for p in pairs if pairs[p]['label'] == 'E2' and hq(p)],
          'E2_low_quality': [p for p in pairs if pairs[p]['label'] == 'E2' and not hq(p)],
          'split_vote': [p for p in pairs if len(pairs[p]['votes']) > 1],
          'E1': [p for p in pairs if pairs[p]['label'] in ('E0', 'E1')]}
quota = {'E2_both_HQ': 5, 'E2_low_quality': 2, 'split_vote': 2, 'E1': 3}
sample, why = [], {}
for s, q in quota.items():
    for p in rng.sample([x for x in strata[s] if x not in sample], min(q, len([x for x in strata[s] if x not in sample]))):
        sample.append(p); why[p] = s
for p in rng.sample([x for x in pairs if x not in sample], max(0, args.n_sample - len(sample))):
    sample.append(p); why[p] = 'random'
rng.shuffle(sample)

items = []
for p in sample:
    x = pairs[p]
    items.append({'pair_id': p, 'set': x['set'], 'task': x['packet']['task'],
                  'label': x['label'], 'votes': x['votes'], 'a_hq': x['a_hq'], 'b_hq': x['b_hq'],
                  'digest': {k: done[p][k] for k in ('A', 'B', 'key_differences')},
                  'judges': {j: {k: v[k] for k in ('label', 'a_high_quality', 'b_high_quality', 'difference_summary', 'rationale')} for j, v in x['judges'].items()},
                  'raw': {'A': x['packet']['A'], 'B': x['packet']['B']}})
(ROOT / 'annotation/judge_review_data.js').write_text('window.JUDGE_REVIEW_DATA = ' + json.dumps(items, ensure_ascii=False) + ';\n')
json.dump({'sample': sample, 'stratum': why, 'strata_sizes': {k: len(v) for k, v in strata.items()},
           'consensus': {p: {k: pairs[p][k] for k in ('set', 'label', 'votes', 'a_hq', 'b_hq')} | {'proxy': pairs[p]['hidden']['proxy']} for p in pairs}},
          open(OUT / 'consensus_and_sample.json', 'w'), indent=1, ensure_ascii=False)
tok = [(d['usage']['prompt_tokens'], d['usage']['completion_tokens']) for d in done.values()]
print('pairs', len(pairs), '| strata', {k: len(v) for k, v in strata.items()}, '| sample', len(sample),
      f"| digest cost ≈ ${sum(i for i, _ in tok) * 4e-6 + sum(o for _, o in tok) * 20e-6:.2f}")
