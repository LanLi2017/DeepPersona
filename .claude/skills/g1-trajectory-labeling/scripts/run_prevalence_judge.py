#!/usr/bin/env python3
"""Blind pairwise E0/E1/E2 + quality judge for the prevalence audit.

Backends: Bedrock converse (kimi, haiku) or the local Claude CLI (opus).
Resumable: pairs already in the output file are skipped.
  ../.venv/bin/python scripts/run_prevalence_judge.py --judge kimi --input data/g1/prevalence_audit_smoke/judge_packets.jsonl
"""
import argparse, hashlib, json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import requests
from botocore.config import Config
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / '.env')
JUDGES = {'kimi': ('bedrock', 'moonshot.kimi-k2-thinking', 12000),
          'haiku': ('bedrock', 'us.anthropic.claude-haiku-4-5-20251001-v1:0', 4000),
          'opus': ('cli', 'opus', None),
          'gpt56': ('openai', 'gpt-5.6-sol', 16000)}

SYSTEM = """You are an independent trajectory-equivalence annotator. You see two agent
trajectories A and B for the same task. Both passed the benchmark's success check, but
success alone does not mean high quality or the same strategy.

Labels (decision-relevance boundary, frozen):
- E0: only wording, identifiers or serialization differ; the same calls with the same effects.
- E1: same strategy. Allowed differences: extra or missing read-only lookups whose results
  are not used in a decision; reordering of independent steps; equivalent authentication
  routes; retries of the same call. Writes/state effects, the key information each decision
  rests on, and the recovery policy are materially the same.
- E2: a decision-relevant difference. Examples: different state-changing actions (e.g. a
  one-step API vs a delete+add sequence); decisions grounded in different key information;
  a different order between *dependent* subgoals; a different recovery branch after a failure.
- ambiguous: the evidence cannot support a reliable call.

Also judge each trajectory's quality: high quality means it completes the user's actual
request, respects the supplied policy, makes no unnecessary or harmful state changes, and
invents nothing. Reward 1 alone is not enough.
Return JSON only."""
SCHEMA = {"pair_id": "...", "a_high_quality": "yes|no|unclear", "b_high_quality": "yes|no|unclear",
          "same_state_effect": "yes|mostly|no|unclear", "same_key_information": "yes|mostly|no|unclear",
          "same_dependency_order": "yes|mostly|no|unclear", "same_recovery": "yes|mostly|no|not_applicable|unclear",
          "label": "E0|E1|E2|ambiguous", "confidence": "high|medium|low",
          "difference_summary": "one sentence: the most important difference, if any",
          "rationale": "concise", "evidence": ["A step k vs B step j ..."]}


def parse(text, pid):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    v = json.loads(text[text.find('{'): text.rfind('}') + 1])
    assert v['pair_id'] == pid and v['label'] in {'E0', 'E1', 'E2', 'ambiguous'}
    assert v['a_high_quality'] in {'yes', 'no', 'unclear'} and v['b_high_quality'] in {'yes', 'no', 'unclear'}
    return v


def call(judge, packet, client):
    kind, model, max_tok = JUDGES[judge]
    user = 'Return exactly this JSON shape:\n' + json.dumps(SCHEMA) + '\n\nPAIR:\n' + json.dumps(packet, ensure_ascii=False)
    for attempt in range(3):
        try:
            if kind == 'bedrock':
                r = client.converse(modelId=model, system=[{'text': SYSTEM}], messages=[{'role': 'user', 'content': [{'text': user}]}],
                                    inferenceConfig={'maxTokens': max_tok})
                text = ''.join(c.get('text', '') for c in r['output']['message']['content'])
                usage = r['usage']
            elif kind == 'openai':
                key = os.environ.get('OPENAI_API_KEY') or os.environ['openai_api_key']  # .env stores it lowercase
                resp = requests.post('https://api.openai.com/v1/chat/completions', timeout=900, headers={'Authorization': f'Bearer {key}'},
                                     json={'model': model, 'max_completion_tokens': max_tok,
                                           'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]})
                resp.raise_for_status(); raw = resp.json()
                text, usage = raw['choices'][0]['message']['content'], raw['usage']
            else:
                cp = subprocess.run(['claude', '-p', '--model', model, '--tools', '', '--output-format', 'json'],
                                    input=SYSTEM + '\n\n' + user, text=True, capture_output=True, timeout=900, check=True)
                w = json.loads(cp.stdout)
                text, usage = w['result'], {'cost_usd': w.get('total_cost_usd'), **(w.get('usage') or {})}
            return {'judge': judge, 'model': model, 'usage': usage, 'prompt_sha256': hashlib.sha256(user.encode()).hexdigest(),
                    **parse(text, packet['pair_id'])}
        except Exception as e:
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{judge} failed on {packet['pair_id']}: {err}")


p = argparse.ArgumentParser()
p.add_argument('--judge', choices=list(JUDGES), required=True)
p.add_argument('--input', type=Path, default=ROOT / 'data/g1/prevalence_audit/judge_packets.jsonl')
p.add_argument('--workers', type=int, default=4)
args = p.parse_args()
out = args.input.parent / 'judgments' / f'{args.judge}.jsonl'
out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)['pair_id'] for l in open(out)} if out.exists() else set()
packets = [json.loads(l) for l in open(args.input) if json.loads(l)['pair_id'] not in done]
client = boto3.client('bedrock-runtime', region_name='us-east-2', config=Config(read_timeout=900, retries={'max_attempts': 2}))
with ThreadPoolExecutor(args.workers) as ex, open(out, 'a') as fh:
    for row in ex.map(lambda pk: call(args.judge, pk, client), packets):
        fh.write(json.dumps(row, ensure_ascii=False) + '\n'); fh.flush()
        print(args.judge, row['pair_id'], row['label'], row['a_high_quality'], row['b_high_quality'], row['usage'], flush=True)
print(f'{args.judge}: {len(packets)} new -> {out}')
