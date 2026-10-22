#!/usr/bin/env python3
"""G1 targeted E2 collection on AppWorld train/dev.

stage plans:    GPT-5.6 proposes up to 3 decision-distinct plans per task (--axes v1: route / information
                source / recovery; v2: route / subgoal order / write parameters / recovery).
stage rollouts: a ReAct code agent runs each (task, plan|none, model); saves messages, api_calls, evaluation.

  APPWORLD_ROOT=envs/appworld_root envs/appworld/bin/python scripts/appworld_targeted_rollout.py --stage plans --smoke
  APPWORLD_ROOT=envs/appworld_root envs/appworld/bin/python scripts/appworld_targeted_rollout.py --stage rollouts --smoke
"""
import argparse, json, os, random, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import boto3, requests
from botocore.config import Config
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
AW = ROOT / 'envs/appworld_root/data'
load_dotenv(ROOT.parent / '.env')
MODELS = {'gpt56': ('openai', 'gpt-5.6-sol'), 'haiku': ('bedrock', 'us.anthropic.claude-haiku-4-5-20251001-v1:0'),
          'kimi': ('bedrock', 'moonshot.kimi-k2-thinking')}
ROUTE_TASKS = ['34d9492_1', '34d9492_2', '34d9492_3', '68ee2c9_1', '68ee2c9_2', '68ee2c9_3',
               'cf6abd2_1', 'cf6abd2_2', 'cf6abd2_3', '396c5a2_1', '396c5a2_2', '396c5a2_3']  # from the ground-truth route scan
bedrock = boto3.client('bedrock-runtime', region_name='us-east-2', config=Config(read_timeout=900, retries={'max_attempts': 3}))


def chat(model, messages, max_tokens):
    kind, mid = MODELS[model]
    for attempt in range(4):
        try:
            if kind == 'openai':
                key = os.environ.get('OPENAI_API_KEY') or os.environ['openai_api_key']
                r = requests.post('https://api.openai.com/v1/chat/completions', timeout=900, headers={'Authorization': f'Bearer {key}'},
                                  json={'model': mid, 'messages': messages, 'max_completion_tokens': max_tokens})
                r.raise_for_status(); j = r.json()
                return j['choices'][0]['message']['content'] or '', {'in': j['usage']['prompt_tokens'], 'out': j['usage']['completion_tokens']}
            # AppWorld freezes the process clock to the task date, which breaks AWS request signing,
            # so Bedrock calls run in a fresh subprocess with a real clock.
            cp = subprocess.run([sys.executable, __file__, '--_bedrock'], input=json.dumps([mid, messages, max_tokens]),
                                capture_output=True, text=True, check=True)
            text, usage = json.loads(cp.stdout)
            return text, usage
        except Exception as e:
            err = e; time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'{model} failed: {err}')


def bedrock_call():
    mid, messages, max_tokens = json.loads(sys.stdin.read())
    sys_ = [{'text': m['content']} for m in messages if m['role'] == 'system']
    conv = [{'role': m['role'], 'content': [{'text': m['content'] or '(empty)'}]} for m in messages if m['role'] != 'system']
    r = bedrock.converse(modelId=mid, system=sys_, messages=conv, inferenceConfig={'maxTokens': max_tokens})
    text = ''.join(c.get('text', '') for c in r['output']['message']['content'])
    print(json.dumps([text, {'in': r['usage']['inputTokens'], 'out': r['usage']['outputTokens']}]))


def task_spec(tid):
    spec = json.load(open(AW / 'tasks' / tid / 'specs.json'))
    apps = json.load(open(AW / 'tasks' / tid / 'ground_truth' / 'required_apps.json'))
    return spec, apps


def api_index(apps):
    out = []
    for app in apps:
        for a in json.load(open(AW / 'api_docs/standard' / f'{app}.json')).values():
            out.append(f"{app}.{a['api_name']} [{a['method']} {a['path']}]: {a['description'][:120]}")
    return '\n'.join(out)


PLAN_SYS_V1 = """You design alternative solution plans for an AppWorld task (a simulated phone with apps,
used through Python API calls). Propose 3 plans. Each plan must fully and correctly complete the
task. Each plan must also be one that a careful agent could follow without policy violations or
unnecessary side effects.

The plans must differ from each other in a DECISION-RELEVANT way, using one or more of these axes:
- route: different state-changing APIs that reach the same end state (e.g. a one-step move vs a
  delete+add sequence; an append vs a full rewrite);
- information_source: a key decision is grounded in information obtained from a different app or
  endpoint (e.g. a contact's phone number from Phone contacts vs from an email thread);
- recovery: a different but valid way to handle a likely failure or ambiguity.

These do NOT count as different: extra read-only lookups, reordering independent steps, a
different login order, or cosmetic differences. If the task truly admits only one strategy,
say so and return fewer plans.

Return JSON only: {"plans": [{"id": "P1", "axis": "route|information_source|recovery",
"summary": "one sentence", "steps": ["high-level step", ...]}], "notes": "..."}"""

# v2 (batch 2): information_source plans never changed the writes in batch 1, so they are replaced.
PLAN_SYS_V2 = """You design alternative solution plans for an AppWorld task (a simulated phone with apps,
used through Python API calls). Propose 3 plans. Each plan must fully and correctly complete the
task. Each plan must also be one that a careful agent could follow without policy violations or
unnecessary side effects.

The plans must differ from each other in a DECISION-RELEVANT way, using one or more of these axes:
- route: different state-changing APIs that reach the same end state (e.g. a one-step move vs a
  delete+add sequence; an append vs a full rewrite);
- subgoal_order: a different order of DEPENDENT subgoals, where the result of an earlier
  subgoal changes what a later step does (not merely reordering independent steps);
- write_parameters: the same state-changing APIs but different decision-relevant arguments that the
  task leaves open (e.g. keep vs reset an optional flag), while both still satisfy the task;
- recovery: a different but valid way to handle a likely failure or ambiguity.

These do NOT count as different: extra read-only lookups, reordering independent steps, reading the
same information from another source when the decision comes out the same, a different login order, or cosmetic differences. If the task truly admits only one strategy,
say so and return fewer plans.

Return JSON only: {"plans": [{"id": "P1", "axis": "route|subgoal_order|write_parameters|recovery",
"summary": "one sentence", "steps": ["high-level step", ...]}], "notes": "..."}"""

AGENT_SYS = """You are an AI assistant who acts on behalf of the supervisor on their phone, using Python
code that calls the app APIs. Each turn, write exactly ONE ```python code block. You will see ONLY
what it prints, so always print() the results you need; bare expressions show nothing. Variables
persist between turns.

- `apis.api_docs.show_app_descriptions()`, `apis.api_docs.show_api_descriptions(app_name=...)` and
  `apis.api_docs.show_api_doc(app_name=..., api_name=...)` list the available APIs. Read the doc
  before calling an API you have not used.
- Account passwords: `apis.supervisor.show_account_passwords()`. Log in with `apis.<app>.login(...)`
  and pass the access_token on to later calls.
- Paginated APIs need a loop over page_index until the result is empty.
- Do exactly what is asked, with no extra side effects. Do not invent information.
- When done, call `apis.supervisor.complete_task(answer=<answer or None>)`. For tasks without a
  question, use answer=None.

Format: reply with a short thought and then ONE fenced block, e.g.
```python
print(apis.api_docs.show_api_descriptions(app_name="spotify"))
```
Never use <function_calls>, XML or any other tool-call syntax, and never imagine outputs. Only the
printed output you receive back is real. Do not call complete_task until real outputs confirm the
work is done."""


def first_block(text):
    # Models sometimes script several turns at once; run only the first block and drop the
    # imagined continuation from history.
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.S)
    return (m.group(1), text[:m.end()]) if m else (None, text)


def rollout(tid, plan, model, outdir, max_turns, seed):
    from appworld import AppWorld
    spec, _ = task_spec(tid)
    sup = spec['supervisor']
    user = (f"Supervisor: {sup['first_name']} {sup['last_name']} ({sup['email']}, {sup['phone_number']}).\n"
            f"Current date-time: {spec['datetime']}.\nTask: {spec['instruction']}")
    if plan:
        user += '\n\nFollow this high-level plan (fill in the details yourself):\n' + plan['summary'] + '\n- ' + '\n- '.join(plan['steps'])
    msgs = [{'role': 'system', 'content': AGENT_SYS}, {'role': 'user', 'content': user}]
    usage = {'in': 0, 'out': 0}
    exp = f"g1_targeted/{outdir.parent.parent.name}/{outdir.parent.name}/{outdir.name}"
    with AppWorld(task_id=tid, experiment_name=exp, random_seed=seed) as w:
        for turn in range(max_turns):
            text, u = chat(model, msgs, 8000)
            usage = {k: usage[k] + u[k] for k in usage}
            code, text = first_block(text)
            msgs.append({'role': 'assistant', 'content': text})
            obs = w.execute(code) if code else 'No ```python code block found. Write exactly one code block.'
            msgs.append({'role': 'user', 'content': 'Output:\n' + str(obs)[:4000]})
            if w.task_completed():
                break
        completed = w.task_completed()
        ev = w.evaluate().to_dict()
    outdir.mkdir(parents=True, exist_ok=True)
    rec = {'task_id': tid, 'model': model, 'plan': plan, 'turns': turn + 1, 'completed': completed,
           'success': ev['success'], 'evaluation': ev, 'usage': usage, 'messages': msgs, 'appworld_experiment': exp, 'seed': seed}
    json.dump(rec, open(outdir / 'rollout.json', 'w'), ensure_ascii=False, default=str)
    return rec


def run_job(j):
    return rollout(j[0], j[1], j[2], j[3], j[4], j[5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['plans', 'rollouts'], required=True)
    p.add_argument('--run', default='pilot')
    p.add_argument('--n-other', type=int, default=12)
    p.add_argument('--models', nargs='+', default=['gpt56', 'haiku'])
    p.add_argument('--max-turns', type=int, default=30)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--only', choices=['all', 'route', 'other'], default='all')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--axes', choices=['v1', 'v2'], default='v1')
    args = p.parse_args()
    run = ROOT / 'outputs/appworld_targeted' / (args.run + ('-smoke' if args.smoke else ''))
    run.mkdir(parents=True, exist_ok=True)

    if args.stage == 'plans':
        rng = random.Random(args.seed)
        pool = [t for sp in ('train', 'dev') for t in (AW / f'datasets/{sp}.txt').read_text().split()]
        scen = {}
        for t in pool:
            if t.split('_')[0] not in {r.split('_')[0] for r in ROUTE_TASKS}:
                scen.setdefault(t.split('_')[0], []).append(t)
        other = [rng.choice(v) for k, v in sorted(rng.sample(sorted(scen.items()), args.n_other))]
        tasks = {'all': ROUTE_TASKS + other, 'route': ROUTE_TASKS, 'other': other}[args.only]
        tasks = tasks[:1] if args.smoke else tasks
        plans = {}
        for tid in tasks:
            spec, apps = task_spec(tid)
            text, u = chat('gpt56', [{'role': 'system', 'content': PLAN_SYS_V2 if args.axes == 'v2' else PLAN_SYS_V1},
                                     {'role': 'user', 'content': f"Current date-time: {spec['datetime']}\nTask: {spec['instruction']}\nApps involved: {apps}\n\nAPIs:\n{api_index(apps)}"}], 12000)
            plans[tid] = {**json.loads(text[text.find('{'): text.rfind('}') + 1]), 'usage': u, 'route_task': tid in ROUTE_TASKS}
            print(tid, len(plans[tid]['plans']), [x['axis'] for x in plans[tid]['plans']], u, flush=True)
        json.dump(plans, open(run / 'plans.json', 'w'), indent=1, ensure_ascii=False)
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        json.dump({'git': sha, 'args': vars(args), 'plan_model': MODELS['gpt56'], 'time': datetime.now(timezone.utc).isoformat()},
                  open(run / 'manifest_plans.json', 'w'), indent=1)
    else:
        plans = json.load(open(run / 'plans.json'))
        jobs = []
        for tid, pl in plans.items():
            if args.only != 'all' and pl['route_task'] != (args.only == 'route'):
                continue
            for plan in [None] + pl['plans']:
                for m in args.models:
                    jobs.append((tid, plan, m, run / tid / f"{m}_{plan['id'] if plan else 'noplan'}"))
        if args.smoke:
            jobs = [j for j in jobs if j[2] == args.models[0] and j[1] is not None][:1]
        jobs = [j for j in jobs if not (j[3] / 'rollout.json').exists()]
        print(f'{len(jobs)} rollouts', flush=True)
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        json.dump({'git': sha, 'args': vars(args), 'models': MODELS, 'agent_max_tokens': 8000, 'time': datetime.now(timezone.utc).isoformat()},
                  open(run / 'manifest_rollouts.json', 'w'), indent=1)
        # AppWorld uses SIGALRM, so each rollout must run in a process's main thread.
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(run_job, [(*j, args.max_turns, args.seed) for j in jobs]):
                print(r['task_id'], r['model'], r['plan']['id'] if r['plan'] else 'noplan', 'success', r['success'], 'turns', r['turns'], r['usage'], flush=True)



if __name__ == '__main__':
    bedrock_call() if sys.argv[1:] == ['--_bedrock'] else main()