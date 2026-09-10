#!/usr/bin/env python3
"""Fresh MBPP training pools; no reference bodies or test assertions in prompts."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path('runs/swe-diversity-selection/code-learning-pilot')
CACHE = Path('/scratch/yirenl2/.cache/huggingface/hub')
DATA_REV = '4bb6404fdc6cacfda99d4ac4205087b89d32030c'
MODEL_REV = 'b968826d9c46dd6066d109eabc6255188de91218'
MODEL = CACHE / 'models--Qwen--Qwen3-8B/snapshots' / MODEL_REV
DATA = CACHE / 'datasets--google-research-datasets--mbpp/snapshots' / DATA_REV / 'sanitized'
SEED = 20260909


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def interface(row):
    tree = ast.parse(row['code'])
    calls = {n.func.id for t in row['test_list'] for n in ast.walk(ast.parse(t))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    targets = [n for n in defs if n.name in calls]
    if not targets:
        raise ValueError(f'No unambiguous test-referenced function: {row["task_id"]}')
    signatures = [f'def {n.name}({ast.unparse(n.args)}): ...' for n in targets]
    return signatures


def prepare():
    import pyarrow.parquet as pq
    if (ROOT / 'protocol.json').exists():
        raise FileExistsError('Protocol already frozen')
    ROOT.mkdir(parents=True, exist_ok=True)
    parts = {s: pq.read_table(DATA / f'{s}-00000-of-00001.parquet').to_pylist()
             for s in ['train', 'validation', 'test']}
    train = sorted(parts['train'], key=lambda r: digest(f'{SEED}:train:{r["task_id"]}'))[:32]
    val = [r for r in parts['validation'] if r['task_id'] != 569]
    val = sorted(val, key=lambda r: digest(f'{SEED}:validation:{r["task_id"]}'))
    groups = {'train': train, 'calibration': val[:21], 'measurement': val[21:]}
    for a, rows in groups.items():
        for b, others in groups.items():
            if a != b:
                assert not ({r['prompt'].strip() for r in rows} & {r['prompt'].strip() for r in others})
    assert not ({r['prompt'].strip() for r in train} & {r['prompt'].strip() for r in parts['test']})
    inputs, targets = [], []
    for split, rows in groups.items():
        for row in rows:
            signatures = interface(row)
            prompt = (row['prompt'] + '\n\nImplement the following Python interface:\n'
                      + '\n'.join(signatures)
                      + '\n\nReturn the complete solution in a single ```python code block.')
            inputs.append({'split': split, 'task_id': row['task_id'], 'prompt': prompt,
                           'interfaces': signatures, 'interface_source': 'Reference function signatures selected by function-name calls in test AST; no function bodies or assertions included.'})
            targets.append({'split': split, **row})
    for name, rows in [('inputs.jsonl', inputs), ('evaluator_targets.jsonl', targets)]:
        (ROOT / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    audit = json.loads(Path('runs/swe-diversity-selection/learning-infrastructure/mbpp_partition_audit.json').read_text())
    protocol = {'seed': SEED, 'dataset': 'google-research-datasets/mbpp', 'config': 'sanitized',
                'dataset_revision': DATA_REV, 'model': 'Qwen/Qwen3-8B', 'model_revision': MODEL_REV,
                'train_count': 32, 'k': 8, 'max_new_tokens': 768, 'temperature': 1.0,
                'top_p': 1.0, 'enable_thinking': False, 'logprobs': 1,
                'generation_seed_formula': 'SEED + task_id * 100 + sample',
                'train_task_ids': [r['task_id'] for r in train],
                'calibration_task_ids': [r['task_id'] for r in val[:21]],
                'measurement_task_ids': [r['task_id'] for r in val[21:]],
                'excluded_validation_task_ids': [569],
                'later_test_task_ids': audit['test_task_ids_unexposed_in_listed_traces'],
                'api_spend_usd': 0, 'scope': 'Code mechanism study; not SWE repair evidence',
                'grader': 'Python subprocess, temporary cwd, credential-free environment, RLIMIT_CPU/AS/FSIZE/NOFILE; no network or filesystem security isolation',
                'frozen_inputs_sha256': digest((ROOT / 'inputs.jsonl').read_text()),
                'targets_sha256': digest((ROOT / 'evaluator_targets.jsonl').read_text()),
                'script_sha256': digest(Path(__file__).read_text()),
                'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    save(ROOT / 'protocol.json', protocol)
    print(json.dumps({'train': len(train), 'calibration': 21, 'measurement': 21,
                      'interfaces': [r['interfaces'] for r in inputs if r['split'] == 'train']}), flush=True)


def grade(text, target):
    import re
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    code = blocks[-1] if blocks else text
    try:
        ast.parse(code)
    except SyntaxError:
        return {'valid_python': False, 'correct': False, 'status': 'syntax_error'}
    program = '\n'.join(target['test_imports']) + '\n' + code + '\n' + '\n'.join(target['test_list'])
    wrapper = ('import resource,runpy; resource.setrlimit(resource.RLIMIT_CPU,(5,6)); '
               'resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824)); '
               'resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576)); '
               'resource.setrlimit(resource.RLIMIT_NOFILE,(64,64)); runpy.run_path("solution.py",run_name="__main__")')
    with tempfile.TemporaryDirectory(prefix='mbpp-grade-') as tmp:
        Path(tmp, 'solution.py').write_text(program)
        try:
            result = subprocess.run([sys.executable, '-I', '-c', wrapper], cwd=tmp,
                                    env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
                                         'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                                    capture_output=True, text=True, timeout=10)
            return {'valid_python': True, 'correct': result.returncode == 0,
                    'status': 'pass' if result.returncode == 0 else 'test_error',
                    'returncode': result.returncode, 'stderr_tail': result.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {'valid_python': True, 'correct': False, 'status': 'timeout'}


def generate(smoke):
    from importlib.metadata import version
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert digest((ROOT / 'inputs.jsonl').read_text()) == protocol['frozen_inputs_sha256']
    assert digest((ROOT / 'evaluator_targets.jsonl').read_text()) == protocol['targets_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3', 'This run is assigned physical GPU3'
    rows = [json.loads(l) for l in (ROOT / 'inputs.jsonl').open()]
    rows = [r for r in rows if r['split'] == 'train'][:2 if smoke else 32]
    targets = {r['task_id']: r for l in (ROOT / 'evaluator_targets.jsonl').open() if (r := json.loads(l))['split'] == 'train'}
    output = ROOT / ('smoke_raw.jsonl' if smoke else 'raw.jsonl')
    grades = ROOT / ('smoke_grades.jsonl' if smoke else 'grades.jsonl')
    if not smoke:
        gate = json.loads((ROOT / 'smoke_summary.json').read_text())
        assert gate['rows'] == 8 and gate['valid_python_rate'] >= .5 and gate['all_generated_tokens_have_logprobs']
    done = {(r['task_id'], r['sample']) for l in output.open() if (r := json.loads(l))} if output.exists() else set()
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    jobs = []
    for row in rows:
        messages = [{'role': 'user', 'content': row['prompt']}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for sample in range(4 if smoke else 8):
            if (row['task_id'], sample) not in done:
                jobs.append({'task_id': row['task_id'], 'sample': sample,
                             'user_prompt': row['prompt'], 'rendered_prompt': prompt})
    manifest = {**protocol, 'smoke': smoke, 'physical_gpu': 3, 'dtype': 'bfloat16',
                'python': sys.version, 'versions': {p: version(p) for p in ['torch', 'transformers', 'vllm']},
                'chat_template_sha256': digest(tok.chat_template), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'generation_script_sha256': digest(Path(__file__).read_text()), 'pid': os.getpid()}
    save(ROOT / ('smoke_manifest.json' if smoke else 'generation_manifest.json'), manifest)
    if not jobs:
        return summarize(smoke)
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), dtype='bfloat16',
              gpu_memory_utilization=.8, max_model_len=2048, max_num_seqs=8,
              enforce_eager=True, disable_log_stats=True)
    start = time.time()
    with output.open('a') as f, grades.open('a') as g:
        for offset in range(0, len(jobs), 8):
            batch = jobs[offset:offset + 8]
            params = [SamplingParams(temperature=protocol['temperature'], top_p=protocol['top_p'], max_tokens=768,
                                     seed=SEED + j['task_id'] * 100 + j['sample'], logprobs=1) for j in batch]
            replies = llm.generate([j['rendered_prompt'] for j in batch], params, use_tqdm=False)
            for job, reply in zip(batch, replies):
                seq = reply.outputs[0]
                assert seq.logprobs is not None
                lp = [float(p[t].logprob) for t, p in zip(seq.token_ids, seq.logprobs)]
                assert len(lp) == len(seq.token_ids)
                record = {**job, 'prompt_token_ids': reply.prompt_token_ids, 'generation': seq.text,
                          'generated_token_ids': list(seq.token_ids), 'chosen_token_logprobs': lp,
                          'finish_reason': seq.finish_reason, 'stop_reason': seq.stop_reason}
                f.write(json.dumps(record) + '\n'); f.flush()
                result = grade(seq.text, targets[job['task_id']])
                g.write(json.dumps({'task_id': job['task_id'], 'sample': job['sample'], **result}) + '\n'); g.flush()
            print(json.dumps({'completed': len(done) + offset + len(batch), 'total': len(done) + len(jobs), 'seconds': round(time.time() - start, 1)}), flush=True)
    summarize(smoke)


def summarize(smoke):
    raw = [json.loads(l) for l in (ROOT / ('smoke_raw.jsonl' if smoke else 'raw.jsonl')).open()]
    grades = [json.loads(l) for l in (ROOT / ('smoke_grades.jsonl' if smoke else 'grades.jsonl')).open()]
    assert len({(r['task_id'], r['sample']) for r in raw}) == len(raw)
    assert {(r['task_id'], r['sample']) for r in raw} == {(r['task_id'], r['sample']) for r in grades}
    result = {'rows': len(raw), 'tasks': len({r['task_id'] for r in raw}),
              'valid_python_rate': sum(r['valid_python'] for r in grades) / len(grades),
              'test_accuracy': sum(r['correct'] for r in grades) / len(grades),
              'truncation_rate': sum(r['finish_reason'] == 'length' for r in raw) / len(raw),
              'generated_tokens': sum(len(r['generated_token_ids']) for r in raw),
              'all_generated_tokens_have_logprobs': all(len(r['generated_token_ids']) == len(r['chosen_token_logprobs']) for r in raw),
              'api_spend_usd': 0}
    save(ROOT / ('smoke_summary.json' if smoke else 'summary.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'generate', 'summarize'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'generate':
        generate(args.smoke)
    else:
        summarize(args.smoke)
