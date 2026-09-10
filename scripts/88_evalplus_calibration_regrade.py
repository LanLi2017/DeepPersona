#!/usr/bin/env python3
"""Bounded preflight, freeze, and CPU regrade of 20 existing calibration outputs."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/swe-diversity-selection/code-evalplus-calibration'
FEAS = OUT.parent / 'code-evalplus-feasibility'
REVISION = '26d6d00bb1fd0fa37f39c99d5290da67891d1c5e'
SOURCE = OUT / 'source' / ('evalplus-' + REVISION)
BASE = OUT.parent / 'code-functional-development/grid-evaluation/base/eval_raw.jsonl'
TRAINED = OUT.parent / 'code-corrected-sft-control/full/calibration/trained_raw.jsonl'
PATTERN = r'```(?:python)?\s*\n(.*?)```'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    os.environ['MBPP_OVERRIDE_PATH'] = str(OUT / 'calibration20.raw.jsonl')
    os.environ['XDG_CACHE_HOME'] = str(OUT / 'cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ.pop('EVALPLUS_TIMEOUT_PER_TASK', None)
    sys.path.insert(0, str(SOURCE))
    from evalplus.data.mbpp import get_mbpp_plus
    problems = get_mbpp_plus()
    expected_ids = read(FEAS / 'membership.json')['splits']['calibration']['supported_ids']
    assert sorted(int(t.split('/')[1]) for t in problems) == expected_ids
    return problems


def extract(record):
    required = set(re.findall(r'def\s+(\w+)\s*\(', record['user_prompt']))
    blocks = re.findall(PATTERN, record['generation'], re.S)
    candidates = blocks or [record['generation']]
    selected, defs = None, {}
    for index, code in enumerate(candidates):
        try:
            defs = {n.name: n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        except SyntaxError:
            defs = {}
        if required <= defs.keys():
            selected = index
            break
    code = candidates[selected if selected is not None else 0]
    return code, {'blocks': len(blocks), 'selected_index': selected, 'required': sorted(required),
                  'missing_interface': selected is None, 'finish_reason': record['finish_reason']}


def checker(problem, code, reference):
    from evalplus.eval import untrusted_check
    result = {}
    for split in ['base', 'plus']:
        status, details = untrusted_check('mbpp', code, copy.deepcopy(problem[split + '_input']),
            problem['entry_point'], reference[split]['expected'], problem['atol'],
            reference[split]['time'], fast_check=False, min_time_limit=4.0, gt_time_limit_factor=4.0)
        result[split] = {'status': status, 'details': [bool(v) for v in details],
                         'passed_inputs': sum(bool(v) for v in details), 'inputs': len(problem[split + '_input'])}
    result['combined_pass'] = all(result[s]['status'] == 'pass' for s in ['base', 'plus'])
    return result


def oracle_worker(task):
    problems = setup()
    from evalplus.gen.util import trusted_exec
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    p = problems[f'Mbpp/{task}']
    function = next(n for n in ast.parse(p['canonical_solution']).body
                    if isinstance(n, ast.FunctionDef) and n.name == p['entry_point'])
    source = 'def contract_check(' + ast.unparse(function.args) + '):\n' + textwrap.indent(textwrap.dedent(p['contract']).strip(), '    ') + '\n'
    namespace = {}
    exec(source, namespace)
    reference, contracts, domains = {}, {}, {}
    for split in ['base', 'plus']:
        invalid, outside = [], []
        for i, inp in enumerate(p[split + '_input']):
            try:
                namespace['contract_check'](*copy.deepcopy(inp))
            except Exception as e:
                invalid.append({'index': i, 'exception': type(e).__name__})
            # Explicit natural-language input promises absent from the weak contracts.
            if task == 597 and any(list(a) != sorted(a) for a in inp[:2]):
                outside.append({'index': i, 'reason': 'input arrays not sorted'})
            if task == 593 and not (len(inp[0].split('.')) == 4 and all(s.isdecimal() and 0 <= int(s) <= 255 for s in inp[0].split('.'))):
                outside.append({'index': i, 'reason': 'not an IPv4 address (four decimal octets, each0..255)'})
        contracts[split], domains[split] = invalid, outside
        try:
            expected, durations = trusted_exec(p['prompt'] + p['canonical_solution'],
                p[split + '_input'], p['entry_point'], record_time=True,
                output_not_none=p['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS)
            reference[split] = {'expected': expected, 'time': durations}
        except Exception as e:
            save(OUT / 'oracle' / f'{task}.json', {'task_id': task, 'oracle_valid': False,
                'exception': type(e).__name__, 'message': str(e), 'split': split,
                'contract_invalid': contracts, 'outside_prompt_domain': domains})
            return
    with (OUT / 'oracle' / f'{task}.pkl').open('wb') as handle:
        pickle.dump(reference, handle)
    control = checker(p, p['canonical_solution'], reference)
    save(OUT / 'oracle' / f'{task}.json', {'task_id': task, 'oracle_valid': True,
        'contract_invalid': contracts, 'outside_prompt_domain': domains, 'canonical_control': control})


def prepare():
    assert not (OUT / 'protocol.json').exists()
    (OUT / 'oracle').mkdir(exist_ok=True)
    problems = setup()
    original = {r['task_id']: r for r in rows(BASE)}
    trained = {r['task_id']: r for r in rows(TRAINED)}
    programs, metadata = [], []
    for key, problem in problems.items():
        task = int(key.split('/')[1])
        a, b = original[task], trained[task]
        assert a['user_prompt'] == b['user_prompt'] and a['prompt_token_ids'] == b['prompt_token_ids']
        public = {n.name: ast.unparse(n.args) for n in ast.parse('\n'.join(re.findall(r'def[^\n]+', a['user_prompt']))).body if isinstance(n, ast.FunctionDef)}
        canonical = {n.name: ast.unparse(n.args) for n in ast.parse(problem['canonical_solution']).body if isinstance(n, ast.FunctionDef)}
        description = problem['prompt'].strip().strip('"').strip().split('\nassert')[0]
        same_description = a['user_prompt'].split('\n\nImplement')[0] == description
        meta = {'task_id': task, 'entry_point': problem['entry_point'], 'description_equal': same_description,
            'public_interface': public.get(problem['entry_point']), 'canonical_interface': canonical.get(problem['entry_point']),
            'contract': problem['contract'], 'base_inputs': len(problem['base_input']), 'plus_inputs': len(problem['plus_input'])}
        meta['interface_equal'] = meta['public_interface'] == meta['canonical_interface']
        metadata.append(meta)
        for label, r in [('base', a), ('trained', b)]:
            code, extraction = extract(r)
            programs.append({'task_id': task, 'model': label, 'solution': code, 'extraction': extraction,
                'prompt_sha256': hashlib.sha256(r['user_prompt'].encode()).hexdigest(),
                'raw_generation_sha256': hashlib.sha256(r['generation'].encode()).hexdigest()})
    save(OUT / 'programs.json', programs)
    save(OUT / 'interfaces_contracts.json', metadata)
    for task in sorted(original.keys() & {int(k.split('/')[1]) for k in problems}):
        if (OUT / 'oracle' / f'{task}.json').exists():
            continue
        with (OUT / 'oracle' / f'{task}.log').open('w') as handle:
            try:
                run = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'oracle-worker', str(task)],
                    stdout=handle, stderr=subprocess.STDOUT, timeout=150, check=False)
                if run.returncode:
                    save(OUT / 'oracle' / f'{task}.json', {'task_id': task, 'oracle_valid': False, 'worker_exit': run.returncode})
            except subprocess.TimeoutExpired:
                save(OUT / 'oracle' / f'{task}.json', {'task_id': task, 'oracle_valid': False, 'worker_timeout': True})
        print('preflight', task, flush=True)
    print('Preflight complete; inspect only interfaces/contracts/oracle controls before freeze.', flush=True)


def freeze():
    assert not (OUT / 'protocol.json').exists()
    dispositions = []
    for meta in read(OUT / 'interfaces_contracts.json'):
        task = meta['task_id']
        control = read(OUT / 'oracle' / f'{task}.json')
        reasons = []
        if task == 559:
            reasons.append('Pinned v0.2.0 oracle is superseded by official June2024 v0.2.1 correction; oracle-version-ambiguous diagnostic only')
        if not meta['interface_equal'] or not meta['description_equal']:
            reasons.append('Interface or description mismatch')
        if not control['oracle_valid']:
            reasons.append('Canonical oracle execution invalid')
        elif not control['canonical_control']['combined_pass']:
            reasons.append('Canonical control fails official checker')
        if any(control.get('contract_invalid', {}).values()):
            reasons.append('Dataset includes contract-invalid inputs')
        if any(control.get('outside_prompt_domain', {}).values()):
            reasons.append('Dataset includes inputs outside explicit prompt domain')
        dispositions.append({'task_id': task, 'primary_compatible': not reasons, 'reasons': reasons})
    save(OUT / 'dispositions.json', dispositions)
    files = [Path(__file__).resolve(), BASE, TRAINED, OUT / 'calibration20.raw.jsonl',
        OUT / 'programs.json', OUT / 'interfaces_contracts.json', OUT / 'dispositions.json',
        OUT / 'runner.tar.gz', OUT / 'oracle_correction_commit.json', OUT / 'oracle559_corrected.py']
    files += sorted((OUT / 'oracle').glob('*.json')) + sorted((OUT / 'oracle').glob('*.pkl'))
    versions = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (OUT / 'runtime_requirements.txt').write_text(versions)
    files.append(OUT / 'runtime_requirements.txt')
    protocol = {'scope': '20 calibration tasks,40 existing programs; no generation, measurement or final decoding/evaluation',
        'runner_revision': REVISION, 'primary_task_ids': [d['task_id'] for d in dispositions if d['primary_compatible']],
        'all_task_ids': [d['task_id'] for d in dispositions],
        'extraction': 'First Python block containing all requested public function definitions; missing interface retained as failure',
        'endpoint': 'Combined official base AND plus test success; primary compatibility stratum only',
        'gate': {'trained_combined_failures_at_least': 3, 'net_combined_gain_over_base_at_least': 1,
                 'interpretation': 'Engineering headroom/learner feasibility, not statistical power or diversity evidence'},
        'canary_task_ids': [556, 600], 'canary_code': 'def ENTRY(*args, **kwargs):\n    return None\n',
        'oracle_controls': 'Canonical20 preflight preserved; no outcome-based exclusion',
        'incompatible_strata': 'Retain official statuses as descriptive diagnostics only; not model-error counts or primary gate',
        'time_limits': {'official_min_input_seconds': 4.0, 'official_gt_factor': 4.0, 'official_task_cap_seconds': 60},
        'sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'runner_python_sha256': {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob('*.py')}}
    save(OUT / 'protocol.json', protocol)
    print(json.dumps({'frozen_primary': protocol['primary_task_ids'], 'dispositions': dispositions}))


def run():
    assert not (OUT / 'results.json').exists()
    protocol = read(OUT / 'protocol.json')
    for name, digest in protocol['sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in protocol['runner_python_sha256'].items():
        assert sha(SOURCE / name) == digest, name
    problems = setup()
    start = time.monotonic()
    controls = []
    for task in protocol['canary_task_ids']:
        p = problems[f'Mbpp/{task}']
        with (OUT / 'oracle' / f'{task}.pkl').open('rb') as handle:
            reference = pickle.load(handle)
        result = checker(p, protocol['canary_code'].replace('ENTRY', p['entry_point']), reference)
        controls.append({'task_id': task, **result})
        assert not result['combined_pass'] and result['base']['status'] == 'fail'
    save(OUT / 'negative_canaries.json', controls)
    results = []
    for program in read(OUT / 'programs.json'):
        task = program['task_id']
        oracle = read(OUT / 'oracle' / f'{task}.json')
        result = {'task_id': task, 'model': program['model'],
                  'primary_compatible': task in protocol['primary_task_ids'], 'extraction': program['extraction']}
        if oracle['oracle_valid']:
            with (OUT / 'oracle' / f'{task}.pkl').open('rb') as handle:
                reference = pickle.load(handle)
            result.update(checker(problems[f'Mbpp/{task}'], program['solution'], reference))
        else:
            result['status'] = 'not_graded_invalid_oracle'
        results.append(result)
        print('graded', task, program['model'], flush=True)
    save(OUT / 'results.json', results)
    pairs = {task: {r['model']: r for r in results if r['task_id'] == task} for task in protocol['primary_task_ids']}
    totals = {label: sum(pair[label]['combined_pass'] for pair in pairs.values()) for label in ['base', 'trained']}
    gains = [task for task, pair in pairs.items() if pair['trained']['combined_pass'] and not pair['base']['combined_pass']]
    losses = [task for task, pair in pairs.items() if pair['base']['combined_pass'] and not pair['trained']['combined_pass']]
    failures = len(pairs) - totals['trained']
    summary = {'primary_tasks': len(pairs), 'combined_pass': totals, 'gained': gains, 'lost': losses,
        'trained_failures': failures, 'net_gain': totals['trained'] - totals['base'],
        'engineering_gate_passed': failures >= 3 and totals['trained'] - totals['base'] >= 1,
        'seconds': time.monotonic() - start, 'protocol_sha256': sha(OUT / 'protocol.json'),
        'all20_dispositions_retained': True, 'gpu_used': False, 'api_spend_usd': 0,
        'measurement_or_final_fields_decoded': False}
    save(OUT / 'summary.json', summary)
    files = [OUT / name for name in ['protocol.json', 'results.json', 'summary.json', 'negative_canaries.json']]
    save(OUT / 'completed_manifest.json', {'sha256': {str(p.relative_to(OUT)): sha(p) for p in files}})
    print(json.dumps(summary))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'oracle-worker':
        oracle_worker(int(sys.argv[2]))
    else:
        {'prepare': prepare, 'freeze': freeze, 'run': run}[mode]()
