#!/usr/bin/env python3
"""CPU-only completed SFT accounting, boundary, FIRST-extraction and grade audit."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path('runs/swe-diversity-selection')
OUT = ROOT / 'code-corrected-sft-control'
FULL = OUT / 'full'
PATTERN = r'```(?:python)?\s*\n(.*?)```'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interface(record):
    required = set(re.findall(r'def\s+(\w+)\s*\(', record['user_prompt']))
    blocks = re.findall(PATTERN, record['generation'], re.S)
    code = blocks[0] if blocks else record['generation']
    try:
        definitions = {n.name for n in ast.parse(code).body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    except SyntaxError:
        definitions = set()
    return code, {'task_id': record['task_id'], 'blocks': len(blocks),
                  'required': sorted(required), 'missing': sorted(required - definitions),
                  'finish_reason': record['finish_reason'],
                  'generated_tokens': len(record['generated_token_ids'])}


def run():
    protocol, summary = read(OUT / 'full_protocol.json'), read(FULL / 'summary.json')
    assert summary['technical_checks_passed'] and not summary['measurement_or_final_scored']
    completed = read(FULL / 'completed_manifest.json')
    for name, h in completed['sha256'].items():
        assert sha(FULL / name) == h, name
    for name, h in protocol['input_sha256'].items():
        assert sha(Path(name)) == h, name
    assert read(FULL / 'manifest.json')['protocol_sha256'] == sha(OUT / 'full_protocol.json')
    training = rows(OUT / 'training_inputs.jsonl')
    items, steps = rows(FULL / 'training_items.jsonl'), rows(FULL / 'training_steps.jsonl')
    assert len(training) == 94 and len(items) == 282 and len(steps) == 72
    assert [r['task_id'] for r in training] == protocol['training_tasks']
    assert [r['task_id'] for r in items] == protocol['training_tasks'] * 3
    for index, step in enumerate(steps):
        epoch, batch_index = divmod(index, 24)
        expected = training[batch_index * 4:batch_index * 4 + 4]
        actual = [r for r in items if r['step'] == index + 1]
        assert step['step'] == index + 1 and step['epoch'] == epoch + 1
        assert step['task_ids'] == [r['task_id'] for r in expected]
        assert len(actual) == len(expected) == (2 if batch_index == 23 else 4)
        for a, b in zip(actual, expected):
            assert a['record_id'] == b['record_id'] and a['epoch'] == epoch + 1
            assert a['accumulation_divisor'] == len(expected)
            assert a['prompt_tokens'] == len(b['prompt_token_ids'])
            assert a['completion_tokens'] == len(b['completion_token_ids'])
            assert math.isfinite(a['nll'])
        assert math.isclose(step['mean_nll'], math.fsum(r['nll'] for r in actual) / len(actual), abs_tol=1e-12)
        assert math.isfinite(step['gradient_norm'])
    budgets = {k: sum(r[k] for r in items) for k in ['prompt_tokens', 'completion_tokens']}
    assert budgets['prompt_tokens'] == protocol['training_prompt_tokens']
    assert budgets['completion_tokens'] == protocol['training_completion_tokens']
    cal_ids = protocol['evaluation']['calibration_ids']
    held_ids = protocol['evaluation']['headroom_ids']
    assert len(cal_ids) == 21 and len(held_ids) == 16
    assert set(held_ids) <= set(protocol['training_tasks'])
    assert set(cal_ids).isdisjoint(protocol['training_tasks'])
    pool = ROOT / 'code-learning-pilot'
    cal_targets = {r['task_id']: r for r in rows(pool / 'evaluator_targets.jsonl') if r['task_id'] in cal_ids}
    held_targets = {int(k): v for k, v in read(OUT / 'headroom_targets.json').items()}
    cal_base = {r['task_id']: r['first_correct'] for r in rows(ROOT / 'code-extraction-audit/outcomes.jsonl') if r['split'] == 'calibration'}
    held_base = {r['task_id']: r['correct'] for r in rows(OUT / 'headroom/grades_first_block.jsonl')}
    spec = importlib.util.spec_from_file_location('grader52', 'scripts/52_code_learning_pool.py')
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    findings = {}
    for split, ids, targets, baseline, base_path in [
        ('headroom', held_ids, held_targets, held_base, OUT / 'headroom/base_raw.jsonl'),
        ('calibration', cal_ids, cal_targets, cal_base, ROOT / 'code-functional-development/grid-evaluation/base/eval_raw.jsonl')]:
        old, new = rows(base_path), rows(FULL / split / 'trained_raw.jsonl')
        grades = rows(FULL / split / 'grades_first_block.jsonl')
        assert [r['task_id'] for r in old] == [r['task_id'] for r in new] == ids
        assert [r['task_id'] for r in grades] == ids
        details, base_details, changed, gained, lost = [], [], [], [], []
        for a, b, grade in zip(old, new, grades):
            for field in ['task_id', 'user_prompt', 'rendered_prompt', 'batch_index', 'batch_task_ids', 'prompt_token_ids', 'padded_prompt_token_ids', 'attention_mask']:
                assert a[field] == b[field], (split, b['task_id'], field)
            code, diagnostic = interface(b)
            _, old_diagnostic = interface(a)
            assert diagnostic['required'], b['task_id']
            wrapped = '```python\n' + code + '```' if re.findall(PATTERN, b['generation'], re.S) else code
            rerun = grader.grade(wrapped, targets[b['task_id']])
            assert rerun['correct'] == grade['correct'] and rerun['status'] == grade['status']
            assert grade['base_correct'] == baseline[b['task_id']]
            assert grade['exact_output_same'] == (a['generated_token_ids'] == b['generated_token_ids'])
            if not grade['exact_output_same']:
                changed.append(b['task_id'])
            if grade['correct'] and not grade['base_correct']:
                gained.append(b['task_id'])
            if grade['base_correct'] and not grade['correct']:
                lost.append(b['task_id'])
            details.append({**diagnostic, 'correct': grade['correct'], 'base_correct': grade['base_correct'], 'regrade_status': rerun['status']})
            base_details.append(old_diagnostic)
        original = summary['evaluation'][split]
        assert gained == original['gained'] and lost == original['lost']
        assert len(changed) == original['outputs_changed']
        assert sum(r['correct'] for r in grades) == original['correct']
        assert sum(r['generated_tokens'] for r in details) == original['generated_tokens']
        findings[split] = {'tasks': len(ids), 'base_correct': sum(baseline[k] for k in ids),
                          'trained_correct': sum(r['correct'] for r in grades), 'gained': gained, 'lost': lost,
                          'output_changed_ids': changed, 'exact_prompt_boundaries': True,
                          'grade_reexecution_agrees': True, 'per_task': details}
        for prefix, records in [('base', base_details), ('trained', details)]:
            findings[split][prefix + '_missing_interface_ids'] = [r['task_id'] for r in records if r['missing']]
            findings[split][prefix + '_multiblock_ids'] = [r['task_id'] for r in records if r['blocks'] > 1]
            findings[split][prefix + '_non_eos_ids'] = [r['task_id'] for r in records if r['finish_reason'] != 'eos']
            findings[split][prefix + '_truncated_ids'] = [r['task_id'] for r in records if r['finish_reason'] == 'length']
    gate = findings['headroom']['trained_correct'] >= 10 and findings['calibration']['trained_correct'] >= 19
    assert gate == summary['feasibility_gate_passed']
    report = {'passed': True, 'training_items': 282, 'optimizer_steps': 72,
              'same94_order_each_epoch': True, 'final_batch_actual_divisor2_each_epoch': True,
              'training_budget': budgets, 'all37_prompt_boundaries_verified': True,
              'all37_grades_reexecuted': True, 'checkpoint_and_input_manifest_hashes_verified': True,
              'feasibility_gate_passed': gate, 'evaluation': findings,
              'scope': 'Held-in16 learner sensitivity and calibration21 development only; no final or measurement grading, no parameter/recipe changes.',
              'source_sha256': {str(p): sha(p) for p in [Path(__file__), OUT / 'full_protocol.json', FULL / 'completed_manifest.json']}}
    (OUT / 'independent_full_audit83.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'held_in': findings['headroom']['trained_correct'], 'calibration': findings['calibration']['trained_correct'], 'gate': gate}))


if __name__ == '__main__':
    run()
