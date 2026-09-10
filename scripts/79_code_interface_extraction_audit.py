#!/usr/bin/env python3
"""Posthoc interface-based extraction check; never change frozen reward labels."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path('runs/swe-diversity-selection')
OUT = ROOT / 'code-extraction-interface-audit'
HELPER = Path('scripts/52_code_learning_pool.py')
PATTERN = r'```(?:python)?\s*\n(.*?)```'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return [json.loads(line) for line in p.open()]


def save(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def definitions(code):
    try:
        return {n.name for n in ast.parse(code).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    except SyntaxError:
        return set()


assert not OUT.exists()
OUT.mkdir()
sources = [Path(__file__), HELPER]
cases = []
count = 0
for pool in ['code-learning-pilot', 'code-learning-expansion']:
    folder = ROOT / pool
    paths = [folder / name for name in ['inputs.jsonl', 'raw.jsonl', 'evaluator_targets.jsonl', 'protocol.json']]
    sources.extend(paths)
    inputs = {r['task_id']: r for r in read(folder / 'inputs.jsonl') if r['split'] == 'train'}
    targets = {r['task_id']: r for r in read(folder / 'evaluator_targets.jsonl') if r['split'] == 'train'}
    for row in read(folder / 'raw.jsonl'):
        count += 1
        task = row['task_id']
        required = set(re.findall(r'def\s+(\w+)\s*\(', '\n'.join(inputs[task]['interfaces'])))
        blocks = re.findall(PATTERN, row['generation'], re.S)
        first = blocks[0] if blocks else row['generation']
        if required <= definitions(first):
            continue
        matching = [i for i, code in enumerate(blocks) if required <= definitions(code)]
        cases.append(dict(record_id=f'{task}:{row["sample"]}', task_id=task,
                          required=sorted(required), block_definitions=[sorted(definitions(c)) for c in blocks],
                          selected_index=matching[0] if matching else None,
                          selected_code=blocks[matching[0]] if matching else None,
                          target=targets[task]))
assert count == 960
save(OUT / 'protocol.json', dict(
    utc=datetime.now(timezone.utc).isoformat(), python=sys.version, executable=sys.executable,
    scope='Posthoc reproducible follow-up to root static/middle-block diagnostic; no new reward labels or active groups modified',
    population='All960 training outputs, no calibration/measurement/final outputs',
    rule='When first block lacks public interface definitions, select first block containing all those top-level definitions; no test-result-based selection',
    regex=PATTERN, cases=[r['record_id'] for r in cases], seed=None,
    source_sha256={str(p): sha(p) for p in sources}, api_spend_usd=0))
save(OUT / 'cases.json', cases)
spec = importlib.util.spec_from_file_location('grader52', HELPER)
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
results = []
for row in cases:
    if row['selected_code'] is None:
        result = dict(correct=False, status='no_interface_block')
    else:
        text = '```python\n' + row['selected_code'] + '\n```'
        assert re.findall(PATTERN, text, re.S) == [row['selected_code'] + '\n']
        result = grader.grade(text, row['target'])
    results.append(dict(record_id=row['record_id'], selected_index=row['selected_index'], grade=result))
save(OUT / 'results.json', results)
summary = dict(records_checked=count, first_missing_interface=len(cases),
               selected_blocks_pass=[r['record_id'] for r in results if r['grade']['correct']],
               active_first_block_groups_unchanged=True, api_spend_usd=0)
save(OUT / 'summary.json', summary)
save(OUT / 'completed_manifest.json', dict(sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}))
print(json.dumps(summary, indent=2))
