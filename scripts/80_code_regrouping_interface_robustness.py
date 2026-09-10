#!/usr/bin/env python3
"""Supplementary exclude739 sensitivity; retain unchanged primary17 analysis."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import types

PRIMARY = Path('runs/swe-diversity-selection/code-regrouping-first-block')
OUT = PRIMARY / 'robustness-excluding-739'
WRAPPER = Path('scripts/77_code_first_block_regrouping_analysis.py')
NOTE = Path('docs/swe_regrouping_interface_robustness_protocol.md')
AUDIT = Path('runs/swe-diversity-selection/code-extraction-interface-audit')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def implementation():
    source = WRAPPER.read_text()
    old = "t._replace(string='17')"
    assert source.count(old) == 1
    source = source.replace(old, "t._replace(string='16')")
    wrapper = types.ModuleType('supplementary16_wrapper')
    wrapper.__file__ = str(WRAPPER)
    exec(compile(source, str(WRAPPER) + ':supplementary16', 'exec'), wrapper.__dict__)
    return wrapper.implementation()


def freeze():
    assert not OUT.exists(), 'Supplement already frozen'
    primary = read(PRIMARY / 'analysis_protocol.json')
    for name, expected in primary['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    assert len(primary['task_ids']) == 17 and 739 in primary['task_ids']
    for name, expected in read(AUDIT / 'completed_manifest.json')['sha256'].items():
        assert sha(AUDIT / name) == expected
    evidence = read(AUDIT / 'summary.json')
    assert evidence['selected_blocks_pass'] == ['739:4']
    sources = [Path(__file__), WRAPPER, Path('scripts/74_code_regrouping_analysis.py'), NOTE,
               PRIMARY / 'analysis_protocol.json', PRIMARY / 'protocol.json', PRIMARY / 'groups.json',
               Path('scripts/79_code_interface_extraction_audit.py'),
               Path('runs/swe-diversity-selection/code-extraction-audit/first_block_interface_diagnostic.json')]
    sources += [AUDIT / f'{name}.json' for name in ['protocol', 'cases', 'results', 'summary', 'completed_manifest']]
    OUT.mkdir()
    save(OUT / 'protocol.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'real_regrouping_outcomes_read': False,
        'status': 'Supplementary robustness; primary17 remains fully retained and unchanged',
        'excluded_task_ids': [739], 'task_ids': [k for k in primary['task_ids'] if k != 739],
        'etas': [1.0, 10.0], 'reason': 'External all960 interface audit before any finite-outcome inspection: selected739:4 passes with an interface-containing block, so the frozen739 group is not2+/2- under that alternative execution policy.',
        'statistics': 'Unchanged77 statistics, seeds and thresholds; only source count17to16; no relabeling or new policy selection.',
        'source_sha256': {str(p): sha(p) for p in sources}})
    print(json.dumps({'frozen': str(OUT / 'protocol.json'), 'sha256': sha(OUT / 'protocol.json')}))


def analyze():
    protocol = read(OUT / 'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    completed = read(PRIMARY / 'completed_manifest.json')
    for name, expected in completed['sha256'].items():
        assert sha(PRIMARY / name) == expected, name
    summary, manifest = read(PRIMARY / 'summary.json'), read(PRIMARY / 'manifest.json')
    primary_protocol = read(PRIMARY / 'analysis_protocol.json')
    assert summary['tasks'] == 17 and summary['technical_checks_passed']
    assert summary['task_ids'] == primary_protocol['task_ids']
    rows = [json.loads(line) for line in (PRIMARY / 'per_task.jsonl').read_text().splitlines()]
    assert [r['task_id'] for r in rows] == primary_protocol['task_ids']
    subset = [r for r in rows if r['task_id'] != 739]
    assert [r['task_id'] for r in subset] == protocol['task_ids'] and len(subset) == 16
    derived = OUT / 'derived_analysis_inputs'
    assert not derived.exists(), 'Prior supplementary results retained'
    derived.mkdir()
    save(derived / 'analysis_protocol.json', {
        'source_sha256': protocol['source_sha256'], 'task_ids': protocol['task_ids'],
        'etas': protocol['etas'], 'inference': 'Supplementary16-task descriptive sensitivity excluding739 per external interface audit; original17-task primary preserved; fixed shared21 measurement tasks.'})
    save(derived / 'summary.json', {**summary, 'tasks': 16, 'task_ids': protocol['task_ids'],
                                  'derivation': 'Filtered analysis copy, not a separate16-task GPU run'})
    save(derived / 'manifest.json', manifest)
    (derived / 'per_task.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in subset))
    save(derived / 'completed_manifest.json', {'sha256': {
        name: sha(derived / name) for name in ['analysis_protocol.json', 'summary.json', 'manifest.json', 'per_task.jsonl']}})
    implementation().analyze(derived)
    result = read(derived / 'analysis.json')
    assert result['source_tasks'] == 16 and len(result['per_task']) == 32
    result['supplementary_provenance'] = {
        'excluded_task_ids': [739], 'primary_source_tasks': 17, 'primary_outcomes_modified': False,
        'interpretation': 'Sensitivity to one externally identified interface-extraction case; no primary reward relabeling, no claim that all remaining rewards are perfect.',
        'source_sha256': {str(p): sha(p) for p in [OUT / 'protocol.json', PRIMARY / 'completed_manifest.json', PRIMARY / 'per_task.jsonl']}}
    save(OUT / 'analysis.json', result)
    print(json.dumps({'supplementary': str(OUT / 'analysis.json'), 'primary17_preserved': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'analyze', 'self-test'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'analyze':
        analyze()
    else:
        implementation().self_test()
