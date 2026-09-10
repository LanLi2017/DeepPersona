#!/usr/bin/env python3
"""Prospective required-test endpoint projection of frozen40 host-readiness logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'runs/swe-diversity-selection/swe-localize-repair-support'
PRIOR = SUPPORT / 'evaluation_readiness_versions'
OUT = SUPPORT / 'evaluation_required_tests'
PLAN = ROOT / 'docs/swe_required_test_control_plan.md'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = load('required_test_grading141', ROOT / 'scripts/141_swe_required_test_grading.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def verify(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT / name) == digest, name


def prospective():
    # Changing the endpoint is allowed only before actual learner/control outcomes.
    for branch in ['full_training', 'full_training_required_tests', 'autonomous_inference', 'autonomous_inference_required_tests']:
        folder = SUPPORT / branch
        assert not (folder / 'run').exists(), str(folder)
        assert not (folder / 'outer_dispatch.json').exists(), str(folder)
        assert not (folder / 'episodes').exists(), str(folder)
        assert not (folder / 'run_started.json').exists(), str(folder)


def freeze(review_path):
    prospective()
    assert not OUT.exists(), 'New endpoint uses an exclusive new namespace'
    review_path = Path(review_path).resolve()
    review = read(review_path)
    assert review['allowed'] is True and review['new_endpoint_not_strict_gate_override'] is True
    required = [Path(__file__).resolve(), Path(g.__file__).resolve(), PLAN,
                PRIOR / 'summary.json', PRIOR / 'completed_manifest.json',
                PRIOR / 'independent_readiness_audit.json', PRIOR / 'official_resolution_semantics_audit.json',
                PRIOR / 'official_resolution_pair_validation.json']
    for path in required:
        assert review['source_sha256'][str(path)] == sha(path), str(path)
    verify(review['source_sha256'])
    audit = read(PRIOR / 'independent_readiness_audit.json')
    assert audit['all_checks_passed'] and audit['evaluation_tasks_audited'] == 20
    for field, name in [('summary_sha256', 'summary.json'), ('completed_manifest_sha256', 'completed_manifest.json'),
                        ('initial_manifest_sha256', 'initial_manifest.json')]:
        assert audit[field] == sha(PRIOR / name)
    verify(read(PRIOR / 'completed_manifest.json'))
    old = read(PRIOR / 'summary.json')
    assert old['evaluation_runtime_valid'] == 18 and old['readiness_gate_passed'] is False
    initial = read(PRIOR / 'initial_manifest.json')
    assert initial['all20_public_initials_frozen'] and initial['evaluation_private_values_read'] is False
    assert initial['protocol_sha256'] == sha(PRIOR / 'initial_protocol.json')
    verify(initial['source_sha256'])
    ids = [t['instance_id'] for t in initial['tasks']]
    assert len(ids) == len(set(ids)) == 20 and ids == [t['instance_id'] for t in old['tasks']]
    selection = read(PRIOR / 'selection.json')
    assert selection['evaluation_ids'] == ids
    OUT.mkdir()
    for name in ['initial_manifest.json', 'initial_protocol.json', 'protocol.json', 'assets_manifest.json', 'export_manifest.json']:
        shutil.copyfile(PRIOR / name, OUT / name)
    for name in ['initials', 'public', 'private']:
        (OUT / name).symlink_to(PRIOR / name, target_is_directory=True)
    # protocol.json is byte-identical original execution attestation. Its selection
    # hash continues to reference PRIOR selection, not this new endpoint projection.
    selection.update(endpoint='required_test_passed_with_exit_status_integrity', endpoint_rule=g.RULE,
                     prior_strict_readiness=str(PRIOR), prior_strict_runtime_valid=18,
                     original_runtime_protocol_sha256=sha(PRIOR / 'protocol.json'),
                     original_runtime_selection_sha256=sha(PRIOR / 'selection.json'),
                     endpoint_changes_runtime_execution=False)
    save(OUT / 'selection.json', selection)
    evidence = required + [review_path, PRIOR / 'protocol.json', PRIOR / 'selection.json',
                           PRIOR / 'initial_manifest.json', PRIOR / 'initial_protocol.json']
    evidence += [g.HARNESS / name for name in g.PINNED]
    save(OUT / 'endpoint_protocol.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), endpoint=selection['endpoint'], rule=g.RULE,
        source_sha256={str(p): sha(p) for p in evidence}, evaluation_ids=ids,
        initial_manifest_sha256=sha(OUT / 'initial_manifest.json'),
        assigned_evaluation_slots=20, prior_strict_runtime_valid=18,
        new_runtime_execution=False, new_test_runs=0, models_trained_or_generated=False,
        primary_exact_required_status='PASSED', official_skip_xfail_semantics='diagnostic_only',
        sensitivity='Completed normal exit0 plus exact required PASSED; prior strict result retained18/20',
        note='Prospective endpoint change after readiness outcomes, before model outcomes; not validation of the original strict gate.'))
    save(OUT / 'kernel_synthetic_checks.json', g.synthetic())
    print('Frozen separate required-test endpoint; original strict18/20 preserved; no tests or models run.')


def project():
    prospective()
    ep = read(OUT / 'endpoint_protocol.json')
    verify(ep['source_sha256'])
    assert not (OUT / 'summary.json').exists()
    assert sha(OUT / 'protocol.json') == sha(PRIOR / 'protocol.json')
    assert sha(OUT / 'initial_manifest.json') == ep['initial_manifest_sha256']
    manifest = read(PRIOR / 'completed_manifest.json')
    verify(manifest)
    old = read(PRIOR / 'summary.json')
    protocol = read(PRIOR / 'protocol.json')
    selection = read(OUT / 'selection.json')
    ids = ep['evaluation_ids']
    assert ids == [t['instance_id'] for t in protocol['tasks']]
    assert all(group['exit_code'] == 0 and not group['timed_out'] and group['results_complete'] for group in old['groups'])
    rows = []
    evidence = dict(manifest)
    for task in protocol['tasks']:
        iid = task['instance_id']
        assert task['ready'] and task['role'] == 'evaluation'
        arms = []
        for arm in ['baseline', 'gold']:
            folder = PRIOR / 'private/tasks' / iid / arm
            result = read(folder / 'result.json')
            parsed = g.parse_log((folder / 'test.log').read_text())
            assert parsed == result['all_parsed_statuses']
            grade = g.grade(task['tests'], parsed, result['exit_code'], completed=True, timed_out=False)
            assert grade['required_statuses'] == result['statuses']
            arms.append(dict(arm=arm, grade=grade, result_sha256=sha(folder / 'result.json'),
                             log_sha256=sha(folder / 'test.log')))
        baseline, gold = [a['grade'] for a in arms]
        passed = baseline['baseline_required_pattern'] and gold['primary_resolved']
        strict = baseline['baseline_required_pattern'] and baseline['exit_code'] == 1 and gold['strict_exit0_resolved']
        prior = next(t for t in old['tasks'] if t['instance_id'] == iid)
        assert strict == (prior['disposition'] == 'baseline_gold_passed')
        row = dict(instance_id=iid, role='evaluation', slot=task['slot'], family=task['family'], replacement=False,
                   disposition='required_test_baseline_gold_passed' if passed else 'required_test_baseline_gold_failed',
                   required_test_baseline_gold_passed=passed, strict_exit_sensitivity_passed=strict,
                   prior_strict_disposition=prior['disposition'], arms=arms)
        path = OUT / 'records' / (iid + '.json')
        save(path, row)
        evidence[str(path.relative_to(ROOT))] = sha(path)
        rows.append({k: v for k, v in row.items() if k != 'arms'} | {'record_path': str(path.relative_to(ROOT)), 'record_sha256': sha(path)})
    runnable = [r['instance_id'] for r in rows if r['required_test_baseline_gold_passed']]
    summary = dict(utc=datetime.now(timezone.utc).isoformat(), assigned_slots=20, assigned_evaluation_slots=20,
                   assigned_source_slots=0, evaluation_runtime_valid=len(runnable), evaluation_runnable_ids=runnable,
                   all20_endpoints_ready=len(runnable) == 20, readiness_gate_passed=len(runnable) == 20,
                   initial_index_fits=old['initial_index_fits'], tasks=rows, endpoint=selection['endpoint'], rule=g.RULE,
                   prior_strict_runtime_valid=old['evaluation_runtime_valid'], prior_strict_readiness_gate_passed=False,
                   strict_exit_sensitivity_valid=sum(t['strict_exit_sensitivity_passed'] for t in rows),
                   original_runtime_protocol_sha256=sha(PRIOR / 'protocol.json'),
                   initial_manifest_sha256=sha(OUT / 'initial_manifest.json'),
                   endpoint_protocol_sha256=sha(OUT / 'endpoint_protocol.json'),
                   runtime_tests_rerun=0, models_trained_or_generated=False, api_spend_usd=0, gpu_used=False,
                   official_container_evaluation=False,
                   interpretation='Saved-log readiness under new prospective required-PASSED endpoint; original strict gate failed18/20. No learner effect or official Docker claim.')
    save(OUT / 'summary.json', summary)
    for path in OUT.glob('*.json'):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    save(OUT / 'completed_manifest.json', evidence)
    print(json.dumps(dict(required_test_pairs=len(runnable), strict_exit_pairs=summary['strict_exit_sensitivity_valid'], tests_rerun=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'project'])
    parser.add_argument('--review')
    args = parser.parse_args()
    if args.stage == 'freeze':
        assert args.review
        freeze(args.review)
    else:
        project()
