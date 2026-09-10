#!/usr/bin/env python3
"""New source preparation namespace correcting only distribution provenance collection."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'runs/swe-diversity-selection/swe-localize-repair-support'
PREVIOUS = SUPPORT / 'source_runtime'
OUT = SUPPORT / 'source_runtime_metadata'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load('quality128corrected', ROOT / 'scripts/128_swe_localize_source_quality.py')
probe = load('provenance131', ROOT / 'scripts/131_swe_environment_provenance.py')
q = prior.q
q.OUT = OUT
q.w.OUT = OUT
q.w.m.OUT = OUT
q.w.__file__ = str(Path(__file__).resolve())
sha, read, save, utc = q.sha, q.read, q.save, q.utc
OLD_QUERY = 'import importlib.metadata as m,json,platform,sympy;print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"]:d.version for d in m.distributions()},sympy_file=sympy.__file__)))'


def verify(mapping):
    for path, digest in mapping.items():
        assert sha(ROOT / path) == digest, path


def freeze():
    assert not OUT.exists(), 'One corrected preparation namespace only'
    review_path = SUPPORT / 'metadata_correction_review.json'
    review = read(review_path)
    assert review['all_checks_passed'] and review['allow_metadata_correction']
    verify(review['source_sha256'])
    verify(review['evidence_sha256'])
    selection = read(PREVIOUS / 'selection.json')
    protocol = read(PREVIOUS / 'protocol.json')
    assets = read(PREVIOUS / 'assets_manifest.json')
    summary = read(PREVIOUS / 'private/assets_review_summary.json')
    assert len(selection['all_slots']) == 32 and len(selection['source_prequalified_ids']) == 23
    assert selection['evaluation_ids'] == [] and all(t['role'] == 'source' for t in selection['tasks'])
    assert [t['instance_id'] for t in protocol['tasks']] == selection['source_prequalified_ids']
    assert len(protocol['tasks']) == 23 and sum(t['ready'] for t in protocol['tasks']) == 7
    failures = [t for t in protocol['tasks'] if not t['ready']]
    assert all(t['preparation_failure'] == dict(type='AssertionError',message='',replacement=False) for t in failures)
    assert not (PREVIOUS / 'groups').exists(), 'No previous baseline/gold test dispatch allowed'
    assert not list((PREVIOUS / 'private/tasks').rglob('test.log'))
    assert not (PREVIOUS / 'summary.json').exists()
    assert [t['instance_id'] for t in assets['tasks']] == selection['source_prequalified_ids']
    assert all(t['ready'] for t in assets['tasks'])
    assert assets['selection_sha256'] == sha(PREVIOUS / 'selection.json')
    assert summary['assets_manifest_sha256'] == sha(PREVIOUS / 'assets_manifest.json')
    verify(selection['source_sha256'])
    verify(protocol['source_sha256'])
    verify(summary['reviewed_source_sha256'])
    assert probe.PINS == q.w.m.b.PACKAGES
    new = copy.deepcopy(selection)
    evidence = [Path(__file__).resolve(), ROOT / 'scripts/131_swe_environment_provenance.py', review_path,
        PREVIOUS / 'selection.json', PREVIOUS / 'protocol.json', PREVIOUS / 'preparation_terminal.json',
        PREVIOUS / 'preparation_progress.json', PREVIOUS / 'assets_manifest.json',
        PREVIOUS / 'private/assets_review_summary.json', PREVIOUS / 'independent_asset_review.json',
        PREVIOUS / 'root_wheel_metadata_diagnostic.json']
    for path in evidence:
        new['source_sha256'][str(path)] = sha(path)
    new.update(utc=utc(), prior_preparation=str(PREVIOUS),
        source_preparation_correction='Replace only90 exact environment-query command with131 direct-site distribution provenance probe. Same23source IDs, same source/assets/pins/setup/test commands. No test outcomes existed before correction.',
        original_preparation_dispositions=dict(ready=7,collector_assertions=16),
        prior_baseline_gold_tests=0, environment_retuned=False,
        no_retry='No test retry or environment retuning; one new preparation attempt corrects a proven metadata collector defect and preserves the entire original attempt.')
    OUT.mkdir()
    (OUT / 'private').mkdir()
    (OUT / 'private/base').symlink_to(prior.POP / 'private/base', target_is_directory=True)
    (OUT / 'public').symlink_to(prior.POP / 'public', target_is_directory=True)
    shutil.copyfile(prior.POP / 'export_manifest.json', OUT / 'export_manifest.json')
    save(OUT / 'selection.json', new)
    # Reuse every frozen official byte, without fetching or choosing alternatives.
    for task in assets['tasks']:
        iid = task['instance_id']
        folder = OUT / 'private/review_assets' / iid
        folder.mkdir(parents=True)
        for name, info in task['files'].items():
            original = PREVIOUS / 'private/review_assets' / iid / name
            assert sha(original) == info['sha256']
            shutil.copyfile(original, folder / name)
        assert q.w.asset_contract(folder) == task['command_contract']
        save(OUT / 'private/asset_records' / (iid + '.json'), task)
    assets['selection_sha256'] = sha(OUT / 'selection.json')
    save(OUT / 'assets_manifest.json', assets)
    summary['selection_sha256'] = sha(OUT / 'selection.json')
    summary['assets_manifest_sha256'] = sha(OUT / 'assets_manifest.json')
    summary['reviewed_source_sha256'].update({str(p):sha(p) for p in [Path(__file__).resolve(), ROOT/'scripts/131_swe_environment_provenance.py']})
    summary['gold_and_test_body_included'] = True
    summary['body_scope_correction'] = 'Embedded SOURCE test patches occur in eval_script strings. Operator-private only; full eval.sh never executed and no evaluation20 assets read.'
    save(OUT / 'private/assets_review_summary.json', summary)
    save(OUT / 'correction_protocol.json', dict(utc=utc(), selection_sha256=sha(OUT/'selection.json'),
        review_sha256=sha(review_path), original_query=OLD_QUERY, replacement_probe_sha256=sha(ROOT/'scripts/131_swe_environment_provenance.py'),
        affected_operation='Only the exact post-install environment provenance query; unchanged package installation and later baseline/gold execution',
        source_tasks=23, retained_slots=32, prior_tests=0, evaluation_private_values_read=False,
        old_attempt_preserved=True, new_asset_downloads=0, api_spend_usd=0, gpu_used=False))
    print(json.dumps(dict(frozen=True, active_sources=23, retained_sources=32, correction='distribution provenance only')), flush=True)


def selection():
    chosen = q.selection()
    assert chosen['source_preparation_correction'] and chosen['evaluation_ids'] == []
    assert len(chosen['all_slots']) == 32 and len(chosen['tasks']) == 23
    assert q.w.__file__ == str(Path(__file__).resolve())
    return chosen


def prepare(review):
    chosen = selection()
    selected_ids = set(chosen['source_prequalified_ids'])
    assert probe.PINS == q.w.m.b.PACKAGES
    original = q.w.m.b.call
    count = 0
    def corrected(cmd, *args, **kwargs):
        nonlocal count
        if len(cmd) == 3 and cmd[1] == '-c' and cmd[2] == OLD_QUERY:
            count += 1
            cmd = list(cmd)
            cmd[2] = probe.PROBE
            repo = Path(kwargs['cwd']).resolve()
            assert repo.name == 'repo' and repo.is_relative_to(OUT/'private/tasks')
            iid, arm = repo.parent.parent.name, repo.parent.name
            assert arm in {'baseline','gold'} and iid in selected_ids
            assert Path(cmd[0]) == repo.parent/'venv/bin/python'
            assert kwargs['env']['VIRTUAL_ENV'] == str(repo.parent/'venv')
            assert kwargs['env']['PATH'].split(':')[0] == str(repo.parent/'venv/bin')
            raw = original(cmd, *args, **kwargs)
            evidence = json.loads(raw)
            save(OUT/'private/environment_provenance'/iid/(arm+'.json'), evidence)
            assert evidence['provenance_all_checks_passed'] is True, 'Strict direct environment provenance failed; raw evidence retained'
            return raw
        return original(cmd, *args, **kwargs)
    q.w.m.b.call = corrected
    try:
        q.prepare(review)
    finally:
        q.w.m.b.call = original
        save(OUT/'metadata_query_dispatch_receipt.json', dict(utc=utc(), substituted_queries=count,
            exact_old_query_only=True, other_commands_unchanged=True, package_pins_unchanged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','prepare','smoke','run','aggregate','_run-group'])
    parser.add_argument('--review')
    parser.add_argument('--group', choices=['smoke','remaining'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(); return
    selection()
    if args.stage == 'prepare': prepare(args.review)
    elif args.stage == 'aggregate': q.aggregate()
    elif args.stage == '_run-group': q.w.child_group(args.group)
    else: q.w.run_group('smoke' if args.stage == 'smoke' else 'remaining', args.review)


if __name__ == '__main__':
    main()
