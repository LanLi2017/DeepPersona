#!/usr/bin/env python3
"""Fixed40 private candidate scoring, only after all autonomous generation closes."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT/'runs/swe-diversity-selection/swe-localize-repair-support'
GEN = SUPPORT/'autonomous_inference'
READY = SUPPORT/'evaluation_readiness_versions'
OUT = SUPPORT/'candidate_evaluation'
PUBLIC = SUPPORT.parent/'swe-sympy-patch-sft/public'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c = load('candidate93', ROOT/'scripts/93_swe_candidate_evaluation.py')
k = load('candidate125', ROOT/'scripts/125_swe_localize_repair_support.py')
b = c.b
sha, read = k.sha, k.read


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary,path)  # Atomic publication and fail rather than overwrite.
    temporary.unlink()


def verify(mapping):
    for path, digest in mapping.items():
        assert sha(ROOT/path) == digest, path


def candidate_environment(venv):
    # Do not create/refresh caches in the readiness-verified shared virtualenv.
    env = b.environment(venv)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def validate_slots(slots):
    assert len(slots) == 40 and len({row['episode_id'] for row in slots}) == 40
    ids = list(dict.fromkeys(row['instance_id'] for row in slots))
    assert len(ids) == 20
    for iid in ids:
        rows = [row for row in slots if row['instance_id'] == iid]
        assert len(rows) == 2 and {row['arm'] for row in rows} == {'base', 'trained'}
    return ids


def resolved(tests, parsed, exit_code):
    assert set(tests) == {'FAIL_TO_PASS','PASS_TO_PASS'}
    if not tests['FAIL_TO_PASS'] or not all(isinstance(names,list) for names in tests.values()):
        return False
    return exit_code == 0 and all(parsed.get(name) == 'PASSED' for names in tests.values() for name in names)


def prepare():
    """No candidate import or generated-code execution; git/AST preparation only."""
    assert not (OUT/'protocol.json').exists() and not (OUT/'preparation_started.json').exists()
    # The two closed manifests are required BEFORE opening any generated patch.
    assert (GEN/'outer_terminal.json').is_file() and (GEN/'completed_manifest.json').is_file()
    inference = read(GEN/'summary.json')
    assert inference['assigned_tasks'] == 20 and inference['assigned_episodes'] == 40 and inference['all_slots_retained']
    verify(read(GEN/'completed_manifest.json'))
    generation_protocol = read(GEN/'protocol.json')
    rows = inference['episodes']
    validate_slots(rows)
    assert inference['outer_terminal_sha256'] == sha(GEN/'outer_terminal.json')
    assert [r['episode_id'] for r in rows] == [r['episode_id'] for r in generation_protocol['schedule']]
    ready = read(READY/'summary.json')
    assert ready['all20_endpoints_ready'] and ready['readiness_gate_passed']
    tasks = {t['instance_id']:t for t in read(READY/'protocol.json')['tasks']}
    assert len(tasks) == 20 and set(tasks) == {r['instance_id'] for r in rows}
    assert all(t['ready'] for t in tasks.values())
    audit = read(READY/'independent_readiness_audit.json')
    assert audit['all_checks_passed'] and audit['summary_sha256'] == sha(READY/'summary.json')
    assert audit['completed_manifest_sha256'] == sha(READY/'completed_manifest.json')
    assert audit['evaluation_tasks_audited'] == 20 and audit['readiness_gate_passed']
    assert set(audit['evaluation_runnable_ids']) == set(tasks)
    assert audit['initial_manifest_sha256'] == sha(SUPPORT/'evaluation_readiness/initial_manifest.json')
    readiness_manifest = read(READY/'completed_manifest.json')
    verify(readiness_manifest)
    verify(read(READY/'protocol.json')['source_sha256'])
    for path in [READY/'protocol.json',READY/'selection.json',READY/'summary.json']:
        assert readiness_manifest[str(path.relative_to(ROOT))] == sha(path)
    checks = read(OUT/'synthetic_checks.json')
    assert checks['all_checks_passed'] and checks['source_sha256'] == sha(Path(__file__))
    launcher_checks = read(OUT/'launcher_checks.json')
    assert launcher_checks['all_checks_passed']
    assert launcher_checks['launcher_sha256'] == sha(ROOT/'scripts/140_swe_localize_candidate_launcher.py')
    assert launcher_checks['audit_sha256'] == sha(OUT/'audit_launcher.py')
    scoring_checks = read(OUT/'scoring_regression_checks.json')
    assert scoring_checks['all_checks_passed']
    assert scoring_checks['scorer_sha256'] == sha(Path(__file__))
    assert scoring_checks['launcher_sha256'] == sha(ROOT/'scripts/140_swe_localize_candidate_launcher.py')
    assert scoring_checks['audit_sha256'] == sha(ROOT/'scripts/audit_swe_localize_candidate_scoring.py')
    OUT.mkdir(exist_ok=True)
    save(OUT/'preparation_started.json', dict(generation_summary_sha256=sha(GEN/'summary.json'),
         generation_completed_manifest_sha256=sha(GEN/'completed_manifest.json'),
         no_candidate_code_execution=True))
    sources = [Path(__file__), ROOT/'scripts/140_swe_localize_candidate_launcher.py',
        ROOT/'scripts/93_swe_candidate_evaluation.py', ROOT/'scripts/85_swe_execution_bridge.py',
        ROOT/'scripts/125_swe_localize_repair_support.py', ROOT/'scripts/110_swe_patch_sft_targets.py',
        ROOT/'scripts/131_swe_environment_provenance.py',
        ROOT/'docs/swe_localize_candidate_scoring_plan.md', b.PARSER, OUT/'synthetic_checks.json', OUT/'launcher_checks.json', OUT/'audit_launcher.py',
        GEN/'protocol.json', GEN/'summary.json', GEN/'outer_terminal.json', GEN/'completed_manifest.json',
        OUT/'scoring_regression_checks.json', ROOT/'scripts/audit_swe_localize_candidate_scoring.py',
        READY/'protocol.json', READY/'summary.json', READY/'independent_readiness_audit.json',
        READY/'completed_manifest.json', ROOT/'scripts/137_swe_evaluation_python_versions.py']
    # Version-aware helper/contract file names are pinned by the readiness handoff.
    provenance = load('candidate138', ROOT/'scripts/138_swe_versioned_environment_probe.py')
    readiness_selection = read(READY/'selection.json')
    sources += [Path(provenance.__file__), READY/'selection.json']
    slots = []
    for row in rows:
        iid, key = row['instance_id'], row['episode_id']
        item = dict(instance_id=iid, arm=row['arm'], episode_id=key, slot=row['slot'],
                    stage='not_eligible', generation_status=row['status'], risks=[], reason=None)
        task = tasks[iid]
        official = readiness_selection['official_environment_by_task'][iid]
        contract = dict(python=official['python'], packages=provenance.pins(official['packages']),
             package_versions=official['packages'], docker_sha256=official['docker_sha256'],
             venv=str(READY/'private/tasks'/iid/'baseline/venv'), base=str(READY/'private/base'/iid))
        item['runtime_contract'] = contract
        assert contract['python'] in ('3.9.20','3.9.21') and len(contract['packages']) == 9
        assert task['tests'] == read(READY/'private/official_tasks'/iid/'tests.json')
        commands = [line for line in (READY/'private/official_tasks'/iid/'eval.sh').read_text().splitlines() if line.startswith('PYTHONWARNINGS=')]
        assert commands == [task['test_command']]
        for name in ['test.patch','tests.json','eval.sh']:
            sources.append(READY/'private/official_tasks'/iid/name)
        sources += [Path(contract['venv'])/'bin/python', READY/'private/tasks'/iid/'baseline/environment.json']
        if row.get('patch_eligible_for_evaluation'):
            episode = GEN/'run/episodes'/key
            patch, final_path = episode/'candidate.patch', episode/'final_edits.json'
            assert sha(patch) == row['patch_sha256']
            sources += [patch, final_path]
            item['patch_sha256'] = sha(patch)
            folder = OUT/'candidates'/key
            folder.mkdir(parents=True)
            base = Path(contract['base'])
            repo = folder/'repo'
            try:
                assert b.call(['git','rev-parse','HEAD'],cwd=base).strip() == task['base_commit']
                assert b.call(['git','status','--porcelain'],cwd=base) == ''
                edits = read(final_path)
                names = sorted({e['path'] for e in edits['edits']})
                assert names and all(k.TARGETS.path_allowed(name) for name in names)
                initial = read(SUPPORT/'evaluation_readiness/initials'/f'{iid}.json')
                original = {name:k.source_bytes(PUBLIC/iid,name,initial) for name in names}
                assert all(sha(base/name) == k.digest(blob) for name,blob in original.items())
                expected = k.TARGETS.apply_edits(original, edits['edits'])
                assert {name:k.digest(blob) for name,blob in expected.items()} == edits['patched_sha256']
                expected_changed = sorted(name for name in names if expected[name] != original[name])
                assert expected_changed
                log = b.call(['git','clone','--shared','--no-checkout',str(base),str(repo)])
                log += b.call(['git','checkout','--detach',task['base_commit']],cwd=repo)
                assert b.call(['git','status','--porcelain'],cwd=repo) == ''
                log += b.call(['git','apply','--check',str(patch)],cwd=repo)
                log += b.call(['git','apply',str(patch)],cwd=repo)
                changed = sorted(filter(None,b.call(['git','diff','--name-only','-z'],cwd=repo).split('\0')))
                assert changed == expected_changed
                for name in names:
                    assert not (repo/name).is_symlink() and (repo/name).read_bytes() == expected[name]
                for name in changed:
                    after = expected[name].decode()
                    ast.parse(after,filename=name)
                    item['risks'] += [dict(file=name,operation=op) for op in c.new_risks(original[name].decode(),after)]
                (folder/'preparation.log').write_text(log)
                (folder/'candidate.diff').write_text(b.call(['git','diff',task['base_commit']],cwd=repo))
                item.update(stage='ready', candidate_file_sha256={name:sha(repo/name) for name in changed},
                     candidate_diff_sha256=sha(folder/'candidate.diff'), files=changed)
                sources += [folder/'candidate.diff', folder/'preparation.log']
            except Exception as exc:
                item.update(stage='invalid_patch', reason=f'{type(exc).__name__}: {exc}')
        slots.append(item)
        save(OUT/'preparation_slots'/f'{key}.json', item)
    save(OUT/'protocol.json', dict(slots=slots, tasks=list(tasks.values()), slot_count=40,
         assigned_tasks=20, source_sha256={str(path):sha(path) for path in sources},
         inference_stopped_before_candidate_read=True, no_gold_applied=True,
         runtime='Unchanged official task command, strict task-specific Python/nine-package provenance; candidate checkout import.',
         candidate_environment_override={'PYTHONDONTWRITEBYTECODE':'1'},
         bytecode_policy='Disable candidate import/test bytecode writes to preserve readiness virtualenv caches; same setting for both arms.',
         external_timeout=dict(global_seconds=7200, per_candidate_seconds=210, kill_grace_seconds=5),
         test_timeout_seconds=180, import_timeout_seconds=30, cpu_limit_seconds=120,
         address_space_bytes=4*1024**3, log_file_bytes=20*1024**2,
         resolution='Exit0 AND nonempty required target list AND all targets/regressions PASSED.',
         ungraded='All unexecuted slots unresolved operationally; correctness_observed false, no replacement.',
         risk_review='Exact patch hash + reason required for introduced flagged external IO/dynamic/process operations.',
         no_retry=True, no_feedback_to_generation=True, api_spend_usd=0, gpu_used=False))
    save(OUT/'preparation_manifest.json', {str(path):sha(path) for path in OUT.rglob('*')
         if path.is_file() and 'repo' not in path.relative_to(OUT).parts})


def score_one(key):
    assert os.environ['SWE_CANDIDATE_EXTERNAL_SLOT'] == key
    protocol = read(OUT/'protocol.json')
    auth = read(OUT/'root_execution_review.json')
    assert auth['allow_candidate_correctness'] and auth['protocol_sha256'] == sha(OUT/'protocol.json')
    assert auth['preparation_manifest_sha256'] == sha(OUT/'preparation_manifest.json')
    verify(protocol['source_sha256'])
    slot = next(row for row in protocol['slots'] if row['episode_id'] == key)
    task = next(row for row in protocol['tasks'] if row['instance_id'] == slot['instance_id'])
    folder = OUT/'execution'/key
    folder.mkdir(parents=True, exist_ok=True)
    save(folder/'started.json',dict(episode_id=key,protocol_sha256=sha(OUT/'protocol.json')))
    result = dict(episode_id=key,instance_id=slot['instance_id'],arm=slot['arm'],resolved=False,
                  correctness_observed=False,execution_status=slot['stage'],generation_status=slot['generation_status'])
    tests = task['tests']
    assert tests['FAIL_TO_PASS'] and set(tests) == {'FAIL_TO_PASS','PASS_TO_PASS'}
    parsed = {}
    tick = time.monotonic()
    try:
        if slot['stage'] != 'ready':
            return
        if slot['risks']:
            approval = auth.get('external_io_approvals',{}).get(key,{})
            if approval.get('patch_sha256') != slot['patch_sha256'] or not approval.get('reason'):
                result['execution_status'] = 'withheld_operation_review'
                return
        repo = OUT/'candidates'/key/'repo'
        contract = slot['runtime_contract']
        assert b.call(['git','rev-parse','HEAD'],cwd=repo).strip() == task['base_commit']
        assert sorted(filter(None,b.call(['git','diff','--name-only','-z'],cwd=repo).split('\0'))) == slot['files']
        assert all(sha(repo/name)==digest for name,digest in slot['candidate_file_sha256'].items())
        test_patch = READY/'private/official_tasks'/slot['instance_id']/'test.patch'
        b.call(['git','apply','--check',str(test_patch)],cwd=repo)
        b.call(['git','apply',str(test_patch)],cwd=repo)
        assert all(sha(repo/name)==digest for name,digest in slot['candidate_file_sha256'].items())
        (folder/'evaluated.diff').write_text(b.call(['git','diff',task['base_commit']],cwd=repo))
        venv = Path(contract['venv'])
        helper = load('candidate138_runtime', ROOT/'scripts/138_swe_versioned_environment_probe.py')
        probe = helper.probe(contract['package_versions'])
        with (folder/'import.log').open('x') as log:
            process = subprocess.run([str(venv/'bin/python'),'-c',probe],cwd=repo,
                env=candidate_environment(venv),stdout=log,stderr=subprocess.STDOUT,timeout=30,preexec_fn=b.limits)
        if process.returncode != 0:
            result.update(execution_status='candidate_import_error',exit_code=process.returncode)
            return
        environment = json.loads((folder/'import.log').read_text())
        save(folder/'environment.json',environment)
        assert environment['provenance_all_checks_passed']
        assert environment['python'] == contract['python']
        assert Path(environment['sympy_file']).resolve().is_relative_to(repo.resolve())
        assert environment['packages'] == dict(spec.split('==') for spec in contract['packages'])
        with (folder/'test.log').open('x') as log:
            process = subprocess.run(['/bin/bash','-c',task['test_command']],cwd=repo,
                env=candidate_environment(venv),stdout=log,stderr=subprocess.STDOUT,timeout=180,preexec_fn=b.limits)
        parsed = c.parser_function()((folder/'test.log').read_text(),None)
        complete = all(parsed.get(name) in {'PASSED','FAILED','ERROR'} for names in tests.values() for name in names)
        result.update(execution_status='tested' if complete else 'missing_test_status', exit_code=process.returncode,
            correctness_observed=complete, resolved=resolved(tests,parsed,process.returncode))
    except subprocess.TimeoutExpired as exc:
        result.update(execution_status='test_timeout' if (folder/'test.log').exists() else 'import_timeout',reason=str(exc))
    except Exception as exc:
        result.update(execution_status='candidate_evaluation_error',reason=f'{type(exc).__name__}: {exc}')
    finally:
        if (folder/'test.log').exists() and not parsed:
            parsed = c.parser_function()((folder/'test.log').read_text(),None)
        result['test_statuses'] = {suite:{name:parsed.get(name,'MISSING' if (folder/'test.log').exists() else 'NOT_RUN')
                                       for name in names} for suite,names in tests.items()}
        result['seconds'] = time.monotonic()-tick
        save(folder/'result.json',result)


def collect():
    protocol = read(OUT/'protocol.json')
    validate_slots(protocol['slots'])
    rows = []
    for slot in protocol['slots']:
        path = OUT/'execution'/slot['episode_id']/'result.json'
        result = read(path) if path.exists() else dict(episode_id=slot['episode_id'],instance_id=slot['instance_id'],
            arm=slot['arm'],resolved=False,correctness_observed=False,
            execution_status='interrupted' if (path.parent/'started.json').exists() else 'unstarted',
            generation_status=slot['generation_status'])
        assert all(result[field] == slot[field] for field in ['episode_id','instance_id','arm'])
        receipt = OUT/'receipts'/f"{slot['episode_id']}.json"
        if receipt.exists():
            outer = read(receipt)
            if outer['timed_out'] or outer['error'] is not None or outer['exit_code'] != 0:
                result['worker_execution_status'] = result['execution_status']
                result['resolved'] = False
                result['execution_status'] = 'outer_interrupted'
                result['correctness_observed'] = False
        elif path.exists() and slot['stage'] == 'ready':
            result.update(resolved=False, correctness_observed=False,
                worker_execution_status=result['execution_status'], execution_status='missing_outer_receipt')
        rows.append(result)
    pairs = []
    for iid in dict.fromkeys(row['instance_id'] for row in rows):
        arms = {r['arm']:r for r in rows if r['instance_id']==iid}
        assert set(arms)=={'base','trained'}
        base, trained = arms['base']['resolved'], arms['trained']['resolved']
        outcome = 'both_solved' if base and trained else 'trained_win' if trained else 'base_win' if base else 'neither_solved'
        pairs.append(dict(instance_id=iid, outcome=outcome, arms=arms))
    save(OUT/'summary.json',dict(assigned_tasks=20,assigned_episodes=40,episodes=rows,pairs=pairs,
        resolved_counts={arm:sum(r['arm']==arm and r['resolved'] for r in rows) for arm in ['base','trained']},
        paired_counts={name:sum(p['outcome']==name for p in pairs) for name in ['trained_win','base_win','both_solved','neither_solved']},
        net_solve_difference=sum(r['resolved']*(1 if r['arm']=='trained' else -1) for r in rows),
        denominator=20,development_only=True,diversity_effect_tested=False,no_feedback_to_generation=True))
    save(OUT/'completed_manifest.json',{str(path):sha(path) for path in OUT.rglob('*')
        if path.is_file() and 'repo' not in path.relative_to(OUT).parts})


def checks():
    import tempfile
    tests = {'FAIL_TO_PASS':['target'],'PASS_TO_PASS':['regression']}
    cases = [resolved(tests,{'target':'PASSED','regression':'PASSED'},0),
        not resolved(tests,{'target':'PASSED','regression':'PASSED'},1),
        not resolved(tests,{'target':'PASSED'},0),
        not resolved(tests,{'target':'FAILED','regression':'PASSED'},0),
        not resolved({'FAIL_TO_PASS':[],'PASS_TO_PASS':['regression']},{'regression':'PASSED'},0),
        bool(c.new_risks('x=1\n','import os\nos.system("echo x")\n')),
        not c.new_risks('x=1\n','x=2\n'),
        candidate_environment(Path('/fake/venv'))['PYTHONDONTWRITEBYTECODE'] == '1',
        candidate_environment(Path('/fake/venv'))['VIRTUAL_ENV'] == '/fake/venv']
    assert all(cases)
    global OUT
    original_out = OUT
    with tempfile.TemporaryDirectory() as tmp:
        OUT = Path(tmp)
        slots = [dict(episode_id=f'{i}_{arm}',instance_id=str(i),arm=arm,generation_status='final_patch',stage='ready')
                 for i in range(20) for arm in ['base','trained']]
        save(OUT/'protocol.json',dict(slots=slots))
        for row in slots:
            i = int(row['instance_id'])
            solved = (i<3 and row['arm']=='trained') or (i==3 and row['arm']=='base') or i in (4,5)
            if i<6:
                save(OUT/'execution'/row['episode_id']/'result.json',dict(**row,resolved=solved,
                     correctness_observed=True,execution_status='tested'))
                save(OUT/'receipts'/f"{row['episode_id']}.json",dict(timed_out=False,error=None,exit_code=0))
        collect()
        summary = read(OUT/'summary.json')
        assert summary['resolved_counts']==dict(base=3,trained=5)
        assert summary['paired_counts']==dict(trained_win=3,base_win=1,both_solved=2,neither_solved=14)
        assert summary['net_solve_difference']==2 and len(summary['episodes'])==40
        assert sum(r['execution_status']=='unstarted' for r in summary['episodes'])==28
        cases.append(True)
    OUT = original_out
    save(OUT/'synthetic_checks.json',dict(all_checks_passed=True,checks=len(cases),source_sha256=sha(Path(__file__)),
         no_model_or_actual_candidate_read=True,no_tests_executed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','score-one','collect','checks'])
    parser.add_argument('--episode-id')
    args = parser.parse_args()
    score_one(args.episode_id) if args.stage=='score-one' else globals()[args.stage]()
