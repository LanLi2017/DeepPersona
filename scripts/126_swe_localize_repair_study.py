#!/usr/bin/env python3
"""Frozen public-initial barrier then source-only constructed trajectory support audit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/swe-diversity-selection/swe-sympy-patch-sft'
OUT = OLD.parent / 'swe-localize-repair-support'
KERNEL = ROOT / 'scripts/125_swe_localize_repair_support.py'
WORKERS = 8
REPORTS = ['kernel_synthetic_checks.json', 'independent_native_checks.json', 'independent_boundary_checks.json']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    global kernel, tok
    spec = importlib.util.spec_from_file_location('localize125', KERNEL)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    tok = kernel.tokenizer()


def verify(hashes):
    for path, value in hashes.items():
        assert sha(path) == value, path


def freeze():
    assert {p.name for p in OUT.iterdir()} == set(REPORTS + ['static_review.json'])
    review = read(OUT / 'static_review.json')
    assert review['all_checks_passed']
    verify(review['source_sha256'])
    for name in REPORTS:
        report = read(OUT / name)
        assert report['all_checks_passed'] and report['kernel_sha256'] == sha(KERNEL), name
    milestone = ROOT / 'runs/swe-diversity-selection/paper-program/milestone_eleven.json'
    prior = read(milestone)
    verify(prior['evidence_sha256'])
    assert prior['static_retrieval_branch_closed'] and prior['chosen_policy'] is None
    tasks = [r for r in read(OLD / 'selection.json')['tasks'] if r['role'] == 'source']
    assert len(tasks) == len({r['instance_id'] for r in tasks}) == 32
    initialize()
    assert (kernel.INITIAL_LIMIT, kernel.PROMPT_LIMIT, kernel.HISTORY_LIMIT, kernel.FINAL_LIMIT,
            kernel.ACTION_LIMIT, kernel.MAX_ACTIONS, kernel.READ_LINES,
            kernel.OBSERVATION_TOKENS, kernel.OBSERVATION_BYTES) == (12288,24576,26624,2048,256,7,160,4096,16384)
    paths = [Path(__file__).resolve(), KERNEL, ROOT / 'scripts/110_swe_patch_sft_targets.py',
             ROOT / 'scripts/audit_swe_localize_native.py', ROOT / 'scripts/audit_swe_localize_boundary.py',
             OLD / 'selection.json', OLD / 'export_manifest.json',
             OLD / 'source_targets/completed_manifest.json', milestone]
    paths += [OUT / name for name in REPORTS + ['static_review.json']]
    paths += [kernel.MODEL / n for n in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']
              if (kernel.MODEL / n).exists()]
    for task in tasks:
        paths += [OLD / 'public' / task['instance_id'] / n for n in ['task.json', 'problem_statement.md']]
    save(OUT / 'protocol.json', dict(utc=utc(), source_tasks=tasks, workers=WORKERS,
        source_sha256={str(p):sha(p) for p in paths}, model=str(kernel.MODEL),
        system=kernel.SYSTEM, tools=kernel.TOOLS,
        limits=dict(initial_prompt=12288, every_generation_prefix=24576, generation_reservation=2048,
            causal_row=26624, final_patch_including_eos=2048, action_including_eos=256,
            total_outline_read_calls=7, tile_LF_lines=160, observation_raw_tokens=4096,
            observation_UTF8_bytes=16384),
        minimum_prequalified_sources=16,
        stages='Freeze; first public initial smoke; remaining31 public initials; immutable all32 manifest; first source construction smoke; remaining31 source constructions; summary. No source record content before initial manifest.',
        labels='Known source maintainer patches choose lexical changed-file outlines then ascending exact public tiles covering unchanged canonical110/111 old anchors. Initials have complete issue and all allowed filepaths only.',
        objective='Separate causal row per current assistant turn, exact native prefix fully masked, current action/final through EOS supervised. Future training proposed task mean of turn mean losses; no training in this audit.',
        qualification='>=16 complete valid demonstrations and zero technical errors permits considering a new GPU smoke and runtime source quality protocol. No replacement, cap changes, rescue reads or static policy requalification.',
        interpretation='Constructed maintainer-supervision feasibility on previously exposed32training sources. Not observed debugging traces, autonomous repair, learner benefit or diversity evidence.',
        evaluation_private_values_read=False, api_spend_usd=0, gpu_used=False,
        versions={n:importlib.metadata.version(n) for n in ['transformers','tokenizers']}))
    (OUT / 'initials').mkdir()
    (OUT / 'demonstrations').mkdir()
    print('Protocol frozen', sha(OUT / 'protocol.json'), flush=True)


def validate():
    protocol = read(OUT / 'protocol.json')
    verify(protocol['source_sha256'])
    return protocol


def initial_one(task):
    iid = task['instance_id']
    path = OUT / 'initials' / (iid + '.json')
    assert not path.exists()
    start = time.monotonic()
    try:
        public = OLD / 'public' / iid
        meta = read(public / 'task.json')
        assert meta['role'] == 'source' and meta['base_commit'] == task['base_commit']
        assert sha(public / 'problem_statement.md') == meta['problem_statement_sha256']
        record = kernel.initial(public, tok)
        save(path, record)
        return dict(instance_id=iid, status='ready', prompt_tokens=record['prompt_tokens'],
            initial_index_fits=record['initial_index_fits'], path=str(path), sha256=sha(path),
            elapsed_seconds=time.monotonic()-start)
    except Exception as error:
        record = dict(instance_id=iid, status='technical_failure', error_type=type(error).__name__, error=str(error))
        if not path.exists(): save(path, record)
        return record


def parallel(function, tasks):
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(function, task):task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row), flush=True)
    return results


def initials():
    protocol = validate()
    save(OUT / 'initials_started.json', dict(utc=utc(), protocol_sha256=sha(OUT / 'protocol.json')))
    initialize()
    first = initial_one(protocol['source_tasks'][0])
    save(OUT / 'initial_smoke.json', first)
    print(json.dumps(first), flush=True)
    assert first['status'] == 'ready', 'First public initial technical failure'
    rows = [first] + parallel(initial_one, protocol['source_tasks'][1:])
    order = {t['instance_id']:i for i,t in enumerate(protocol['source_tasks'])}
    rows.sort(key=lambda r:order[r['instance_id']])
    assert len(rows) == 32 and all(r['status'] == 'ready' for r in rows)
    save(OUT / 'initial_manifest.json', dict(utc=utc(), protocol_sha256=sha(OUT / 'protocol.json'),
        tasks=rows, source_sha256={r['path']:r['sha256'] for r in rows}, all32_public_initials_frozen=True,
        source_label_records_opened=False, evaluation_private_values_read=False))
    print('All32 public initials frozen', flush=True)


def source_barrier():
    protocol = validate()
    manifest = read(OUT / 'initial_manifest.json')
    assert manifest['all32_public_initials_frozen'] and manifest['source_label_records_opened'] is False
    assert manifest['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert [r['instance_id'] for r in manifest['tasks']] == [r['instance_id'] for r in protocol['source_tasks']]
    verify(manifest['source_sha256'])
    # Read only hashes here. Source record content is opened below, after the barrier.
    verify(read(OLD / 'source_targets/completed_manifest.json'))
    return protocol


def construct_one(task):
    iid = task['instance_id']
    path = OUT / 'demonstrations' / (iid + '.json')
    assert not path.exists()
    start = time.monotonic()
    source = OLD / 'source_targets' / iid / 'record.json'
    initial = OUT / 'initials' / (iid + '.json')
    try:
        result = kernel.construct(OLD / 'public' / iid, read(source), read(initial), tok)
        result.update(status='complete', source_record_sha256=sha(source), initial_sha256=sha(initial),
            protocol_sha256=sha(OUT / 'protocol.json'), initial_manifest_sha256=sha(OUT / 'initial_manifest.json'))
    except Exception as error:
        result = dict(instance_id=iid, status='technical_failure', prequalified=False,
            error_type=type(error).__name__, error=str(error))
    result['elapsed_seconds'] = time.monotonic()-start
    save(path, result)
    return {k:result.get(k) for k in ['instance_id','status','prequalified','required_actions','required_reads','errors','error','elapsed_seconds']}


def smoke():
    protocol = source_barrier()
    initialize()
    result = construct_one(protocol['source_tasks'][0])
    save(OUT / 'source_smoke.json', dict(utc=utc(), result=result,
        technical_pass=result['status']=='complete',
        record_sha256={str(OUT / 'demonstrations' / (result['instance_id']+'.json')):
            sha(OUT / 'demonstrations' / (result['instance_id']+'.json'))}))
    print(json.dumps(result), flush=True)
    assert result['status'] == 'complete', 'Source smoke technical failure; no replacement'


def construct():
    protocol = source_barrier()
    smoke_record = read(OUT / 'source_smoke.json')
    assert smoke_record['technical_pass']
    verify(smoke_record['record_sha256'])
    save(OUT / 'construction_started.json', dict(utc=utc(), jobs=31, workers=WORKERS,
        protocol_sha256=sha(OUT / 'protocol.json'), initial_manifest_sha256=sha(OUT / 'initial_manifest.json')))
    parallel(construct_one, protocol['source_tasks'][1:])
    paths = [OUT / 'demonstrations' / (t['instance_id']+'.json') for t in protocol['source_tasks']]
    rows = [read(p) for p in paths]
    failures = sum(r['status']!='complete' for r in rows)
    support = sum(r['prequalified'] for r in rows)
    flags = ['conversion_supported','initial_index_fits','final_patch_fits','action_count_fits',
             'observations_fit','history_fits','anchor_visible','replay_verified']
    summary = dict(utc=utc(), source_tasks=32, technical_failures=failures, prequalified=support,
        minimum_required=16, support_gate_passed=support>=16 and failures==0,
        marginal_flags={k:sum(bool(r.get(k)) for r in rows) for k in flags},
        flags_note='Later flags are unassessed/false after early rejection; not independent marginal ceilings.',
        rejection_counts=dict(Counter(e for r in rows for e in r.get('errors',[]))),
        first_rejection_counts=dict(Counter(r.get('errors',['technical_failure'])[0] for r in rows if not r['prequalified'])),
        tasks=[{k:r.get(k) for k in ['instance_id','status','prequalified','required_actions','required_reads',
            'max_sequence_tokens','supervised_tokens','errors']} for r in rows],
        quality_tests_run=0, gpu_used=False, api_spend_usd=0, evaluation_private_values_read=False,
        learner_benefit_established=False, diversity_benefit_established=False)
    save(OUT / 'support_summary.json', summary)
    evidence = paths + [OUT/n for n in ['protocol.json','initial_manifest.json','initial_smoke.json',
        'source_smoke.json','initials_started.json','construction_started.json','support_summary.json']]
    save(OUT / 'completed_manifest.json', {str(p):sha(p) for p in evidence})
    print(json.dumps({k:v for k,v in summary.items() if k!='tasks'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze','initials','smoke','construct'])
    globals()[parser.parse_args().stage]()
