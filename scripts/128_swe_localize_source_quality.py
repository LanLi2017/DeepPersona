#!/usr/bin/env python3
"""Conditional source-only quality checks after audited localize/read support and real learner smoke.

Code preparation only does not acquire assets or open runtime/evaluation labels. CLI
freeze requires completed source-support and learner-smoke evidence; runtime stages
inherit the unchanged119/112/90 host bridge and explicit operation reviews.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/swe-diversity-selection'
POP=BASE/'swe-sympy-patch-sft'
STUDY=BASE/'swe-localize-repair-support'
SMOKE=STUDY/'learner_smoke'
OUT=STUDY/'source_runtime'
REUSE=ROOT/'scripts/119_swe_retrieval_policy_quality.py'
SPEC=importlib.util.spec_from_file_location('localize_source_quality119',REUSE)
q=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)
q.STUDY=STUDY
q.SMOKE=SMOKE
q.OUT=OUT
q.w.OUT=OUT
q.w.m.OUT=OUT
#112's child launcher must call this wrapper with the new OUT, not119 or112.
q.w.__file__=str(Path(__file__).resolve())
sha,read,save,utc=q.sha,q.read,q.save,q.utc


def verify(mapping):
    for name,value in mapping.items():
        assert sha(ROOT/name)==value,name


def freeze():
    assert not OUT.exists()
    support=read(STUDY/'support_summary.json')
    protocol=read(STUDY/'protocol.json')
    audit=read(STUDY/'independent_support_audit.json')
    assert audit['all_checks_passed'] is True
    for field,name in [('support_summary_sha256','support_summary.json'),
                       ('completed_manifest_sha256','completed_manifest.json'),
                       ('initial_manifest_sha256','initial_manifest.json')]:
        assert audit[field]==sha(STUDY/name),field
    audit_started=read(STUDY/'independent_support_audit.started.json')
    audit_terminal=read(STUDY/'independent_support_audit.terminal.json')
    assert audit['source_tasks_audited']==32 and audit['prequalified']==23
    assert audit['summary_checks'] and all(audit['summary_checks'].values())
    assert audit_started['protocol_sha256']==audit['protocol_sha256']==sha(STUDY/'protocol.json')
    assert audit_started['completed_manifest_sha256']==sha(STUDY/'completed_manifest.json')
    assert audit_started['initial_manifest_sha256']==sha(STUDY/'initial_manifest.json')
    assert audit_started['driver_sha256']==audit['driver_sha256']==sha(ROOT/'scripts/audit_swe_localize_support.py')
    assert audit_terminal['exit_code']==0 and audit_terminal['failure'] is None
    assert audit_terminal['tasks_returned']==32 and audit_terminal['report_exists'] is True
    assert support['support_gate_passed'] is True and support['minimum_required']==16
    assert support['source_tasks']==32 and support['prequalified']==23 and support['technical_failures']==0
    assert support['quality_tests_run']==0 and support['gpu_used'] is False
    assert support['evaluation_private_values_read'] is False
    assert support['learner_benefit_established'] is False and support['diversity_benefit_established'] is False
    population=read(POP/'selection.json')
    sources=[t for t in population['tasks'] if t['role']=='source']
    ids_all=[t['instance_id'] for t in sources]
    assert len(ids_all)==len(set(ids_all))==32 and protocol['source_tasks']==sources
    assert [r['instance_id'] for r in support['tasks']]==ids_all
    assert all(r['status']=='complete' for r in support['tasks'])
    screened={r['instance_id']:r for r in support['tasks']}
    active=[t for t in sources if screened[t['instance_id']]['prequalified']]
    ids=[t['instance_id'] for t in active]
    assert len(ids)==23
    assert protocol['limits']==dict(initial_prompt=12288,every_generation_prefix=24576,
        generation_reservation=2048,causal_row=26624,final_patch_including_eos=2048,
        action_including_eos=256,total_outline_read_calls=7,tile_LF_lines=160,
        observation_raw_tokens=4096,observation_UTF8_bytes=16384)
    initial_manifest=read(STUDY/'initial_manifest.json')
    assert initial_manifest['all32_public_initials_frozen'] is True
    assert initial_manifest['source_label_records_opened'] is False
    assert [r['instance_id'] for r in initial_manifest['tasks']]==ids_all
    support_completed=read(STUDY/'completed_manifest.json')
    old_targets=read(POP/'source_targets/completed_manifest.json')
    for iid in ids_all:
        demonstration=STUDY/'demonstrations'/(iid+'.json')
        assert support_completed[str(demonstration)]==sha(demonstration)
        record=read(demonstration)
        assert record['instance_id']==iid and record['role']=='source' and record['status']=='complete'
        assert record['prequalified']==screened[iid]['prequalified']
        target=POP/'source_targets'/iid/'record.json'
        assert record['source_record_sha256']==old_targets[str(target)]==sha(target)
        assert record['initial_sha256']==sha(STUDY/'initials'/(iid+'.json'))
        assert record['protocol_sha256']==sha(STUDY/'protocol.json')
        assert record['initial_manifest_sha256']==sha(STUDY/'initial_manifest.json')
        if iid in ids:
            assert all(record[k] is True for k in ['conversion_supported','initial_index_fits',
                'final_patch_fits','action_count_fits','observations_fit','history_fits',
                'anchor_visible','replay_verified','prior_independent_reconstruction_verified'])
            assert record['quality_verified'] is False and not record['errors']
    technical=read(SMOKE/'run/summary.json')
    technical_protocol=read(SMOKE/'protocol.json')
    outer=read(SMOKE/'outer_terminal.json')
    started=read(SMOKE/'run/started.json')
    smoke_completed=read(SMOKE/'run/completed_manifest.json')
    assert technical['technical_checks_passed'] is True
    assert technical['quality_verified'] is False and technical['full_training_run'] is False
    assert technical['inference_run'] is False and technical['evaluation_private_access'] is False
    assert started['protocol_sha256']==sha(SMOKE/'protocol.json')
    normalized={str(ROOT/name):digest for name,digest in smoke_completed.items()}
    for path in [SMOKE/'run/started.json',SMOKE/'run/summary.json']:
        assert normalized[str(path)]==sha(path)
    assert technical_protocol['prequalified_source_ids']==ids
    assert technical_protocol['interface']=='public_outline_read_then_canonical_patch'
    assert technical_protocol['initial_prompt_cap']==12288
    assert technical_protocol['max_prompt_tokens']==24576 and technical_protocol['max_action_tokens']==256
    assert technical_protocol['max_completion_tokens']==2048 and technical_protocol['max_sequence_tokens']==26624
    assert technical_protocol['total_shared_gpu_budget_seconds']==3600 and technical_protocol['prior_gpu_seconds']==0
    assert technical['future_max_dose_fits_internal_budget'] is True
    assert outer['protocol_sha256']==sha(SMOKE/'protocol.json')
    assert outer['exit_code']==0 and outer['timed_out'] is False and outer['error'] is None
    assert outer['hard_limit_seconds']==3600 and outer['soft_limit_seconds']==3595
    internal,external=technical['total_elapsed_seconds'],outer['elapsed_seconds']
    estimate=technical['future_max_prequalified_dose_estimate_seconds']
    assert all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in [internal,external,estimate])
    charged=max(internal,external)
    assert charged<=3600 and estimate<=3600-charged,'Future full source dose exceeds externally charged shared GPU budget'
    model_root=Path(technical_protocol['model'])
    nonmodel={name:digest for name,digest in technical_protocol['inputs_sha256'].items()
              if not (ROOT/name).is_relative_to(model_root)}
    # The executed smoke must bind this exact completed support corpus, not an older interface.
    normalized_inputs={str(ROOT/name):digest for name,digest in nonmodel.items()}
    for path in [STUDY/'protocol.json',STUDY/'support_summary.json',STUDY/'completed_manifest.json',
                 STUDY/'independent_support_audit.json']:
        assert normalized_inputs[str(path)]==sha(path),path
    maps=[protocol['source_sha256'],initial_manifest['source_sha256'],support_completed,
          old_targets,nonmodel,smoke_completed]
    for mapping in maps:verify(mapping)
    exports=read(POP/'export_manifest.json')
    assert exports['selection_sha256']==sha(POP/'selection.json')
    exported={t['instance_id']:t for t in exports['tasks']}
    assert all(exported[i]['ready'] for i in ids_all)
    paths=[Path(__file__).resolve(),REUSE,ROOT/'scripts/112_swe_patch_sft_runtime.py',
        ROOT/'scripts/90_swe_fresh_development.py',ROOT/'scripts/85_swe_execution_bridge.py',
        ROOT/'scripts/125_swe_localize_repair_support.py',ROOT/'scripts/126_swe_localize_repair_study.py',
        ROOT/'scripts/127_swe_localize_repair_smoke.py',ROOT/'scripts/129_swe_localize_smoke_launcher.py',
        q.w.m.b.PARSER,q.w.m.b.DATA,POP/'selection.json',POP/'selection_protocol.json',POP/'export_manifest.json',
        STUDY/'protocol.json',STUDY/'initial_manifest.json',STUDY/'support_summary.json',
        STUDY/'completed_manifest.json',STUDY/'independent_support_audit.json',
        STUDY/'independent_support_audit.started.json',STUDY/'independent_support_audit.terminal.json',
        ROOT/'scripts/audit_swe_localize_support.py',
        SMOKE/'protocol.json',SMOKE/'run/summary.json',SMOKE/'run/started.json',
        SMOKE/'run/completed_manifest.json',SMOKE/'outer_terminal.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for mapping in maps:
        for name,digest in mapping.items():
            path=ROOT/name;key=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            assert key not in hashes or hashes[key]==digest
            hashes[key]=digest
    OUT.mkdir()
    (OUT/'private').mkdir()
    (OUT/'private/base').symlink_to(POP/'private/base',target_is_directory=True)
    (OUT/'public').symlink_to(POP/'public',target_is_directory=True)
    shutil.copyfile(POP/'export_manifest.json',OUT/'export_manifest.json')
    save(OUT/'selection.json',dict(utc=utc(),tasks=active,all_slots=sources,
        source_prequalification=support['tasks'],source_prequalified_ids=ids,evaluation_ids=[],
        smoke_ids=ids[:1],remaining_ids=ids[1:],chosen_policy='constructed_outline_read_repair',chosen_prompt_cap=24576,
        source_interface=technical_protocol['interface'],
        dataset_revision=population['dataset_revision'],harness_revision=population['harness_revision'],
        task_repo_revision=population['task_repo_revision'],source_sha256=hashes,
        export_manifest_sha256=sha(OUT/'export_manifest.json'),
        export_manifest_original=str((POP/'export_manifest.json').relative_to(ROOT)),
        gpu_accounting=dict(shared_limit_seconds=3600,technical_internal_seconds=internal,
            technical_external_seconds=external,charged_prior_seconds=charged,remaining_seconds=3600-charged,
            future_max_source_dose_estimate_seconds=estimate),
        scope='Source-only host baseline/gold quality: fixed32 source slots,23 supported active sources. No evaluation assets or model calls.',
        admission='Only frozen126 complete supported SOURCE IDs acquire assets; no replacement or evaluation admission.',
        private_access='Inherited112/90 selected-source preparation only; source-gold hash must match original111 gold.patch.',
        asset_guard='Inherited112 cached URL getter admits only ready rows after exact command-contract/hash review; failed rows cannot execute.',
        limits=dict(per_test_wall_seconds=180,per_test_cpu_seconds=120,per_test_memory_bytes=4*1024**3,
            aggregate_outer_test_seconds=7200,outer_termination_grace_seconds=5),
        stages='freeze -> assets -> root-reviewed exact preparation -> firstsource smoke -> reviewed remaining -> aggregate',
        no_retry=True,api_spend_usd=0,gpu_used=False))
    print(json.dumps(dict(frozen=True,active_sources=len(ids),retained_sources=32)),flush=True)


def selection():
    chosen=q.selection()
    assert chosen['chosen_policy']=='constructed_outline_read_repair'
    assert len(chosen['all_slots'])==32 and len(chosen['source_prequalified_ids'])==23
    assert q.w.__file__==str(Path(__file__).resolve())
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','assets','prepare','smoke','run','aggregate','_run-group'])
    parser.add_argument('--review');parser.add_argument('--group',choices=['smoke','remaining'])
    args=parser.parse_args()
    if args.stage=='freeze':freeze();return
    selection()
    if args.stage=='prepare':q.prepare(args.review)
    elif args.stage=='aggregate':q.aggregate()
    elif args.stage=='assets':q.w.assets()
    elif args.stage=='_run-group':q.w.child_group(args.group)
    else:q.w.run_group('smoke' if args.stage=='smoke' else 'remaining',args.review)

if __name__=='__main__':main()
