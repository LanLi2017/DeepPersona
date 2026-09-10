#!/usr/bin/env python3
"""Frozen source-only target feasibility after all public prompts are immutable."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/swe-diversity-selection'
OUT = BASE/'swe-sympy-patch-sft'
DEST = OUT/'source_targets'
DATA = BASE/'swe-sympy-learning-metadata/full_test.parquet'
TECHNICAL_CODES = {'converter_roundtrip_mismatch','target_roundtrip_mismatch','invalid_visibility_intervals','missing_single_eos_token','nonunique_original_anchor','nonunique_sequential_anchor','invalid_edit_schema','invalid_edit_strings','unsupported_edit_path','target_not_canonical_converter_output','base_must_be_mapping'}
MODEL = Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
spec = importlib.util.spec_from_file_location('targets110', ROOT/'scripts/110_swe_patch_sft_targets.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    with path.open('x') as f: f.write(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def verify_context():
    protocol=read(OUT/'context/protocol.json')
    for name,digest in protocol['source_sha256'].items(): assert sha(name)==digest,name
    amendment=read(OUT/'context/parallel_execution_amendment.json')
    for name,digest in amendment['source_sha256'].items(): assert sha(name)==digest,name
    assert read(OUT/'context/parallel_parity.json')['all_checks_passed']
    assert read(OUT/'context/parallel_completed.json')['manifest_sha256']==sha(OUT/'context/manifest.json')
    selection=read(OUT/'selection.json')
    manifest=read(OUT/'context/manifest.json')
    expected={(t['instance_id'],t['role']) for t in selection['tasks']}
    assert len(manifest['tasks'])==52 and {(t['instance_id'],t['role']) for t in manifest['tasks']}==expected
    assert manifest['selection_sha256']==sha(OUT/'selection.json') and manifest['private_inputs_read'] is False
    for task in manifest['tasks']:
        iid=task['instance_id'];path=OUT/'context'/(iid+'.json');record=read(path)
        assert sha(path)==task['record_sha256']
        assert (record['instance_id'],record['role'])==(iid,task['role'])
        assert record['protocol_sha256']==sha(OUT/'context/protocol.json')
        if record['status']=='ready':
            assert sha(OUT/'context'/(iid+'.prompt.txt'))==record['prompt_sha256']
            assert sha(OUT/'context'/(iid+'.packet.txt'))==record['packet_sha256']
        else:
            assert record.get('error')=='Issue alone exceeds packet budget', 'Unexpected context error is technical, not scientific exclusion: '+iid


def freeze():
    verify_context()
    context = read(OUT/'context/manifest.json')
    selection = read(OUT/'selection.json')
    checks=read(OUT/'target_converter_checks/final/summary.json')
    assert checks['passed'] and checks['real_data_accessed'] is False
    assert checks['source_sha256']==sha(ROOT/'scripts/110_swe_patch_sft_targets.py')
    assert context['completed_slots'] == context['assigned_slots'] == len(selection['tasks']) == 52
    assert context['selection_sha256'] == sha(OUT/'selection.json')
    for task in context['tasks']:
        assert sha(OUT/'context'/(task['instance_id']+'.json')) == task['record_sha256']
    inputs = [Path(__file__).resolve(),ROOT/'scripts/110_swe_patch_sft_targets.py',
        OUT/'selection.json',OUT/'selection_protocol.json',OUT/'export_manifest.json',
        OUT/'context/protocol.json',OUT/'context/manifest.json',
        OUT/'target_converter_checks/final/summary.json',DATA,
        OUT/'context/parallel_execution_amendment.json',OUT/'context/parallel_parity.json',OUT/'context/parallel_completed.json']
    inputs += list((OUT/'context').glob('sympy__*.json'))
    inputs += list((OUT/'context').glob('sympy__*.prompt.txt'))
    inputs += list((OUT/'context').glob('sympy__*.packet.txt'))
    inputs += list((OUT/'public').glob('*/task.json'))
    DEST.mkdir()
    save(DEST/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=sha(OUT/'selection.json'),context_manifest_sha256=sha(OUT/'context/manifest.json'),
        source_ids=[t['instance_id'] for t in selection['tasks'] if t['role']=='source'],
        projection=['instance_id','patch'],evaluation_private_columns_materialized=False,
        source_patch_access='After all52 public prompt records are frozen. Only selected source IDs; smoke firstfixed source then remaining31, no task replacement.',
        source_contract='110canonical exact sequential edits; wholeanchor visible in frozen intervals; <=2048target tokens includingEOS; independent gitapply byte reconstruction for supported targets.',
        early_stop='If fewer than16 of32 targets satisfy conversion, visibility and completionbudget, requiredadmission is impossible even if everyqualitycheckpasses. Close fixedcontrol as datafeasibilityfailure; do not run unnecessaryqualitytests or open evaluationgold. This is not a learningnull result.',
        quality='Not measured by this stage. A prequalified target still needs baselinefail/goldpass and requiredregressions before training.',
        model_tokenizer=str(MODEL),inputs_sha256={str(p):sha(p) for p in inputs},api_spend_usd=0,gpu_used=False))
    print('Source-only target access protocol frozen',flush=True)


def validate():
    verify_context()
    p=read(DEST/'protocol.json')
    for path,digest in p['inputs_sha256'].items(): assert sha(path)==digest,path
    return p


def independent_git(base,patch,edits):
    changed=sorted({e['path'] for e in edits})
    expected=c.apply_edits(base,edits)
    with tempfile.TemporaryDirectory(prefix='swe-sft-target111-') as temp:
        root=Path(temp)
        for name in changed:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base[name])
        path=root/'target.patch';path.write_bytes(patch)
        env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
        for check in [True,False]:
            cmd=['git','apply','--unidiff-zero','--whitespace=nowarn']+(['--check'] if check else [])+[str(path)]
            r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,timeout=30)
            assert r.returncode==0,r.stderr.decode()[-2000:]
        assert all((root/name).read_bytes()==expected[name] for name in changed),'git_reconstruction_mismatch'
    return {name:hashlib.sha256(expected[name]).hexdigest() for name in changed}


def one(iid, patch, tok):
    folder=DEST/iid;folder.mkdir()
    (folder/'gold.patch').write_bytes(patch)
    record=dict(instance_id=iid,role='source',patch_sha256=sha(folder/'gold.patch'),
        conversion_supported=False,anchor_visible=None,completion_fits=None,independent_git_reconstruction=None,
        prequalified=False,quality_verified=False,errors=[])
    try:
        public=OUT/'public'/iid;meta=read(public/'task.json')
        base={name:(public/'source'/name).read_bytes() for name in meta['source_sha256']}
        assert all(hashlib.sha256(blob).hexdigest()==meta['source_sha256'][name] for name,blob in base.items())
        context=read(OUT/'context'/(iid+'.json'))
        edits=c.convert(base,patch)
        record.update(conversion_supported=True,edits=edits,edit_count=len(edits))
        record['patched_sha256']=independent_git(base,patch,edits)
        record['independent_git_reconstruction']=True
        serialized=c.serialize_target(edits,tok)
        record.update(serialized)
        record['completion_fits']=serialized['target_tokens_including_eos']<=2048
        if not record['completion_fits']: record['errors'].append('completion_token_budget_exceeded')
        if context['status']!='ready':
            record['anchor_visible']=False;record['errors'].append('context_infeasible')
        else:
            visible=defaultdict(list)
            for span in context['retrieval']['source_spans']:
                visible[span['path']].append((span['start_line'],span['end_line']))
            try:
                record['anchors']=c.validate_visibility(base,edits,visible)
                record['anchor_visible']=True
            except c.TargetError as error:
                if str(error)!='anchor_not_wholly_visible': raise
                record['anchor_visible']=False;record['errors'].append(str(error))
            record['prompt_tokens']=context['prompt_tokens']
        record['prequalified']=bool(record['anchor_visible'] and record['completion_fits'])
    except c.TargetError as error:
        record['errors'].append(str(error))
        if str(error) in TECHNICAL_CODES:
            record['technical_failure']=dict(type=type(error).__name__,message=str(error))
            record['errors'].append('technical_failure')
    except Exception as error:
        record['technical_failure']=dict(type=type(error).__name__,message=str(error))
        record['errors'].append('technical_failure')
    save(folder/'record.json',record)
    print(json.dumps({k:record[k] for k in ['instance_id','conversion_supported','anchor_visible','completion_fits','prequalified','errors']}),flush=True)
    return record


def screen(smoke):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    p=validate(); ids=p['source_ids'];assert len(ids)==32
    assert not (DEST/'summary.json').exists()
    if not smoke: assert read(DEST/'smoke.json')['technical_pass']
    selected=ids[:1] if smoke else ids[1:]
    rows=pq.read_table(DATA,columns=p['projection'],filters=[('instance_id','in',selected)]).to_pylist()
    assert {r['instance_id'] for r in rows}==set(selected)
    patches={r['instance_id']:r['patch'].encode('utf-8') for r in rows}
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    for iid in selected: one(iid,patches[iid],tok)
    if smoke:
        record=read(DEST/ids[0]/'record.json')
        save(DEST/'smoke.json',dict(instance_id=ids[0],record_sha256=sha(DEST/ids[0]/'record.json'),
            technical_pass='technical_failure' not in record,source_target_eligibility_is_not_smoke_gate=True))
        assert 'technical_failure' not in record
    else:
        records=[read(DEST/iid/'record.json') for iid in ids]
        count=sum(r['prequalified'] for r in records)
        technical=sum('technical_failure' in r for r in records)
        summary=dict(assigned_source_slots=32,conversion_supported=sum(r['conversion_supported'] for r in records),
            anchor_visible=sum(r['anchor_visible'] is True for r in records),completion_fits=sum(r['completion_fits'] is True for r in records),
            prequalified=count,minimum_required=16,technical_failures=technical,prequalification_gate_passed=count>=16 and technical==0,
            quality_tests_run=0,admitted_training_targets=0,trained=False,evaluation_private_columns_materialized=False,
            errors=dict(Counter(e for r in records for e in r['errors'])),
            decision=('Resolve technical failures before scientific interpretation' if technical else ('Proceed to frozen baseline/gold quality and evaluation runtime preparation' if count>=16 else 'Fixed control fails data feasibility; no training or learning-null claim; no replacements or context expansion')),
            tasks=[{k:r.get(k) for k in ['instance_id','conversion_supported','anchor_visible','completion_fits','prequalified','errors']} for r in records],
            api_spend_usd=0,gpu_used=False)
        save(DEST/'summary.json',summary)
        files=[DEST/'protocol.json',DEST/'summary.json',DEST/'smoke.json']+list(DEST.glob('sympy__*/*'))
        save(DEST/'completed_manifest.json',{str(f):sha(f) for f in files})
        print(json.dumps({k:v for k,v in summary.items() if k!='tasks'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','smoke','screen']);a=parser.parse_args()
    if a.stage=='freeze': freeze()
    else: screen(a.stage=='smoke')
