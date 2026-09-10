#!/usr/bin/env python3
"""Two metadata-selected SymPy1.11 baseline/gold host compatibility checks."""
import argparse
from datetime import datetime,timezone
import importlib.util
import io
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/swe-diversity-selection'
OUT=BASE/'swe-sympy-compatibility'
META=BASE/'swe-sympy-learning-metadata'
spec=importlib.util.spec_from_file_location('compat90',ROOT/'scripts/90_swe_fresh_development.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=OUT;m.b.DATA=META/'full_test.parquet'
sha,read,save=m.sha,m.read,m.save
ASSETS=['task.yaml','tests.json','gold.patch','test.patch','eval.sh','Dockerfile']

def freeze():
    assert not OUT.exists();OUT.mkdir()
    pair=read(META/'compatibility_canary_selection.json');rule=META/'compatibility_canary_selection_protocol.json'
    assert pair['selection_protocol_sha256']==sha(rule) and pair['version']=='1.11'
    assert pair['source_family']!=pair['evaluation_family']
    for name,digest in read(rule)['inputs_sha256'].items():assert sha(ROOT/name)==digest,name
    prior=read(BASE/'swe-fresh-development/selection.json')
    tasks=[dict(**pair[key],canary_role=role) for key,role in [('source_canary','source'),('evaluation_canary','evaluation')]]
    files=[Path(__file__).resolve(),ROOT/'scripts/90_swe_fresh_development.py',ROOT/'scripts/85_swe_execution_bridge.py',META/'protocol.json',META/'completed_manifest.json',META/'compatibility_canary_selection.json',rule,m.b.DATA,BASE/'swe-fresh-development/selection.json']
    selection=dict(utc=datetime.now(timezone.utc).isoformat(),seed=20260909,tasks=tasks,dataset_revision=read(META/'protocol.json')['revision'],harness_revision=prior['harness_revision'],task_repo_revision=prior['task_repo_revision'],
        rule='Use unchanged metadata-selected22934source/23413evaluation compatibility canaries, before repair/test columns. No replacements.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        scope='Two infrastructure-only gold-exposed canaries; both excluded from future heldout model-performance reporting. No final training/evaluation split, no learning/diversity outcome.',
        package_rule='Require exact existing90 Python3.9.20/allnine packagepins and official task command. Any version/asset mismatch is retained as compatibility failure; no silent environment retuning.',
        stages='Freeze -> public source exports and separately cached official assets -> root exact operation review -> prepare clean baseline/gold environments -> four host test runs.',
        limits='Same90per-command180s, per-test180wall/120CPU/4GiB, aggregate1800steststage. No model/API/GPU. Stop affectedtask on infrastructure or expected baseline/gold mismatch; retainbothslots.',api_spend_usd=0,gpu_used=False)
    save(OUT/'selection.json',selection);print(json.dumps(dict(selection_sha256=sha(OUT/'selection.json'),ids=[t['instance_id'] for t in tasks])),flush=True)

def assets():
    selection=m.frozen_selection();assert not (OUT/'assets_manifest.json').exists();records=[]
    for task in selection['tasks']:
        iid=task['instance_id'];folder=OUT/'review_assets'/iid;folder.mkdir(parents=True);row=dict(instance_id=iid,files={})
        try:
            for name in ASSETS:
                url=f"https://raw.githubusercontent.com/SWE-bench/swe-bench-tasks/{selection['task_repo_revision']}/tasks/{iid}/{name}"
                with urllib.request.urlopen(url,timeout=30) as response:body=response.read(10_000_001)
                assert len(body)<=10_000_000
                (folder/name).write_bytes(body);row['files'][name]=dict(url=url,sha256=sha(folder/name))
            row['ready']=True
        except Exception as error:row.update(ready=False,error=dict(type=type(error).__name__,message=str(error),replacement=False))
        records.append(row);print(json.dumps(dict(instance_id=iid,ready=row['ready'])),flush=True)
    save(OUT/'assets_manifest.json',dict(selection_sha256=sha(OUT/'selection.json'),tasks=records))

def prepare(review):
    auth=read(Path(review));assert auth['allow_preparation'] and auth['selection_sha256']==sha(OUT/'selection.json') and auth['assets_manifest_sha256']==sha(OUT/'assets_manifest.json')
    assert auth['reason'];assets=read(OUT/'assets_manifest.json');cache={}
    for row in assets['tasks']:
        for name,info in row['files'].items():
            path=OUT/'review_assets'/row['instance_id']/name;assert sha(path)==info['sha256'];cache[info['url']]=path
    for name,digest in auth['reviewed_source_sha256'].items():assert sha(ROOT/name)==digest,name
    save(OUT/'root_preparation_review.json',auth)
    def cached(url,timeout=30):
        assert url in cache,'Missing frozen official asset; do not fetch a replacement'
        return io.BytesIO(cache[url].read_bytes())
    m.urllib.request.urlopen=cached
    m.prepare()
    protocol=read(OUT/'protocol.json')
    for path in [Path(__file__).resolve(),OUT/'assets_manifest.json',OUT/'root_preparation_review.json']:
        protocol['source_sha256'][str(path)]=sha(path)
    protocol['scope']=read(OUT/'selection.json')['scope']
    save(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),ready=sum(t['ready'] for t in protocol['tasks']))),flush=True)

def run(review):
    auth=read(Path(review));assert auth['allow_baseline_gold_tests'] and auth['protocol_sha256']==sha(OUT/'protocol.json') and auth['reason']
    save(OUT/'root_test_review.json',auth);m.run()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','exports','assets','prepare','run']);p.add_argument('--review');a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.stage=='exports':m.exports()
    elif a.stage=='assets':assets()
    elif a.stage=='prepare':prepare(a.review)
    else:run(a.review)
