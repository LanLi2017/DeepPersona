#!/usr/bin/env python3
"""Public-only fixed family selection and clean base exports for patch SFT."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/swe-diversity-selection'
OUT = BASE / 'swe-sympy-patch-sft'
META = BASE / 'swe-sympy-learning-metadata'
NAMESPACE = 'swe-sympy-patch-sft-v1:'
VERSIONS = ['1.7', '1.8', '1.9', '1.10', '1.11']
spec = importlib.util.spec_from_file_location('population85', ROOT/'scripts/85_swe_execution_bridge.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
sha, read = b.sha, b.read


def save(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2)+'\n')


def order(kind, value):
    return hashlib.sha256((NAMESPACE+kind+':'+value).encode()).hexdigest()


def validate():
    p = read(OUT/'selection_protocol.json')
    for name, digest in p['inputs_sha256'].items():
        assert sha(ROOT/name) == digest, name
    return p


def freeze():
    OUT.mkdir(exist_ok=True)
    audit = OUT/'public_family_review.json'
    assert audit.exists(), 'Complete public-only family review before freezing selection'
    inputs = [Path(__file__).resolve(), ROOT/'scripts/85_swe_execution_bridge.py',
              META/'sympy_metadata.jsonl', META/'dispositions.json', META/'family_edges.json',
              META/'protocol.json', META/'post_canary_capacity.json', audit,
              BASE/'swe-fresh-development/selection.json']
    save(OUT/'selection_protocol.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        namespace=NAMESPACE, versions=VERSIONS, source_slots=32, evaluation_slots=20,
        representative_rule='Minimum SHA256(namespace + representative: + instance_id) within each eligible public family.',
        family_order_rule='Ascending SHA256(namespace + role + -family: + family), then family; first32source/20evaluation. No outcome-based replacement.',
        family_review='Use reviewed_family_by_id and quarantined_family_ids from the frozen public review; no private columns.',
        source_rule='106source_candidate; version1.7–1.11; neither107canary family nor any reviewed quarantine.',
        evaluation_rule='Exact Verified identity with no106family exclusions except any_Verified_family; same versions and canary/quarantine exclusions.',
        scope='Nonstandard fullTEST source outside allVerified/knownexposed families; Verified development evaluation, not leaderboard holdout.',
        exports='Exact original tracked UTF8 base source bytes, no symlinks/binaries/.git/addedtests/gold. Smoke first selectedsource; then four concurrent exports. Preserve failures without replacements.',
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        private_columns_materialized=False, api_spend_usd=0, gpu_used=False))
    print('Selection protocol frozen', flush=True)


def select():
    p = validate(); review = read(OUT/'public_family_review.json')
    assert review['private_columns_materialized'] is False
    families = review['reviewed_family_by_id']; quarantine = set(review['quarantined_family_ids'])
    rows = read(META/'dispositions.json')
    metadata = {r['instance_id']:r for r in map(json.loads, (META/'sympy_metadata.jsonl').read_text().splitlines())}
    canaries = {families[i] for i in ['sympy__sympy-22934','sympy__sympy-23413']}
    selected = []; capacities = {}
    for role, count in [('source',32),('evaluation',20)]:
        groups = defaultdict(list)
        for r in rows:
            iid = r['instance_id']; family = families[iid]
            if r['version'] not in VERSIONS or family in canaries|quarantine: continue
            eligible = r['source_candidate'] if role=='source' else (r['exact_Verified_identity'] and not (set(r['exclusion_reasons'])-{'any_Verified_family'}))
            if eligible: groups[family].append(iid)
        capacities[role] = dict(tasks=sum(map(len,groups.values())),families=len(groups))
        assert len(groups)>=count, (role, capacities[role])
        ordered = sorted(groups,key=lambda f:(order(role+'-family',f),f))
        for slot, family in enumerate(ordered[:count]):
            iid = min(groups[family],key=lambda i:(order('representative',i),i))
            row = metadata[iid]
            selected.append(dict(instance_id=iid,repo=row['repo'],base_commit=row['base_commit'],
                environment_setup_commit=row['environment_setup_commit'],version=row['version'],
                created_at=row['created_at'],role=role,slot=slot,family=family,
                family_order_sha256=order(role+'-family',family),representative_sha256=order('representative',iid)))
    assert not ({t['family'] for t in selected if t['role']=='source'} & {t['family'] for t in selected if t['role']=='evaluation'})
    old = read(BASE/'swe-fresh-development/selection.json')
    save(OUT/'selection.json',dict(protocol_sha256=sha(OUT/'selection_protocol.json'), tasks=selected,
        capacities=capacities, version_counts={role:dict(Counter(t['version'] for t in selected if t['role']==role)) for role in capacities},
        dataset_revision=read(META/'protocol.json')['revision'],harness_revision=old['harness_revision'],
        task_repo_revision=old['task_repo_revision'],private_columns_materialized=False,
        source_sha256=p['inputs_sha256']))
    print(json.dumps(dict(capacities=capacities,selected=len(selected),selection_sha256=sha(OUT/'selection.json'))),flush=True)


def export_one(task):
    iid = task['instance_id']; record_path = OUT/'export_records'/f'{iid}.json'
    assert not record_path.exists(), 'Do not retry or overwrite a frozen slot'
    row = dict(instance_id=iid,role=task['role'],ready=False)
    try:
        base = OUT/'private/base'/iid; base.mkdir(parents=True)
        log = b.call(['git','init',str(base)])
        log += b.call(['git','remote','add','origin','https://github.com/sympy/sympy.git'],cwd=base)
        log += b.call(['git','fetch','--depth','1','origin',task['base_commit']],cwd=base)
        log += b.call(['git','checkout','--detach','FETCH_HEAD'],cwd=base)
        assert b.call(['git','rev-parse','HEAD'],cwd=base).strip()==task['base_commit']
        assert b.call(['git','status','--porcelain'],cwd=base)==''
        (base.parent/(iid+'_fetch.log')).write_text(log)
        public = OUT/'public'/iid; public.mkdir(parents=True)
        hashes = {}
        for name in filter(None,b.call(['git','ls-files','-z'],cwd=base).split('\0')):
            path = base/name
            if not path.is_file() or path.is_symlink(): continue
            blob = path.read_bytes()
            if b'\x00' in blob: continue
            try: blob.decode('utf-8')
            except UnicodeDecodeError: continue
            dest = public/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(blob); hashes[name]=sha(dest)
        metadata = next(r for r in map(json.loads,(META/'sympy_metadata.jsonl').read_text().splitlines()) if r['instance_id']==iid)
        (public/'problem_statement.md').write_bytes(metadata['problem_statement'].encode())
        save(public/'task.json',dict(**task,source_sha256=hashes,problem_statement_sha256=sha(public/'problem_statement.md')))
        row.update(ready=True,source_files=len(hashes),metadata_sha256=sha(public/'task.json'))
    except Exception as error:
        row['failure']=dict(type=type(error).__name__,message=str(error),replacement=False)
    save(record_path,row); print(json.dumps(row),flush=True)
    return row


def exports(smoke):
    validate(); selection=read(OUT/'selection.json')
    assert selection['protocol_sha256']==sha(OUT/'selection_protocol.json')
    (OUT/'export_records').mkdir(exist_ok=True)
    tasks=selection['tasks']
    if smoke:
        row=export_one(tasks[0]); save(OUT/'export_smoke.json',row); assert row['ready']
        return
    assert read(OUT/'export_smoke.json')['ready']
    assert not (OUT/'export_manifest.json').exists()
    pending=[t for t in tasks if not (OUT/'export_records'/f"{t['instance_id']}.json").exists()]
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(export_one,pending))
    records=[read(OUT/'export_records'/f"{t['instance_id']}.json") for t in tasks]
    save(OUT/'export_manifest.json',dict(selection_sha256=sha(OUT/'selection.json'),tasks=records,
        ready=sum(r['ready'] for r in records),private_columns_materialized=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','select','smoke','exports']);args=parser.parse_args()
    if args.stage=='freeze': freeze()
    elif args.stage=='select': select()
    else: exports(args.stage=='smoke')
