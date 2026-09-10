#!/usr/bin/env python3
"""Pinned full-TEST metadata acquisition; never project repair/test/outcome columns."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import time
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/swe-diversity-selection'
OUT = BASE / 'swe-sympy-learning-metadata'
REV = '02e6f7c57a4aef57bcddb40fc38c3d71578a6951'
DATASET = 'princeton-nlp/SWE-bench'
FILE = 'data/test-00000-of-00001.parquet'
TREE = f'https://huggingface.co/api/datasets/{DATASET}/tree/{REV}/data'
RESOLVE = f'https://huggingface.co/datasets/{DATASET}/resolve/{REV}/{FILE}'
PROJECTION = ['instance_id', 'repo', 'base_commit', 'version', 'environment_setup_commit', 'problem_statement', 'created_at']
CAP, TIMEOUT = 200_000_000, 180
VERIFIED = Path('/scratch/yirenl2/.cache/huggingface/hub/datasets--princeton-nlp--SWE-bench_Verified/snapshots/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet')
INPUTS = [Path(__file__).resolve(), VERIFIED, BASE/'public-audit/metadata.jsonl', BASE/'s1/issue_metadata.jsonl', BASE/'s1/split.json', BASE/'swe-fresh-development/selection.json', BASE/'swe-execution-bridge/selection.json']


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p, value):
    p = Path(p); assert not p.exists(), f'Preserve existing artifact: {p}'
    p.write_text(json.dumps(value, indent=2)+'\n')
def request(url): return urllib.request.Request(url, headers={'User-Agent': 'DeepPersona-public-metadata-audit/1.0'})
def deadline(signum, frame): raise TimeoutError('180-second acquisition deadline')
def validate():
    p = read(OUT/'protocol.json')
    for path, digest in p['source_sha256'].items(): assert sha(path) == digest, path
    return p


def prepare():
    assert not OUT.exists(); OUT.mkdir()
    frozen = dict(utc=utc(),script_sha256=sha(__file__),tree_url=TREE,resolve_url=RESOLVE,
        revision=REV,file=FILE,projection=PROJECTION,max_file_bytes=CAP,timeout_seconds=TIMEOUT,
        privacy='Raw Parquet may store repair/test/outcome columns, but only the seven listed metadata columns may be materialized; never inspect other column values/statistics.',
        family_rule='Connected components: exact repo/base commit, normalized issue text, and explicit same-repository GitHub issue/PR references or issue/PR/fixes/closes/resolves #number cues. Reference links are conservative quarantine evidence, not proved equivalent bugs.',
        selection='No final train/evaluation split. Full-TEST source candidates exclude every Verified family, all Nebius identities/known metadata families, eight exposed and other project-used identities. Unused Verified candidates remain evaluation-only metadata, no runtime claim.',
        source_sha256={str(p):sha(p) for p in INPUTS},model_API_GPU_test_calls=0)
    save(OUT/'acquisition_source_protocol.json', frozen)
    signal.signal(signal.SIGALRM, deadline); signal.alarm(TIMEOUT)
    try:
        with urllib.request.urlopen(request(TREE),timeout=30) as response:
            body=response.read(1_000_001); assert len(body)<=1_000_000
            metadata=json.loads(body)
        assert isinstance(metadata,list)
        matches=[r for r in metadata if r.get('path')==FILE]; assert len(matches)==1
        info=matches[0]; lfs=info['lfs']; size=int(lfs['size']); digest=lfs['oid']
        assert re.fullmatch('[0-9a-f]{64}',digest) and size==int(info['size']) and 0<size<=CAP
        save(OUT/'remote_tree.json',metadata)
        save(OUT/'protocol.json',dict(**frozen,lfs_sha256=digest,expected_bytes=size,
            tree_response_sha256=hashlib.sha256(body).hexdigest(),selected_file_metadata=info,
            projection_smoke='Read identity/repo metadata, choose SHA-first SymPy ID, project seven columns for that ID and freeze smoke, then project all SymPy rows. No candidate patches or test labels.'))
        print(json.dumps(dict(stage='prepared',protocol_sha256=sha(OUT/'protocol.json'),expected_bytes=size,lfs_sha256=digest)),flush=True)
    except Exception as error:
        save(OUT/'prepare_failure.json',dict(utc=utc(),type=type(error).__name__,detail=str(error))); raise
    finally: signal.alarm(0)


def acquire():
    p=validate(); partial=OUT/'full_test.parquet.part'; final=OUT/'full_test.parquet'
    assert not partial.exists() and not final.exists()
    signal.signal(signal.SIGALRM,deadline);signal.alarm(TIMEOUT);start=time.monotonic();size=0;h=hashlib.sha256()
    try:
        with urllib.request.urlopen(request(RESOLVE),timeout=30) as response, partial.open('xb') as f:
            length=response.headers.get('Content-Length')
            if length is not None: assert int(length)<=CAP
            host=urllib.parse.urlparse(response.url).hostname
            while True:
                b=response.read(1024*1024)
                if not b:break
                size+=len(b);assert size<=CAP and size<=p['expected_bytes']
                f.write(b);h.update(b)
        assert size==p['expected_bytes'] and h.hexdigest()==p['lfs_sha256']
        partial.rename(final)
        save(OUT/'download.json',dict(utc=utc(),stable_url=RESOLVE,redirect_host=host,bytes=size,sha256=h.hexdigest(),elapsed_seconds=time.monotonic()-start,
            raw_contains_unprojected_columns=True,unprojected_values_inspected=False,auth_headers_or_env_credentials_used=False))
        print(json.dumps(dict(stage='downloaded',bytes=size,sha256=h.hexdigest())),flush=True)
    except Exception as error:
        save(OUT/'download_failure.json',dict(utc=utc(),type=type(error).__name__,detail=str(error),bytes_received=size,elapsed_seconds=time.monotonic()-start));raise
    finally:signal.alarm(0)


def normtext(s):return ' '.join(s.split())
def refs(row):
    repo=row['repo'].lower();text=row.get('problem_statement','');found=set()
    for owner,name,number in re.findall(r'https?://github\.com/([^/\s]+)/([^/\s]+)/(?:issues|pull)/(\d+)',text,re.I):
        if f'{owner}/{name}'.lower()==repo:found.add(repo+'#'+number)
    for number in re.findall(r'\b(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?|issue|PR|pull\s+request)\s*:?\s*#(\d+)\b',text,re.I):found.add(repo+'#'+number)
    return sorted(found)


def inventory():
    import pyarrow.parquet as pq
    p=validate();file=OUT/'full_test.parquet';assert sha(file)==p['lfs_sha256']
    assert not (OUT/'smoke.json').exists() and not (OUT/'summary.json').exists()
    identity=pq.read_table(file,columns=['instance_id','repo']).to_pylist()
    ids=[r['instance_id'] for r in identity if r['repo']=='sympy/sympy'];assert len(ids)==len(set(ids)) and ids
    first=min(ids,key=lambda s:hashlib.sha256(('swe-sympy-metadata-smoke-v1:'+s).encode()).hexdigest())
    smoke=pq.read_table(file,columns=PROJECTION,filters=[('instance_id','=',first)]).to_pylist()
    assert len(smoke)==1 and set(smoke[0])==set(PROJECTION) and smoke[0]['repo']=='sympy/sympy'
    assert all(isinstance(smoke[0][k],str) and smoke[0][k] for k in ['instance_id','repo','base_commit','version','problem_statement'])
    save(OUT/'smoke.json',dict(utc=utc(),first_metadata_ordered_id=first,columns=PROJECTION,row=smoke[0],passed=True,private_columns_materialized=False))
    rows=pq.read_table(file,columns=PROJECTION,filters=[('repo','=','sympy/sympy')]).to_pylist()
    assert {r['instance_id'] for r in rows}==set(ids) and all(set(r)==set(PROJECTION) for r in rows)
    rows=sorted(rows,key=lambda r:r['instance_id'])
    (OUT/'sympy_metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    verified=pq.read_table(VERIFIED,columns=PROJECTION,filters=[('repo','=','sympy/sympy')]).to_pylist();assert len(verified)==75
    verified_ids={r['instance_id'] for r in verified}
    nebius=set()
    for line in (BASE/'public-audit/metadata.jsonl').open():
        match=re.search(r'"instance_id"\s*:\s*"([^"\\]+)"',line)
        if match:nebius.add(match.group(1)) # Never parse target/eval fields.
    fresh={r['instance_id'] for r in read(BASE/'swe-fresh-development/selection.json')['tasks']}
    exposed=fresh|set(read(BASE/'swe-execution-bridge/selection.json')['ids']);assert len(exposed)==8
    oldplan=read(BASE/'s1/split.json');projectused={i for i,r in oldplan['issues'].items() if r['split']=='dev'}|exposed
    prior=[]
    for line in (BASE/'s1/issue_metadata.jsonl').open():
        if re.search(r'"repo"\s*:\s*"sympy/sympy"',line):
            r=json.loads(line)
            if r['instance_id'] in nebius or r['instance_id'] in projectused:prior.append(r)
    catalog={};variants=[]
    for origin,rs in [('full_test',rows),('Verified',verified),('known_previous_metadata',prior)]:
        for r in rs:
            iid=r['instance_id'];catalog.setdefault(iid,dict(r));variants.append((origin,r))
    for iid in sorted(nebius | exposed | projectused | verified_ids):
        if iid.startswith('sympy__sympy-') and iid not in catalog:
            row=dict(instance_id=iid,repo='sympy/sympy',base_commit='',problem_statement='')
            catalog[iid]=row;variants.append(('known_excluded_identity_only',row))
    parent={i:i for i in catalog}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={};edges=[]
    for origin,r in variants:
        iid=r['instance_id'];repo=r['repo'];own=re.search(r'-(\d+)$',iid)
        keys=[]
        if r.get('base_commit'):keys.append(('base',repo,r['base_commit']))
        if normtext(r.get('problem_statement','')):keys.append(('text',repo,hashlib.sha256(normtext(r['problem_statement']).encode()).hexdigest()))
        links=refs(r)
        if own:links.append(repo.lower()+'#'+own.group(1))
        keys.extend(('reference',repo,v) for v in links)
        for key in keys:
            if key in seen and seen[key]!=iid:
                other=seen[key];parent[find(iid)]=find(other);edges.append(dict(a=iid,b=other,evidence=list(key),origin=origin))
            else:seen[key]=iid
    families=defaultdict(list)
    for iid in catalog:families[find(iid)].append(iid)
    family={i:min(group) for group in families.values() for i in group};reasons=defaultdict(set)
    for iid in catalog:
        if iid in verified_ids:reasons[family[iid]].add('any_Verified_family')
        if iid in nebius:reasons[family[iid]].add('Nebius_identity_or_known_family')
        if iid in exposed:reasons[family[iid]].add('eight_exposed_family')
        if iid in projectused:reasons[family[iid]].add('project_used_family')
    dispositions=[]
    for r in rows:
        iid=r['instance_id'];why=sorted(reasons[family[iid]])
        dispositions.append(dict(instance_id=iid,version=r['version'],base_commit=r['base_commit'],environment_setup_commit=r['environment_setup_commit'],family=family[iid],linked_references=refs(r),source_candidate=not why,exclusion_reasons=why,exact_Verified_identity=iid in verified_ids,exact_Nebius_identity=iid in nebius))
    versions=[]
    for version in sorted({r['version'] for r in rows},key=lambda s:tuple(int(x) if x.isdigit() else -1 for x in s.split('.'))):
        rs=[r for r in dispositions if r['version']==version];left=[r for r in rs if r['source_candidate']]
        versions.append(dict(version=version,full_TEST_tasks=len(rs),exact_Verified=sum(r['exact_Verified_identity'] for r in rs),exact_Nebius=sum(r['exact_Nebius_identity'] for r in rs),source_candidates=len(left),source_candidate_families=len({r['family'] for r in left}),excluded_families_or_identities=len(rs)-len(left)))
    evalc=[r for r in verified if r['instance_id'] not in exposed and r['instance_id'] not in nebius];assert len(evalc)==67
    missing=sorted(verified_ids-set(ids))
    result=dict(utc=utc(),full_TEST_total_metadata_rows=len(identity),full_TEST_SymPy_tasks=len(rows),source_candidates=sum(r['source_candidate'] for r in dispositions),source_candidate_families=len({r['family'] for r in dispositions if r['source_candidate']}),versions=versions,Verified_SymPy_metadata=75,unused_Verified_evaluation_metadata_candidates=67,Verified_ids_missing_from_historical_full_TEST=missing,
        family_definition=p['family_rule'],family_uncertainty=['Explicit links are conservative quarantine signals, not proved same-bug families.','Missing PR/linked-issue fields and implicit backport/revert relationships can leave unresolved semantic overlap; metadata eligibility is not a clean final split.','Repeated base/text/reference families are kept intact; no training/evaluation assignment is made.'],runtime_compatibility='Unknown for every new candidate; only the eight already-used1.12 environments have been exercised.',scope='Explicit nonstandard source capacity from fullTEST outside ALLVerified and known exposed/Nebius/project families. Not nativeTRAIN, not a standardVerifiedleaderboard split.',next_step='Review version-specific source/evaluation metadata and unresolved families; freeze one metadata-first source/evaluation compatibility pair before maintainer patches or new outcome access. No auto execution.',private_columns_materialized=False,raw_Parquet_stored_but_unprojected_columns_not_inspected=True,API_paid_model_GPU_test_patch_application_calls=0)
    save(OUT/'family_edges.json',edges);save(OUT/'dispositions.json',dispositions);save(OUT/'summary.json',result)
    save(OUT/'completed_manifest.json',dict(utc=utc(),protocol_sha256=sha(OUT/'protocol.json'),source_revalidated=True,output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='completed_manifest.json'}))
    print(json.dumps(dict(stage='complete',full_TEST_SymPy_tasks=len(rows),source_candidates=result['source_candidates'],families=result['source_candidate_families'],versions=versions)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','acquire','inventory']);args=parser.parse_args()
    {'prepare':prepare,'acquire':acquire,'inventory':inventory}[args.stage]()
