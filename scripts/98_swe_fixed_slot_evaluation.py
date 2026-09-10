#!/usr/bin/env python3
"""Additive fixed-slot SWE host grading; reuse93 semantics with explicit risk review."""
import argparse
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
COHORT=ROOT/'runs/swe-diversity-selection/swe-fresh-development'
GEN=None
OUT=None
HELPER=ROOT/'scripts/85_swe_execution_bridge.py'
spec=importlib.util.spec_from_file_location('candidate_bridge85',HELPER)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
sha=b.sha;read=b.read;save=b.save
RISK_MODULES={'os','sys','subprocess','socket','urllib','requests','http','ftplib','paramiko',
              'pathlib','shutil','ctypes','importlib','pickle','marshal','multiprocessing','io'}
RISK_CALLS={'open','eval','exec','compile','__import__','breakpoint','input'}
RISK_METHODS={'read_text','read_bytes','write_text','write_bytes','unlink','rmdir','mkdir',
              'system','popen','connect','send','sendall','recv','urlopen','urlretrieve'}


def path_allowed(name):
    p=PurePosixPath(name)
    return (not p.is_absolute() and p.parts and p.parts[0]=='sympy' and '..' not in p.parts
            and '\\' not in name and p.suffix=='.py'
            and not any(x in {'test','tests','testing'} or x.startswith('test_') for x in p.parts)
            and not p.name.endswith('_test.py') and p.name!='conftest.py')


def new_risks(before,after):
    def operations(code):
        found=Counter();tree=ast.parse(code);aliases={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                aliases.update({n.asname or n.name.split('.')[0]:n.name.split('.')[0] for n in node.names})
            elif isinstance(node,ast.ImportFrom):
                aliases.update({n.asname or n.name:(node.module or '').split('.')[0] for n in node.names})
        for node in ast.walk(tree):
            flag=False
            if isinstance(node,ast.Import):flag=any(n.name.split('.')[0] in RISK_MODULES for n in node.names)
            elif isinstance(node,ast.ImportFrom):flag=(node.module or '').split('.')[0] in RISK_MODULES
            elif isinstance(node,ast.Call):
                label=ast.unparse(node.func);root=label.split('.')[0];root=aliases.get(root,root)
                flag=(label in RISK_CALLS or root in RISK_MODULES or
                      isinstance(node.func,ast.Attribute) and node.func.attr in RISK_METHODS)
            elif isinstance(node,ast.Attribute):flag=node.attr in {'environ','__builtins__'}
            if flag:found[ast.unparse(node)]+=1
        return found
    return sorted((operations(after)-operations(before)).elements())


def prepare():
    assert not (OUT/'protocol.json').exists()
    jobs=read(GEN/'jobs.json');patches=read(GEN/'patch_manifest.json')
    assert 0<len(jobs)==len(patches)<=48
    assert len({r['key'] for r in jobs})==len(jobs)
    for name,digest in read(GEN/'protocol.json').get('source_sha256',{}).items():assert sha(ROOT/name)==digest,name
    assert [r['key'] for r in jobs]==[r['key'] for r in patches]
    parent=read(COHORT/'protocol.json');tasks={r['instance_id']:r for r in parent['tasks']}
    assert len(tasks)==6 and all(r['ready'] for r in tasks.values())
    for p,h in parent['source_sha256'].items():assert sha(Path(p))==h
    assert read(COHORT/'results.json')['host_bridge_passed']==6
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=[]
    sources=[Path(__file__).resolve(),ROOT/'scripts/93_swe_candidate_evaluation.py',HELPER,b.PARSER,COHORT/'protocol.json',COHORT/'selection.json',
             COHORT/'results.json',GEN/'jobs.json',GEN/'patch_manifest.json',GEN/'protocol.json']
    for job,row in zip(jobs,patches):
        iid=job['instance_id'];key=job['key'];task=tasks[iid]
        assert key==f"{iid}__{job['sample']}" and 0<=job['sample']<8
        assert row['instance_id']==iid and row['sample']==job['sample']
        item=dict(key=key,instance_id=iid,sample=job['sample'],base_commit=task['base_commit'],
                  stage='ungenerated' if row.get('ungenerated') else 'invalid_generation',
                  reason=row.get('invalid_reason'),suspicious_additions=[])
        patch=GEN/'patches'/(key+'.patch')
        if row['valid']:
            try:
                assert patch.is_file() and sha(patch)==row['patch_sha256']
                item['patch_sha256']=sha(patch);sources.append(patch)
                files=row['files'];assert files and len(files)==len(set(files)) and all(path_allowed(n) for n in files)
                base=COHORT/'private/base'/iid;public=COHORT/'public'/iid
                assert b.call(['git','rev-parse','HEAD'],cwd=base).strip()==task['base_commit']
                assert b.call(['git','status','--porcelain'],cwd=base)==''
                metadata=read(public/'task.json');sources.append(public/'task.json')
                for name in files:
                    assert (base/name).is_file() and not (base/name).is_symlink()
                    assert sha(base/name)==sha(public/'source'/name)==metadata['source_sha256'][name]==row['base_file_sha256'][name]
                folder=OUT/'candidates'/key;folder.mkdir(parents=True)
                repo=folder/'repo'
                log=b.call(['git','clone','--shared','--no-checkout',str(base),str(repo)])
                log+=b.call(['git','checkout','--detach',task['base_commit']],cwd=repo)
                assert b.call(['git','status','--porcelain'],cwd=repo)==''
                log+=b.call(['git','apply','--check',str(patch)],cwd=repo)
                log+=b.call(['git','apply',str(patch)],cwd=repo)
                changed=sorted(filter(None,b.call(['git','diff','--name-only','-z'],cwd=repo).split('\0')))
                assert changed==sorted(files)
                hashes={}
                for name in changed:
                    assert (repo/name).is_file() and not (repo/name).is_symlink()
                    before=(base/name).read_text();after=(repo/name).read_text();ast.parse(after,filename=name)
                    item['suspicious_additions'] += [dict(file=name,operation=x) for x in new_risks(before,after)]
                    hashes[name]=sha(repo/name)
                (folder/'preparation.log').write_text(log)
                (folder/'candidate.diff').write_text(b.call(['git','diff',task['base_commit']],cwd=repo))
                item.update(stage='ready',reason=None,files=changed,candidate_file_sha256=hashes,
                    candidate_diff_sha256=sha(folder/'candidate.diff'),base_file_sha256=row['base_file_sha256'])
            except Exception as error:
                item.update(stage='invalid_patch',reason=f'{type(error).__name__}: {error}')
        frozen.append(item);save(OUT/'preparation_progress.json',frozen)
    sources+=list((COHORT/'private/tasks').glob('*/baseline/environment.json'))
    sources+=[COHORT/'private/official_tasks'/iid/name for iid in tasks for name in ['test.patch','tests.json','eval.sh']]
    sources.append(b.PYTHON)
    protocol=dict(slots=frozen,slot_count=len(frozen),tasks=parent['tasks'],source_sha256={str(p):sha(p) for p in sources},
        benchmark='Frozen SymPy host-development cohort, official target/regression commands and parser from90/85',
        reward='Per required test:1 PASS,0 observed FAIL/ERROR,null MISSING/NOT_RUN; resolved iff completed execution reports every target and regression PASS. Other statuses remain explicit.',
        nonexecuted='Invalid or ungenerated slots remain resolved=false with null test rewards and explicit reason; review-blocked and budget-skipped slots are ungraded, never silently repaired or replaced.',
        no_candidate_test_changes=True,no_gold_applied=True,workers=1,max_test_wall_seconds=180,
        max_cpu_seconds=120,max_address_space_bytes=4*1024**3,total_execution_wall_seconds=1800,
        runtime='Reuse task baseline venv only after candidate-checkout sympy.__file__ check; same minimal environment and resource limits as90.',
        authorization='Execution record pins this protocol. Two blinded reviews belonged to old48 study only; not required here. Flagged operations still require exact-patch root review.',
        suspicious_policy='New risky imports/calls are recorded before grading. Block unless root explicitly approves this key and exact patch SHA. Static scan is not a security sandbox.',
        input_exports='Only public base source was used by generator; private evaluator never modifies public exports.',api_spend_usd=0,gpu_used=False)
    save(OUT/'protocol.json',protocol)
    print(json.dumps(dict(slots=len(frozen),ready=sum(r['stage']=='ready' for r in frozen),review_flags=sum(bool(r['suspicious_additions']) for r in frozen),protocol_sha256=sha(OUT/'protocol.json'))),flush=True)


def parser_function():
    node=next(n for n in ast.parse(b.PARSER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parse_log_sympy')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]);ast.fix_missing_locations(module)
    namespace=dict(re=re,TestStatus=SimpleNamespace(**{s:SimpleNamespace(value=s) for s in ['PASSED','FAILED','ERROR']}))
    exec(compile(module,str(b.PARSER),'exec'),namespace)
    return namespace['parse_log_sympy']


def run(authorization):
    assert authorization and not (OUT/'started.json').exists(),'No run without root gate; no retries'
    p=read(OUT/'protocol.json');auth=read(authorization)
    assert auth['allow_candidate_correctness'] is True and auth['protocol_sha256']==sha(OUT/'protocol.json')
    reviews=auth.get('blinded_review_sha256',{})
    for f,h in reviews.items():assert sha(ROOT/f)==h
    for f,h in p['source_sha256'].items():assert sha(Path(f))==h,f
    save(OUT/'started.json',dict(protocol_sha256=sha(OUT/'protocol.json'),authorization_sha256=sha(authorization),blinded_review_sha256=reviews,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    tasks={t['instance_id']:t for t in p['tasks']};parse=parser_function();started=time.monotonic();results=[]
    for slot in p['slots']:
        task=tasks[slot['instance_id']];key=slot['key'];folder=OUT/'candidates'/key;repo=folder/'repo'
        row=dict(key=key,instance_id=slot['instance_id'],sample=slot['sample'],execution_status=slot['stage'],reason=slot['reason'],resolved=False,correctness_observed=False)
        statuses={suite:{name:'NOT_RUN' for name in names} for suite,names in task['tests'].items()}
        if slot['stage']=='ready':
            try:
                if slot['suspicious_additions']:
                    approval=auth.get('external_io_approvals',{}).get(key,{})
                    if approval.get('patch_sha256')!=slot['patch_sha256'] or not approval.get('reason'):
                        row.update(execution_status='blocked_root_review',reason='New potentially external I/O, dynamic execution or process operation requires root review')
                        raise StopIteration
                remaining=p['total_execution_wall_seconds']-(time.monotonic()-started)
                if remaining<=0:
                    row.update(execution_status='not_run_budget',reason='Frozen execution-stage wall budget exhausted');raise StopIteration
                for name,h in slot['candidate_file_sha256'].items():assert sha(repo/name)==h
                assert sha(folder/'candidate.diff')==slot['candidate_diff_sha256']
                assert b.call(['git','rev-parse','HEAD'],cwd=repo).strip()==task['base_commit']
                assert sorted(filter(None,b.call(['git','diff','--name-only','-z'],cwd=repo).split('\0')))==slot['files']
                tests=COHORT/'private/official_tasks'/slot['instance_id']/'test.patch'
                b.call(['git','apply','--check',str(tests)],cwd=repo);b.call(['git','apply',str(tests)],cwd=repo)
                (folder/'evaluated.diff').write_text(b.call(['git','diff',task['base_commit']],cwd=repo))
                venv=COHORT/'private/tasks'/slot['instance_id']/'baseline/venv'
                required={s.split('==')[0]:s.split('==')[1] for s in b.PACKAGES}
                code='import pathlib,sys,importlib.metadata as m; required='+repr(required)+'; assert all(m.version(n)==v for n,v in required.items()); import sympy; p=pathlib.Path(sympy.__file__).resolve(); print(p); assert p.is_relative_to(pathlib.Path.cwd()), p; assert sys.version_info[:3]==(3,9,20)'
                with (folder/'import.log').open('w') as log:
                    check=subprocess.run([str(venv/'bin/python'),'-c',code],cwd=repo,env=b.environment(venv),stdout=log,stderr=subprocess.STDOUT,timeout=min(30,remaining),preexec_fn=b.limits)
                if check.returncode:
                    row.update(execution_status='candidate_import_error',reason='Candidate SymPy import or checkout-origin check failed; see import.log');raise StopIteration
                tick=time.monotonic();remaining=p['total_execution_wall_seconds']-(tick-started)
                with (folder/'test.log').open('w') as log:
                    process=subprocess.run(['/bin/bash','-c',task['test_command']],cwd=repo,env=b.environment(venv),stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,min(180,remaining)),preexec_fn=b.limits)
                parsed=parse((folder/'test.log').read_text(),None)
                statuses={suite:{name:parsed.get(name,'MISSING') for name in names} for suite,names in task['tests'].items()}
                complete=all(value in {'PASSED','FAILED','ERROR'} for values in statuses.values() for value in values.values())
                row.update(execution_status='tested' if complete else 'missing_test_status',reason=None if complete else 'Some required tests were not reported',
                    exit_code=process.returncode,seconds=time.monotonic()-tick,correctness_observed=complete,
                    resolved=all(value=='PASSED' for values in statuses.values() for value in values.values()),all_parsed_statuses=parsed)
            except StopIteration:pass
            except subprocess.TimeoutExpired:
                row.update(execution_status='timeout',reason='Frozen import/test timeout; no retry')
                if (folder/'test.log').exists():
                    parsed=parse((folder/'test.log').read_text(),None)
                    statuses={suite:{name:parsed.get(name,'MISSING') for name in names} for suite,names in task['tests'].items()}
                    row['all_parsed_statuses']=parsed
            except Exception as error:
                row.update(execution_status='evaluation_error',reason=f'{type(error).__name__}: {error}')
        row['test_statuses']=statuses
        row['reward_vector']={suite:{name:1 if value=='PASSED' else 0 if value in {'FAILED','ERROR'} else None for name,value in values.items()} for suite,values in statuses.items()}
        results.append(row)
        with (OUT/'results.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        print(json.dumps(dict(completed=len(results),slots=len(p['slots']),key=key,status=row['execution_status'])),flush=True)
    save(OUT/'completed.json',dict(slots=len(results),seconds=time.monotonic()-started,protocol_sha256=sha(OUT/'protocol.json'),api_spend_usd=0,gpu_used=False,official_container_evaluation=False))
    files=[OUT/'protocol.json',OUT/'started.json',OUT/'results.jsonl',OUT/'completed.json']+list((OUT/'candidates').glob('*/*.log'))+list((OUT/'candidates').glob('*/*.diff'))
    save(OUT/'completed_manifest.json',{str(f):sha(f) for f in files})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run'])
    ap.add_argument('--generation',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--authorization',type=Path);args=ap.parse_args()
    GEN=args.generation.resolve();OUT=args.out.resolve()
    assert GEN.is_relative_to(ROOT) and OUT.is_relative_to(ROOT)
    assert OUT != COHORT/'private/candidate-evaluation' and OUT != GEN
    prepare() if args.stage=='prepare' else run(args.authorization)
