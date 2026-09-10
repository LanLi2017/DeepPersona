#!/usr/bin/env python3
"""Frozen six-task SWE host development environments and clean source exports."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/swe-diversity-selection/swe-fresh-development'
HELPER = ROOT/'scripts/85_swe_execution_bridge.py'
spec = importlib.util.spec_from_file_location('bridge85', HELPER)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
sha = b.sha
read = b.read
save = b.save


def frozen_selection():
    p = read(OUT/'selection.json')
    for f,h in p['source_sha256'].items(): assert sha(ROOT/f) == h
    return p


def exports():
    import pyarrow.parquet as pq
    selection = frozen_selection()
    ids = [r['instance_id'] for r in selection['tasks']]
    rows = pq.read_table(b.DATA, columns=['instance_id','problem_statement'], filters=[('instance_id','in',ids)]).to_pylist()
    issues = {r['instance_id']:r['problem_statement'] for r in rows}
    assert not (OUT/'public').exists()
    manifest = []
    for task in selection['tasks']:
        iid = task['instance_id']; base = OUT/'private/base'/iid
        base.mkdir(parents=True)
        log = b.call(['git','init',str(base)])
        log += b.call(['git','remote','add','origin','https://github.com/sympy/sympy.git'],cwd=base)
        log += b.call(['git','fetch','--depth','1','origin',task['base_commit']],cwd=base)
        log += b.call(['git','checkout','--detach','FETCH_HEAD'],cwd=base)
        assert b.call(['git','status','--porcelain'],cwd=base) == ''
        (base.parent/(iid+'_fetch.log')).write_text(log)
        public = OUT/'public'/iid; public.mkdir(parents=True)
        files = b.call(['git','ls-files','-z'],cwd=base).split('\0')
        hashes = {}
        for name in filter(None,files):
            path = Path(name)
            source = base/path
            if not source.is_file() or source.is_symlink(): continue
            try: source.read_bytes().decode('utf-8')
            except UnicodeDecodeError: continue
            dest = public/'source'/path; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest); hashes[name] = sha(dest)
        (public/'problem_statement.md').write_text(issues[iid])
        record = dict(instance_id=iid,repo='sympy/sympy',base_commit=task['base_commit'],
            problem_statement_sha256=sha(public/'problem_statement.md'), source_sha256=hashes,
            export_rule='Exact tracked UTF-8 base source bytes, including original repository tests; no .git or added benchmark test patches/testlists/gold/evaluation logs. Binary files and symlinks omitted.')
        save(public/'task.json',record)
        manifest.append(dict(instance_id=iid,base_commit=task['base_commit'],path=str(public.relative_to(ROOT)),source_files=len(hashes),metadata_sha256=sha(public/'task.json')))
        print(json.dumps(manifest[-1]),flush=True)
    save(OUT/'export_manifest.json',dict(tasks=manifest,selection_sha256=sha(OUT/'selection.json'),generator_files_only=True))


def prepare():
    import pyarrow.parquet as pq
    import yaml
    selection = frozen_selection()
    assert (OUT/'export_manifest.json').exists() and not (OUT/'protocol.json').exists()
    ids = [r['instance_id'] for r in selection['tasks']]
    rows = pq.read_table(b.DATA,filters=[('instance_id','in',ids)]).to_pylist()
    dataset = {r['instance_id']:r for r in rows}
    records=[];uv=shutil.which('uv');started=time.monotonic()
    for task in selection['tasks']:
        iid=task['instance_id'];record=dict(task,ready=False)
        try:
            assets=OUT/'private/official_tasks'/iid;assets.mkdir(parents=True)
            for name in ['task.yaml','tests.json','gold.patch','test.patch','eval.sh','Dockerfile']:
                url=f"https://raw.githubusercontent.com/SWE-bench/swe-bench-tasks/{selection['task_repo_revision']}/tasks/{iid}/{name}"
                with urllib.request.urlopen(url,timeout=30) as response:body=response.read()
                (assets/name).write_bytes(body)
            row=dataset[iid];meta=yaml.safe_load((assets/'task.yaml').read_text())
            assert meta['base_commit']==row['base_commit']==task['base_commit']
            for name,key in [('gold.patch','patch'),('test.patch','test_patch')]:assert (assets/name).read_text()==row[key]
            tests=read(assets/'tests.json')
            assert all(tests[k]==json.loads(row[k]) for k in ['FAIL_TO_PASS','PASS_TO_PASS'])
            commands=[x for x in (assets/'eval.sh').read_text().splitlines() if x.startswith('PYTHONWARNINGS=')]
            assert len(commands)==1 and 'bin/test -C --verbose ' in commands[0]
            docker=(assets/'Dockerfile').read_text()
            assert 'python=3.9.20=' in docker
            for package in b.PACKAGES:
                name,version=package.split('==')
                assert name+'='+version+'=' in docker or name+'=='+version in docker,package
            for arm in ['baseline','gold']:
                folder=OUT/'private/tasks'/iid/arm;folder.mkdir(parents=True)
                repo=folder/'repo';venv=folder/'venv'
                log=b.call(['git','clone','--shared','--no-checkout',str(OUT/'private/base'/iid),str(repo)])
                log+=b.call(['git','checkout','--detach',task['base_commit']],cwd=repo)
                assert (repo/'setup.py').exists() and b.call(['git','status','--porcelain'],cwd=repo)==''
                log+=b.call([uv,'--cache-dir',str(b.OUT/'uv-cache'),'venv','--python',str(b.PYTHON),str(venv)])
                log+=b.call([uv,'--cache-dir',str(b.OUT/'uv-cache'),'pip','install','--python',str(venv/'bin/python'),*b.PACKAGES])
                log+=b.call([str(venv/'bin/python'),'-m','pip','install','--no-deps','--no-build-isolation','-e','.'],cwd=repo,env=b.environment(venv))
                (folder/'setup.log').write_text(log)
                versions=json.loads(b.call([str(venv/'bin/python'),'-c','import importlib.metadata as m,json,platform,sympy;print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"]:d.version for d in m.distributions()},sympy_file=sympy.__file__)))'],cwd=repo,env=b.environment(venv)))
                assert versions['python']=='3.9.20' and Path(versions['sympy_file']).is_relative_to(repo)
                for package in b.PACKAGES:
                    name,version=package.split('==');assert versions['packages'][name]==version
                save(folder/'environment.json',versions)
            record.update(ready=True,test_command=commands[0],tests=tests)
        except Exception as error:
            record['preparation_failure']=dict(type=type(error).__name__,message=str(error),replacement=False)
        records.append(record);save(OUT/'preparation_progress.json',records)
        print(json.dumps(dict(task=iid,ready=record['ready'])),flush=True)
    paths=[Path(__file__).resolve(),HELPER,b.PARSER,OUT/'selection.json',OUT/'export_manifest.json']
    paths+=list((OUT/'private/official_tasks').glob('*/*'))+list((OUT/'private/tasks').glob('*/*/environment.json'))
    paths+=list((OUT/'public').glob('*/task.json'))
    save(OUT/'protocol.json',dict(tasks=records,selection_sha256=sha(OUT/'selection.json'),
        source_sha256={str(p):sha(p) for p in paths},python='3.9.20',packages=b.PACKAGES,
        test_timeout_seconds=180,cpu_limit_seconds=120,memory_limit_bytes=4*1024**3,
        total_test_stage_wall_seconds=1800,preparation_seconds=time.monotonic()-started,
        baseline='Exact base plus official test.patch',gold='Exact base plus official gold.patch and same test.patch',
        failures='Stop affected task on any load-bearing mismatch; retain all selected tasks; never replace by outcomes',
        scope='Gold-exposed SWE host development, not heldout, official-container evaluation, or learned repair gain',
        deviations=read(b.OUT/'protocol.json')['deviations'][:3],api_spend_usd=0,gpu_used=False))


def run():
    protocol=read(OUT/'protocol.json');assert not (OUT/'results.json').exists()
    for p,h in protocol['source_sha256'].items():assert sha(Path(p))==h,p
    parsed=ast.parse(b.PARSER.read_text());node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='parse_log_sympy')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]);ast.fix_missing_locations(module)
    namespace=dict(re=re,TestStatus=SimpleNamespace(**{s:SimpleNamespace(value=s) for s in ['PASSED','FAILED','ERROR']}));exec(compile(module,str(b.PARSER),'exec'),namespace)
    parser=namespace['parse_log_sympy'];results=[];started=time.monotonic()
    for task in protocol['tasks']:
        iid=task['instance_id'];item=dict(instance_id=iid,host_bridge_passed=False,arms=[])
        try:
            assert task['ready'],task.get('preparation_failure')
            for arm in ['baseline','gold']:
                remaining=protocol['total_test_stage_wall_seconds']-(time.monotonic()-started);assert remaining>0,'Global test-stage budget exhausted'
                folder=OUT/'private/tasks'/iid/arm;repo=folder/'repo';assets=OUT/'private/official_tasks'/iid
                assert not (folder/'test.log').exists()
                for patch in (['gold.patch'] if arm=='gold' else [])+['test.patch']:
                    b.call(['git','apply','--check',str(assets/patch)],cwd=repo)
                    b.call(['git','apply',str(assets/patch)],cwd=repo)
                (folder/'applied.diff').write_text(b.call(['git','diff',task['base_commit']],cwd=repo))
                tick=time.monotonic()
                with (folder/'test.log').open('w') as log:
                    proc=subprocess.run(['/bin/bash','-c',task['test_command']],cwd=repo,env=b.environment(folder/'venv'),stdout=log,stderr=subprocess.STDOUT,timeout=min(180,remaining),preexec_fn=b.limits)
                statuses=parser((folder/'test.log').read_text(),None)
                selected={k:{test:statuses.get(test,'MISSING') for test in v} for k,v in task['tests'].items()}
                row=dict(arm=arm,exit_code=proc.returncode,seconds=time.monotonic()-tick,statuses=selected,all_parsed_statuses=statuses)
                save(folder/'result.json',row);item['arms'].append(row)
                assert all(v=='PASSED' for v in selected['PASS_TO_PASS'].values()),'Regression/environment mismatch'
                expected={'PASSED'} if arm=='gold' else {'FAILED','ERROR'}
                assert all(v in expected for v in selected['FAIL_TO_PASS'].values()),'Target status mismatch'
                assert proc.returncode==(0 if arm=='gold' else 1),'Unexpected test runner exit'
            item['host_bridge_passed']=True
        except Exception as error:
            item['failure']=dict(type=type(error).__name__,message=str(error),replacement=False)
        results.append(item);save(OUT/'results_progress.json',results)
        print(json.dumps(dict(task=iid,passed=item['host_bridge_passed'],seconds=round(time.monotonic()-started,2))),flush=True)
    save(OUT/'results.json',dict(selected_tasks=len(results),host_bridge_passed=sum(r['host_bridge_passed'] for r in results),tasks=results,seconds=time.monotonic()-started,official_container_evaluation=False,api_spend_usd=0,gpu_used=False))
    files=[OUT/'protocol.json',OUT/'results.json']+list((OUT/'private/tasks').glob('*/*/test.log'))+list((OUT/'private/tasks').glob('*/*/result.json'))
    save(OUT/'completed_manifest.json',{str(p):sha(p) for p in files})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['exports','prepare','run']);args=ap.parse_args()
    {'exports':exports,'prepare':prepare,'run':run}[args.stage]()
