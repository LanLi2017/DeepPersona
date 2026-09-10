#!/usr/bin/env python3
"""Two frozen SymPy gold-patch host reproductions; not container evaluation."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time
from types import SimpleNamespace

OUT = Path('runs/swe-diversity-selection/swe-execution-bridge').resolve()
DATA = Path('/scratch/yirenl2/.cache/huggingface/hub/datasets--princeton-nlp--SWE-bench_Verified/snapshots/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet')
PYTHON = OUT/'python/cpython-3.9.20-linux-x86_64-gnu/bin/python3.9'
PACKAGES = ['pip==24.2', 'setuptools==75.1.0', 'wheel==0.44.0', 'mpmath==1.3.0',
            'flake8==7.1.1', 'mccabe==0.7.0', 'pycodestyle==2.12.1',
            'pyflakes==3.2.0', 'flake8-comprehensions==3.15.0']
PARSER = OUT/'harness/swebench/harness/log_parsers/python.py'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def save(p, value): p.write_text(json.dumps(value, indent=2)+'\n')


def environment(venv=None):
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
           'TZ': 'UTC', 'PYTHONHASHSEED': '0', 'PYTHONNOUSERSITE': '1',
           'PIP_CONFIG_FILE': '/dev/null', 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': '/dev/null'}
    if venv:
        env['PATH'] = str(venv/'bin')+':'+env['PATH']
        env['VIRTUAL_ENV'] = str(venv)
    return env


def call(cmd, cwd=None, env=None, timeout=180):
    r = subprocess.run(cmd, cwd=cwd, env=env or environment(), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    assert r.returncode == 0, f'{cmd}: {r.stdout[-3000:]}'
    return r.stdout


def prepare():
    import pyarrow.parquet as pq
    import yaml
    assert not (OUT/'protocol.json').exists()
    selection = read(OUT/'selection.json')
    tasks = pq.read_table(DATA, filters=[('instance_id', 'in', selection['ids'])]).to_pylist()
    tasks = {r['instance_id']: r for r in tasks}
    audit = {}
    for name, cmd in {'docker': ['docker', 'info', '--format', '{{json .ServerVersion}}'],
                      'user_namespace': ['unshare', '-Ur', 'true']}.items():
        r = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        audit[name] = dict(command=cmd, exit_code=r.returncode, output=r.stdout)
    audit['runtimes'] = {n: shutil.which(n) for n in ['docker', 'podman', 'apptainer', 'singularity', 'nerdctl', 'rootlesskit', 'newuidmap', 'newgidmap']}
    save(OUT/'runtime_access.json', audit)
    assert call([str(PYTHON), '-c', 'import sys;print(sys.version.split()[0])']).strip() == '3.9.20'
    records = []
    uv = shutil.which('uv')
    for task in selection['ids']:
        source = OUT/'official_tasks'/task
        meta = yaml.safe_load((source/'task.yaml').read_text())
        row = tasks[task]
        assert meta['base_commit'] == row['base_commit']
        for file, key in [('gold.patch', 'patch'), ('test.patch', 'test_patch')]:
            assert (source/file).read_text() == row[key], file
        tests = read(source/'tests.json')
        assert all(tests[k] == json.loads(row[k]) for k in ['FAIL_TO_PASS', 'PASS_TO_PASS'])
        command = next(line for line in (source/'eval.sh').read_text().splitlines()
                       if line.startswith("PYTHONWARNINGS="))
        assert command == "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose sympy/physics/units/tests/test_quantities.py"
        cache = OUT/'source_repos'/task
        cache.mkdir(parents=True)
        source_log = call(['git', 'init', str(cache)])
        source_log += call(['git', 'remote', 'add', 'origin', 'https://github.com/sympy/sympy.git'], cwd=cache)
        source_log += call(['git', 'fetch', '--depth', '1', 'origin', row['base_commit']], cwd=cache)
        source_log += call(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=cache)
        assert (cache/'setup.py').is_file()
        assert call(['git', 'status', '--porcelain'], cwd=cache) == ''
        (cache.parent/(task+'_fetch.log')).write_text(source_log)
        for arm in ['baseline', 'gold']:
            folder = OUT/'tasks'/task/arm
            folder.mkdir(parents=True)
            repo = folder/'repo'; venv = folder/'venv'
            log = call(['git', 'clone', '--shared', '--no-checkout', str(cache), str(repo)])
            log += call(['git', 'checkout', '--detach', row['base_commit']], cwd=repo)
            assert (repo/'setup.py').is_file() and call(['git', 'status', '--porcelain'], cwd=repo) == ''
            log += call([uv, '--cache-dir', str(OUT/'uv-cache'), 'venv', '--python', str(PYTHON), str(venv)])
            log += call([uv, '--cache-dir', str(OUT/'uv-cache'), 'pip', 'install', '--python', str(venv/'bin/python'), *PACKAGES])
            log += call([str(venv/'bin/python'), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '.'], cwd=repo, env=environment(venv))
            (folder/'setup.log').write_text(log)
            versions = json.loads(call([str(venv/'bin/python'), '-c',
                'import importlib.metadata as m,json,platform,sympy;print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"]:d.version for d in m.distributions()},sympy_file=sympy.__file__)))'], cwd=repo, env=environment(venv)))
            assert Path(versions['sympy_file']).is_relative_to(repo)
            save(folder/'environment.json', versions)
        records.append(dict(instance_id=task, base_commit=row['base_commit'], environment_setup_commit=row['environment_setup_commit'],
                            test_command=command, tests=tests, version=row['version']))
    paths = [Path(__file__).resolve(), DATA, PARSER, OUT/'selection.json', OUT/'source_materialization_amendment.json']
    paths += list((OUT/'official_tasks').glob('*/*')) + list((OUT/'tasks').glob('*/*/environment.json'))
    save(OUT/'protocol.json', dict(seed=20260909, tasks=records, packages=PACKAGES, python='3.9.20',
        harness_revision=selection['harness_revision'], task_repo_revision=selection['task_repo_revision'],
        baseline='Pinned base source plus official test.patch', gold='Same base plus official gold.patch and test.patch',
        log_parser='Exact AST-extracted parse_log_sympy from pinned official source, no code changes',
        test_timeout_seconds=180, cpu_seconds=120, memory_bytes=4*1024**3,
        source_sha256={str(p): sha(p) for p in paths}, api_spend_usd=0,
        scope='Host reproduction with matching Python/package versions and exact official test command; not official Docker evaluation.',
        deviations=['Ubuntu host and python-build-standalone CPython replace Ubuntu22.04 Docker/Conda binary builds.',
                    'Fresh isolated local virtualenv/checkouts with minimal credential-free environment are not a filesystem/network security sandbox.',
                    'Only official execution command/test patches are reproduced; conda activation, container paths and Git-history pruning are omitted.',
                    'Both tasks share the units subsystem/test file; this is infrastructure smoke, not independent benchmark-performance evidence.'],
        stop_rule='Stop on missing target/regression test, dependency/import error, baseline target unexpectedly passing, regression failure, or gold failure; no task replacement.'))
    print(json.dumps({'prepared_tasks':selection['ids'], 'protocol_sha256':sha(OUT/'protocol.json')}), flush=True)


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (20*1024**2, 20*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))


def run():
    assert not (OUT/'results.json').exists() and not (OUT/'failure.json').exists()
    protocol = read(OUT/'protocol.json')
    for name, digest in protocol['source_sha256'].items(): assert sha(Path(name)) == digest, name
    parsed = ast.parse(PARSER.read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_log_sympy')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(re=re, TestStatus=SimpleNamespace(**{s:SimpleNamespace(value=s) for s in ['PASSED','FAILED','ERROR']}))
    exec(compile(module, str(PARSER), 'exec'), namespace)
    parser = namespace['parse_log_sympy']
    results = []; started = time.monotonic()
    for task in protocol['tasks']:
        for arm in ['baseline', 'gold']:
            task_id = task['instance_id']; folder = OUT/'tasks'/task_id/arm
            repo = folder/'repo'; source = OUT/'official_tasks'/task_id
            assert not (folder/'test.log').exists()
            for patch in (['gold.patch'] if arm == 'gold' else [])+['test.patch']:
                call(['git', 'apply', '--check', str(source/patch)], cwd=repo)
                call(['git', 'apply', str(source/patch)], cwd=repo)
            (folder/'applied.diff').write_text(call(['git', 'diff', task['base_commit']], cwd=repo))
            tick = time.monotonic()
            with (folder/'test.log').open('w') as log:
                process = subprocess.run(['/bin/bash', '-c', task['test_command']], cwd=repo,
                    env=environment(folder/'venv'), stdout=log, stderr=subprocess.STDOUT,
                    timeout=180, preexec_fn=limits)
            statuses = parser((folder/'test.log').read_text(), None)
            selected = {k:{test:statuses.get(test, 'MISSING') for test in v} for k,v in task['tests'].items()}
            row = dict(instance_id=task_id, arm=arm, exit_code=process.returncode,
                       elapsed_seconds=time.monotonic()-tick, statuses=selected, all_parsed_statuses=statuses)
            save(folder/'result.json', row); results.append(row)
            assert all(v=='PASSED' for v in selected['PASS_TO_PASS'].values()), 'Regression or environment mismatch'
            expected = {'PASSED'} if arm == 'gold' else {'FAILED','ERROR'}
            assert all(v in expected for v in selected['FAIL_TO_PASS'].values()), 'Target status mismatch'
            assert process.returncode == (0 if arm=='gold' else 1), 'Unexpected test runner exit'
            print(json.dumps({k:v for k,v in row.items() if k not in ['statuses','all_parsed_statuses']}), flush=True)
    save(OUT/'results.json', dict(host_bridge_passed=True, official_container_evaluation=False,
        tasks=2, runs=results, elapsed_seconds=time.monotonic()-started, api_spend_usd=0,
        interpretation='Gold patches fix both frozen targets while all official PASS_TO_PASS tests remain passing in the matched-version host environment. No generated repair or learning gain measured.'))
    files = [OUT/'protocol.json', OUT/'results.json']+list((OUT/'tasks').glob('*/*/result.json'))+list((OUT/'tasks').glob('*/*/test.log'))
    save(OUT/'completed_manifest.json', {str(p):sha(p) for p in files})


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage', choices=['prepare','run']);args=ap.parse_args()
    try: {'prepare':prepare,'run':run}[args.stage]()
    except Exception as error:
        save(OUT/(args.stage+'_failure.json'), dict(type=type(error).__name__, message=str(error), fallback_used=False))
        raise
