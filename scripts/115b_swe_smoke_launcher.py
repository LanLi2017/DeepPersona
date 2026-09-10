#!/usr/bin/env python3
"""One external process-group deadline and conservative smoke budget receipt."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/swe-diversity-selection/swe-sympy-context-capacity/learner_smoke'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    auth=json.loads((OUT/'root_launch_review.json').read_text())
    assert auth['allow_technical_smoke'] and auth['reason']
    assert auth['protocol_sha256']==sha(OUT/'protocol.json')
    assert auth['launcher_sha256']==sha(Path(__file__))
    assert auth['smoke_script_sha256']==sha(ROOT/'scripts/115_swe_sympy_long_context_smoke.py')
    assert protocol['external_timeout']['term_seconds']==3595
    assert protocol['external_timeout']['kill_grace_seconds']==5
    assert protocol['total_shared_gpu_budget_seconds']==3600 and protocol['prior_gpu_seconds']==0
    assert not (OUT/'outer_dispatch.json').exists() and not (OUT/'outer_terminal.json').exists()
    assert not (OUT/'run').exists(), 'No retry of any attempted smoke'
    gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
    assert gpu[0].strip()=='2' and gpu[1].strip()==protocol['hardware']['uuid']
    assert int(gpu[2].strip())<100,'GPU2 is occupied; do not collide with another job'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='2',CUDA_DEVICE_ORDER='PCI_BUS_ID',SWE_SMOKE_EXTERNAL_TIMEOUT='3595+5',
        NVIDIA_TF32_OVERRIDE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0')
    cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/115_swe_sympy_long_context_smoke.py'),'run']
    save(OUT/'outer_dispatch.json',dict(command=cmd,protocol_sha256=sha(OUT/'protocol.json'),
        launcher_sha256=sha(Path(__file__)),review_sha256=sha(OUT/'root_launch_review.json'),
        soft_limit_seconds=3595,hard_limit_seconds=3600,kill_grace_seconds=5,gpu=2,api_spend_usd=0,retry=False))
    start=time.monotonic();timed_out=False;error=None;proc=None;code=None
    try:
        with (OUT/'outer_run.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=proc.wait(timeout=3595)
            except subprocess.TimeoutExpired:
                timed_out=True
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:code=proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    code=proc.wait()
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        if proc is not None and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=proc.wait()
    elapsed=time.monotonic()-start
    result=dict(exit_code=code,timed_out=timed_out,elapsed_seconds=elapsed,hard_limit_seconds=3600,
        soft_limit_seconds=3595,protocol_sha256=sha(OUT/'protocol.json'),error=error,
        future_training_charge='Subtract max(this external elapsed, internal smoke elapsed) from shared3600 seconds',
        retry=False,gpu=2,summary_exists=(OUT/'run/summary.json').exists())
    save(OUT/'outer_terminal.json',result)
    print(json.dumps(result),flush=True)
    assert code==0 and not timed_out and error is None,'Technical smoke failed; preserve artifacts, no retry/fallback'


if __name__=='__main__':main()
