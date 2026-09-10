#!/usr/bin/env python3
"""External per-candidate process groups; one timeout cannot censor all later slots."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/swe-diversity-selection/swe-localize-repair-support/candidate_evaluation'
SOURCE = ROOT/'scripts/139_swe_localize_candidate_scoring.py'
spec = importlib.util.spec_from_file_location('candidate139',SOURCE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
sha, read, save = s.sha, s.read, s.save


def controlled(cmd, cwd, env, log_path, soft_deadline, hard_deadline):
    """Absolute deadlines and deferred cancellation cover Popen's PID-assignment gap."""
    proc, code, cancel = None, None, None
    cleaning, timed_out, residual = False, False, False
    error = None
    started = time.monotonic()
    def cancelled(signum, frame):
        nonlocal cancel
        cancel = signum
        if proc is not None and not cleaning:
            raise InterruptedError(f'Launcher received signal {signum}')
    old = {sig:signal.signal(sig,cancelled) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        with log_path.open('x') as log:
            proc = subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if cancel is not None:
                raise InterruptedError(f'Cancellation during Popen: {cancel}')
            try:
                code = proc.wait(timeout=max(0,soft_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    code = proc.wait(timeout=max(0,hard_deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = proc.wait()
    except BaseException as exc:
        cleaning = True
        error = dict(type=type(exc).__name__,message=str(exc))
    finally:
        cleaning = True
        if proc is not None:
            try:
                os.killpg(proc.pid,signal.SIGKILL)
                residual = True
            except ProcessLookupError:
                pass
            if proc.poll() is None:
                code = proc.wait()
        for sig, previous in old.items():
            signal.signal(sig,previous)
    if cancel is not None and error is None:
        error = dict(type='InterruptedError', message=f'Cancellation during cleanup: {cancel}')
    return dict(exit_code=code,timed_out=timed_out,error=error,cancel_signal=cancel,
        elapsed_seconds=time.monotonic()-started,residual_group_killed=residual,
        soft_deadline=soft_deadline,hard_deadline=hard_deadline,
        all_descendants_share_candidate_process_group=True)


def main():
    protocol = read(OUT/'protocol.json')
    auth = read(OUT/'root_execution_review.json')
    assert auth['allow_candidate_correctness'] and auth['protocol_sha256']==sha(OUT/'protocol.json')
    assert auth['preparation_manifest_sha256']==sha(OUT/'preparation_manifest.json')
    assert auth['launcher_sha256']==sha(Path(__file__)) and auth['scorer_sha256']==sha(SOURCE)
    s.verify(protocol['source_sha256'])
    s.verify(read(OUT/'preparation_manifest.json'))
    assert protocol['external_timeout']==dict(global_seconds=7200,per_candidate_seconds=210,kill_grace_seconds=5)
    s.validate_slots(protocol['slots'])
    assert not (OUT/'outer_dispatch.json').exists() and not (OUT/'outer_terminal.json').exists()
    save(OUT/'outer_dispatch.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
         review_sha256=sha(OUT/'root_execution_review.json'),launcher_sha256=sha(Path(__file__)),
         hard_seconds=7200,per_candidate_hard_seconds=210,kill_grace_seconds=5,no_retry=True))
    started = time.monotonic()
    global_deadline = started+7200
    completed, error = [], None
    def outer_cancel(signum, frame):
        raise InterruptedError(f'Outer candidate launcher received signal {signum}')
    original_handlers = {sig:signal.signal(sig,outer_cancel) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in protocol['slots']:
            key = slot['episode_id']
            if global_deadline-time.monotonic() <= 5:
                break
            folder = OUT/'execution'/key
            folder.mkdir(parents=True)
            if slot['stage'] != 'ready':
                save(folder/'result.json',dict(episode_id=key,instance_id=slot['instance_id'],arm=slot['arm'],
                    resolved=False,correctness_observed=False,execution_status=slot['stage'],
                    generation_status=slot['generation_status']))
                completed.append(key)
                continue
            hard = min(global_deadline,time.monotonic()+210)
            soft = hard-5
            env = s.b.environment()
            env['SWE_CANDIDATE_EXTERNAL_SLOT'] = key
            cmd = [str(ROOT/'.venv/bin/python'),str(SOURCE),'score-one','--episode-id',key]
            receipt = controlled(cmd,ROOT,env,folder/'outer.log',soft,hard)
            save(OUT/'receipts'/f'{key}.json',dict(**receipt,episode_id=key,
                 protocol_sha256=sha(OUT/'protocol.json'),command=cmd))
            completed.append(key)
            if receipt['cancel_signal'] is not None:
                error = receipt['error']
                break
    except BaseException as exc:
        error = dict(type=type(exc).__name__,message=str(exc))
    finally:
        # Once dispatch starts, preserve every terminal/unstarted slot even on cancellation.
        for sig in original_handlers:
            signal.signal(sig,signal.SIG_IGN)
        try:
            try:
                save(OUT/'outer_terminal.json',dict(completed_slots=len(completed),assigned_slots=40,
                     elapsed_seconds=time.monotonic()-started,execution_budget_seconds=7200,
                     global_budget_exhausted=time.monotonic()>=global_deadline-5,
                     error=error,protocol_sha256=sha(OUT/'protocol.json'),no_retry=True))
            finally:
                s.collect()
        finally:
            for sig, previous in original_handlers.items():
                signal.signal(sig,previous)
    assert error is None, 'Candidate execution interrupted; all slots retained, no retry'


if __name__=='__main__':
    main()
