#!/usr/bin/env python3
"""External deadline for fixed40 autonomous inference; preserve all slots on every terminal path."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/swe-diversity-selection/swe-localize-repair-support/autonomous_inference"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2) + "\n")


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    auth = json.loads((OUT / "root_launch_review.json").read_text())
    assert auth["allow_autonomous_inference"] and auth["reason"]
    assert auth["protocol_sha256"] == sha(OUT / "protocol.json")
    assert auth["launcher_sha256"] == sha(Path(__file__))
    assert auth["inference_script_sha256"] == sha(ROOT / "scripts/135_swe_localize_inference.py")
    assert protocol["external_timeout"]["term_seconds"] == 7195
    assert protocol["external_timeout"]["kill_grace_seconds"] == 5
    assert protocol["limits"]["global_wall_seconds"] == 7200
    assert (OUT / "launch_manifest.json").exists()
    assert auth["launch_manifest_sha256"] == sha(OUT / "launch_manifest.json")
    assert not (OUT / "outer_dispatch.json").exists() and not (OUT / "outer_terminal.json").exists()
    assert not (OUT / "run").exists(), "No retry of any attempted inference"
    gpu = (
        subprocess.check_output(
            ["nvidia-smi", "-i", "2", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        .strip()
        .split(",")
    )
    assert gpu[0].strip() == "2" and gpu[1].strip() == protocol["hardware"]["uuid"]
    assert int(gpu[2].strip()) < 100, "GPU2 is occupied; do not collide with another job"
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="2",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        SWE_INFERENCE_EXTERNAL_TIMEOUT="7195+5",
        NVIDIA_TF32_OVERRIDE="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONHASHSEED="0",
    )
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/135_swe_localize_inference.py"), "worker"]
    save(
        OUT / "outer_dispatch.json",
        dict(
            command=cmd,
            protocol_sha256=sha(OUT / "protocol.json"),
            launcher_sha256=sha(Path(__file__)),
            review_sha256=sha(OUT / "root_launch_review.json"),
            soft_limit_seconds=7195,
            hard_limit_seconds=7200,
            kill_grace_seconds=5,
            gpu=2,
            api_spend_usd=0,
            retry=False,
        ),
    )
    start = time.monotonic()
    timed_out = False
    error = None
    proc = None
    code = None
    cancel_signal = None
    cleaning = False

    def cancelled(signum, frame):
        nonlocal cancel_signal
        cancel_signal = signum
        # A signal during Popen creation is handled as soon as its PID is assigned.
        if proc is not None and not cleaning:
            raise InterruptedError(f"Launcher received signal {signum}")

    old_term = signal.signal(signal.SIGTERM, cancelled)
    old_int = signal.signal(signal.SIGINT, cancelled)
    residual_group_killed = False
    try:
        with (OUT / "outer_run.log").open("x") as log:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            if cancel_signal is not None:
                raise InterruptedError(f"Launcher received signal {cancel_signal} during child creation")
            try:
                code = proc.wait(timeout=max(0.0, start + 7195 - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    code = proc.wait(timeout=max(0.0, start + 7200 - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = proc.wait()
    except BaseException as exc:
        cleaning = True
        error = dict(type=type(exc).__name__, message=str(exc))
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = proc.wait()
    cleaning = True
    # The leader can exit while descendants ignore TERM. Kill the entire remaining
    # session group on every terminal path, including otherwise successful exit.
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            residual_group_killed = True
        except ProcessLookupError:
            pass
        if proc.poll() is None:
            code = proc.wait()
    elapsed = time.monotonic() - start
    result = dict(
        exit_code=code,
        timed_out=timed_out,
        elapsed_seconds=elapsed,
        hard_limit_seconds=7200,
        soft_limit_seconds=7195,
        protocol_sha256=sha(OUT / "protocol.json"),
        error=error,
        budget_charge="External elapsed includes worker model loading and all inference; separate from training budget",
        retry=False,
        gpu=2,
        residual_process_group_killed=residual_group_killed,
        absolute_deadline_from_child_creation_start=True,
        worker_summary_exists=(OUT / "run/worker_summary.json").exists(),
    )
    try:
        save(OUT / "outer_terminal.json", result)
        print(json.dumps(result), flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/135_swe_localize_inference.py"), "collect"], cwd=ROOT, check=True)
    assert code == 0 and not timed_out and error is None, "Incomplete inference; all slots preserved, no retry"


if __name__ == "__main__":
    main()
