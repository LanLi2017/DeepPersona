#!/usr/bin/env python3
"""Separate required-test143 control with unchanged133 absolute remaining-budget deadline."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/swe-diversity-selection/swe-localize-repair-support/full_training_required_tests"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2) + "\n")


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    auth = json.loads((OUT / "root_launch_review.json").read_text())
    assert auth["allow_full_training"] and auth["reason"]
    assert auth["protocol_sha256"] == sha(OUT / "protocol.json")
    assert auth["launcher_sha256"] == sha(Path(__file__))
    assert auth["training_script_sha256"] == sha(ROOT / "scripts/143_swe_required_test_train.py")
    assert auth["helper_sha256"] == sha(ROOT / "scripts/swe_localize_learner_ops.py")
    hard_limit = protocol["remaining_seconds"]
    soft_limit = protocol["external_timeout"]["term_seconds"]
    assert protocol["external_timeout"]["kill_grace_seconds"] == 5
    assert protocol["external_timeout"]["hard_ceiling_seconds"] == hard_limit
    assert soft_limit == hard_limit - 5 and soft_limit > 0
    assert protocol["total_shared_gpu_budget_seconds"] == 3600
    assert protocol["charged_prior_seconds"] + hard_limit == 3600
    assert protocol["forecast_seconds"] <= hard_limit
    assert not (OUT / "outer_dispatch.json").exists() and not (OUT / "outer_terminal.json").exists()
    assert not (OUT / "run").exists(), "No retry of any attempted full training"
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
        SWE_FULL_EXTERNAL_TIMEOUT=json.dumps(protocol["external_timeout"], sort_keys=True),
        NVIDIA_TF32_OVERRIDE="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONHASHSEED="0",
    )
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/143_swe_required_test_train.py"), "run"]
    save(
        OUT / "outer_dispatch.json",
        dict(
            command=cmd,
            protocol_sha256=sha(OUT / "protocol.json"),
            launcher_sha256=sha(Path(__file__)),
            review_sha256=sha(OUT / "root_launch_review.json"),
            soft_limit_seconds=soft_limit,
            hard_limit_seconds=hard_limit,
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
                code = proc.wait(timeout=max(0.0, start + soft_limit - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    code = proc.wait(timeout=max(0.0, start + hard_limit - time.monotonic()))
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
        hard_limit_seconds=hard_limit,
        soft_limit_seconds=soft_limit,
        protocol_sha256=sha(OUT / "protocol.json"),
        error=error,
        charged_prior_seconds=protocol["charged_prior_seconds"],
        total_shared_charge_seconds=protocol["charged_prior_seconds"] + elapsed,
        accounting="Any total charged time uses prior smoke plus max(full internal, this outer elapsed). No retries or extra epochs.",
        retry=False,
        gpu=2,
        residual_process_group_killed=residual_group_killed,
        absolute_deadline_from_child_creation_start=True,
        summary_exists=(OUT / "run/summary.json").exists(),
    )
    try:
        save(OUT / "outer_terminal.json", result)
        print(json.dumps(result), flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    assert code == 0 and not timed_out and error is None, "Full training failed; preserve artifacts, no retry/fallback"


if __name__ == "__main__":
    main()
