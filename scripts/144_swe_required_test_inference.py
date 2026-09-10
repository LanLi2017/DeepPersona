#!/usr/bin/env python3
"""Explicit new endpoint/path binding; unchanged135 public autonomous controller."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "runs/swe-diversity-selection/swe-localize-repair-support"
OLD = SUPPORT / "autonomous_inference"
OUT = SUPPORT / "autonomous_inference_required_tests"
SPEC = importlib.util.spec_from_file_location("required_inference135", ROOT / "scripts/135_swe_localize_inference.py")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
h.OUT = OUT
h.READY = SUPPORT / "evaluation_required_tests"
h.TRAIN = SUPPORT / "full_training_required_tests"
PLAN = ROOT / "docs/swe_required_test_control_plan.md"


def freeze():
    assert not (OUT / "protocol.json").exists() and not (OUT / "run").exists()
    old_protocol = h.read(OLD / "protocol.json")
    h.verify(old_protocol["source_sha256"])
    assert not (OLD / "outer_dispatch.json").exists() and not (OLD / "run").exists()
    assert not (SUPPORT / "full_training/outer_dispatch.json").exists()
    assert not (SUPPORT / "full_training/run").exists()
    checks = h.read(OLD / "synthetic_checks.json")
    assert checks["all_checks_passed"] and checks["source_sha256"] == h.sha(
        ROOT / "scripts/135_swe_localize_inference.py"
    )
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "synthetic_checks.json").exists()
    shutil.copyfile(OLD / "synthetic_checks.json", OUT / "synthetic_checks.json")
    for name in ["launcher_checks.json", "audit_launcher.py"]:
        assert not (OUT / name).exists()
        shutil.copyfile(OLD / name, OUT / name)
    h.save(
        OUT / "inherited_controller_checks.json",
        dict(
            original_report_sha256=h.sha(OLD / "synthetic_checks.json"),
            inherited_launcher_report_sha256=h.sha(OLD / "launcher_checks.json"),
            inherited_launcher_driver_sha256=h.sha(OLD / "audit_launcher.py"),
            interpretation="Reuse exact135 controller checks, not a claim of independently rerunning new wrapper.",
            inherited_controller_sha256=h.sha(ROOT / "scripts/135_swe_localize_inference.py"),
            new_wrapper_sha256=h.sha(Path(__file__)),
            model_calls=0,
        ),
    )
    h.freeze()
    protocol = h.read(OUT / "protocol.json")
    assert protocol["limits"] == old_protocol["limits"]
    assert protocol["schedule"] == old_protocol["schedule"]
    assert protocol["evaluation_initial_manifest_sha256"] == old_protocol["evaluation_initial_manifest_sha256"]
    paths = [
        Path(__file__),
        ROOT / "scripts/146_swe_required_test_inference_launcher.py",
        ROOT / "scripts/143_swe_required_test_train.py",
        ROOT / "scripts/145_swe_required_test_train_launcher.py",
        ROOT / "scripts/141_swe_required_test_grading.py",
        ROOT / "scripts/142_swe_required_test_readiness.py",
        ROOT / "scripts/147_swe_required_test_candidates.py",
        ROOT / "scripts/148_swe_required_test_candidate_launcher.py",
        ROOT / "scripts/139_swe_localize_candidate_scoring.py",
        ROOT / "scripts/140_swe_localize_candidate_launcher.py",
        PLAN,
        OLD / "protocol.json",
        OUT / "inherited_controller_checks.json",
    ]
    assert all(path.is_file() for path in paths)
    protocol["source_sha256"].update({str(path): h.sha(path) for path in paths})
    protocol.update(
        prospective_endpoint="Required literal PASS tests plus completed-exit/status integrity; strict whole-command sensitivity retained separately.",
        supersession="New namespace before any model run. The new control plan supersedes endpoint and artifact paths in inherited original documents; public controller/limits are identical.",
        primary_plan=str(PLAN),
        primary_plan_sha256=h.sha(PLAN),
        inherited_protocol_sha256=h.sha(OLD / "protocol.json"),
        worker_script=str(Path(__file__).resolve()),
        external_launcher=str(ROOT / "scripts/146_swe_required_test_inference_launcher.py"),
        original_strict_attempt_preserved=True,
    )
    # Complete this one freeze before any dispatch; the prior protocol is in OLD.
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(dict(frozen=True, slots=40, protocol_sha256=h.sha(OUT / "protocol.json"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "bind", "worker", "collect"])
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "bind":
        h.launch_bind()
    elif args.stage == "worker":
        h.worker()
    else:
        h.collect()


if __name__ == "__main__":
    main()
