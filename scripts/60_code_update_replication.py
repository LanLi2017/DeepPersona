#!/usr/bin/env python3
"""Run unchanged frozen group/learner/analysis code on new source tasks."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path("runs/swe-diversity-selection")
OUT = ROOT / "code-update-replication"
POOL = OUT / "input_pool"
OLD = ROOT / "code-learning-pilot"
NEW = ROOT / "code-learning-expansion"
REF = ROOT / "code-update-transfer-fp32"
HELPERS = {
    "groups": "54_code_update_groups.py",
    "learner": "57_code_update_precision.py",
    "analysis": "56_code_update_analysis.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.open()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def helper(name):
    path = Path(__file__).with_name(HELPERS[name])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POOL, module.OUT = POOL, OUT
    return module


def prepare():
    assert not OUT.exists(), "Replication already prepared"
    POOL.mkdir(parents=True)
    old, new = [json.loads((p / "protocol.json").read_text()) for p in [OLD, NEW]]
    assert not set(old["train_task_ids"]) & set(new["train_task_ids"])
    assert len(new["train_task_ids"]) == 88
    for name in ["raw.jsonl", "grades.jsonl"]:
        shutil.copy2(NEW / name, POOL / name)
    for name in ["inputs.jsonl", "evaluator_targets.jsonl"]:
        rows = read(NEW / name) + [r for r in read(OLD / name) if r["split"] != "train"]
        (POOL / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    save(
        POOL / "protocol.json",
        {
            **new,
            "calibration_task_ids": old["calibration_task_ids"],
            "measurement_task_ids": old["measurement_task_ids"],
        },
    )
    groups = helper("groups")
    groups.prepare(True)
    groups.prepare(False)
    summary = json.loads((OUT / "group_summary.json").read_text())
    protocol = json.loads((REF / "learner_protocol.json").read_text())
    protocol.update(
        utc=datetime.now(timezone.utc).isoformat(),
        input_hashes={
            str(p): sha(p)
            for p in [
                OUT / "groups.json",
                ROOT / "code-update-transfer/initial_adapter.pt",
                POOL / "raw.jsonl",
                POOL / "inputs.jsonl",
                POOL / "evaluator_targets.jsonl",
            ]
        },
    )
    protocol["replication"] = (
        "New source tasks only; same frozen model, exact initial adapter, FP32, group rules, max12 tasks, steps, calibration21 and measurement21. Shared measurement set remains development, not fresh evaluation-task evidence."
    )
    save(OUT / "learner_protocol.json", protocol)
    analysis = json.loads((REF / "analysis_protocol.json").read_text())
    analysis.update(
        utc=protocol["utc"],
        input_sha256={name: sha(OUT / name) for name in ["groups.json", "group_protocol.json", "records.jsonl"]},
    )
    analysis["null"] = (
        "Zero task-mean selection lift; exact two-sided task sign-flips assume sign symmetry; at most12 independent training-task clusters."
    )
    analysis["limitations"] = (
        "Task-disjoint source replication with at most12 source tasks; overlapping groups; reused fixed21 measurement references; same adapter/model/seed; NLL not execution success."
    )
    analysis.pop("wrapper_sha256", None)
    analysis["replication_source_protocol_sha256"] = sha(REF / "analysis_protocol.json")
    save(OUT / "analysis_protocol.json", analysis)
    save(
        OUT / "replication_protocol.json",
        {
            "utc": protocol["utc"],
            "scope": protocol["replication"],
            "helper_sha256": {name: sha(Path(__file__).with_name(filename)) for name, filename in HELPERS.items()},
            "wrapper_sha256": sha(Path(__file__)),
            "source_inputs_sha256": {
                str(p / name): sha(p / name)
                for p in [OLD, NEW]
                for name in ["protocol.json", "inputs.jsonl", "evaluator_targets.jsonl", "raw.jsonl", "grades.jsonl"]
            },
            "independent_training_tasks": summary["selected_tasks"],
            "api_spend_usd": 0,
        },
    )
    print(json.dumps(summary, indent=2))


def verify():
    protocol = json.loads((OUT / "replication_protocol.json").read_text())
    assert sha(Path(__file__)) == protocol["wrapper_sha256"]
    for name, expected in protocol["helper_sha256"].items():
        assert sha(Path(__file__).with_name(HELPERS[name])) == expected
    for name, expected in protocol["source_inputs_sha256"].items():
        assert sha(Path(name)) == expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run", "analyze"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        verify()
        if args.stage == "run":
            helper("learner").run(args.smoke)
        else:
            helper("analysis").analyze()
