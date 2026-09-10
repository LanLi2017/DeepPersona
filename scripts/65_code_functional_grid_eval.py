#!/usr/bin/env python3
"""Evaluate the frozen calibration grid and choose its shared training recipe."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path("runs/swe-diversity-selection/code-functional-development")
EVAL = ROOT / "grid-evaluation"
HELPER = Path(__file__).with_name("62_code_functional_eval.py")
spec = importlib.util.spec_from_file_location("functional_eval62", HELPER)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def prepare():
    assert not (ROOT / "grid_evaluation_protocol.json").exists()
    assert not EVAL.exists(), "Freeze before any functional grid outcomes"
    p = json.loads((ROOT / "functional_protocol.json").read_text())
    t = json.loads((ROOT / "training_protocol.json").read_text())
    assert t["source_protocol_sha256"] == e.sha(ROOT / "functional_protocol.json")
    assert t["runner_sha256"] == e.sha(Path("scripts/63_code_functional_train.py"))
    assert p["seeds"] == [0, 1, 2] and p["grid"]["learning_rates"] == [0.1, 1.0] and p["grid"]["epochs"] == [1, 3]
    assert p["grid"]["tuning_arm"] == "model_typicality" and p["calibration_only"]
    rows, _, _ = e.load_inputs(p["reference_ids_used"])
    assert len(rows) == 21
    jobs = [dict(name="base", adapter=None)]
    for seed in p["seeds"]:
        for epochs in p["grid"]["epochs"]:
            for lr in p["grid"]["learning_rates"]:
                name = f"model_typicality_seed{seed}_lr{lr:g}_epochs{epochs}"
                jobs.append(
                    dict(
                        name=name,
                        adapter=str(ROOT / "checkpoints" / name / "adapter"),
                        seed=seed,
                        epochs=epochs,
                        learning_rate=lr,
                    )
                )
    paths = [
        Path(__file__),
        HELPER,
        e.HELPER,
        ROOT / "functional_protocol.json",
        ROOT / "training_protocol.json",
        ROOT / "selections.json",
        e.POOL / "inputs.jsonl",
        e.POOL / "evaluator_targets.jsonl",
        e.POOL / "protocol.json",
    ]
    e.save(
        ROOT / "grid_evaluation_protocol.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            jobs=jobs,
            task_ids=[r["task_id"] for r in rows],
            split="calibration_only",
            physical_gpu=2,
            batch_size=4,
            dtype="float32",
            tf32=False,
            model_snapshot=str(e.MODEL),
            seed=e.SEED,
            generation=dict(do_sample=False, max_new_tokens=768, enable_thinking=False),
            adapter_loading="Load standardPEFT adapter then unload without merge; one unchanged base model for all jobs",
            choose="Maximize sum correct across all3seeds (equivalent mean accuracy on identical21tasks); exact ties fewer epochs then lowerLR; retain all4recipes",
            no_measurement_or_final_test=True,
            checkpoint_hashes="Capture all12 completed checkpoints before generation",
            input_sha256={str(p): e.sha(p) for p in paths},
            api_spend_usd=0,
        ),
    )
    print("Frozen13 calibration jobs before functional grid outcomes")


def protocol():
    p = json.loads((ROOT / "grid_evaluation_protocol.json").read_text())
    for name, sha in p["input_sha256"].items():
        assert e.sha(Path(name)) == sha
    return p


def run():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = protocol()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    assert json.loads((ROOT / "smoke_gate.json").read_text())["technical_checks_passed"]
    assert not EVAL.exists(), "Use fresh grid outputs; inspect partial runs before restarting"
    checkpoints = {}
    for job in p["jobs"][1:]:
        folder = Path(job["adapter"]).parent
        summary = json.loads((folder / "summary.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        assert summary["technical_checks_passed"] and summary["exact_saved_adapter_tensors"]
        assert manifest["arm"] == "model_typicality" and not manifest["smoke"]
        assert manifest["training_protocol_sha256"] == e.sha(ROOT / "training_protocol.json")
        for key, other in [("seed", "selection_order_seed"), ("epochs", "epochs"), ("learning_rate", "learning_rate")]:
            assert job[key] == manifest[other]
        assert summary["steps"] == 8 * job["epochs"]
        adapter = Path(job["adapter"])
        actual = {file.name: e.sha(file) for file in adapter.iterdir() if file.is_file()}
        assert actual == summary["adapter_sha256"]
        checkpoints[job["name"]] = dict(
            adapter_sha256=actual,
            summary_sha256=e.sha(folder / "summary.json"),
            manifest_sha256=e.sha(folder / "manifest.json"),
        )
    rows, targets, grader = e.load_inputs(p["task_ids"])
    EVAL.mkdir()
    e.save(EVAL / "checkpoint_manifest.json", checkpoints)
    torch.manual_seed(p["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(e.MODEL, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            e.MODEL, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = True
    base_state = {name: (tensor.data_ptr(), tensor._version) for name, tensor in model.named_parameters()}
    versions = {k: importlib.metadata.version(k) for k in ["torch", "transformers", "peft"]}
    for job in p["jobs"]:
        out = EVAL / job["name"]
        out.mkdir()
        e.save(
            out / "manifest.json",
            dict(
                job=job,
                task_ids=p["task_ids"],
                batch_size=4,
                physical_gpu=2,
                precision="FP32,TF32off",
                grid_protocol_sha256=e.sha(ROOT / "grid_evaluation_protocol.json"),
                helper_sha256=e.sha(HELPER),
                grader_sha256=e.sha(e.HELPER),
                versions=versions,
                checkpoint=checkpoints.get(job["name"]),
                api_spend_usd=0,
            ),
        )
        active = (
            PeftModel.from_pretrained(model, job["adapter"], is_trainable=False).eval() if job["adapter"] else model
        )
        assert all(t.dtype == torch.float32 for t in active.parameters() if t.is_floating_point())
        records = e.greedy(active, tokenizer, rows, 4, out, "eval")
        if job["adapter"]:
            model = active.unload()
            del active
        assert {name: (tensor.data_ptr(), tensor._version) for name, tensor in model.named_parameters()} == base_state
        grades = []
        with (out / "grades.jsonl").open("w") as handle:
            for row in records:
                result = dict(task_id=row["task_id"], **grader.grade(row["generation"], targets[row["task_id"]]))
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                grades.append(result)
        summary = dict(
            tasks=len(grades),
            correct=sum(r["correct"] for r in grades),
            pass_at_1=sum(r["correct"] for r in grades) / len(grades),
            valid_python=sum(r["valid_python"] for r in grades),
            truncated=sum(r["finish_reason"] == "length" for r in records),
            generated_tokens=sum(len(r["generated_token_ids"]) for r in records),
            base_parameter_pointers_and_versions_unchanged=True,
            measurement_or_final_test_scored=False,
            api_spend_usd=0,
        )
        e.save(out / "summary.json", summary)
        e.save(
            out / "completed_manifest.json",
            dict(sha256={file.name: e.sha(file) for file in out.iterdir() if file.is_file()}),
        )
        print(json.dumps(dict(name=job["name"], **summary)), flush=True)


def choose():
    p = protocol()
    assert not (ROOT / "chosen_recipe.json").exists(), "Shared recipe already selected"
    candidates = {}
    hashes = {}
    for job in p["jobs"]:
        folder = EVAL / job["name"]
        for name, sha in json.loads((folder / "completed_manifest.json").read_text())["sha256"].items():
            assert e.sha(folder / name) == sha
        summary = json.loads((folder / "summary.json").read_text())
        grades = [json.loads(line) for line in (folder / "grades.jsonl").open()]
        assert [r["task_id"] for r in grades] == p["task_ids"] and len(grades) == 21
        assert sum(r["correct"] for r in grades) == summary["correct"]
        hashes[job["name"]] = e.sha(folder / "completed_manifest.json")
        if job["name"] == "base":
            base = summary
            continue
        key = (job["learning_rate"], job["epochs"])
        candidates.setdefault(key, []).append(
            dict(seed=job["seed"], correct=summary["correct"], tasks=21, pass_at_1=summary["pass_at_1"])
        )
    options = []
    for (lr, epochs), rows in candidates.items():
        assert sorted(r["seed"] for r in rows) == [0, 1, 2]
        total = sum(r["correct"] for r in rows)
        options.append(
            dict(
                learning_rate=lr,
                epochs=epochs,
                total_correct=total,
                total_evaluations=63,
                mean_calibration_accuracy=total / 63,
                per_seed=rows,
            )
        )
    assert len(options) == 4
    options.sort(key=lambda r: (-r["total_correct"], r["epochs"], r["learning_rate"]))
    e.save(
        ROOT / "chosen_recipe.json",
        dict(
            learning_rate=options[0]["learning_rate"],
            epochs=options[0]["epochs"],
            selected=options[0],
            all_recipes=options,
            base=base,
            tuning_arm="model_typicality",
            shared_across_all_arms=True,
            selection_rule=p["choose"],
            calibration_only=True,
            measurement_or_final_test_scored=False,
            evaluation_manifest_sha256=hashes,
            grid_protocol_sha256=e.sha(ROOT / "grid_evaluation_protocol.json"),
            api_spend_usd=0,
        ),
    )
    print(json.dumps(options[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run", "choose"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "choose": choose}[args.stage]()
