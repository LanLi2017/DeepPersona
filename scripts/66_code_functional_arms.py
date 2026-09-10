#!/usr/bin/env python3
"""Complete the frozen four-arm calibration development experiment."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("runs/swe-diversity-selection/code-functional-development")
EVAL = ROOT / "arm-evaluation"
TRAIN = Path("scripts/63_code_functional_train.py")
HELPER = Path("scripts/62_code_functional_eval.py")
spec = importlib.util.spec_from_file_location("eval62", HELPER)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def prepare():
    assert not (ROOT / "arms_protocol.json").exists() and not EVAL.exists()
    p = json.loads((ROOT / "functional_protocol.json").read_text())
    assert p["calibration_only"] and p["seeds"] == [0, 1, 2]
    paths = [Path(__file__), TRAIN, HELPER, e.HELPER, Path("scripts/65_code_functional_grid_eval.py")]
    paths += [
        ROOT / n
        for n in [
            "functional_protocol.json",
            "training_protocol.json",
            "grid_evaluation_protocol.json",
            "selections.json",
        ]
    ]
    paths += [e.POOL / n for n in ["raw.jsonl", "inputs.jsonl", "evaluator_targets.jsonl", "protocol.json"]]
    e.save(
        ROOT / "arms_protocol.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            arms=list(p["arms"]),
            seeds=p["seeds"],
            task_ids=p["reference_ids_used"],
            recipe="Use the verified65 chosen_recipe unchanged for every arm/seed; no further tuning",
            training="Nine non-typicality arm/seed jobs via frozen63 sequential subprocess; reuse three chosen typicality checkpoints",
            inference="Reuse selected typicality and base grid outputs; evaluate nine new adapters, oneFP32 base load, batch4, TF32off, PEFT unload without merge",
            summaries="All4arms×3seeds, pertask correctness/token-output changes against base and matched-seed typicality, exact realized training token exposure",
            interpretation="Descriptive calibration-tuned development only; no significance tests or functional generalization claim",
            input_sha256={str(p): e.sha(p) for p in paths},
            physical_gpu=2,
            api_spend_usd=0,
        ),
    )
    print("Frozen remaining-arm training/evaluation before arm functional outcomes")


def context():
    p = json.loads((ROOT / "arms_protocol.json").read_text())
    for name, sha in p["input_sha256"].items():
        assert e.sha(Path(name)) == sha
    recipe = json.loads((ROOT / "chosen_recipe.json").read_text())
    assert recipe["grid_protocol_sha256"] == e.sha(ROOT / "grid_evaluation_protocol.json")
    assert recipe["shared_across_all_arms"] and recipe["calibration_only"]
    for name, sha in recipe["evaluation_manifest_sha256"].items():
        folder = ROOT / "grid-evaluation" / name
        assert e.sha(folder / "completed_manifest.json") == sha
        complete(folder)
    assert len(recipe["all_recipes"]) == 4
    assert {(r["learning_rate"], r["epochs"]) for r in recipe["all_recipes"]} == {
        (0.1, 1),
        (0.1, 3),
        (1.0, 1),
        (1.0, 3),
    }
    assert len(recipe["evaluation_manifest_sha256"]) == 13
    for option in recipe["all_recipes"]:
        assert sorted(r["seed"] for r in option["per_seed"]) == p["seeds"]
        assert option["total_correct"] == sum(r["correct"] for r in option["per_seed"])
        for row in option["per_seed"]:
            name = f"model_typicality_seed{row['seed']}_lr{option['learning_rate']:g}_epochs{option['epochs']}"
            summary = json.loads((ROOT / "grid-evaluation" / name / "summary.json").read_text())
            assert row["correct"] == summary["correct"] and summary["tasks"] == 21
    best = min(recipe["all_recipes"], key=lambda r: (-r["total_correct"], r["epochs"], r["learning_rate"]))
    assert (recipe["learning_rate"], recipe["epochs"]) == (best["learning_rate"], best["epochs"])
    jobs = [
        dict(
            arm=arm,
            seed=seed,
            learning_rate=recipe["learning_rate"],
            epochs=recipe["epochs"],
            name=f"{arm}_seed{seed}_lr{recipe['learning_rate']:g}_epochs{recipe['epochs']}",
        )
        for seed in p["seeds"]
        for arm in p["arms"]
    ]
    assert len(jobs) == 12 and sum(j["arm"] != "model_typicality" for j in jobs) == 9
    binding = ROOT / "arms_execution_manifest.json"
    if binding.exists():
        assert json.loads(binding.read_text())["chosen_recipe_sha256"] == e.sha(ROOT / "chosen_recipe.json")
    else:
        e.save(
            binding,
            dict(
                chosen_recipe_sha256=e.sha(ROOT / "chosen_recipe.json"),
                arms_protocol_sha256=e.sha(ROOT / "arms_protocol.json"),
                jobs=jobs,
            ),
        )
    return p, jobs


def complete(folder):
    for name, sha in json.loads((folder / "completed_manifest.json").read_text())["sha256"].items():
        assert e.sha(folder / name) == sha


def checkpoint(job):
    folder = ROOT / "checkpoints" / job["name"]
    m = json.loads((folder / "manifest.json").read_text())
    s = json.loads((folder / "summary.json").read_text())
    assert m["arm"] == job["arm"] and m["selection_order_seed"] == job["seed"] and not m["smoke"]
    assert m["learning_rate"] == job["learning_rate"] and m["epochs"] == job["epochs"]
    assert m["training_protocol_sha256"] == e.sha(ROOT / "training_protocol.json")
    initial = Path("runs/swe-diversity-selection/code-update-transfer-fp32/initial_adapter.pt")
    assert m["initial_adapter_sha256"] == s["initial_adapter_sha256"] == e.sha(initial)
    assert s["technical_checks_passed"] and s["exact_saved_adapter_tensors"] and s["steps"] == 8 * job["epochs"]
    selections = {r["seed"]: r for r in json.loads((ROOT / "selections.json").read_text())["selections"]}
    selected = selections[job["seed"]]["arms"][job["arm"]]
    assert m["groups"] == [[g["group_id"]] for g in selected] * job["epochs"]
    actual = {f.name: e.sha(f) for f in (folder / "adapter").iterdir() if f.is_file()}
    assert actual == s["adapter_sha256"]
    return dict(
        adapter_sha256=actual,
        manifest_sha256=e.sha(folder / "manifest.json"),
        summary_sha256=e.sha(folder / "summary.json"),
    )


def train():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    assert json.loads((ROOT / "smoke_gate.json").read_text())["technical_checks_passed"]
    _, jobs = context()
    for job in jobs:
        if job["arm"] != "model_typicality":
            subprocess.run(
                [
                    sys.executable,
                    str(TRAIN),
                    "train",
                    "--arm",
                    job["arm"],
                    "--seed",
                    str(job["seed"]),
                    "--lr",
                    str(job["learning_rate"]),
                    "--epochs",
                    str(job["epochs"]),
                ],
                check=True,
            )
        checkpoint(job)


def evaluate():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    p, jobs = context()
    checks = {j["name"]: checkpoint(j) for j in jobs}
    assert not EVAL.exists(), "Prior arm outputs immutable; inspect partial runs before restarting"
    rows, targets, grader = e.load_inputs(p["task_ids"])
    EVAL.mkdir()
    e.save(EVAL / "checkpoint_manifest.json", checks)
    torch.manual_seed(e.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tok = AutoTokenizer.from_pretrained(e.MODEL, local_files_only=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            e.MODEL, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = True
    base = {n: (v.data_ptr(), v._version) for n, v in model.named_parameters()}
    for job in [j for j in jobs if j["arm"] != "model_typicality"]:
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
                arms_protocol_sha256=e.sha(ROOT / "arms_protocol.json"),
                helper_sha256=e.sha(HELPER),
                grader_sha256=e.sha(e.HELPER),
                checkpoint=checks[job["name"]],
                chosen_recipe_sha256=e.sha(ROOT / "chosen_recipe.json"),
                versions={k: importlib.metadata.version(k) for k in ["torch", "transformers", "peft"]},
                api_spend_usd=0,
            ),
        )
        active = PeftModel.from_pretrained(
            model, str(ROOT / "checkpoints" / job["name"] / "adapter"), is_trainable=False
        ).eval()
        assert all(v.dtype == torch.float32 for v in active.parameters() if v.is_floating_point())
        records = e.greedy(active, tok, rows, 4, out, "eval")
        model = active.unload()
        del active
        assert {n: (v.data_ptr(), v._version) for n, v in model.named_parameters()} == base
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
        e.save(out / "completed_manifest.json", dict(sha256={f.name: e.sha(f) for f in out.iterdir() if f.is_file()}))
        print(json.dumps(dict(name=job["name"], **summary)), flush=True)


def summarize():
    p, jobs = context()
    assert not (ROOT / "functional_comparison.json").exists()
    raw = {f"{r['task_id']}:{r['sample']}": r for line in (e.POOL / "raw.jsonl").open() if (r := json.loads(line))}
    complete(ROOT / "grid-evaluation/base")
    base_records = {
        r["task_id"]: r for line in (ROOT / "grid-evaluation/base/eval_raw.jsonl").open() if (r := json.loads(line))
    }
    base_grades = {
        r["task_id"]: r for line in (ROOT / "grid-evaluation/base/grades.jsonl").open() if (r := json.loads(line))
    }
    all_rows = {}
    budgets = {}
    hashes = {}
    for job in jobs:
        checkpoint(job)
        folder = (ROOT / "grid-evaluation" if job["arm"] == "model_typicality" else EVAL) / job["name"]
        complete(folder)
        hashes[job["name"]] = e.sha(folder / "completed_manifest.json")
        records = {r["task_id"]: r for line in (folder / "eval_raw.jsonl").open() if (r := json.loads(line))}
        grades = {r["task_id"]: r for line in (folder / "grades.jsonl").open() if (r := json.loads(line))}
        assert set(records) == set(grades) == set(p["task_ids"])
        all_rows[job["name"]] = [
            dict(
                task_id=t,
                correct=int(grades[t]["correct"]),
                base_correct=int(base_grades[t]["correct"]),
                output_changed_vs_base=records[t]["generated_token_ids"] != base_records[t]["generated_token_ids"],
                generated_token_ids=records[t]["generated_token_ids"],
            )
            for t in p["task_ids"]
        ]
        steps = [json.loads(line) for line in (ROOT / "checkpoints" / job["name"] / "steps.jsonl").open()]
        assert len(steps) == 8 * job["epochs"]
        keys = [r["record_id"] for step in steps for r in step["records"]]
        assert len(keys) == 32 * job["epochs"]
        selected = next(
            r for r in json.loads((ROOT / "selections.json").read_text())["selections"] if r["seed"] == job["seed"]
        )["arms"][job["arm"]]
        for step, group in zip(steps, selected * job["epochs"]):
            assert [r["record_id"] for r in step["records"]] == group["record_ids"]
            assert [r["reward"] for r in step["records"]] == group["rewards"]
        budgets[job["name"]] = dict(
            updates=len(steps),
            trajectory_exposures=len(keys),
            correct_exposures=sum(r["reward"] for step in steps for r in step["records"]),
            completion_tokens=sum(len(raw[k]["generated_token_ids"]) for k in keys),
            prompt_tokens=sum(len(raw[k]["prompt_token_ids"]) for k in keys),
        )
    report = []
    for job in jobs:
        rows = all_rows[job["name"]]
        typical = all_rows[f"model_typicality_seed{job['seed']}_lr{job['learning_rate']:g}_epochs{job['epochs']}"]
        for row, t in zip(rows, typical):
            row["typicality_correct"] = t["correct"]
            row["output_changed_vs_typicality"] = row["generated_token_ids"] != t["generated_token_ids"]
        report.append(
            dict(
                **job,
                correct=sum(r["correct"] for r in rows),
                tasks=21,
                pass_at_1=sum(r["correct"] for r in rows) / 21,
                output_changes_vs_base=sum(r["output_changed_vs_base"] for r in rows),
                output_changes_vs_typicality=sum(r["output_changed_vs_typicality"] for r in rows),
                wrong_to_right_vs_base=sum(r["correct"] > r["base_correct"] for r in rows),
                right_to_wrong_vs_base=sum(r["correct"] < r["base_correct"] for r in rows),
                training_budget=budgets[job["name"]],
                per_task=rows,
            )
        )
    for seed in p["seeds"]:
        values = [r["training_budget"]["completion_tokens"] for r in report if r["seed"] == seed]
        assert max(values) / min(values) <= 1.05
    e.save(
        ROOT / "functional_comparison.json",
        dict(
            base_correct=sum(r["correct"] for r in base_grades.values()),
            calibration_tasks=21,
            arms=report,
            aggregate={
                arm: dict(
                    correct_sum=sum(r["correct"] for r in report if r["arm"] == arm),
                    evaluations=63,
                    mean_accuracy=sum(r["correct"] for r in report if r["arm"] == arm) / 63,
                )
                for arm in p["arms"]
            },
            selected_recipe=json.loads((ROOT / "chosen_recipe.json").read_text()),
            evaluation_manifest_sha256=hashes,
            interpretation=p["interpretation"],
            measurement_or_final_test_scored=False,
            api_spend_usd=0,
        ),
    )
    print(json.dumps({r["name"]: r["correct"] for r in report}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "summarize", "run"])
    args = parser.parse_args()
    if args.stage == "run":
        train()
        evaluate()
        summarize()
    else:
        {"prepare": prepare, "train": train, "evaluate": evaluate, "summarize": summarize}[args.stage]()
