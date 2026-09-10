#!/usr/bin/env python3
"""Independent CPU audit of the completed four-arm calibration experiment."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("runs/swe-diversity-selection/code-functional-development")
POOL = Path("runs/swe-diversity-selection/code-learning-pilot")
OUT = ROOT / "independent_audit"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(line) for line in p.open()]


def save(p, value):
    p.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    assert not OUT.exists()
    OUT.mkdir()
    paths = [
        Path(__file__),
        Path("scripts/63_code_functional_train.py"),
        Path("scripts/65_code_functional_grid_eval.py"),
        Path("scripts/66_code_functional_arms.py"),
    ]
    paths += [
        ROOT / n for n in ["functional_protocol.json", "chosen_recipe.json", "selections.json", "arms_protocol.json"]
    ]
    save(
        OUT / "protocol.json",
        dict(
            scope="CPU artifact audit only; no new metric, grading run or model execution",
            checks=[
                "All checkpoint/evaluation source hashes",
                "Exact shared recipe and original adapter",
                "Realized group order, reward counts, token budgets",
                "Every per-task output/correctness comparison",
                "Calibration-only prompts and task IDs",
            ],
            input_sha256={str(p): sha(p) for p in paths},
            api_spend_usd=0,
        ),
    )


def audit():
    import torch
    from safetensors.torch import load_file

    assert not (OUT / "results.json").exists()
    for name, expected in read(OUT / "protocol.json")["input_sha256"].items():
        assert sha(Path(name)) == expected
    protocol = read(ROOT / "functional_protocol.json")
    pool = read(POOL / "protocol.json")
    chosen = read(ROOT / "chosen_recipe.json")
    report = read(ROOT / "functional_comparison.json")
    assert report["selected_recipe"] == chosen
    calibration = protocol["reference_ids_used"]
    assert len(calibration) == len(set(calibration)) == 21
    assert not set(calibration) & set(pool["measurement_task_ids"] + pool["later_test_task_ids"])
    for name, expected in chosen["evaluation_manifest_sha256"].items():
        folder = ROOT / "grid-evaluation" / name
        assert sha(folder / "completed_manifest.json") == expected
        for filename, filehash in read(folder / "completed_manifest.json")["sha256"].items():
            assert sha(folder / filename) == filehash
    options = []
    for option in chosen["all_recipes"]:
        correct = []
        for seed in [0, 1, 2]:
            name = f"model_typicality_seed{seed}_lr{option['learning_rate']:g}_epochs{option['epochs']}"
            grades = lines(ROOT / "grid-evaluation" / name / "grades.jsonl")
            assert [r["task_id"] for r in grades] == calibration
            correct.append(sum(r["correct"] for r in grades))
        assert sum(correct) == option["total_correct"]
        options.append((sum(correct), option["epochs"], option["learning_rate"]))
    assert len(options) == 4
    best = min(options, key=lambda x: (-x[0], x[1], x[2]))
    assert (chosen["epochs"], chosen["learning_rate"]) == best[1:]
    selections = {r["seed"]: r for r in read(ROOT / "selections.json")["selections"]}
    originals = {f"{r['task_id']}:{r['sample']}": r for r in lines(POOL / "raw.jsonl")}
    source_grades = {f"{r['task_id']}:{r['sample']}": r for r in lines(POOL / "grades.jsonl")}
    prompts = {r["task_id"]: r["prompt"] for r in lines(POOL / "inputs.jsonl") if r["split"] == "calibration"}
    initial_path = Path("runs/swe-diversity-selection/code-update-transfer-fp32/initial_adapter.pt")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    evaluations = {}
    hashes_checked = 0
    for name in ["base"] + [r["name"] for r in report["arms"]]:
        folder = (
            ROOT / "grid-evaluation"
            if name == "base" or name.startswith("model_typicality_")
            else ROOT / "arm-evaluation"
        ) / name
        for file, expected in read(folder / "completed_manifest.json")["sha256"].items():
            assert sha(folder / file) == expected
            hashes_checked += 1
        manifest = read(folder / "manifest.json")
        summary = read(folder / "summary.json")
        raw = lines(folder / "eval_raw.jsonl")
        grades = lines(folder / "grades.jsonl")
        assert manifest["task_ids"] == [r["task_id"] for r in raw] == [r["task_id"] for r in grades] == calibration
        assert manifest["batch_size"] == 4 and manifest["precision"] == "FP32,TF32off"
        assert not summary["measurement_or_final_test_scored"]
        assert summary["correct"] == sum(r["correct"] for r in grades)
        assert summary["generated_tokens"] == sum(len(r["generated_token_ids"]) for r in raw)
        for i, r in enumerate(raw):
            assert r["user_prompt"] == prompts[r["task_id"]]
            assert r["batch_task_ids"] == calibration[i // 4 * 4 : i // 4 * 4 + 4]
        evaluations[name] = (raw, grades)
    result = []
    for arm in report["arms"]:
        name = arm["name"]
        seed = arm["seed"]
        epochs = chosen["epochs"]
        assert arm["learning_rate"] == chosen["learning_rate"] and arm["epochs"] == epochs
        checkpoint = ROOT / "checkpoints" / name
        m = read(checkpoint / "manifest.json")
        s = read(checkpoint / "summary.json")
        assert m["arm"] == arm["arm"] and m["selection_order_seed"] == seed
        assert m["initial_adapter_sha256"] == s["initial_adapter_sha256"] == sha(initial_path)
        assert m["learning_rate"] == chosen["learning_rate"] and m["epochs"] == epochs and not m["smoke"]
        assert s["steps"] == 8 * epochs
        for file, expected in s["adapter_sha256"].items():
            assert sha(checkpoint / "adapter" / file) == expected
            hashes_checked += 1
        weights = load_file(str(checkpoint / "adapter/adapter_model.safetensors"))
        assert set(weights) == {k.replace(".default.", ".") for k in initial}
        delta = float(
            torch.sqrt(sum((weights[k.replace(".default.", ".")] - v).square().sum() for k, v in initial.items()))
        )
        assert abs(delta - s["parameter_delta_l2"]) < 1e-8 and delta > 0
        selected = selections[seed]["arms"][arm["arm"]]
        assert [g["task_id"] for g in selected] == selections[seed]["task_order"]
        assert m["groups"] == [[g["group_id"]] for g in selected] * epochs
        steps = lines(checkpoint / "steps.jsonl")
        assert len(steps) == 8 * epochs
        prompt_tokens = completion_tokens = correct_exposures = 0
        for step, g in zip(steps, selected * epochs):
            assert [r["record_id"] for r in step["records"]] == g["record_ids"]
            assert [r["reward"] for r in step["records"]] == g["rewards"] and sum(g["rewards"]) == 2
            for record in step["records"]:
                key = record["record_id"]
                assert record["reward"] == int(source_grades[key]["correct"])
                prompt_tokens += len(originals[key]["prompt_token_ids"])
                completion_tokens += len(originals[key]["generated_token_ids"])
                correct_exposures += record["reward"]
        budget = dict(
            updates=8 * epochs,
            trajectory_exposures=32 * epochs,
            correct_exposures=16 * epochs,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )
        assert correct_exposures == 16 * epochs and arm["training_budget"] == budget
        rows, grades = evaluations[name]
        base_rows, base_grades = evaluations["base"]
        typical_name = f"model_typicality_seed{seed}_lr{chosen['learning_rate']:g}_epochs{epochs}"
        typical_rows, typical_grades = evaluations[typical_name]
        changed_base = changed_typical = up = down = 0
        for recorded, r, g, b, bg, t, tg in zip(
            arm["per_task"], rows, grades, base_rows, base_grades, typical_rows, typical_grades
        ):
            assert (
                r["prompt_token_ids"] == b["prompt_token_ids"]
                and r["padded_prompt_token_ids"] == b["padded_prompt_token_ids"]
                and r["attention_mask"] == b["attention_mask"]
            )
            cb = r["generated_token_ids"] != b["generated_token_ids"]
            ct = r["generated_token_ids"] != t["generated_token_ids"]
            assert recorded == dict(
                task_id=r["task_id"],
                correct=int(g["correct"]),
                base_correct=int(bg["correct"]),
                output_changed_vs_base=cb,
                generated_token_ids=r["generated_token_ids"],
                typicality_correct=int(tg["correct"]),
                output_changed_vs_typicality=ct,
            )
            changed_base += cb
            changed_typical += ct
            up += g["correct"] > bg["correct"]
            down += g["correct"] < bg["correct"]
        assert (
            arm["output_changes_vs_base"],
            arm["output_changes_vs_typicality"],
            arm["wrong_to_right_vs_base"],
            arm["right_to_wrong_vs_base"],
        ) == (changed_base, changed_typical, up, down)
        assert arm["correct"] == sum(g["correct"] for g in grades)
        result.append(
            dict(
                name=name,
                seed=seed,
                correct=arm["correct"],
                output_changes_vs_base=changed_base,
                output_changes_vs_typicality=changed_typical,
                training_budget=budget,
                adapter_delta_l2=delta,
            )
        )
    for seed in [0, 1, 2]:
        seed_rows = [r for r in result if r["seed"] == seed]
        assert len(seed_rows) == 4
        lengths = [r["training_budget"]["completion_tokens"] for r in seed_rows]
        assert max(lengths) / min(lengths) <= 1.05
        for task_index in range(8):
            vals = [selections[seed]["arms"][a][task_index]["total_completion_tokens"] for a in protocol["arms"]]
            assert max(vals) / min(vals) <= 1.05
    for name, total in report["aggregate"].items():
        correct = sum(r["correct"] for r in report["arms"] if r["arm"] == name)
        assert total == dict(correct_sum=correct, evaluations=63, mean_accuracy=correct / 63)
    save(
        OUT / "results.json",
        dict(
            all_checks_passed=True,
            checkpoints=12,
            calibration_output_records=273,
            per_task_comparisons=252,
            hashes_checked=hashes_checked,
            selected_recipe=dict(learning_rate=chosen["learning_rate"], epochs=chosen["epochs"]),
            runs=result,
            measurement_or_final_scored=False,
            calibration_tuned_descriptive_only=True,
            limitation="Checks preserved execution labels and source/task boundaries; does not independently rerun grader or establish generalization",
            api_spend_usd=0,
        ),
    )
    print(json.dumps(dict(all_checks_passed=True, runs=result), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "audit"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else audit()
