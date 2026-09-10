#!/usr/bin/env python3
"""Freeze calibration-only functional selectors from original development groups."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("runs/swe-diversity-selection")
SOURCE = ROOT / "code-update-transfer-fp32"
POOL = ROOT / "code-learning-pilot"
OUT = ROOT / "code-functional-development"
SEEDS = [0, 1, 2]
ARMS = ["random", "model_typicality", "gradient_dispersion", "additive_calibration_utility"]


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(smoke):
    dest = OUT / "preparation-smoke" if smoke else OUT
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / "selections.json").exists(), "Functional selections already frozen"
    frozen = json.loads((SOURCE / "groups.json").read_text())
    group_protocol = json.loads((SOURCE / "group_protocol.json").read_text())
    pool_protocol = json.loads((POOL / "protocol.json").read_text())
    groups = {g["group_id"]: g for g in frozen["groups"]}
    tasks = sorted({g["task_id"] for g in groups.values()})
    assert len(groups) == 64 and len(tasks) == 8
    calibration = group_protocol["calibration_task_ids"]
    measurement = group_protocol["measurement_task_ids"]
    assert set(tasks).isdisjoint(calibration) and set(tasks).isdisjoint(measurement)
    bank = np.load(SOURCE / "gradients.npz")
    index = {key: i for i, key in enumerate(bank["ids"])}
    record_ids = sorted({k for g in groups.values() for k in g["record_ids"]})
    vectors = {k: bank["gradients"][index[k]].astype(np.float64) for k in record_ids}
    cal = np.mean([bank["gradients"][index[f"calibration:{t}"]].astype(np.float64) for t in calibration], axis=0)
    baseline = json.loads((SOURCE / "baseline_nll.json").read_text())
    individual = {k: float(cal @ v) for k, v in vectors.items()}
    scores, errors = {}, []
    for gid, g in groups.items():
        v = np.stack([vectors[k] for k in g["record_ids"]])
        rewards = np.array(g["rewards"], dtype=float)
        h = ((rewards - 0.5)[:, None] * v).mean(0)
        direct = float(cal @ h)
        additive = float(np.mean([(r - 0.5) * individual[k] for r, k in zip(rewards, g["record_ids"])]))
        errors.append(abs(direct - additive))
        unit = v / np.linalg.norm(v, axis=1)[:, None]
        dispersion = float((1 - unit @ unit.T)[np.triu_indices(4, 1)].mean())
        scores[gid] = dict(
            model_typicality=-float(np.mean([baseline[k] for k in g["record_ids"]])),
            gradient_dispersion=dispersion,
            additive_calibration_utility=additive,
            direct_group_calibration_alignment=direct,
        )
    assert max(errors) < 1e-12
    selections, diagnostics = [], []
    for seed in SEEDS[:1] if smoke else SEEDS:
        order = sorted(tasks, key=lambda t: sha(f"20260909:functional-order:{seed}:{t}"))
        if smoke:
            order = order[:2]
        arms = {a: [] for a in ARMS}
        pair_map = {}
        for task in order:
            options = [p for p in frozen["matched_pairs"] if p["task_id"] == task]
            pair = min(options, key=lambda p: sha(f"20260909:functional-pair:{seed}:{task}:{p['pair_id']}"))
            pair_map[str(task)] = pair["pair_id"]
            ids = pair["group_ids"]

            def tie(gid):
                return sha(f"20260909:functional-choice:{seed}:{task}:{gid}")

            for arm in ARMS:
                selected = (
                    min(ids, key=tie) if arm == "random" else min(ids, key=lambda gid: (-scores[gid][arm], tie(gid)))
                )
                arms[arm].append(dict(groups[selected], selector_scores=scores[selected]))
            utility_choice = arms["additive_calibration_utility"][-1]["group_id"]
            direct_choice = min(ids, key=lambda gid: (-scores[gid]["direct_group_calibration_alignment"], tie(gid)))
            assert utility_choice == direct_choice
            lengths = [arms[a][-1]["total_completion_tokens"] for a in ARMS]
            assert (max(lengths) - min(lengths)) / min(lengths) <= 0.05
        totals = {a: sum(g["total_completion_tokens"] for g in gs) for a, gs in arms.items()}
        assert (max(totals.values()) - min(totals.values())) / min(totals.values()) <= 0.05
        agreements = {
            a: sum(x["group_id"] == y["group_id"] for x, y in zip(arms[a], arms["model_typicality"])) for a in ARMS
        }
        selections.append(dict(seed=seed, task_order=order, pairs_by_task=pair_map, arms=arms))
        diagnostics.append(
            dict(
                seed=seed,
                tasks=len(order),
                completion_tokens_per_epoch=totals,
                max_relative_arm_token_gap=(max(totals.values()) - min(totals.values())) / min(totals.values()),
                identical_to_typicality_tasks=agreements,
                selected_correct_per_arm=2 * len(order),
                selected_failed_per_arm=2 * len(order),
                no_effective_arm_difference_tasks=sum(
                    len({arms[a][j]["group_id"] for a in ARMS}) == 1 for j in range(len(order))
                ),
            )
        )
    paths = [
        SOURCE / n
        for n in ["groups.json", "group_protocol.json", "gradients.npz", "baseline_nll.json", "initial_adapter.pt"]
    ]
    paths += [POOL / n for n in ["raw.jsonl", "protocol.json", "inputs.jsonl"]]
    protocol = dict(
        schema_version=1,
        utc=datetime.now(timezone.utc).isoformat(),
        status="Frozen development selectors and calibration-only functional recipe; no functional outputs used",
        scope="Original8 source tasks only, separately from new88 replication; does not modify or validate frozen predictor replication",
        seeds=SEEDS,
        seed_role="Shared pair choice, tie break, and task order; original adapter tensors identical across arms/seeds, so these are selection/order replicates, not independent initialization seeds",
        arms={
            a: {
                "random": "SHA-seeded uniform choice within the common pair",
                "model_typicality": "Maximize negative per-trajectory mean completion NLL, averaged equally over4 trajectories",
                "gradient_dispersion": "Maximize mean pairwise cosine distance of raw unweighted initial adapter gradients",
                "additive_calibration_utility": "Maximize mean((reward-.5)*calibration-gradient dot trajectory gradient); supervised additive utility, not diversity",
            }[a]
            for a in ARMS
        },
        proposal_amendment="Use four pure frozen scores, with no quality/diversity mixture or fitted weights, as authorized for this calibration-only development task",
        shared_constraints="One original frozen matched pair per task/seed; one four-member group per arm, exactly2correct+2failed; <=5% completion-token difference per task and aggregate; no relaxation",
        numerical_scores="Stored original FP32 gradients cast to float64 for CPU dot products; no measurement gradients accessed; exact additive/direct identity checked algebraically and numerically",
        score_supervision="Calibration arm uses21 reference solutions; model-typicality arm does not. Additive individual-utility optimum must equal group alignment under identical constraints",
        reference_ids_used=calibration,
        prohibited_measurement_task_ids=measurement,
        final107_locked=True,
        final_test_task_ids_sha256=sha(json.dumps(pool_protocol["later_test_task_ids"])),
        model_revision=pool_protocol["model_revision"],
        dataset_revision=pool_protocol["dataset_revision"],
        precision="Full modelFP32, TF32off; exact original adapter tensors from code-update-transfer-fp32/initial_adapter.pt",
        adapter=dict(
            last_layer=35, target_modules=["q_proj", "v_proj"], rank=8, alpha=16, dropout=0, parameters=106496
        ),
        training_objective="At each task, J=mean_i((reward_i-.5)*mean_completion_token_NLL_i); recompute gradients at current theta, update theta-=lr*gradientJ; no fresh sampling or policy-ratio claim",
        optimizer=dict(name="SGD", momentum=0, weight_decay=0, gradient_clipping=None),
        grid=dict(
            learning_rates=[0.1, 1.0],
            epochs=[1, 3],
            tuning_arm="model_typicality",
            seeds=SEEDS,
            tuning_endpoint="Mean greedy calibration execution accuracy over21tasks and3 selection/order seeds",
            tie_break="Fewer epochs, then lower learning rate",
            apply_shared_recipe_to_all_arms=True,
            all_grid_results_retained=True,
        ),
        epoch_order="Repeat each seed-specific frozen task order without reshuffling across epochs",
        controls=[
            "Base model",
            "No-update adapter saved/reloaded; greedy outputs must exactly match base",
            "Reset identical adapter and fresh SGD state for every arm/seed/grid trial",
            "Save/reload trained checkpoint before greedy evaluation",
        ],
        calibration_only=True,
        inference=dict(
            task_ids=calibration,
            do_sample=False,
            max_new_tokens=768,
            enable_thinking=False,
            source_prompts=str(POOL / "inputs.jsonl"),
            grading="Existing script52 hidden-test grader; no assertions/reference bodies in prompt",
            output_capture=[
                "generated_token_ids",
                "generation",
                "finish_reason",
                "checkpoint_sha256",
                "grading_status",
                "correct",
                "generated_token_count",
            ],
        ),
        technical_smoke=dict(
            training_task_ids=selections[0]["task_order"][:2],
            calibration_task_ids=calibration[:1],
            seed=0,
            learning_rate=0.1,
            update_steps=1,
            group_batch="Average objectives of the first2 typicality-selected training groups in one step",
            checks=[
                "Finite gradient and parameter update",
                "Exact no-update base generation",
                "Checkpoint save/reload identity",
                "Executable grading end to end",
            ],
        ),
        stop_conditions=[
            "Any nonfinite loss/gradient/parameters stops that trial; record failure, do not change objective or hyperparameters",
            "Truncations, parse failures and execution failures remain outcomes; no repeat generation until correct",
            "No measurement21 or final107 evaluation in this stage",
        ],
        interpretation="Calibration-only development outcomes reused for tuning; not unbiased functional generalization, diversity proof, or SWE improvement",
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        api_spend_usd=0,
        gpu_used=False,
    )
    save(dest / "functional_protocol.json", protocol)
    save(dest / "selections.json", dict(schema_version=1, selections=selections))
    save(dest / "selector_scores.json", scores)
    save(dest / "individual_calibration_utilities.json", individual)
    save(
        dest / "selection_summary.json",
        dict(
            seeds=diagnostics,
            source_tasks=8,
            source_groups=64,
            max_additive_identity_absolute_error=max(errors),
            all_direct_and_additive_choices_identical=True,
            scores_used_for_fitting=False,
            new88_outcomes_used=False,
            measurement_gradients_used=False,
            api_spend_usd=0,
        ),
    )
    names = [
        "functional_protocol.json",
        "selections.json",
        "selector_scores.json",
        "individual_calibration_utilities.json",
        "selection_summary.json",
    ]
    save(
        dest / "selection_manifest.json",
        dict(files_sha256={n: hashlib.sha256((dest / n).read_bytes()).hexdigest() for n in names}),
    )
    print(json.dumps(json.loads((dest / "selection_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.smoke)
