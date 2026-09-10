#!/usr/bin/env python3
"""Prespecified task-clustered analysis of fixed single-update groups."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

OUT = Path("runs/swe-diversity-selection/code-update-transfer")
SEED = 20260909
MODES = ["sgd_0.1", "sgd_1.0", "norm_0.02"]
METRICS = [
    "cheap_token_diversity",
    "cheap_ast_node_type_diversity",
    "gradient_dispersion",
    "gradient_norm",
    "cancellation",
    "calibration_alignment",
    "calibration_cosine",
    "quality_negative_mean_nll",
    "budget_negative_completion_tokens",
]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert not (OUT / "analysis_protocol.json").exists(), "Analysis already frozen"
    assert not (OUT / "updates.jsonl").exists() and not (OUT / "learner_summary.json").exists(), (
        "Full outcomes already exist"
    )
    save(
        "analysis_protocol.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            seed=SEED,
            primary_mode=MODES[0],
            sensitivity_modes=MODES[1:],
            primary_predictor="gradient_dispersion",
            mechanistic_comparator="calibration_alignment",
            predictors=METRICS,
            selection="For each frozen pair select larger metric; exact ties choose uniformly; quality is negative mean NLL, budget is negative completion tokens",
            effect="Selection lift versus uniform random member = .5*sign(metricA-metricB)*(improvementA-improvementB); average pairs within training task then average tasks",
            null="Zero task-mean selection lift; exact two-sided sign-flip over training tasks assumes sign symmetry; exploratory with only8 tasks",
            uncertainty="10000 percentile resamples of training-task clusters; preserve all pairs per task; conditional on same fixed21 measurement tasks",
            multiplicity="Holm adjustment across all9 predictors within each step size; sensitivity steps remain exploratory",
            paired_association="Pearson correlation of matched within-task pair score differences with actual loss-decrease differences; task-cluster bootstrap interval; undefined constant vectors reported null",
            controls="Verify declared repeated groups, exact no-update reset, two correct/two failed, token calipers; compare negative mean NLL and negative token length; no regression fitting",
            missing="If AST distance is null for either group drop that pair only for AST and report support; no imputation; report undefined gradient correlations",
            limitations="At most8 independent training-task clusters, overlapping groups, fixed shared measurement set, one model state/adapter/seed, reference NLL not execution success; calibration used only as fixed predictor",
            outputs=["analysis_pairs.jsonl", "analysis.json"],
            input_sha256={n: digest(OUT / n) for n in ["groups.json", "group_protocol.json", "records.jsonl"]},
            script_sha256=digest(Path(__file__)),
            api_spend_usd=0,
        ),
    )
    print("Frozen analysis_protocol.json before full outcomes")


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else None


def stats(by_task):
    tasks = sorted(by_task)
    means = np.array([np.mean([r["selection_lift"] for r in by_task[t]]) for t in tasks])
    if not tasks:
        return dict(tasks=0, pairs=0, mean_selection_lift=None, ci95=None, exact_task_signflip_p=None)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(tasks), (10000, len(tasks)))
    boots = means[draws].mean(1)
    observed = float(means.mean())
    null = np.array([np.mean(means * np.array(signs)) for signs in product([-1, 1], repeat=len(tasks))])
    pvalue = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    all_rows = [r for t in tasks for r in by_task[t]]
    association = corr([r["metric_difference"] for r in all_rows], [r["loss_decrease_difference"] for r in all_rows])
    correlations = []
    for indices in draws[:2000]:
        rows = [r for i in indices for r in by_task[tasks[i]]]
        value = corr([r["metric_difference"] for r in rows], [r["loss_decrease_difference"] for r in rows])
        if value is not None:
            correlations.append(value)
    return dict(
        tasks=len(tasks),
        pairs=len(all_rows),
        mean_selection_lift=observed,
        ci95=np.percentile(boots, [2.5, 97.5]).tolist(),
        exact_task_signflip_p=pvalue,
        task_lifts={str(t): float(v) for t, v in zip(tasks, means)},
        positive_tasks=int(np.sum(means > 0)),
        negative_tasks=int(np.sum(means < 0)),
        zero_tasks=int(np.sum(means == 0)),
        pair_metric_ties=sum(r["metric_difference"] == 0 for r in all_rows),
        positive_pair_lifts=sum(r["selection_lift"] > 0 for r in all_rows),
        negative_pair_lifts=sum(r["selection_lift"] < 0 for r in all_rows),
        pair_difference_correlation=association,
        correlation_ci95=np.percentile(correlations, [2.5, 97.5]).tolist() if correlations else None,
        valid_correlation_bootstraps=len(correlations),
    )


def analyze():
    protocol = json.loads((OUT / "analysis_protocol.json").read_text())
    assert protocol["script_sha256"] == digest(Path(__file__)), "Analysis code changed after freeze"
    for name, sha in protocol["input_sha256"].items():
        assert digest(OUT / name) == sha
    assert not (OUT / "analysis.json").exists(), "Completed analysis already saved"
    summary = json.loads((OUT / "learner_summary.json").read_text())
    assert summary["technical_checks_passed"] and summary["exact_group_repeat_and_reset"]
    frozen = json.loads((OUT / "groups.json").read_text())
    groups = {g["group_id"]: g for g in frozen["groups"]}
    updates = [json.loads(line) for line in (OUT / "updates.jsonl").open()]
    rows = {r["group_id"]: r for r in updates}
    assert len(rows) == len(updates)
    assert all(k in rows for k in groups)
    for control in frozen["controls"]["identical_group_repeats"]:
        key = control["source_group_id"]
        assert rows[key]["updates"] == rows[control["control_id"]]["updates"]
    for group in groups.values():
        assert sum(group["rewards"]) == 2 and len(group["record_ids"]) == 4
    paired = []
    effects = {}
    for mode in MODES:
        effects[mode] = {}
        for metric in METRICS:
            clusters = defaultdict(list)
            for pair in frozen["matched_pairs"]:
                a, b = pair["group_ids"]
                ga, gb = groups[a], groups[b]

                def value(key):
                    if metric.startswith("cheap_"):
                        return groups[key][metric]
                    if metric == "quality_negative_mean_nll":
                        return -rows[key]["mean_nll"]
                    if metric == "budget_negative_completion_tokens":
                        return -rows[key]["total_completion_tokens"]
                    return rows[key][metric]

                va, vb = value(a), value(b)
                if va is None or vb is None:
                    continue
                assert (
                    abs(ga["total_completion_tokens"] - gb["total_completion_tokens"])
                    / min(ga["total_completion_tokens"], gb["total_completion_tokens"])
                    <= 0.05
                )
                ya, yb = [rows[k]["updates"][mode]["mean_loss_decrease"] for k in (a, b)]
                delta = va - vb
                record = dict(
                    mode=mode,
                    metric=metric,
                    pair_id=pair["pair_id"],
                    task_id=pair["task_id"],
                    group_ids=[a, b],
                    metric_difference=delta,
                    loss_decrease_difference=ya - yb,
                    random_member_loss_decrease=(ya + yb) / 2,
                    selected_member_loss_decrease=(ya + yb) / 2 + 0.5 * float(np.sign(delta)) * (ya - yb),
                    selection_lift=0.5 * float(np.sign(delta)) * (ya - yb),
                    mean_nll_difference=rows[a]["mean_nll"] - rows[b]["mean_nll"],
                    completion_token_difference=ga["total_completion_tokens"] - gb["total_completion_tokens"],
                )
                paired.append(record)
                clusters[pair["task_id"]].append(record)
            effects[mode][metric] = stats(clusters)
        ordered = sorted(
            ((m, r) for m, r in effects[mode].items() if r["exact_task_signflip_p"] is not None),
            key=lambda item: item[1]["exact_task_signflip_p"],
        )
        adjusted = 0.0
        for rank, (metric, result) in enumerate(ordered):
            adjusted = max(adjusted, min(1.0, (len(ordered) - rank) * result["exact_task_signflip_p"]))
            result["holm_p_within_mode"] = adjusted
    absolute = {
        mode: {
            str(task): float(
                np.mean(
                    [rows[k]["updates"][mode]["mean_loss_decrease"] for k, g in groups.items() if g["task_id"] == task]
                )
            )
            for task in sorted({g["task_id"] for g in groups.values()})
        }
        for mode in MODES
    }
    (OUT / "analysis_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in paired))
    result = dict(
        primary_mode=MODES[0],
        primary_predictor="gradient_dispersion",
        independent_training_tasks=summary["training_tasks"],
        matched_pairs=len(frozen["matched_pairs"]),
        groups=len(groups),
        measurement_tasks=summary["measurement_tasks"],
        effects=effects,
        mean_update_loss_decrease_by_training_task=absolute,
        checks=dict(all_declared_repeats_match=True, learner_technical_checks_passed=True),
        scope=protocol["limitations"],
        api_spend_usd=0,
        inputs_sha256={
            n: digest(OUT / n)
            for n in ["analysis_protocol.json", "updates.jsonl", "learner_summary.json", "groups.json"]
        },
    )
    save("analysis.json", result)
    print(
        json.dumps(
            dict(
                primary=effects[MODES[0]]["gradient_dispersion"], calibration=effects[MODES[0]]["calibration_alignment"]
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "analyze"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else analyze()
