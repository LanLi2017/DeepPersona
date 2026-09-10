#!/usr/bin/env python3
"""Frozen numerical audit of source-task replication, without predictor tuning."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("runs/swe-diversity-selection/code-update-replication")
INITIAL = Path("runs/swe-diversity-selection/code-update-transfer/initial_adapter.pt")
MODES = ["sgd_0.1", "sgd_1.0", "norm_0.02"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def read(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines()]


def prepare():
    assert not (OUT / "independent_numerical_protocol.json").exists()
    paths = [
        Path(__file__),
        INITIAL,
        OUT / "groups.json",
        OUT / "replication_protocol.json",
        OUT / "learner_protocol.json",
        OUT / "analysis_protocol.json",
    ]
    save(
        "independent_numerical_protocol.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "scope": "All groups, all three unchanged steps, all repeat controls. Same-measurement gradient used only for numerical sanity. Independently verify frozen primary and comparator selection arithmetic/signflip results; no new predictor or threshold.",
            "full_outcomes_read": False,
            "hashes": {str(p): sha(p) for p in paths},
        },
    )
    print("Numerical replication audit frozen before reading full outcomes.")


def stats(values):
    a = np.asarray(values)
    return dict(min=float(a.min()), median=float(np.median(a)), p90=float(np.quantile(a, 0.9)), max=float(a.max()))


def run():
    protocol = json.loads((OUT / "independent_numerical_protocol.json").read_text())
    assert not (OUT / "independent_numerical_audit.json").exists()
    assert all(sha(Path(p)) == expected for p, expected in protocol["hashes"].items())
    summary = json.loads((OUT / "learner_summary.json").read_text())
    assert summary["technical_checks_passed"] and summary["exact_group_repeat_and_reset"]
    frozen = json.loads((OUT / "groups.json").read_text())
    groups = {g["group_id"]: g for g in frozen["groups"]}
    rows = {r["group_id"]: r for r in read("updates.jsonl")}
    assert len(rows) == len(read("updates.jsonl"))
    original = torch.load(INITIAL, map_location="cpu", weights_only=True)
    current = torch.load(OUT / "initial_adapter.pt", map_location="cpu", weights_only=True)
    assert set(original) == set(current) and all(torch.equal(original[k], current[k]) for k in original)
    z = np.load(OUT / "gradients.npz")
    gradients = dict(zip(z["ids"], z["gradients"]))
    measure = np.mean([v for k, v in gradients.items() if k.startswith("measurement:")], axis=0)
    controls = []
    for c in frozen["controls"]["identical_group_repeats"]:
        assert rows[c["source_group_id"]]["updates"] == rows[c["control_id"]]["updates"]
        controls.append(dict(control=c, updates=rows[c["control_id"]]["updates"]))
    output = []
    for gid, g in groups.items():
        h = sum((reward - 0.5) * gradients[k] for k, reward in zip(g["record_ids"], g["rewards"])) / 4
        assert float(measure @ h) == rows[gid]["same_measurement_alignment_sanity_only"]
        for mode in MODES:
            r = rows[gid]["updates"][mode]
            changes = np.asarray(summary["base_measurement_losses"]) - r["measurement_losses"]
            np.testing.assert_array_equal(changes, r["per_task_loss_decrease"])
            assert float(changes.mean()) == r["mean_loss_decrease"]
            actual = float(changes.mean())
            prediction = r["scale"] * float(measure @ h)
            output.append(
                dict(
                    group_id=gid,
                    task_id=g["task_id"],
                    mode=mode,
                    actual_loss_decrease=actual,
                    same_measurement_linear_prediction=prediction,
                    absolute_residual=abs(actual - prediction),
                    update_norm=r["scale"] * rows[gid]["gradient_norm"],
                    gain_per_scale=actual / r["scale"],
                )
            )
    modes = {}
    pair_rows = []
    for mode in MODES:
        rs = [r for r in output if r["mode"] == mode]
        for pair in frozen["matched_pairs"]:
            a, b = pair["group_ids"]
            delta = rows[a]["updates"][mode]["mean_loss_decrease"] - rows[b]["updates"][mode]["mean_loss_decrease"]
            pair_rows.append(
                dict(pair_id=pair["pair_id"], task_id=pair["task_id"], mode=mode, actual_loss_decrease_difference=delta)
            )
        margins = [abs(p["actual_loss_decrease_difference"]) for p in pair_rows if p["mode"] == mode]
        modes[mode] = dict(
            groups=len(rs),
            positive_groups=sum(r["actual_loss_decrease"] > 0 for r in rs),
            mean_loss_decrease=float(np.mean([r["actual_loss_decrease"] for r in rs])),
            absolute_actual_gain=stats([abs(r["actual_loss_decrease"]) for r in rs]),
            absolute_residual=stats([r["absolute_residual"] for r in rs]),
            absolute_pair_margin=stats(margins),
            update_norm=stats([r["update_norm"] for r in rs]),
        )
    result = dict(
        protocol=protocol["scope"],
        exact_initial_tensors=True,
        gradient_alignments_checked=len(groups),
        loss_changes_checked=len(output),
        source_tasks=summary["training_tasks"],
        measurement_tasks=summary["measurement_tasks"],
        modes=modes,
        per_group_step=output,
        per_pair_step=pair_rows,
        repeated_group_controls=controls,
        numerical_limit="Residual includes finite-step curvature and numerical error; no exclusion or significance threshold is inferred. Tiny pair margins remain unresolved.",
        source_hashes={str(OUT / n): sha(OUT / n) for n in ["gradients.npz", "updates.jsonl", "learner_summary.json"]},
    )
    save("independent_numerical_audit.json", result)
    print(json.dumps(dict(outcomes=len(output), modes=modes), indent=2))


def effects():
    result = json.loads((OUT / "analysis.json").read_text())
    paired = read("analysis_pairs.jsonl")
    checked = []
    for mode in MODES:
        for metric in ["gradient_dispersion", "calibration_alignment"]:
            rs = [r for r in paired if r["mode"] == mode and r["metric"] == metric]
            ts = sorted({r["task_id"] for r in rs})
            means = np.array(
                [
                    np.mean(
                        [
                            0.5 * np.sign(r["metric_difference"]) * r["loss_decrease_difference"]
                            for r in rs
                            if r["task_id"] == t
                        ]
                    )
                    for t in ts
                ]
            )
            reported = result["effects"][mode][metric]
            assert float(means.mean()) == reported["mean_selection_lift"]
            signs = 1 - 2 * ((np.arange(2 ** len(ts))[:, None] >> np.arange(len(ts))) & 1)
            p = float(np.mean(np.abs((signs * means).mean(1)) >= abs(means.mean()) - 1e-15))
            assert p == reported["exact_task_signflip_p"]
            checked.append(
                dict(
                    mode=mode,
                    metric=metric,
                    task_mean_lift=float(means.mean()),
                    exact_task_signflip_p=p,
                    ci95=reported["ci95"],
                    holm_p=reported["holm_p_within_mode"],
                )
            )
    save(
        "independent_effect_checks.json",
        dict(
            independent_task_mean_and_signflip_checks=checked,
            source_analysis_sha256=sha(OUT / "analysis.json"),
            new_hypotheses_or_tuning=False,
        ),
    )
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run", "effects"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "effects": effects}[args.stage]()
