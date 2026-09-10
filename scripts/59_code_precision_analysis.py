#!/usr/bin/env python3
"""Compare frozen BF16/FP32 update outcomes without selecting a method."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path("runs/swe-diversity-selection/code-update-transfer")
OUT = Path("runs/swe-diversity-selection/code-update-transfer-fp32")
MODES = ["sgd_0.1", "sgd_1.0", "norm_0.02"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def prepare():
    path = OUT / "precision_analysis_protocol.json"
    assert not path.exists()
    paths = [
        Path(__file__),
        Path("scripts/55_code_update_transfer.py"),
        Path("scripts/57_code_update_precision.py"),
        BASE / "groups.json",
        OUT / "groups.json",
        BASE / "initial_adapter.pt",
    ]
    save(
        path,
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "scope": "All frozen groups and all three steps, repeated-group controls retained separately. No fitting, thresholds, method selection, or significance tests.",
            "same_measurement_alignment": "Numerical sanity only; not an independent predictive endpoint.",
            "small_effect_diagnostic": "Report observed magnitudes, residuals, and nominal FP32 baseline-loss ULP. ULP is not a bound on forward-pass numerical error and defines no exclusion or significance threshold.",
            "modes": MODES,
            "hashes": {str(p): sha(p) for p in paths},
            "full_outcomes_read_before_freeze": False,
        },
    )
    print("Precision analysis protocol frozen without reading full outcomes.")


def distribution(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return {
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.1)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.9)),
        "max": float(a.max()),
    }


def read_run(path):
    summary = json.loads((path / "learner_summary.json").read_text())
    assert summary["technical_checks_passed"]
    rows = [json.loads(line) for line in (path / "updates.jsonl").read_text().splitlines()]
    assert len({r["group_id"] for r in rows}) == len(rows)
    return summary, {r["group_id"]: r for r in rows}


def run():
    protocol = json.loads((OUT / "precision_analysis_protocol.json").read_text())
    for p, expected in protocol["hashes"].items():
        assert sha(Path(p)) == expected
    assert not (OUT / "precision_analysis.json").exists()
    bsummary, bf = read_run(BASE)
    fsummary, fp = read_run(OUT)
    initial_b = torch.load(BASE / "initial_adapter.pt", map_location="cpu", weights_only=True)
    initial_f = torch.load(OUT / "initial_adapter.pt", map_location="cpu", weights_only=True)
    assert set(initial_b) == set(initial_f)
    assert all(torch.equal(initial_b[k], initial_f[k]) for k in initial_b)
    assert bsummary["measurement_task_ids"] == fsummary["measurement_task_ids"]
    groups = json.loads((OUT / "groups.json").read_text())["groups"]
    ids = [g["group_id"] for g in groups]
    assert len(ids) == fsummary["groups"] == bsummary["groups"]
    assert set(ids) <= set(bf) and set(bf) == set(fp)
    pairs = {}
    for g in groups:
        pairs.setdefault(g["matched_pair_id"], []).append(g["group_id"])
    assert all(len(v) == 2 for v in pairs.values())
    nominal_ulp = float(np.mean(np.spacing(np.asarray(fsummary["base_measurement_losses"], dtype=np.float32))))
    rows, comparisons, pair_rows = [], {}, []
    for mode in MODES:
        by_mode = []
        for gid in ids:
            b, f = bf[gid]["updates"][mode], fp[gid]["updates"][mode]
            actual = f["mean_loss_decrease"]
            predicted = f["scale"] * fp[gid]["same_measurement_alignment_sanity_only"]
            row = {
                "group_id": gid,
                "task_id": fp[gid]["task_id"],
                "matched_pair_id": fp[gid]["matched_pair_id"],
                "mode": mode,
                "bf16_gain": b["mean_loss_decrease"],
                "fp32_gain": actual,
                "same_measurement_linear_prediction": predicted,
                "residual": actual - predicted,
                "absolute_residual": abs(actual - predicted),
                "relative_residual_to_prediction": abs(actual - predicted) / abs(predicted) if predicted else None,
                "gradient_norm": fp[gid]["gradient_norm"],
                "scale": f["scale"],
                "update_norm": f["scale"] * fp[gid]["gradient_norm"],
                "gain_per_scale": actual / f["scale"] if f["scale"] else None,
                "absolute_gain_over_nominal_baseline_ulp": abs(actual) / nominal_ulp,
            }
            rows.append(row)
            by_mode.append(row)
        for pid, (a, b) in pairs.items():
            bd = bf[b]["updates"][mode]["mean_loss_decrease"] - bf[a]["updates"][mode]["mean_loss_decrease"]
            fd = fp[b]["updates"][mode]["mean_loss_decrease"] - fp[a]["updates"][mode]["mean_loss_decrease"]
            pair_rows.append(
                {
                    "matched_pair_id": pid,
                    "group_a": a,
                    "group_b": b,
                    "mode": mode,
                    "bf16_b_minus_a": bd,
                    "fp32_b_minus_a": fd,
                    "strict_winner_flip": bool(bd * fd < 0),
                    "bf16_tie": bd == 0,
                    "fp32_tie": fd == 0,
                }
            )
        pair_mode = [r for r in pair_rows if r["mode"] == mode]
        comparisons[mode] = {
            "groups": len(ids),
            "bf16_positive": sum(r["bf16_gain"] > 0 for r in by_mode),
            "fp32_positive": sum(r["fp32_gain"] > 0 for r in by_mode),
            "strict_gain_sign_flips": sum(r["bf16_gain"] * r["fp32_gain"] < 0 for r in by_mode),
            "strict_pair_winner_flips": sum(r["strict_winner_flip"] for r in pair_mode),
            "pairs": len(pair_mode),
            "bf16_tied_pairs": sum(r["bf16_tie"] for r in pair_mode),
            "fp32_tied_pairs": sum(r["fp32_tie"] for r in pair_mode),
            "absolute_residual": distribution([r["absolute_residual"] for r in by_mode]),
            "absolute_actual_gain": distribution([abs(r["fp32_gain"]) for r in by_mode]),
            "absolute_gain_over_nominal_baseline_ulp": distribution(
                [r["absolute_gain_over_nominal_baseline_ulp"] for r in by_mode]
            ),
            "update_norm": distribution([r["update_norm"] for r in by_mode]),
            "fp32_gain": distribution([r["fp32_gain"] for r in by_mode]),
        }
    sensitivities = []
    for gid in ids:
        rr = [r for r in rows if r["group_id"] == gid]
        sensitivities.append(
            {
                "group_id": gid,
                "gain_per_scale": {r["mode"]: r["gain_per_scale"] for r in rr},
                "gain_signs": {r["mode"]: int(np.sign(r["fp32_gain"])) for r in rr},
            }
        )
    controls = {gid: {"bf16": bf[gid]["updates"], "fp32": fp[gid]["updates"]} for gid in bf if gid not in ids}
    result = {
        "scope": protocol["scope"],
        "initial_adapter_tensors_exactly_equal": True,
        "measurement_tasks": len(fsummary["measurement_task_ids"]),
        "measurement_task_ids": fsummary["measurement_task_ids"],
        "distinct_groups": len(ids),
        "group_step_outcomes": len(rows),
        "nominal_mean_baseline_loss_fp32_ulp": nominal_ulp,
        "ulp_caveat": protocol["small_effect_diagnostic"],
        "comparisons": comparisons,
        "per_group_step": rows,
        "per_pair_step": pair_rows,
        "step_sensitivity": sensitivities,
        "repeated_group_controls": controls,
        "source_hashes": {
            str(p): sha(p)
            for p in [
                BASE / "updates.jsonl",
                OUT / "updates.jsonl",
                OUT / "initial_adapter.pt",
                OUT / "learner_summary.json",
            ]
        },
    }
    save(OUT / "precision_analysis.json", result)
    print(json.dumps({"outcomes": len(rows), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
