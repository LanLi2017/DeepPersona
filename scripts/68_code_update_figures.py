#!/usr/bin/env python3
"""Standalone descriptive figure of the frozen pilot and source-task replication."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("runs/swe-diversity-selection")
OUT = ROOT / "paper-program/figures"
STUDIES = [
    ("code-update-transfer-fp32", "Original 8 tasks", "#2864a0"),
    ("code-update-replication", "New 12 tasks", "#cc6b25"),
]
METRICS = [
    ("gradient_dispersion", "Gradient diversity"),
    ("calibration_alignment", "Calibration alignment"),
    ("calibration_cosine", "Calibration cosine"),
    ("cheap_token_diversity", "Token diversity"),
    ("cheap_ast_node_type_diversity", "AST node diversity"),
    ("gradient_norm", "Gradient norm"),
    ("cancellation", "Cancellation ratio"),
    ("quality_negative_mean_nll", "Model typicality"),
    ("budget_negative_completion_tokens", "Shorter completions"),
]
MODES = [
    ("sgd_0.1", "Primary: SGD step 0.1"),
    ("sgd_1.0", "Sensitivity: SGD step 1"),
    ("norm_0.02", "Sensitivity: update norm 0.02"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "update_selection_manifest.json").exists()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.8), sharey=True)
    rows = []
    for study_index, (folder, label, color) in enumerate(STUDIES):
        data = json.loads((ROOT / folder / "analysis.json").read_text())
        for axis, (mode, title) in zip(axes, MODES):
            for i, (metric, name) in enumerate(METRICS):
                effect = data["effects"][mode][metric]
                mean = effect["mean_selection_lift"] * 1e6
                low, high = np.array(effect["ci95"]) * 1e6
                axis.errorbar(
                    mean,
                    i + (study_index - 0.5) * 0.23,
                    xerr=[[mean - low], [high - mean]],
                    fmt="o",
                    markersize=4,
                    capsize=2,
                    color=color,
                    label=label if i == 0 else None,
                )
                rows.append({"study": folder, "mode": mode, "metric": metric, **effect})
            axis.set_title(title, fontsize=10)
    for axis in axes:
        axis.axvline(0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
        axis.grid(axis="x", alpha=0.15)
        axis.set_xlabel("Selection lift (NLL/token × 10⁻⁶)", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(METRICS)), [name for _, name in METRICS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.59, 0.95), ncol=2, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.19, top=0.84, wspace=0.05)
    fig.suptitle("Matched-group update selection: diversity versus additive utility", fontsize=13)
    fig.supxlabel(
        "Positive = more loss reduction than random selection within matched pairs.\n95% task-bootstrap intervals; shared 21 measurement references; FP32. Reference loss is not functional accuracy.",
        fontsize=8,
    )
    outputs = []
    for suffix in ["png", "pdf"]:
        target = OUT / f"update_selection_replication.{suffix}"
        fig.savefig(target, dpi=180)
        outputs.append(target)
    plt.close(fig)
    (OUT / "update_selection_data.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "update_selection_manifest.json").write_text(
        json.dumps(
            {
                "purpose": "Descriptive visualization of frozen analyses; no new fitting or hypothesis testing",
                "matplotlib": importlib.metadata.version("matplotlib"),
                "numpy": importlib.metadata.version("numpy"),
                "sources": {folder: sha(ROOT / folder / "analysis.json") for folder, _, _ in STUDIES},
                "script_sha256": sha(Path(__file__)),
                "outputs": {str(p): sha(p) for p in outputs},
                "api_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(map(str, outputs)))


if __name__ == "__main__":
    main()
