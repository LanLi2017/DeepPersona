"""
Step 5 of docs/aime_passk_diversity_design.md: correlate each diversity metric (from scripts/31)
against pass@k, both RAW (naive, potentially confounded by answer entropy) and PARTIALLED
(controlling for answer entropy) per §4.3/§4.5.

Partial correlation method: group ALL rows (random + controlled) into cells sharing the same
(condition, item_idx, k, n_distinct_answers) -- this is the actual realized entropy of the drawn
group, regardless of which strategy produced it. Within each cell with >=2 rows and pass_k
variance, demean diversity and pass_k by the cell mean; pool the residuals across all cells and
compute Pearson r on the pooled residuals. This isolates diversity's association with pass@k
after removing both item-specific and entropy-specific baseline levels -- directly answering what
N1-dry could only ask observationally.

Simplification flagged: the bootstrap CI resamples rows, not cells/items -- doesn't fully respect
the clustered structure (multiple groups per item). Adequate for a first read, not a final number.

Usage:
  python scripts/32_aime_correlation.py <run_dir>
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def pearson_r(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def bootstrap_ci(xs, ys, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.array(xs), np.array(ys)
    n = len(xs)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = pearson_r(xs[idx], ys[idx])
        if not math.isnan(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def partial_residuals(rows, metric_key):
    """Demean metric_key and pass_k within (condition, item_idx, k, n_distinct_answers) cells."""
    cells = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["item_idx"], r["k"], r["n_distinct_answers"])
        cells[key].append(r)
    dx, dy = [], []
    for key, cell_rows in cells.items():
        if len(cell_rows) < 2:
            continue
        pass_ks = [r["pass_k"] for r in cell_rows]
        if len(set(pass_ks)) < 2:
            continue  # no within-cell variance to explain
        mx = np.mean([r[metric_key] for r in cell_rows])
        my = np.mean(pass_ks)
        for r in cell_rows:
            dx.append(r[metric_key] - mx)
            dy.append(r["pass_k"] - my)
    return dx, dy


def report(name, xs, ys):
    if len(xs) < 5:
        print(f"  {name:30s} n={len(xs):5d}  (too few for a stable estimate)")
        return
    r = pearson_r(xs, ys)
    lo, hi = bootstrap_ci(xs, ys)
    sig = "*" if (lo > 0 or hi < 0) else " "
    print(f"  {name:30s} n={len(xs):5d}  r={r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--file", default="groups_scored.jsonl")
    ap.add_argument("--metrics", nargs="+", default=["dtw_div", "wass_div"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(args.run_dir) / "raw" / args.file)]
    models = sorted(set(r["condition"] for r in rows))
    metrics = args.metrics

    print(f"total groups: {len(rows)}  models: {models}\n")

    for metric in metrics:
        print(f"=== {metric} ===")
        print("-- RAW correlation (naive, confounded by answer entropy) --")
        report("ALL (pooled)", [r[metric] for r in rows], [r["pass_k"] for r in rows])
        for m in models:
            sub = [r for r in rows if r["condition"] == m]
            report(m, [r[metric] for r in sub], [r["pass_k"] for r in sub])

        print("-- PARTIAL correlation (controlling for answer entropy, per §4.3 fix) --")
        dx, dy = partial_residuals(rows, metric)
        report("ALL (pooled)", dx, dy)
        for m in models:
            sub = [r for r in rows if r["condition"] == m]
            dx, dy = partial_residuals(sub, metric)
            report(m, dx, dy)
        print()

    print("* = 95% CI excludes 0")
    print("\nDecision rule (design §5): PASS needs CI excluding 0 (positive) for >=1 metric,")
    print("consistent across a majority of models. Compare RAW vs PARTIAL: if RAW is significant")
    print("but PARTIAL is not, the apparent effect is just answer-entropy in disguise (the N1-dry")
    print("failure mode) and doesn't support Assumption 3 for this metric.")


if __name__ == "__main__":
    main()
