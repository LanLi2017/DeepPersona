#!/usr/bin/env python3
"""Aggregate per-cell scaffolding runs (one 01_baseline.py run dir per cell)
into a single scaffolding.json + console summary.

The HPC path submits each (none / basic-i / structured-i) cell as its own job,
all sharing --run-name <EXP_ID>. This script globs those run dirs, aligns
per-item correctness by item idx, and computes the same structured-vs-basic
contrast as the single-process 03_scaffolding.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeppersona.scaffolding_stats import contrast_summary


def load_cell(run_dir: Path) -> dict | None:
    manifest = run_dir / "manifest.json"
    raw = run_dir / "raw" / "items.jsonl"
    if not manifest.exists() or not raw.exists():
        return None
    cfg = json.loads(manifest.read_text())["config"]
    correct = {}
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        correct[rec["idx"]] = int(rec["correct"])
    metrics = {}
    if (run_dir / "metrics.json").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text())
    return {"cfg": cfg, "correct": correct, "metrics": metrics, "dir": run_dir.name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True, help="shared --run-name tag of the cell jobs")
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--out", default=None, help="output json (default runs/scaffolding-<exp_id>.json)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    cell_dirs = sorted(d for d in runs_dir.glob(f"*{args.exp_id}*") if d.is_dir())
    if not cell_dirs:
        raise SystemExit(f"no run dirs matching *{args.exp_id}* under {runs_dir}")

    none_correct = None
    by_cell: dict[tuple[str, int], dict] = {}  # (level, persona_idx) -> cell
    for d in cell_dirs:
        cell = load_cell(d)
        if cell is None:
            print(f"[skip] incomplete run dir: {d.name}", flush=True)
            continue
        cfg = cell["cfg"]
        if cfg["condition"] == "none":
            none_correct = cell["correct"]
        elif cfg["condition"] == "prompt":
            by_cell[(cfg["persona_level"], cfg["persona_idx"])] = cell
        print(f"[cell] {d.name}  cond={cfg['condition']} "
              f"pidx={cfg['persona_idx']} level={cfg.get('persona_level')} "
              f"n={len(cell['correct'])} acc={cell['metrics'].get('accuracy', float('nan')):.3f}", flush=True)

    # pair basic[i] with structured[i] on the intersection of item idxs
    persona_idxs = sorted({pidx for (_, pidx) in by_cell})
    pairs = []
    cells_out = []
    for pidx in persona_idxs:
        bc = by_cell.get(("basic", pidx))
        sc = by_cell.get(("structured", pidx))
        if bc is None or sc is None:
            print(f"[warn] persona {pidx} missing a level (basic={bc is not None}, "
                  f"structured={sc is not None}) — excluded from contrast", flush=True)
            continue
        common = sorted(set(bc["correct"]) & set(sc["correct"]))
        if not common:
            print(f"[warn] persona {pidx}: no overlapping items between levels", flush=True)
            continue
        b_arr = [bc["correct"][i] for i in common]
        s_arr = [sc["correct"][i] for i in common]
        pairs.append((pidx, b_arr, s_arr))
        cells_out.append({"persona_idx": pidx, "n_aligned": len(common),
                          "basic_acc": bc["metrics"].get("accuracy"),
                          "structured_acc": sc["metrics"].get("accuracy")})

    if not pairs:
        raise SystemExit("no complete basic/structured pairs found — nothing to aggregate")

    none_acc = (sum(none_correct.values()) / len(none_correct)) if none_correct else None
    stats = contrast_summary(pairs, none_acc, seed=0)
    summary = {"exp_id": args.exp_id, "n_personas_paired": len(pairs), **stats, "cells": cells_out}

    out = Path(args.out) if args.out else runs_dir / f"scaffolding-{args.exp_id}.json"
    out.write_text(json.dumps(summary, indent=2))

    lo, hi = stats["pooled_delta_ci95"]
    none_str = f"{none_acc:.3f}" if none_acc is not None else "n/a"
    print("\n[summary] personas={}  none={}  basic(mean)={:.3f}  structured(mean)={:.3f}  "
          "Δ(struct-basic)={:+.3f} ci95=[{:+.3f},{:+.3f}]  pooled_mcnemar_p={:.4g}".format(
              len(pairs), none_str, stats["basic_mean_acc"], stats["structured_mean_acc"],
              stats["pooled_delta_structured_minus_basic"], lo, hi,
              stats["pooled_mcnemar"]["p"]), flush=True)
    print(f"[summary] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
