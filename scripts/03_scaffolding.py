#!/usr/bin/env python3
"""Persona scaffolding readout (Wharton-style, but varying persona *richness*).

Sweeps three conditions on a split, holding persona identity fixed and varying
only the scaffolding level:
  - C-none:       no persona.
  - C-basic:      the original one-line persona  (level="basic").
  - C-structured: the same identity as a multi-section block (level="structured").

The load-bearing contrast is structured vs basic, per identity and pooled. We
keep per-item correctness so the paired McNemar test (design doc §8) is exact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeppersona.config import add_overrides, apply_overrides, load_config
from deeppersona.data import load_items
from deeppersona.generate import build_chat_prompts, generate_iter, load_model
from deeppersona.manifest import append_item, make_run_dir, write_manifest
from deeppersona.personas import num_personas, system_message
from deeppersona.scaffolding_stats import contrast_summary
from deeppersona.verifiers import extract_pred, score


def run_cell(model, tok, items, sys_msg, cfg, label, run_dir):
    prompts = build_chat_prompts(tok, sys_msg, [it["question"] for it in items])
    n_batches = math.ceil(len(prompts) / cfg.batch_size)
    correct: list[int] = []
    parse_fail = 0
    t0 = time.time()
    for bi, batch in enumerate(generate_iter(model, tok, prompts, cfg)):
        for j, gen in enumerate(batch):
            i = bi * cfg.batch_size + j
            it = items[i]
            pred = extract_pred(cfg.task, gen)
            ok = score(cfg.task, pred, it["gold_answer"])
            parse_fail += pred is None
            correct.append(int(ok))
            append_item(run_dir, {
                "cell": label, "idx": it["idx"], "gold": it["gold_answer"],
                "prompt": prompts[i], "generation": gen, "pred": pred, "correct": ok,
            })
        acc = sum(correct) / len(correct)
        elapsed = time.time() - t0
        eta = elapsed / (bi + 1) * (n_batches - bi - 1)
        print(f"[gen] {label:<22} batch={bi+1:>2d}/{n_batches} n={len(correct):>3d}/{len(prompts)} "
              f"acc={acc:.3f} pf={parse_fail} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    return correct, parse_fail, round(time.time() - t0, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    add_overrides(parser)  # --persona-level is ignored here; this script sweeps both levels
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    cfg.condition = "prompt"
    if cfg.split == "test":
        print("[warn] running scaffolding sweep on TEST — selection should be on val", flush=True)
    cfg.run_name = "scaffolding"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = make_run_dir(REPO_ROOT, cfg)
    write_manifest(run_dir, cfg, REPO_ROOT)
    print(f"[run] {run_dir}", flush=True)

    items = load_items(cfg.task, cfg.split, n_items=cfg.n_items, seed=cfg.seed)
    print(f"[data] {len(items)} items from {cfg.task}/{cfg.split}", flush=True)

    model, tok = load_model(cfg)
    print(f"[model] {cfg.model_id}@{cfg.model_revision[:8]} on {model.device}", flush=True)

    n = num_personas(cfg.task)
    cells: list[tuple[str, int]] = [("none", -1)]
    cells += [(lvl, i) for lvl in ("basic", "structured") for i in range(n)]

    rows: list[dict] = []
    sweep_t0 = time.time()
    for si, (level, pidx) in enumerate(cells):
        sys_msg = system_message(cfg.task, pidx, cfg.neutral_template_idx, level)
        label = f"{level}/p{pidx}" if pidx >= 0 else "none"
        correct, parse_fail, wall = run_cell(model, tok, items, sys_msg, cfg, label, run_dir)
        rows.append({
            "level": level, "persona_idx": pidx, "persona_text": sys_msg,
            "accuracy": sum(correct) / len(correct), "n_correct": sum(correct),
            "n": len(correct), "n_parse_fail": parse_fail, "wall_s": wall,
            "correct": correct,
        })
        el = time.time() - sweep_t0
        print(f"[cell] {si+1}/{len(cells)} {label:<14} acc={rows[-1]['accuracy']:.3f} "
              f"pf={parse_fail} took={wall}s sweep_elapsed={el:.0f}s "
              f"eta={el/(si+1)*(len(cells)-si-1):.0f}s", flush=True)

    # ---- analysis: structured vs basic, per identity and pooled ----
    by = {(r["level"], r["persona_idx"]): r["correct"] for r in rows}
    none_acc = next(r["accuracy"] for r in rows if r["level"] == "none")
    pairs = [(i, by[("basic", i)], by[("structured", i)]) for i in range(n)]
    stats = contrast_summary(pairs, none_acc, seed=cfg.seed)

    summary = {
        "task": cfg.task, "split": cfg.split, "n_items": len(items),
        "model_id": cfg.model_id, "model_revision": cfg.model_revision,
        **stats, "cells": rows,
    }
    (run_dir / "scaffolding.json").write_text(json.dumps(summary, indent=2))

    lo, hi = stats["pooled_delta_ci95"]
    print("\n[summary] none={:.3f}  basic(mean)={:.3f}  structured(mean)={:.3f}  "
          "Δ(struct-basic)={:+.3f} ci95=[{:+.3f},{:+.3f}]  pooled_mcnemar_p={:.4g}".format(
              none_acc, stats["basic_mean_acc"], stats["structured_mean_acc"],
              stats["pooled_delta_structured_minus_basic"], lo, hi,
              stats["pooled_mcnemar"]["p"]), flush=True)
    print(f"[summary] wrote {run_dir / 'scaffolding.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
