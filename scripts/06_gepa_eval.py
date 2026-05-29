#!/usr/bin/env python3
"""Native-harness verdict for a GEPA-optimized persona prompt.

Re-scores the GEPA winner with the SAME greedy decoding, matched 512-token
budget, chat template, and CSQA verifier as the headline pipeline (§5.2), so the
comparison is apples-to-apples and free of DSPy chat-adapter formatting effects.

Conditions (persona identity held fixed at --persona-idx):
  none        - no persona
  basic       - the one-line persona
  structured  - the multi-section persona (the under-performer)
  gepa        - GEPA's evolved instruction (from --gepa-prompt-file)

Reports per-cell accuracy + the load-bearing paired contrasts against gepa
(McNemar exact + paired bootstrap CI), mirroring scaffolding_stats. Run on
--split val first, then --split test (1221) for the headline.
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

from deeppersona.config import apply_overrides, load_config  # noqa: E402
from deeppersona.data import load_items  # noqa: E402
from deeppersona.generate import build_chat_prompts, generate_iter, load_model  # noqa: E402
from deeppersona.manifest import write_manifest  # noqa: E402
from deeppersona.personas import ANSWER_INSTRUCTION, system_message  # noqa: E402
from deeppersona.scaffolding_stats import mcnemar_exact_p, paired_boot_ci  # noqa: E402
from deeppersona.verifiers import extract_pred, score  # noqa: E402


def run_cell(model, tok, items, sys_msg, cfg, label, raw_fh):
    prompts = build_chat_prompts(tok, sys_msg, [it["question"] for it in items])
    n_batches = math.ceil(len(prompts) / cfg.batch_size)
    correct: list[int] = []
    gen_words: list[int] = []
    parse_fail = 0
    t0 = time.time()
    for bi, batch in enumerate(generate_iter(model, tok, prompts, cfg)):
        for j, gen in enumerate(batch):
            i = bi * cfg.batch_size + j
            it = items[i]
            pred = extract_pred(cfg.task, gen)
            ok = bool(score(cfg.task, pred, it["gold_answer"]))
            parse_fail += pred is None
            correct.append(int(ok))
            gen_words.append(len(gen.split()))
            raw_fh.write(json.dumps({
                "cell": label, "idx": it["idx"], "gold": it["gold_answer"],
                "pred": pred, "correct": ok, "gen_words": gen_words[-1], "generation": gen,
            }) + "\n")
        acc = sum(correct) / len(correct)
        print(f"[gen] {label:<11} batch={bi+1:>3d}/{n_batches} n={len(correct):>4d}/{len(prompts)} "
              f"acc={acc:.3f} pf={parse_fail} elapsed={time.time()-t0:.0f}s", flush=True)
    return {
        "label": label, "sys_msg": sys_msg, "n": len(correct),
        "accuracy": sum(correct) / len(correct), "n_parse_fail": parse_fail,
        "mean_gen_words": float(np.mean(gen_words)), "correct": correct,
    }


def paired_contrast(x_correct, g_correct, seed=0):
    x, g = np.asarray(x_correct), np.asarray(g_correct)
    b = int(np.sum((x == 1) & (g == 0)))   # X right, gepa wrong
    c = int(np.sum((x == 0) & (g == 1)))   # X wrong, gepa right
    lo, hi = paired_boot_ci(g - x, seed=seed)
    return {
        "delta_gepa_minus_x": float(g.mean() - x.mean()),
        "delta_ci95": [lo, hi],
        "mcnemar": {"b_x_only": b, "c_gepa_only": c, "p": mcnemar_exact_p(b, c)},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/qwen25_7b_csqa.yaml")
    p.add_argument("--gepa-prompt-file", required=True, help="evolved_prompt.txt from 05_gepa_optimize")
    p.add_argument("--persona-idx", type=int, default=1)
    p.add_argument("--split", default="val", choices=["calib", "val", "test"])
    p.add_argument("--n-items", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)  # picks up --split/--n-items/--batch-size/--seed if set
    task = cfg.task
    idx = args.persona_idx

    gepa_body = Path(args.gepa_prompt_file).read_text().strip()
    # Append the SAME boxed-letter format instruction the structured renderer uses.
    gepa_msg = gepa_body + "\n\n" + ANSWER_INSTRUCTION[task].strip()

    cells = {
        "none": system_message(task, -1, cfg.neutral_template_idx, "basic"),
        "basic": system_message(task, idx, cfg.neutral_template_idx, "basic"),
        "structured": system_message(task, idx, cfg.neutral_template_idx, "structured"),
        "gepa": gepa_msg,
    }

    ts = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime())
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "runs" / f"gepa-eval-{ts}-{task}-p{idx}-{cfg.split}"
    )
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, cfg, REPO_ROOT)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    items = load_items(task, cfg.split, n_items=cfg.n_items, seed=cfg.seed)
    print(f"[data] {len(items)} items from {task}/{cfg.split}; persona #{idx}", flush=True)
    if cfg.split == "test":
        print("[note] TEST split — headline number; selection was on val.", flush=True)

    model, tok = load_model(cfg)
    print(f"[model] {cfg.model_id}@{cfg.model_revision[:8]} on {model.device}", flush=True)

    raw_fh = (out_dir / "raw" / "items.jsonl").open("w")
    results = {}
    for label, sys_msg in cells.items():
        results[label] = run_cell(model, tok, items, sys_msg, cfg, label, raw_fh)
    raw_fh.close()

    g = results["gepa"]["correct"]
    contrasts = {x: paired_contrast(results[x]["correct"], g, seed=cfg.seed)
                 for x in ("none", "basic", "structured")}

    summary = {
        "task": task, "split": cfg.split, "n_items": len(items), "persona_idx": idx,
        "model_id": cfg.model_id, "model_revision": cfg.model_revision,
        "gepa_prompt_file": str(args.gepa_prompt_file),
        "accuracy": {k: results[k]["accuracy"] for k in cells},
        "mean_gen_words": {k: results[k]["mean_gen_words"] for k in cells},
        "n_parse_fail": {k: results[k]["n_parse_fail"] for k in cells},
        "contrasts_vs_gepa": contrasts,
        "cells": {k: {kk: vv for kk, vv in results[k].items() if kk != "correct"} for k in cells},
        "correct": {k: results[k]["correct"] for k in cells},
    }
    (out_dir / "gepa_eval.json").write_text(json.dumps(summary, indent=2))

    print("\n========== VERDICT (native harness) ==========", flush=True)
    print(f" task={task} split={cfg.split} n={len(items)} persona#{idx}", flush=True)
    for k in ("none", "basic", "structured", "gepa"):
        print(f"   {k:<11} acc={results[k]['accuracy']:.4f}  "
              f"pf={results[k]['n_parse_fail']}  mean_words={results[k]['mean_gen_words']:.1f}", flush=True)
    print(" Δ(gepa − X), 95% CI, McNemar p:", flush=True)
    for x in ("none", "basic", "structured"):
        ct = contrasts[x]
        lo, hi = ct["delta_ci95"]
        print(f"   vs {x:<11} Δ={ct['delta_gepa_minus_x']:+.4f}  ci95=[{lo:+.4f},{hi:+.4f}]  "
              f"b/c={ct['mcnemar']['b_x_only']}/{ct['mcnemar']['c_gepa_only']}  p={ct['mcnemar']['p']:.4g}",
              flush=True)
    print(f" wrote {out_dir / 'gepa_eval.json'}", flush=True)
    print("==============================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
