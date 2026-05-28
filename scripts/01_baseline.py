#!/usr/bin/env python3
"""Phase 1 baseline: C-none and C-prompt on a verifiably-scored task.

Currently supports GSM8K. Greedy, matched max_new_tokens. Writes:
  runs/<...>/manifest.json
  runs/<...>/raw/items.jsonl
  runs/<...>/metrics.json
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeppersona.config import RunConfig, add_overrides, apply_overrides, load_config
from deeppersona.data import load_gsm8k_items
from deeppersona.generate import build_chat_prompts, generate_iter, load_model
from deeppersona.manifest import append_item, make_run_dir, write_manifest, write_metrics
from deeppersona.personas import MATH_PERSONAS, system_message
from deeppersona.verifiers import extract_gsm8k_pred, gsm8k_score


def bootstrap_ci(correct: list[bool], n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0):
    arr = np.array(correct, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(arr)
    samples = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    add_overrides(parser)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)

    if cfg.condition == "prompt" and not (0 <= cfg.persona_idx < len(MATH_PERSONAS)):
        raise SystemExit(f"--persona-idx must be in [0, {len(MATH_PERSONAS)}) for condition=prompt")
    if cfg.condition == "none":
        cfg.persona_idx = -1

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run_dir = make_run_dir(REPO_ROOT, cfg)
    write_manifest(run_dir, cfg, REPO_ROOT)
    print(f"[run] {run_dir}", flush=True)

    if cfg.task != "gsm8k":
        raise SystemExit(f"task={cfg.task} not implemented in 01_baseline.py")

    items = load_gsm8k_items(cfg.split, n_items=cfg.n_items, seed=cfg.seed)
    print(f"[data] {len(items)} items from gsm8k/{cfg.split}", flush=True)

    model, tok = load_model(cfg)
    print(f"[model] {cfg.model_id}@{cfg.model_revision[:8]} on {model.device}", flush=True)

    sys_msg = system_message(cfg.persona_idx, cfg.neutral_template_idx)
    prompts = build_chat_prompts(tok, sys_msg, [it["question"] for it in items])

    if cfg.smoke:
        print(f"[smoke] system: {sys_msg!r}")
        print(f"[smoke] first prompt:\n{prompts[0]}\n---")

    correct: list[bool] = []
    parse_fail = 0
    n_batches = math.ceil(len(prompts) / cfg.batch_size)
    t0 = time.time()
    for bi, batch in enumerate(generate_iter(model, tok, prompts, cfg)):
        for j, gen in enumerate(batch):
            i = bi * cfg.batch_size + j
            it = items[i]
            prompt = prompts[i]
            pred = extract_gsm8k_pred(gen)
            ok = gsm8k_score(pred, it["gold_answer"])
            if pred is None:
                parse_fail += 1
            correct.append(ok)
            append_item(run_dir, {
                "idx": it["idx"],
                "question": it["question"],
                "gold": it["gold_answer"],
                "prompt": prompt,
                "generation": gen,
                "pred": pred,
                "correct": ok,
            })
            if cfg.smoke:
                print(f"[smoke] idx={it['idx']} gold={it['gold_answer']} pred={pred} ok={ok}")
                print(f"[smoke] gen: {gen[:400]!r}")
                print("---")
        n_so_far = len(correct)
        acc_so_far = sum(correct) / n_so_far
        elapsed = time.time() - t0
        eta = elapsed / (bi + 1) * (n_batches - bi - 1)
        print(f"[gen] batch={bi+1:>2d}/{n_batches} n={n_so_far:>4d}/{len(prompts)} "
              f"acc={acc_so_far:.3f} parse_fail={parse_fail} "
              f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    n = len(correct)
    acc = sum(correct) / n if n else 0.0
    lo, hi = bootstrap_ci(correct, seed=cfg.seed)
    metrics = {
        "n": n,
        "accuracy": acc,
        "accuracy_ci95_lo": lo,
        "accuracy_ci95_hi": hi,
        "parse_fail_rate": parse_fail / n if n else 0.0,
        "n_correct": int(sum(correct)),
        "n_parse_fail": parse_fail,
    }
    write_metrics(run_dir, metrics)
    print(f"[result] task={cfg.task} split={cfg.split} cond={cfg.condition} "
          f"persona={cfg.persona_idx} n={n} acc={acc:.3f} "
          f"ci95=[{lo:.3f},{hi:.3f}] parse_fail={metrics['parse_fail_rate']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
