#!/usr/bin/env python3
"""Phase 1 persona selection: sweep all MATH_PERSONAS on the val split,
write selection.json with per-persona accuracies + the best persona.

This locks the C-prompt persona used on test. Selection vs. reporting on
disjoint data (design doc §4: no cherry-picking).
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
from deeppersona.data import load_gsm8k_items
from deeppersona.generate import build_chat_prompts, generate_iter, load_model
from deeppersona.manifest import make_run_dir, write_manifest
from deeppersona.personas import MATH_PERSONAS, system_message
from deeppersona.verifiers import extract_gsm8k_pred, gsm8k_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    add_overrides(parser)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    cfg.condition = "prompt"
    cfg.split = "val"
    cfg.run_name = "persona-select"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = make_run_dir(REPO_ROOT, cfg)
    write_manifest(run_dir, cfg, REPO_ROOT)
    print(f"[run] {run_dir}", flush=True)

    items = load_gsm8k_items(cfg.split, n_items=cfg.n_items, seed=cfg.seed)
    user_msgs = [it["question"] for it in items]
    print(f"[data] {len(items)} val items", flush=True)

    model, tok = load_model(cfg)
    print(f"[model] {cfg.model_id}@{cfg.model_revision[:8]} on {model.device}", flush=True)

    # Also include C-none as a reference baseline in the sweep.
    sweep: list[tuple[int, str]] = [(-1, system_message(-1, cfg.neutral_template_idx))]
    sweep += [(i, system_message(i, cfg.neutral_template_idx)) for i in range(len(MATH_PERSONAS))]

    results: list[dict] = []
    sweep_t0 = time.time()
    for si, (persona_idx, sys_msg) in enumerate(sweep):
        prompts = build_chat_prompts(tok, sys_msg, user_msgs)
        n_batches = math.ceil(len(prompts) / cfg.batch_size)
        print(f"[sweep-start] {si+1}/{len(sweep)} persona={persona_idx:>3d} "
              f"text={sys_msg[:60]!r}{'...' if len(sys_msg)>60 else ''}", flush=True)
        gens: list[str] = []
        correct: list[bool] = []
        t0 = time.time()
        for bi, batch in enumerate(generate_iter(model, tok, prompts, cfg)):
            gens.extend(batch)
            for g in batch:
                it = items[len(correct)]
                correct.append(gsm8k_score(extract_gsm8k_pred(g), it["gold_answer"]))
            n_so_far = len(gens)
            acc_so_far = sum(correct) / n_so_far
            elapsed = time.time() - t0
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f"[gen] persona={persona_idx:>3d} batch={bi+1:>2d}/{n_batches} "
                  f"n={n_so_far:>3d}/{len(prompts)} acc={acc_so_far:.3f} "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
        acc = sum(correct) / len(correct)
        n_parse_fail = sum(1 for g in gens if extract_gsm8k_pred(g) is None)
        row = {
            "persona_idx": persona_idx,
            "persona_text": sys_msg,
            "accuracy": acc,
            "n_correct": int(sum(correct)),
            "n": len(correct),
            "n_parse_fail": n_parse_fail,
            "wall_s": round(time.time() - t0, 1),
        }
        results.append(row)
        total_elapsed = time.time() - sweep_t0
        sweep_eta = total_elapsed / (si + 1) * (len(sweep) - si - 1)
        print(f"[sweep] {si+1}/{len(sweep)} persona={persona_idx:>3d} acc={acc:.3f} "
              f"parse_fail={n_parse_fail}/{len(correct)} took={row['wall_s']}s "
              f"sweep_elapsed={total_elapsed:.0f}s eta={sweep_eta:.0f}s", flush=True)

    persona_rows = [r for r in results if r["persona_idx"] >= 0]
    best = max(persona_rows, key=lambda r: r["accuracy"])
    none_row = next(r for r in results if r["persona_idx"] == -1)

    out = {
        "task": cfg.task,
        "split": cfg.split,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "best_persona_idx": best["persona_idx"],
        "best_persona_text": best["persona_text"],
        "best_persona_accuracy": best["accuracy"],
        "none_accuracy": none_row["accuracy"],
        "per_persona": results,
    }
    (run_dir / "selection.json").write_text(json.dumps(out, indent=2))
    print(f"[best] persona={best['persona_idx']} acc={best['accuracy']:.3f} "
          f"(vs none={none_row['accuracy']:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
