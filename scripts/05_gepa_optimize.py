#!/usr/bin/env python3
"""GEPA optimization of a (structured) persona prompt on CSQA.

Tests the under-optimization hypothesis: seed a DSPy predictor's instruction
with a structured persona, let GEPA's reflective loop rewrite it, and save the
evolved prompt for a native-harness verdict (scripts/06_gepa_eval.py).

Task LM   : Qwen2.5-7B-Instruct served by a local vLLM OpenAI server (greedy).
Reflect LM: a strong API model (default OpenAI), reads its key from the env.

Search uses calib (train) + val (GEPA's Pareto set). The held-out test split is
NEVER seen here, so the verdict on test stays clean.

Outputs under runs/gepa-<ts>-.../:
  config.json          - full run config + lib versions
  evolved_prompt.txt   - the winning instruction (persona body only)
  seed_prompt.txt      - the seed instruction, for diffing
  gepa_result.json     - seed vs evolved val scores + GEPA stats
  optimized_program.json - dspy program state (reloadable)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import dspy  # noqa: E402

from deeppersona.gepa_program import (  # noqa: E402
    build_examples,
    build_program,
    get_instruction,
    make_metric,
)

_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="csqa")
    p.add_argument("--persona-idx", type=int, default=1,
                   help="seed persona (default 1 = the biggest structured regressor)")
    p.add_argument("--seed-level", default="structured", choices=["structured", "basic"])
    # data
    p.add_argument("--train-split", default="calib")
    p.add_argument("--val-split", default="val")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    # task LM (vLLM OpenAI server)
    p.add_argument("--task-model", default="openai/Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--task-api-base", default="http://localhost:8000/v1")
    p.add_argument("--task-api-key", default="EMPTY")
    p.add_argument("--max-new-tokens", type=int, default=512)
    # reflection LM (API)
    p.add_argument("--reflection-model", default="openai/gpt-5",
                   help="litellm model id; key read from the matching *_API_KEY env var")
    p.add_argument("--reflection-temperature", type=float, default=1.0)
    p.add_argument("--reflection-max-tokens", type=int, default=32000)
    # GEPA budget / behavior
    p.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    p.add_argument("--max-metric-calls", type=int, default=None,
                   help="if set, overrides --auto with an explicit budget")
    p.add_argument("--reflection-minibatch-size", type=int, default=3)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def require_reflection_key(model: str) -> None:
    provider = model.split("/", 1)[0].lower()
    env = _KEY_ENV.get(provider)
    if env and not os.environ.get(env):
        raise SystemExit(
            f"Reflection model '{model}' needs {env} in the environment. "
            f"Export it (e.g. `export {env}=...`) before launching; it is read from the env, "
            f"never passed on the command line."
        )


def main() -> int:
    args = parse_args()
    require_reflection_key(args.reflection_model)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "runs" / f"gepa-{ts}-{args.task}-p{args.persona_idx}-{args.seed_level}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}", flush=True)

    # ── LMs ────────────────────────────────────────────────────────────────
    # Task LM: greedy (temperature 0), matched token budget, talking to vLLM.
    task_lm = dspy.LM(
        args.task_model,
        api_base=args.task_api_base,
        api_key=args.task_api_key,
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )
    dspy.configure(lm=task_lm)
    reflection_lm = dspy.LM(
        args.reflection_model,
        temperature=args.reflection_temperature,
        max_tokens=args.reflection_max_tokens,
    )

    # ── program + data ───────────────────────────────────────────────────────
    program = build_program(args.task, args.persona_idx, args.seed_level)
    seed_instruction = get_instruction(program)
    trainset = build_examples(args.task, args.train_split, args.n_train, args.seed)
    valset = build_examples(args.task, args.val_split, args.n_val, args.seed)
    print(f"[data] train={len(trainset)} ({args.train_split})  val={len(valset)} ({args.val_split})", flush=True)
    print(f"[seed] persona #{args.persona_idx} ({args.seed_level}), {len(seed_instruction)} chars", flush=True)

    (out_dir / "seed_prompt.txt").write_text(seed_instruction)

    # ── GEPA ──────────────────────────────────────────────────────────────────
    budget = {"max_metric_calls": args.max_metric_calls} if args.max_metric_calls else {"auto": args.auto}
    gepa = dspy.GEPA(
        metric=make_metric(args.task),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.reflection_minibatch_size,
        candidate_selection_strategy="pareto",
        num_threads=args.num_threads,
        track_stats=True,
        track_best_outputs=True,
        log_dir=str(out_dir / "gepa_log"),
        seed=args.seed,
        **budget,
    )

    print(f"[gepa] starting ({budget}) reflection={args.reflection_model}", flush=True)
    t0 = time.time()
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    wall = time.time() - t0
    print(f"[gepa] done in {wall:.0f}s", flush=True)

    evolved_instruction = get_instruction(optimized)
    (out_dir / "evolved_prompt.txt").write_text(evolved_instruction)
    optimized.save(str(out_dir / "optimized_program.json"))

    # ── stats ────────────────────────────────────────────────────────────────
    result = {
        "task": args.task,
        "persona_idx": args.persona_idx,
        "seed_level": args.seed_level,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "n_train": len(trainset),
        "n_val": len(valset),
        "reflection_model": args.reflection_model,
        "task_model": args.task_model,
        "budget": budget,
        "wall_s": round(wall, 1),
        "seed_instruction": seed_instruction,
        "evolved_instruction": evolved_instruction,
        "instruction_changed": evolved_instruction != seed_instruction,
        "dspy_version": dspy.__version__,
    }
    det = getattr(optimized, "detailed_results", None)
    if det is not None:
        try:
            scores = list(getattr(det, "val_aggregate_scores", []) or [])
            result["val_aggregate_scores"] = scores
            result["best_val_score"] = max(scores) if scores else None
            result["seed_val_score"] = scores[0] if scores else None
            result["best_idx"] = int(getattr(det, "best_idx", -1)) if scores else None
            result["num_candidates"] = len(scores)
        except Exception as e:  # pragma: no cover - stats are best-effort
            result["stats_error"] = repr(e)

    (out_dir / "gepa_result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    sv, bv = result.get("seed_val_score"), result.get("best_val_score")
    print("\n========== GEPA SUMMARY ==========", flush=True)
    print(f" persona #{args.persona_idx} ({args.seed_level}) on {args.task}", flush=True)
    if sv is not None and bv is not None:
        print(f" val score (DSPy-adapter scoring): seed={sv:.4f} -> best={bv:.4f}  (Δ={bv - sv:+.4f})", flush=True)
    print(f" instruction changed: {result['instruction_changed']}", flush=True)
    print(f" evolved prompt -> {out_dir / 'evolved_prompt.txt'}", flush=True)
    print(" NEXT: verdict in native harness:", flush=True)
    print(f"   python scripts/06_gepa_eval.py --gepa-prompt-file {out_dir / 'evolved_prompt.txt'} \\", flush=True)
    print(f"       --persona-idx {args.persona_idx} --split val   # then --split test", flush=True)
    print("==================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
