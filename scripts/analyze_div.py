#!/usr/bin/env python3
"""Summarize F1 diversity-GRPO runs into a dose-response table, aggregated across seeds.

Reads runs/divgrpo-*<glob>*/metrics.jsonl, groups by (arm, prompt_source, n_distinct),
and reports mean +/- std across seeds for:
  Link 1 (input->output diversity): train-group distinct4 / self_bleu / answer_entropy (mean over steps).
  Link 2 (input->performance): held-out MATH-500 pass@1 last, pass@1 (last-first) delta, pass@8 last.
Usage: .venv/bin/python scripts/analyze_div.py [glob_substr=deep]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sub = sys.argv[1] if len(sys.argv) > 1 else "deep"


def per_run(run):
    rows = [json.loads(l) for l in (run / "metrics.jsonl").open()]
    if not rows:
        return None
    evals = [x for x in rows if "eval" in x]
    tg = lambda k: float(np.mean([s["train"]["div_group"][k] for s in rows]))
    return {
        "arm": rows[0]["arm"], "src": rows[0]["prompt_source"], "nd": rows[0]["n_distinct"],
        "n_steps": len(rows),
        "reward": float(np.mean([s["train"]["avg_reward"] for s in rows[-5:]])),
        "trD4": tg("distinct4"), "trSBLEU": tg("self_bleu"), "trAnsH": tg("answer_entropy"),
        "evP1_last": evals[-1]["eval"]["pass@1"] if evals else np.nan,
        "evP1_delta": (evals[-1]["eval"]["pass@1"] - evals[0]["eval"]["pass@1"]) if len(evals) >= 2 else np.nan,
        "evP8_last": evals[-1]["eval"]["pass@k"][-1] if evals else np.nan,
    }


groups = defaultdict(list)
for r in sorted(glob.glob(str(REPO / "runs" / f"divgrpo-*{sub}*"))):
    run = Path(r)
    if not (run / "metrics.jsonl").exists() or (run / "metrics.jsonl").stat().st_size == 0:
        continue
    rec = per_run(run)
    if rec:
        groups[(rec["arm"], rec["src"], rec["nd"])].append(rec)


def ms(vals):  # mean+/-std string
    v = [x for x in vals if x == x]  # drop nan
    if not v:
        return "    -    "
    return f"{np.mean(v):.3f}±{np.std(v):.3f}" if len(v) > 1 else f"{np.mean(v):.3f}     "


cols = ["reward", "trD4", "trSBLEU", "trAnsH", "evP1_last", "evP1_delta", "evP8_last"]
print(f"\nF1 dose-response (glob *{sub}*), mean±std across seeds  "
      f"[Link1=trD4↑/SBLEU↓/AnsH↑ ; Link2=evP1/evP8]\n")
print(f"{'level':>16} {'seeds':>5} {'steps':>5}  " + "  ".join(f"{c:>11}" for c in cols))
for key in sorted(groups, key=lambda k: (k[0], k[2])):
    g = groups[key]
    label = f"{key[0][:3]}/{key[1][:4]}/nd{key[2]}"
    steps = int(np.median([r["n_steps"] for r in g]))
    print(f"{label:>16} {len(g):>5} {steps:>5}  " + "  ".join(ms([r[c] for r in g]) for c in cols))
print("\nevP1_delta = held-out pass@1 (last eval - first eval); the Link-2 dose-response is the trend across nd.")
