#!/usr/bin/env python3
"""F0a via Tinker sampling API (remote, no local GPU) on Qwen/Qwen3-8B.

Same protocol as scripts/07_f0a_axis_passk.py (vLLM) but generation goes through
tinker.SamplingClient with the qwen3_disable_thinking renderer (direct answers,
comparable to an instruct model). Three conditions at matched budget n:
  vanilla_T / vanilla_Thi (entropy control) / axis_T (personas or strategies).
Decision: axis_T pass@k beats BOTH controls by paired-bootstrap CI at some k>=4.

Run from .venv (has tinker + tinker_cookbook), sourcing the key file:
  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/08_f0a_tinker.py --task gsm8k --calibrate-only ...
Tinker is remote -> no CUDA, runs alongside local GPU jobs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeppersona.config import RunConfig
from deeppersona.data import load_items
from deeppersona.manifest import write_manifest
from deeppersona.passk import pass_at_k
from deeppersona.personas import num_personas, num_strategies, strategy_message, system_message
from deeppersona.scaffolding_stats import paired_boot_ci
from deeppersona.verifiers import extract_pred, score


# ── axis pools (shared semantics with script 07) ──────────────────────────
def axis_size(pool: str, task: str) -> int:
    return num_strategies() if pool == "strategies" else num_personas(task)


def axis_sys(pool: str, task: str, axis_idx: int, neutral_idx: int) -> str:
    if axis_idx < 0 or pool == "personas":
        return system_message(task, axis_idx, neutral_idx, "basic")
    return strategy_message(task, axis_idx)


def split_counts(n: int, k: int) -> list[int]:
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def distinct_n(generations: list[str], n: int = 4) -> float:
    grams, total = set(), 0
    for g in generations:
        toks = g.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    return len(grams) / total if total else 0.0


# ── scoring + data (task-aware; math needs sympy+pylatexenc in this venv) ──
def score_one(task, gen, gold):
    if task == "math":
        from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
        try:
            pred = extract_boxed(gen)
        except ValueError:
            pred = None
        ok = bool(run_with_timeout_signal(grade_answer, args=(pred, gold), timeout_seconds=2)) if pred else False
        return pred, int(ok)
    pred = extract_pred(task, gen)
    return pred, int(score(task, pred, gold))


def score_requests(requests, task):
    for r in requests:
        scored = [score_one(task, g, r["gold"]) for g in r["generations"]]
        r["preds"] = [p for p, _ in scored]
        r["correct"] = [c for _, c in scored]


def load_task_items(task, split, n_items, seed):
    if task == "math":
        from deeppersona.data_math import load_math_items
        return load_math_items(split, n_items=n_items, seed=seed)
    return load_items(task, split, n_items=n_items, seed=seed)


# ── Tinker backend ────────────────────────────────────────────────────────
def make_sampler(args):
    import tinker
    from tinker_cookbook import renderers
    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(base_model=args.model)
    tok = sc.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tok)
    return sc, renderer, tok


def make_req(task, pool, item, axis_idx, neutral_idx, n_samples, temp, neutral_idx_, condition):
    return dict(item_idx=item["idx"], gold=item["gold_answer"], condition=condition, persona_idx=axis_idx,
                system=axis_sys(pool, task, axis_idx, neutral_idx), question=item["question"],
                n_samples=n_samples, temperature=temp)


def run_and_fill(sc, renderer, tok, requests, top_p, mnt, seed, concurrency):
    """Fan out sc.sample() futures in bounded chunks; fill generations + gen_lens."""
    import tinker
    stop = renderer.get_stop_sequences()
    for r in requests:
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["question"]}]
        r["_mi"] = renderer.build_generation_prompt(msgs)
        r["_sp"] = tinker.SamplingParams(max_tokens=mnt, temperature=r["temperature"], top_p=top_p,
                                         stop=stop, seed=seed)
    done = 0
    for i in range(0, len(requests), concurrency):
        chunk = requests[i : i + concurrency]
        futs = [sc.sample(prompt=r["_mi"], num_samples=r["n_samples"], sampling_params=r["_sp"]) for r in chunk]
        for r, f in zip(chunk, futs):
            resp = f.result()
            r["generations"] = [tok.decode(s.tokens) for s in resp.sequences]
            r["gen_lens"] = [len(s.tokens) for s in resp.sequences]
        done += len(chunk)
        print(f"[gen] {done}/{len(requests)} requests", flush=True)


# ── metrics (shared semantics with script 07) ─────────────────────────────
def aggregate(requests):
    pool = defaultdict(lambda: {"gens": [], "personas": [], "correct": [], "gold": None})
    for r in requests:
        s = pool[(r["condition"], r["item_idx"])]
        s["gens"].extend(r["generations"])
        s["personas"].extend([r["persona_idx"]] * len(r["generations"]))
        s["correct"].extend(r["correct"]); s["gold"] = r["gold"]
    per_item = defaultdict(dict)
    for (cond, item_idx), s in pool.items():
        per_item[cond][item_idx] = {"n": len(s["correct"]), "c": int(sum(s["correct"])),
                                    "distinct4": distinct_n(s["gens"]),
                                    "correct_by_sample": s["correct"], "persona_of_sample": s["personas"]}
    return per_item


def passk_per_item(cond_items, items_sorted, Kmax):
    return {k: np.array([pass_at_k(cond_items[it]["n"], cond_items[it]["c"], k) for it in items_sorted])
            for k in range(1, Kmax + 1)}


def coverage(axis_items, n_axes):
    solved = defaultdict(set)
    for it, s in axis_items.items():
        for pa, ok in zip(s["persona_of_sample"], s["correct_by_sample"]):
            if ok:
                solved[pa].add(it)
    n_items = len(axis_items)
    out = {}
    for pa in range(n_axes):
        mine = solved[pa]
        unique = sum(1 for it in mine if all(it not in solved[o] for o in range(n_axes) if o != pa))
        out[pa] = {"solve_rate": len(mine) / n_items if n_items else 0.0,
                   "n_solved": len(mine), "unique_solved": unique}
    return out


def sample_accuracy(per_item_cond):
    tot_c = sum(s["c"] for s in per_item_cond.values())
    tot_n = sum(s["n"] for s in per_item_cond.values())
    return tot_c / tot_n if tot_n else 0.0


def run_dir_for(task, pool, suffix, smoke):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    name = f"f0a-tinker-{ts}-{task}-{pool}-{suffix}" + ("-smoke" if smoke else "")
    d = REPO_ROOT / "runs" / name
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def manifest_cfg(args):
    return RunConfig(model_id=args.model, model_revision="tinker", task=args.task, condition="axis-passk",
                     seed=args.seed, split="test", n_items=args.n_items, run_name="f0a-tinker", smoke=args.smoke)


def calibrate(args, sc, renderer, tok):
    items = load_task_items(args.task, "calib", args.n_cal_items, args.seed)
    print(f"[calib] {len(items)} calib items; pool={args.axis_pool}; axis@T={args.temp}; grid={args.thi_grid}", flush=True)
    counts = split_counts(args.n_cal_samples, args.n_axes)
    reqs = []
    for it in items:
        for pa in range(args.n_axes):
            reqs.append(make_req(args.task, args.axis_pool, it, pa, args.neutral_idx, counts[pa], args.temp, 0, "axis_T"))
        for thi in args.thi_grid:
            reqs.append(make_req(args.task, args.axis_pool, it, -1, args.neutral_idx, args.n_cal_samples, thi, 0, f"thi_{thi}"))
    run_and_fill(sc, renderer, tok, reqs, args.top_p, args.max_new_tokens, args.seed, args.concurrency)
    score_requests(reqs, args.task)
    per_item = aggregate(reqs)

    d4_axis = float(np.mean([s["distinct4"] for s in per_item["axis_T"].values()]))
    acc_axis = sample_accuracy(per_item["axis_T"])
    grid = []
    for thi in args.thi_grid:
        cond = f"thi_{thi}"
        d4 = float(np.mean([s["distinct4"] for s in per_item[cond].values()]))
        grid.append({"T_hi": thi, "distinct4": d4, "sample_acc": sample_accuracy(per_item[cond])})
        print(f"[calib] T_hi={thi}: distinct4={d4:.4f} sample_acc={grid[-1]['sample_acc']:.3f}", flush=True)
    chosen = min(grid, key=lambda g: abs(g["distinct4"] - d4_axis))["T_hi"]
    print(f"[calib] axis distinct4={d4_axis:.4f} (sample_acc={acc_axis:.3f}) -> chosen T_hi={chosen}", flush=True)

    out = run_dir_for(args.task, args.axis_pool, "calib", args.smoke)
    write_manifest(out, manifest_cfg(args), REPO_ROOT)
    (out / "thi_calibration.json").write_text(json.dumps({
        "task": args.task, "model": args.model, "axis_pool": args.axis_pool, "T": args.temp,
        "n_cal_items": len(items), "n_cal_samples": args.n_cal_samples, "n_axes": args.n_axes,
        "distinct4_axis": d4_axis, "axis_sample_acc": acc_axis, "grid": grid, "chosen_T_hi": chosen, "seed": args.seed,
    }, indent=2))
    print(f"[calib] wrote {out/'thi_calibration.json'}", flush=True)
    return chosen


def evaluate(args, sc, renderer, tok, t_hi):
    items = load_task_items(args.task, "test", args.n_items, args.seed)
    item_idxs = [it["idx"] for it in items]
    print(f"[data] {len(items)} test items; pool={args.axis_pool}; n={args.n_samples}; T={args.temp} T_hi={t_hi}; axes={args.n_axes}", flush=True)
    counts = split_counts(args.n_samples, args.n_axes)
    reqs = []
    for it in items:
        reqs.append(make_req(args.task, args.axis_pool, it, -1, args.neutral_idx, args.n_samples, args.temp, 0, "vanilla_T"))
        reqs.append(make_req(args.task, args.axis_pool, it, -1, args.neutral_idx, args.n_samples, t_hi, 0, "vanilla_Thi"))
        for pa in range(args.n_axes):
            reqs.append(make_req(args.task, args.axis_pool, it, pa, args.neutral_idx, counts[pa], args.temp, 0, "axis_T"))

    out = run_dir_for(args.task, args.axis_pool, "eval", args.smoke)
    if args.smoke:
        print(f"[smoke] first system: {reqs[0]['system']!r}", flush=True)
    t0 = time.time()
    run_and_fill(sc, renderer, tok, reqs, args.top_p, args.max_new_tokens, args.seed, args.concurrency)
    score_requests(reqs, args.task)
    print(f"[gen] {sum(len(r['generations']) for r in reqs)} samples in {time.time()-t0:.1f}s", flush=True)

    with (out / "raw" / "samples.jsonl").open("w") as f:
        for r in reqs:
            for slot, (g, gl, pred, ok) in enumerate(zip(r["generations"], r["gen_lens"], r["preds"], r["correct"])):
                f.write(json.dumps({"item_idx": r["item_idx"], "condition": r["condition"], "persona_idx": r["persona_idx"],
                                    "sample_slot": slot, "temperature": r["temperature"], "gold": r["gold"],
                                    "pred": pred, "correct": ok, "gen_len": gl, "generation": g}) + "\n")
                if args.smoke and r["item_idx"] == reqs[0]["item_idx"]:
                    print(f"[smoke] cond={r['condition']} p={r['persona_idx']} gold={r['gold']} pred={pred} ok={ok} len={gl}", flush=True)

    per_item = aggregate(reqs)
    conds = ["vanilla_T", "vanilla_Thi", "axis_T"]
    items_sorted = sorted(item_idxs)
    Kmax = min(32, args.n_samples)
    K = list(range(1, Kmax + 1))
    ppk = {c: passk_per_item(per_item[c], items_sorted, Kmax) for c in conds}
    passk_mean = {c: {k: float(ppk[c][k].mean()) for k in K} for c in conds}
    distinct4 = {c: float(np.mean([s["distinct4"] for s in per_item[c].values()])) for c in conds}
    sample_acc = {c: sample_accuracy(per_item[c]) for c in conds}
    cov = coverage(per_item["axis_T"], args.n_axes)

    decision_rows, first_pass_k = [], None
    for k in K:
        d1, d2 = ppk["axis_T"][k] - ppk["vanilla_T"][k], ppk["axis_T"][k] - ppk["vanilla_Thi"][k]
        lo1, hi1 = paired_boot_ci(d1, seed=args.seed)
        lo2, hi2 = paired_boot_ci(d2, seed=args.seed)
        beats_both = bool(lo1 > 0 and lo2 > 0)
        decision_rows.append({"k": k, "axis": passk_mean["axis_T"][k], "vanilla_T": passk_mean["vanilla_T"][k],
                              "vanilla_Thi": passk_mean["vanilla_Thi"][k], "d_vs_T": [lo1, hi1], "d_vs_Thi": [lo2, hi2],
                              "beats_both": beats_both})
        if beats_both and k >= 4 and first_pass_k is None:
            first_pass_k = k
    verdict = "PASS" if first_pass_k is not None else "NULL"

    print(f"\n[decision] task={args.task} pool={args.axis_pool} model={args.model}", flush=True)
    print(f"{'k':>3} {'axis':>7} {'van_T':>7} {'van_Thi':>8}  {'d_vs_T CI':>20} {'d_vs_Thi CI':>20}  both", flush=True)
    for r in decision_rows:
        print(f"{r['k']:>3} {r['axis']:>7.3f} {r['vanilla_T']:>7.3f} {r['vanilla_Thi']:>8.3f}  "
              f"[{r['d_vs_T'][0]:+.3f},{r['d_vs_T'][1]:+.3f}]  [{r['d_vs_Thi'][0]:+.3f},{r['d_vs_Thi'][1]:+.3f}]  "
              f"{'YES' if r['beats_both'] else ''}", flush=True)
    print(f"[distinct4] " + "  ".join(f"{c}={distinct4[c]:.4f}" for c in conds), flush=True)
    print(f"[sample_acc] " + "  ".join(f"{c}={sample_acc[c]:.3f}" for c in conds), flush=True)
    print(f"[verdict] {verdict}" + (f" (first k>=4 = {first_pass_k})" if first_pass_k else ""), flush=True)

    write_manifest(out, manifest_cfg(args), REPO_ROOT)
    (out / "per_item.json").write_text(json.dumps({c: {str(it): per_item[c][it] for it in per_item[c]} for c in conds}, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "task": args.task, "model": args.model, "renderer": args.renderer, "axis_pool": args.axis_pool,
        "verdict": verdict, "first_pass_k": first_pass_k,
        "sampling": {"n_samples": args.n_samples, "n_axes": args.n_axes, "T": args.temp, "T_hi": t_hi,
                     "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "Kmax": Kmax,
                     "n_items": len(items), "item_idxs": items_sorted, "seed": args.seed},
        "passk_mean": passk_mean, "distinct4": distinct4, "sample_acc": sample_acc,
        "coverage": {str(pa): cov[pa] for pa in cov}, "decision": decision_rows,
    }, indent=2))
    print(f"[run] wrote {out}", flush=True)
    return verdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--axis-pool", default="personas", choices=["personas", "strategies"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--renderer", default="qwen3_disable_thinking")
    p.add_argument("--n-items", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-axes", type=int, default=12)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--neutral-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--t-hi", type=float, default=None)
    p.add_argument("--thi-file", default=None)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--n-cal-items", type=int, default=16)
    p.add_argument("--n-cal-samples", type=int, default=32)
    p.add_argument("--thi-grid", type=float, nargs="+", default=[1.0, 1.1, 1.2, 1.3])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cap = axis_size(args.axis_pool, args.task)
    if not (1 <= args.n_axes <= cap):
        raise SystemExit(f"--n-axes must be in [1, {cap}] for pool={args.axis_pool} task={args.task}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    sc, renderer, tok = make_sampler(args)
    print(f"[model] {args.model} renderer={args.renderer} (tinker)", flush=True)

    if args.calibrate_only:
        calibrate(args, sc, renderer, tok)
        return 0
    if args.thi_file:
        t_hi = json.loads(Path(args.thi_file).read_text())["chosen_T_hi"]
    elif args.t_hi is not None:
        t_hi = args.t_hi
    else:
        raise SystemExit("need --t-hi or --thi-file (or run --calibrate-only first)")
    evaluate(args, sc, renderer, tok, t_hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
