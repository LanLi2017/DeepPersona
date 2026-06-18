#!/usr/bin/env python3
"""F0a: does fixed human-axis (persona) conditioning lift unbiased pass@k at a
matched per-problem sample budget vs i.i.d. sampling? (docs/axis_evolution_brainstorm.md §8)

Inference only, no training. vLLM sampling. Three conditions at matched budget n:
  vanilla_T    : n i.i.d. samples, neutral system, temp T
  vanilla_Thi  : n i.i.d. samples, neutral system, temp T_hi (entropy control, calibrated)
  axis_T       : n split across the 12 personas (basic), temp T

Decision: axis_T pass@k beats BOTH vanilla conditions by a paired-bootstrap-CI margin
at some k>=4. Run with --calibrate-only first to pick T_hi (matched on distinct-4).

Run from .venv-f0a (the only venv with vLLM):
  CUDA_VISIBLE_DEVICES=3 HF_HOME=/scratch/yirenl2/.cache/huggingface .venv-f0a/bin/python ...
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

from deeppersona.config import RunConfig, load_config
from deeppersona.data import load_items
from deeppersona.generate import build_chat_prompts
from deeppersona.manifest import write_manifest
from deeppersona.passk import pass_at_k
from deeppersona.personas import num_personas, num_strategies, strategy_message, system_message
from deeppersona.scaffolding_stats import paired_boot_ci
from deeppersona.verifiers import extract_pred, score


def axis_size(pool: str, task: str) -> int:
    return num_strategies() if pool == "strategies" else num_personas(task)


def axis_sys(pool: str, task: str, axis_idx: int, neutral_idx: int) -> str:
    """System message for an axis-value. axis_idx<0 -> shared neutral template (both pools)."""
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


def make_req(tok, sp_cls, pool, task, item, persona_idx, neutral_idx, n_samples, temp, top_p, mnt, seed, condition):
    sys_msg = axis_sys(pool, task, persona_idx, neutral_idx)
    prompt = build_chat_prompts(tok, sys_msg, [item["question"]])[0]
    sp = sp_cls(n=n_samples, temperature=temp, top_p=top_p, max_tokens=mnt, seed=seed)
    return dict(item_idx=item["idx"], gold=item["gold_answer"], condition=condition,
                persona_idx=persona_idx, system=sys_msg, prompt=prompt, sp=sp)


def run_and_fill(llm, requests):
    outs = llm.generate([r["prompt"] for r in requests], [r["sp"] for r in requests])
    for r, o in zip(requests, outs):
        r["generations"] = [c.text for c in o.outputs]
        r["gen_lens"] = [len(c.token_ids) for c in o.outputs]
    return requests


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


def aggregate(requests):
    """scored requests -> per-condition per-item {n,c,distinct4,correct_by_sample,persona_of_sample}."""
    pool = defaultdict(lambda: {"gens": [], "personas": [], "correct": [], "gold": None})
    for r in requests:
        s = pool[(r["condition"], r["item_idx"])]
        s["gens"].extend(r["generations"])
        s["personas"].extend([r["persona_idx"]] * len(r["generations"]))
        s["correct"].extend(r["correct"]); s["gold"] = r["gold"]
    per_item = defaultdict(dict)
    for (cond, item_idx), s in pool.items():
        per_item[cond][item_idx] = {
            "n": len(s["correct"]),
            "c": int(sum(s["correct"])),
            "distinct4": distinct_n(s["gens"]),
            "correct_by_sample": s["correct"],
            "persona_of_sample": s["personas"],
        }
    return per_item


def passk_per_item(cond_items: dict, items_sorted: list[int], Kmax: int) -> dict[int, np.ndarray]:
    """k -> array of per-item pass@k aligned to items_sorted."""
    out = {}
    for k in range(1, Kmax + 1):
        out[k] = np.array([pass_at_k(cond_items[it]["n"], cond_items[it]["c"], k) for it in items_sorted])
    return out


def coverage(axis_items: dict, n_axes: int) -> dict:
    solved = defaultdict(set)  # persona_idx -> set(item_idx) with >=1 correct
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


def sample_accuracy(per_item_cond: dict) -> float:
    tot_c = sum(s["c"] for s in per_item_cond.values())
    tot_n = sum(s["n"] for s in per_item_cond.values())
    return tot_c / tot_n if tot_n else 0.0


def make_llm(args, cfg, llm_cls):
    return llm_cls(model=cfg.model_id, revision=cfg.model_revision, dtype=cfg.dtype,
                   gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
                   enforce_eager=(args.smoke or args.enforce_eager), enable_prefix_caching=True,
                   seed=args.seed)


def run_dir_for(task: str, pool: str, suffix: str, smoke: bool) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    name = f"f0a-{ts}-{task}-{pool}-{suffix}" + ("-smoke" if smoke else "")
    d = REPO_ROOT / "runs" / name
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def calibrate(args, cfg, llm, sp_cls):
    """Pick T_hi whose vanilla distinct-4 best matches axis_T's, on a disjoint calib slice."""
    tok = llm.get_tokenizer()
    items = load_task_items(args.task, "calib", args.n_cal_items, args.seed)
    print(f"[calib] {len(items)} calib items; axis@T={args.temp}; grid={args.thi_grid}", flush=True)
    counts = split_counts(args.n_cal_samples, args.n_axes)

    reqs = []
    for it in items:
        for pa in range(args.n_axes):
            reqs.append(make_req(tok, sp_cls, args.axis_pool, args.task, it, pa, args.neutral_idx, counts[pa],
                                 args.temp, args.top_p, cfg.max_new_tokens, args.seed, "axis_T"))
        for thi in args.thi_grid:
            reqs.append(make_req(tok, sp_cls, args.axis_pool, args.task, it, -1, args.neutral_idx, args.n_cal_samples,
                                 thi, args.top_p, cfg.max_new_tokens, args.seed, f"thi_{thi}"))
    run_and_fill(llm, reqs)
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
    write_manifest(out, RunConfig(model_id=cfg.model_id, model_revision=cfg.model_revision, dtype=cfg.dtype,
                                  task=args.task, condition="calib", seed=args.seed, split="calib",
                                  n_items=args.n_cal_items, run_name="f0a-calib", smoke=args.smoke), REPO_ROOT)
    (out / "thi_calibration.json").write_text(json.dumps({
        "task": args.task, "axis_pool": args.axis_pool, "T": args.temp, "n_cal_items": len(items),
        "n_cal_samples": args.n_cal_samples, "n_axes": args.n_axes, "distinct4_axis": d4_axis,
        "axis_sample_acc": acc_axis, "grid": grid, "chosen_T_hi": chosen, "seed": args.seed,
    }, indent=2))
    print(f"[calib] wrote {out/'thi_calibration.json'}", flush=True)
    return chosen


def evaluate(args, cfg, llm, sp_cls, t_hi):
    tok = llm.get_tokenizer()
    items = load_task_items(args.task, "test", args.n_items, args.seed)
    item_idxs = [it["idx"] for it in items]
    print(f"[data] {len(items)} test items; budget n={args.n_samples}; T={args.temp} T_hi={t_hi}; "
          f"axes={args.n_axes}", flush=True)
    counts = split_counts(args.n_samples, args.n_axes)

    reqs = []
    for it in items:
        reqs.append(make_req(tok, sp_cls, args.axis_pool, args.task, it, -1, args.neutral_idx, args.n_samples,
                             args.temp, args.top_p, cfg.max_new_tokens, args.seed, "vanilla_T"))
        reqs.append(make_req(tok, sp_cls, args.axis_pool, args.task, it, -1, args.neutral_idx, args.n_samples,
                             t_hi, args.top_p, cfg.max_new_tokens, args.seed, "vanilla_Thi"))
        for pa in range(args.n_axes):
            reqs.append(make_req(tok, sp_cls, args.axis_pool, args.task, it, pa, args.neutral_idx, counts[pa],
                                 args.temp, args.top_p, cfg.max_new_tokens, args.seed, "axis_T"))

    out = run_dir_for(args.task, args.axis_pool, "eval", args.smoke)
    with (out / "raw" / "prompts.jsonl").open("w") as f:
        for r in reqs:
            f.write(json.dumps({"item_idx": r["item_idx"], "condition": r["condition"],
                                "persona_idx": r["persona_idx"], "system": r["system"],
                                "prompt": r["prompt"]}) + "\n")
    if args.smoke:
        print(f"[smoke] first templated prompt:\n{reqs[0]['prompt']}\n---", flush=True)

    t0 = time.time()
    run_and_fill(llm, reqs)
    print(f"[gen] {sum(len(r['generations']) for r in reqs)} samples in {time.time()-t0:.1f}s", flush=True)
    score_requests(reqs, args.task)

    with (out / "raw" / "samples.jsonl").open("w") as f:
        for r in reqs:
            for slot, (g, gl, pred, ok) in enumerate(zip(r["generations"], r["gen_lens"], r["preds"], r["correct"])):
                f.write(json.dumps({"item_idx": r["item_idx"], "condition": r["condition"],
                                    "persona_idx": r["persona_idx"], "sample_slot": slot,
                                    "temperature": r["sp"].temperature, "gold": r["gold"],
                                    "pred": pred, "correct": ok, "gen_len": gl, "generation": g}) + "\n")
                if args.smoke and r["item_idx"] == reqs[0]["item_idx"]:
                    print(f"[smoke] cond={r['condition']} p={r['persona_idx']} gold={r['gold']} "
                          f"pred={pred} ok={ok}", flush=True)

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
                              "vanilla_Thi": passk_mean["vanilla_Thi"][k],
                              "d_vs_T": [lo1, hi1], "d_vs_Thi": [lo2, hi2], "beats_both": beats_both})
        if beats_both and k >= 4 and first_pass_k is None:
            first_pass_k = k
    verdict = "PASS" if first_pass_k is not None else "NULL"

    print(f"\n[decision] task={args.task}  (axis beats BOTH by paired-CI at some k>=4 ?)", flush=True)
    print(f"{'k':>3} {'axis':>7} {'van_T':>7} {'van_Thi':>8}  {'d_vs_T CI':>20} {'d_vs_Thi CI':>20}  both", flush=True)
    for r in decision_rows:
        print(f"{r['k']:>3} {r['axis']:>7.3f} {r['vanilla_T']:>7.3f} {r['vanilla_Thi']:>8.3f}  "
              f"[{r['d_vs_T'][0]:+.3f},{r['d_vs_T'][1]:+.3f}]  [{r['d_vs_Thi'][0]:+.3f},{r['d_vs_Thi'][1]:+.3f}]  "
              f"{'YES' if r['beats_both'] else ''}", flush=True)
    print(f"[distinct4] " + "  ".join(f"{c}={distinct4[c]:.4f}" for c in conds), flush=True)
    print(f"[sample_acc] " + "  ".join(f"{c}={sample_acc[c]:.3f}" for c in conds), flush=True)
    print(f"[verdict] {verdict}" + (f" (first k>=4 = {first_pass_k})" if first_pass_k else ""), flush=True)

    write_manifest(out, RunConfig(model_id=cfg.model_id, model_revision=cfg.model_revision, dtype=cfg.dtype,
                                  task=args.task, condition="axis-passk", seed=args.seed, split="test",
                                  n_items=args.n_items, run_name="f0a", smoke=args.smoke), REPO_ROOT)
    (out / "per_item.json").write_text(json.dumps(
        {c: {str(it): per_item[c][it] for it in per_item[c]} for c in conds}, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "task": args.task, "axis_pool": args.axis_pool, "verdict": verdict, "first_pass_k": first_pass_k,
        "sampling": {"n_samples": args.n_samples, "n_axes": args.n_axes, "T": args.temp, "T_hi": t_hi,
                     "top_p": args.top_p, "max_new_tokens": cfg.max_new_tokens, "Kmax": Kmax,
                     "n_items": len(items), "item_idxs": items_sorted, "seed": args.seed},
        "passk_mean": {c: passk_mean[c] for c in conds},
        "distinct4": distinct4, "sample_acc": sample_acc,
        "coverage": {str(pa): cov[pa] for pa in cov},
        "decision": decision_rows,
    }, indent=2))
    print(f"[run] wrote {out}", flush=True)
    return verdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--axis-pool", default="personas", choices=["personas", "strategies"])
    p.add_argument("--n-items", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-axes", type=int, default=12)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--neutral-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-hi", type=float, default=None)
    p.add_argument("--thi-file", default=None)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--n-cal-items", type=int, default=16)
    p.add_argument("--n-cal-samples", type=int, default=32)
    p.add_argument("--thi-grid", type=float, nargs="+", default=[1.0, 1.1, 1.2, 1.3])
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cap = axis_size(args.axis_pool, args.task)
    if not (1 <= args.n_axes <= cap):
        raise SystemExit(f"--n-axes must be in [1, {cap}] for pool={args.axis_pool} task={args.task}")

    cfg = load_config(args.config)
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    random.seed(args.seed)
    np.random.seed(args.seed)

    from vllm import LLM, SamplingParams
    llm = make_llm(args, cfg, LLM)
    print(f"[model] {cfg.model_id}@{cfg.model_revision[:8]} dtype={cfg.dtype}", flush=True)

    if args.calibrate_only:
        calibrate(args, cfg, llm, SamplingParams)
        return 0

    if args.thi_file:
        t_hi = json.loads(Path(args.thi_file).read_text())["chosen_T_hi"]
    elif args.t_hi is not None:
        t_hi = args.t_hi
    else:
        raise SystemExit("need --t-hi or --thi-file (or run --calibrate-only first)")

    evaluate(args, cfg, llm, SamplingParams, t_hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
