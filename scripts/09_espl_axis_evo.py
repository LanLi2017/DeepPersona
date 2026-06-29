#!/usr/bin/env python3
"""Axis-structured evolutionary system-prompt optimization (E-SPL, evolution-only / no RL).

Tests two fixes for the F0a negative: (1) E-SPL competition-math benchmarks instead of
GSM8K/MATH-500, (2) *evolving* axis content via LLM self-reflection instead of hand-
authored axes. Programs are system prompts organized by a predefined persona-axis
template (role/method/verification/pitfalls/other + an "other" catch-all); mutation &
crossover (OpenAI) edit within axes. Rollouts/fitness come from Qwen3-8B via Tinker
(thinking ON). TrueSkill drives selection. Headline = does the best evolved prompt beat
the empty-prompt (root) baseline on a held-out test set, in pass@1 and pass@k.

Run from .venv (tinker + tinker_cookbook + openai), sourcing the key file:
  set -a; . ./.tinker_env; set +a
  HF_HOME=/scratch/yirenl2/.cache/huggingface .venv/bin/python scripts/09_espl_axis_evo.py --smoke
Tinker is remote (no local GPU); OpenAI is used only for mutation/crossover.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeppersona import espl_axis as ax
from deeppersona.config import RunConfig
from deeppersona.data_espl import load_espl_items
from deeppersona.manifest import write_manifest
from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
from deeppersona.passk import pass_at_k
from deeppersona.scaffolding_stats import paired_boot_ci
from deeppersona.trueskill_utils import Rating, TrueSkillSystem, scores_to_ties


# ── Tinker rollouts ─────────────────────────────────────────────────────────
def make_sampler(model: str, renderer_name: str):
    import tinker
    from tinker_cookbook import renderers
    sc = tinker.ServiceClient().create_sampling_client(base_model=model)
    tok = sc.get_tokenizer()
    return sc, renderers.get_renderer(renderer_name, tok), tok


def generate(sc, renderer, tok, reqs, top_p, max_new_tokens, seed, concurrency):
    """Each req: {system, problem, n_samples, temperature}. Fills gens + gen_lens in place."""
    import tinker
    stop = renderer.get_stop_sequences()
    for r in reqs:
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["problem"]}]
        r["_mi"] = renderer.build_generation_prompt(msgs)
        r["_sp"] = tinker.SamplingParams(max_tokens=max_new_tokens, temperature=r["temperature"],
                                         top_p=top_p, stop=stop, seed=seed)
    done = 0
    for i in range(0, len(reqs), concurrency):
        chunk = reqs[i:i + concurrency]
        futs = [sc.sample(prompt=r["_mi"], num_samples=r["n_samples"], sampling_params=r["_sp"]) for r in chunk]
        for r, f in zip(chunk, futs):
            resp = f.result()
            r["gens"] = [tok.decode(s.tokens) for s in resp.sequences]
            r["gen_lens"] = [len(s.tokens) for s in resp.sequences]
        done += len(chunk)
        print(f"[gen] {done}/{len(reqs)} reqs", flush=True)


def grade_one(gen: str, gold: str):
    try:
        pred = extract_boxed(gen)
    except ValueError:
        pred = None
    if not pred:
        return None, 0
    ok = bool(run_with_timeout_signal(grade_answer, args=(pred, gold), timeout_seconds=3))
    return pred, int(ok)


def grade_reqs(reqs):
    for r in reqs:
        scored = [grade_one(g, r["gold"]) for g in r["gens"]]
        r["preds"] = [p for p, _ in scored]
        r["correct"] = [c for _, c in scored]


# ── eval (pass@1 + pass@k of one system prompt on a held-out set) ───────────
def eval_system(sc, renderer, tok, system, items, n_samples, args, tag):
    t0 = time.time()
    reqs = [{"item_idx": it["idx"], "system": system, "problem": it["problem"], "gold": it["gold_answer"],
             "n_samples": n_samples, "temperature": args.temp} for it in items]
    generate(sc, renderer, tok, reqs, args.top_p, args.max_new_tokens, args.seed, args.concurrency)
    grade_reqs(reqs)
    per_item = [{"item_idx": r["item_idx"], "n": len(r["gens"]), "c": int(sum(r["correct"])),
                 "gen_lens": r["gen_lens"], "preds": r["preds"]} for r in reqs]
    passk = {k: float(np.mean([pass_at_k(p["n"], p["c"], k) for p in per_item]))
             for k in range(1, n_samples + 1)}
    pass1 = float(np.mean([p["c"] / p["n"] for p in per_item]))
    print(f"[eval:{tag}] pass@1={pass1:.3f} pass@{n_samples}={passk[n_samples]:.3f} "
          f"({sum(p['c'] for p in per_item)}/{sum(p['n'] for p in per_item)} correct, {time.time()-t0:.0f}s)", flush=True)
    return {"pass1": pass1, "passk": passk, "per_item": per_item}, reqs


def per_item_passk(per_item, kmax):
    """item_idxs (sorted) + {k: np.array of per-item pass@k aligned to that order}."""
    order = sorted(p["item_idx"] for p in per_item)
    by = {p["item_idx"]: p for p in per_item}
    return order, {k: np.array([pass_at_k(by[i]["n"], by[i]["c"], k) for i in order])
                   for k in range(1, kmax + 1)}


def dump_eval(reqs, tag, raw_dir):
    with (raw_dir / f"eval_{tag}.jsonl").open("w") as f:
        for r in reqs:
            for slot, (g, gl, pred, ok) in enumerate(zip(r["gens"], r["gen_lens"], r["preds"], r["correct"])):
                f.write(json.dumps({"tag": tag, "item_idx": r["item_idx"], "slot": slot, "gold": r["gold"],
                                    "pred": pred, "correct": ok, "gen_len": gl, "generation": g}) + "\n")


def passk_ci_vs_root(champ_eval, root_eval, kmax, seed):
    """Paired-bootstrap CI of (champion - root) per-item pass@k, per k (same test items)."""
    _, ck = per_item_passk(champ_eval["per_item"], kmax)
    _, rk = per_item_passk(root_eval["per_item"], kmax)
    out, sig_k = {}, None
    for k in range(1, kmax + 1):
        lo, hi = paired_boot_ci(ck[k] - rk[k], seed=seed)
        out[k] = [float(lo), float(hi)]
        if lo > 0 and sig_k is None:
            sig_k = k
    return out, sig_k


# ── one evolution step ──────────────────────────────────────────────────────
def evolve_step(step, pool, train_items, llm, ts, sc, renderer, tok, args, rng, raw_dir):
    sel = pool.select(args.num_parallel, args.recent_k, args.selection, args.lam, rng)
    batch = rng.sample(train_items, min(args.batch_size, len(train_items)))
    reqs = []
    for prog in sel:
        sysmsg = ax.render_system_prompt(prog.principles)
        for it in batch:
            reqs.append({"program_id": prog.program_id, "item_idx": it["idx"], "system": sysmsg,
                         "problem": it["problem"], "gold": it["gold_answer"],
                         "n_samples": args.group_size, "temperature": args.temp})
    generate(sc, renderer, tok, reqs, args.top_p, args.max_new_tokens, args.seed, args.concurrency)
    grade_reqs(reqs)

    # fitness V_i = mean over batch of pass@1 (c / group_size)
    by_prog = {p.program_id: [r for r in reqs if r["program_id"] == p.program_id] for p in sel}
    V = {pid: float(np.mean([sum(r["correct"]) / len(r["gens"]) for r in rs])) for pid, rs in by_prog.items()}
    for prog in sel:
        prog.history.append(V[prog.program_id])

    # TrueSkill tournament on this step's matched batch
    if len(sel) >= 2:
        order = sorted(sel, key=lambda p: V[p.program_id], reverse=True)
        ties = scores_to_ties([V[p.program_id] for p in order], tolerance=0.01)
        new = ts.rate_ranking([p.rating for p in order], ties=ties)
        for p, nr in zip(order, new):
            p.rating = Rating(nr.mu, nr.sigma)

    best = max(sel, key=lambda p: V[p.program_id])
    children = []

    # mutation on the best program
    best_rollouts = [{"problem": r["problem"], "answer": r["gold"],
                      "attempts": [{"text": g, "correct": ok} for g, ok in zip(r["gens"], r["correct"])]}
                     for r in by_prog[best.program_id]]
    child_principles, mut_info = ax.mutate(best.principles, best_rollouts, llm,
                                           max_ops=args.max_ops, max_problems=args.mutate_problems)
    child = ax.AxisProgram(child_principles, pool.new_id(), origin="mutation", parent=best.program_id,
                           timestep=step)
    child.rating = ax.mutation_rating(best.rating, args.mutation_sigma)
    best.children.append(child.program_id)
    pool.add(child)
    children.append(child.program_id)

    # crossover (optional): recombine axis content across the step's programs
    if len(sel) >= 2 and rng.random() < args.crossover_prob:
        winners = {p.program_id: [] for p in sel}
        for it in batch:
            c_by = {p.program_id: sum(r["correct"]) for p in sel
                    for r in by_prog[p.program_id] if r["item_idx"] == it["idx"]}
            wid = max(c_by, key=c_by.get)
            winners[wid].append((it["idx"], it["problem"]))
        order = [best] + [p for p in sel if p.program_id != best.program_id]
        xch_principles, _ = ax.crossover([p.principles for p in order],
                                         [winners[p.program_id] for p in order], llm)
        xch = ax.AxisProgram(xch_principles, pool.new_id(), origin="crossover",
                             parents_list=[p.program_id for p in sel], timestep=step)
        xch.rating = ax.crossover_rating([p.rating for p in sel], args.crossover_sigma)
        pool.add(xch)
        children.append(xch.program_id)

    # raw rollouts for this step (re-scorable offline)
    with (raw_dir / f"rollouts_step{step:03d}.jsonl").open("w") as f:
        for r in reqs:
            for slot, (g, gl, pred, ok) in enumerate(zip(r["gens"], r["gen_lens"], r["preds"], r["correct"])):
                f.write(json.dumps({"step": step, "program_id": r["program_id"], "item_idx": r["item_idx"],
                                    "slot": slot, "gold": r["gold"], "pred": pred, "correct": ok,
                                    "gen_len": gl, "generation": g}) + "\n")

    print(f"[step {step}] sel={[p.program_id for p in sel]} "
          f"V={ {p.program_id: round(V[p.program_id], 3) for p in sel} } "
          f"best={best.program_id} -> children {children} "
          f"(mut: {mut_info['n_problems']} probs, {len(child_principles)} principles)", flush=True)
    return {"step": step, "selected": [p.program_id for p in sel], "V": V, "best": best.program_id,
            "children": children, "ratings": {p.program_id: [p.rating.mu, p.rating.sigma] for p in sel},
            "n_principles": {c: len(pool.programs[-1].principles) for c in [children[-1]]}}


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    # data + models
    p.add_argument("--train-set", default="aime_and_amc")
    p.add_argument("--test-set", default=None, help="None -> disjoint split of train-set")
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test-items", type=int, default=None)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--renderer", default="qwen3")  # thinking ON
    p.add_argument("--mutation-model", default="gpt-4.1")
    p.add_argument("--mutation-temp", type=float, default=0.7)
    p.add_argument("--mutation-max-tokens", type=int, default=4000)
    p.add_argument("--axis-set", default="persona5", choices=["persona5", "compmath", "flat"])
    # evolution
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--num-parallel", type=int, default=3)
    p.add_argument("--max-pool", type=int, default=100)
    p.add_argument("--recent-k", type=int, default=5)
    p.add_argument("--selection", default="ucb", choices=["ucb", "uniform"])
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--max-ops", type=int, default=2)
    p.add_argument("--mutate-problems", type=int, default=6)
    p.add_argument("--crossover-prob", type=float, default=0.3)
    p.add_argument("--mutation-sigma", type=float, default=1.0)
    p.add_argument("--crossover-sigma", type=float, default=1.0)
    # rollouts / eval
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--test-group-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-topk", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:  # tiny end-to-end (plumbing, not accuracy)
        args.n_steps = args.n_steps if args.n_steps != 30 else 2
        args.batch_size, args.group_size, args.num_parallel = 4, 3, 3
        args.n_test, args.n_test_items, args.test_group_size = 6, 6, 3
        args.eval_every, args.crossover_prob, args.eval_topk = 1, 1.0, 1
        args.max_new_tokens = min(args.max_new_tokens, 4096)
        args.mutate_problems = 3

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    ax.configure_axes(args.axis_set)

    ts_ = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out = REPO_ROOT / "runs" / (f"espl-axis-{ts_}-{args.train_set}" + ("-smoke" if args.smoke else ""))
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out, RunConfig(model_id=args.model, model_revision="tinker", task=args.train_set,
                                  condition="espl-axis-evo", seed=args.seed, split="test",
                                  n_items=args.n_test, run_name="espl-axis", smoke=args.smoke), REPO_ROOT)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_items, test_items = load_espl_items(args.train_set, args.test_set, args.n_test,
                                              args.n_train, args.n_test_items, args.seed)
    print(f"[data] train={len(train_items)} test={len(test_items)} "
          f"(train_set={args.train_set} test_set={args.test_set or 'split'})", flush=True)
    print(f"[cfg] model={args.model} renderer={args.renderer} mutation={args.mutation_model} "
          f"axes={ax.AXES} steps={args.n_steps} batch={args.batch_size} group={args.group_size} "
          f"M={args.num_parallel} max_new_tokens={args.max_new_tokens}", flush=True)

    sc, renderer, tok = make_sampler(args.model, args.renderer)
    llm = ax.OpenAIChat(args.mutation_model, args.mutation_temp, args.mutation_max_tokens)

    # root baseline = empty prompt (the F0a-style neutral)
    root = ax.AxisProgram({}, 0, origin="root", timestep=-1)
    base_eval, base_reqs = eval_system(sc, renderer, tok, ax.render_system_prompt({}), test_items,
                                       args.test_group_size, args, "root")
    dump_eval(base_reqs, "root", raw_dir)

    pool = ax.EvolutionPool(args.max_pool)
    pool._next = 1
    pool.add(root)
    ts = TrueSkillSystem()

    metrics_f = (out / "metrics.jsonl").open("w")
    for step in range(args.n_steps):
        row = evolve_step(step, pool, train_items, llm, ts, sc, renderer, tok, args, rng, raw_dir)
        if (step + 1) % args.eval_every == 0 or step == args.n_steps - 1:
            champ = pool.best(by="mean")
            ev, _ = eval_system(sc, renderer, tok, ax.render_system_prompt(champ.principles), test_items,
                                args.test_group_size, args, f"champ@{step}(id{champ.program_id})")
            row["eval"] = {"champion_id": champ.program_id, "pass1": ev["pass1"], "passk": ev["passk"]}
        row["openai_calls"] = llm.n_calls
        metrics_f.write(json.dumps(row) + "\n")
        metrics_f.flush()
    metrics_f.close()

    # final: eval the top-K champions by rating, compare to root (paired-bootstrap CI on pass@k)
    rolled = [p for p in pool.programs if p.history]
    champs = sorted(rolled, key=lambda p: p.rating.mu, reverse=True)[:args.eval_topk]
    K = args.test_group_size
    finals = []
    for c in champs:
        ev, reqs = eval_system(sc, renderer, tok, ax.render_system_prompt(c.principles), test_items,
                               args.test_group_size, args, f"final-id{c.program_id}")
        dump_eval(reqs, f"final-id{c.program_id}", raw_dir)
        ci, sig_k = passk_ci_vs_root(ev, base_eval, K, args.seed)
        finals.append({"program_id": c.program_id, "origin": c.origin, "rating_mu": c.rating.mu,
                       "n_principles": len(c.principles), "pass1": ev["pass1"], "passk": ev["passk"],
                       "passk_ci_vs_root": ci, "first_sig_k": sig_k,
                       "principles": c.principles, "system_prompt": ax.render_system_prompt(c.principles)})

    best_final = max(finals, key=lambda f: f["pass1"]) if finals else None
    any_sig = any(f["first_sig_k"] is not None for f in finals)
    verdict = "NO PROGRAMS" if not best_final else (
        "BEAT_ROOT_CI" if any_sig else
        ("BEAT_ROOT_POINT" if (best_final["pass1"] > base_eval["pass1"] or
                               best_final["passk"][K] > base_eval["passk"][K]) else "NULL"))

    (out / "final_eval.json").write_text(json.dumps({
        "train_set": args.train_set, "test_set": args.test_set, "model": args.model,
        "mutation_model": args.mutation_model, "axis_set": args.axis_set, "axes": ax.AXES,
        "n_steps": args.n_steps, "batch_size": args.batch_size, "group_size": args.group_size,
        "test_group_size": K, "n_test": len(test_items), "openai_calls": llm.n_calls,
        "root_baseline": {"pass1": base_eval["pass1"], "passk": base_eval["passk"]},
        "champions": finals, "verdict": verdict,
    }, indent=2))
    (out / "evolution_pool.json").write_text(json.dumps([p.state_dict() for p in pool.programs], indent=2))

    print(f"\n[final] root pass@1={base_eval['pass1']:.3f} pass@{K}={base_eval['passk'][K]:.3f}", flush=True)
    for f in finals:
        print(f"[final] id{f['program_id']}({f['origin']}, {f['n_principles']}p) "
              f"pass@1={f['pass1']:.3f} pass@{K}={f['passk'][K]:.3f}", flush=True)
    print(f"[verdict] {verdict}   (openai_calls={llm.n_calls}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
