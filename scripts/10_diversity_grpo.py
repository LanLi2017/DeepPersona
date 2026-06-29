#!/usr/bin/env python3
"""F1 — rollout-input diversity as a swept lever in GRPO (Tinker, Qwen3-8B).

Forks the E-SPL GRPO core (refs/E-SPL/.../system_prompt_learning_rl.py) with all
evolution/TrueSkill/mutation stripped. The independent variable is how many DISTINCT
system prompts the G=group_total rollouts of a question are spread across:

  --n-distinct 1/2/4/8 (with --prompt-source strategy)  -> the dose-response
  --prompt-source neutral --n-distinct 1                -> vanilla-GRPO baseline (D0)
  --prompt-source neutral --n-distinct 1 --temp <T_hi>  -> temperature-matched control (D0-Thi)

Two arms (run both, per plan):
  within  : each question's G rollouts split across n_distinct prompts; MARGINALIZED baseline
            (reward - per-question mean over all prompts x rollouts). Train with the diverse
            scaffold; eval on a NEUTRAL prompt -> tests internalization.
  across  : each question homogeneous (one prompt, group_size=G, NORMAL GRPO baseline); the
            prompt varies ACROSS questions (round-robin over the n_distinct pool). Population
            diversity, statistically clean control.

Run (Tinker is remote -> no local GPU):
  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/10_diversity_grpo.py --arm within --prompt-source strategy \
      --n-distinct 2 --n-steps 1 --batch-questions 2 --n-test 4 --max-tokens 512 --smoke
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

from deeppersona.diversity_metrics import answer_entropy, distinct_n, self_bleu, token_surprisal
from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
from deeppersona.manifest import write_manifest
from deeppersona.config import RunConfig
from deeppersona.passk import pass_at_k, pass_at_k_for_range
from deeppersona.personas import num_strategies, strategy_message, system_message

NEUTRAL = system_message("math", -1, 0, "basic")


# ── data ──────────────────────────────────────────────────────────────────
def _level_ok(level: str, allowed: set[str]) -> bool:
    digits = "".join(ch for ch in str(level) if ch.isdigit())
    return digits in allowed if digits else False


def load_train(args, seed):
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

    test_problems = {r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    pieces = []
    for cfg in get_dataset_config_names("EleutherAI/hendrycks_math"):
        for split in ("train", "test"):
            pieces.append(load_dataset("EleutherAI/hendrycks_math", name=cfg, split=split))
    full = concatenate_datasets(pieces)
    allowed = set(args.levels.split(",")) if args.levels else set()
    items = []
    for r in full:
        if r["problem"] in test_problems:
            continue
        if allowed and not _level_ok(r.get("level", ""), allowed):
            continue
        try:
            gold = extract_boxed(r["solution"])
        except ValueError:
            continue
        items.append({"problem": r["problem"], "gold": gold, "level": r.get("level", "")})
    random.Random(seed).shuffle(items)
    if args.train_size > 0:
        items = items[: args.train_size]
    for i, it in enumerate(items):
        it["idx"] = i
    return items


def load_test(args, seed):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for i, r in enumerate(ds):
        gold = r.get("answer") or extract_boxed(r["solution"])
        items.append({"idx": i, "problem": r["problem"], "gold": gold, "level": r.get("level", "")})
    random.Random(seed + 1).shuffle(items)
    if args.n_test > 0:
        items = items[: args.n_test]
    return items


def grade(gen, gold):
    try:
        pred = extract_boxed(gen)
    except ValueError:
        pred = None
    ok = bool(run_with_timeout_signal(grade_answer, args=(pred, gold), timeout_seconds=2)) if pred else False
    return pred, int(ok)


# ── diversity over a question's group + a batch pool ────────────────────────
def group_diversity(comps, preds, lps):
    return {
        "distinct4": distinct_n(comps, 4),
        "self_bleu": self_bleu(comps),
        "answer_entropy": answer_entropy(preds),
        "token_surprisal": token_surprisal(lps),
    }


def mean_dicts(dicts):
    if not dicts:
        return {}
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


# ── sampling one batch into nested storage [P][Q] ───────────────────────────
def sample_cells(sampling_client, renderer, sp, cells):
    import tinker  # noqa: F401

    mis, futs = [], []
    for (_p, _q, sysmsg, nr, item) in cells:
        mi = renderer.build_generation_prompt(
            [{"role": "system", "content": sysmsg}, {"role": "user", "content": item["problem"]}]
        )
        mis.append(mi)
        futs.append(sampling_client.sample(prompt=mi, num_samples=nr, sampling_params=sp))
    out = {}
    for (p, q, _sys, _nr, item), mi, fut in zip(cells, mis, futs):
        ptoks = mi.to_ints()
        res = fut.result()
        rec = {"tokens": [], "ob_lens": [], "logprobs": [], "comps": [], "rewards": [], "preds": []}
        for seq in res.sequences:
            assert seq.logprobs is not None, "sampler returned no logprobs"
            rec["tokens"].append(ptoks + seq.tokens)
            rec["ob_lens"].append(len(ptoks) - 1)
            rec["logprobs"].append(seq.logprobs)
            msg, _ = renderer.parse_response(seq.tokens)
            txt = msg["content"]
            rec["comps"].append(txt)
            pred, ok = grade(txt, item["gold"])
            rec["preds"].append(pred)
            rec["rewards"].append(float(ok))
        out[(p, q)] = rec
    return out


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default="qwen3_disable_thinking")  # non-thinking; deliberate (see plan)
    ap.add_argument("--arm", choices=["within", "across"], default="within")
    ap.add_argument("--prompt-source", choices=["neutral", "strategy"], default="strategy")
    ap.add_argument("--n-distinct", type=int, default=2)
    ap.add_argument("--group-total", type=int, default=8)  # G: rollouts/question (fixed across levels)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)  # RL-stable LoRA lr; get_lr() (~5e-4) is SFT-tuned & DIVERGES here. --lr <=0 -> auto get_lr
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--rl-loss", default="importance_sampling")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--batch-questions", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--test-group-size", type=int, default=8)
    ap.add_argument("--levels", default="3,4,5")  # Hendrycks MATH difficulty band; "" = all
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")  # run-label suffix (e.g. D0thi)
    ap.add_argument("--no-update", action="store_true")  # dry: sample+advantage+eval, skip fwd_bwd
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.prompt_source == "neutral":
        args.n_distinct = 1
    assert args.group_total % args.n_distinct == 0, "group_total must be divisible by n_distinct"
    assert args.n_distinct <= num_strategies(), "n_distinct exceeds strategy pool size"
    G = args.group_total
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    print(f"[setup] model={args.model} renderer={args.renderer} arm={args.arm} "
          f"source={args.prompt_source} n_distinct={args.n_distinct} G={G} T={args.temp}", flush=True)

    tok = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tok)
    svc = tinker.ServiceClient()
    training_client = svc.create_lora_training_client(base_model=args.model, rank=args.lora_rank)

    lr = args.lr
    if lr <= 0:
        try:
            from tinker_cookbook import hyperparam_utils
            lr = hyperparam_utils.get_lr(args.model)
        except Exception:
            lr = 1e-5
        print(f"[setup] auto lr={lr}", flush=True)
    adam = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
                               stop=renderer.get_stop_sequences())
    sp_eval = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p,
                                    stop=renderer.get_stop_sequences())

    # prompt set: strategy axes [0..n_distinct) or the single neutral prompt
    prompt_set = ([NEUTRAL] if args.prompt_source == "neutral"
                  else [strategy_message("math", i) for i in range(args.n_distinct)])

    train_items = load_train(args, args.seed)
    test_items = load_test(args, args.seed)
    print(f"[data] train={len(train_items)} (levels={args.levels or 'all'}) test={len(test_items)}", flush=True)
    order = list(range(len(train_items)))
    random.Random(args.seed).shuffle(order)
    B = args.batch_questions

    cond = f"{args.arm}-{args.prompt_source[:4]}-nd{args.n_distinct}" + (f"-{args.tag}" if args.tag else "")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = REPO_ROOT / "runs" / (f"divgrpo-{ts}-{cond}-s{args.seed}" + ("-smoke" if args.smoke else ""))
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, RunConfig(model_id=args.model, model_revision="tinker", task="math",
                                      condition=cond, seed=args.seed, split="train",
                                      n_items=len(train_items), run_name="divgrpo", smoke=args.smoke), REPO_ROOT)
    (run_dir / "args.json").write_text(json.dumps({**vars(args), "lr_used": lr, "prompt_set": prompt_set}, indent=2))
    metrics_f = (run_dir / "metrics.jsonl").open("w")
    print(f"[run] {run_dir}", flush=True)

    def eval_neutral(sampling_client, step):
        cells = [(0, q, NEUTRAL, args.test_group_size, it) for q, it in enumerate(test_items)]
        data = sample_cells(sampling_client, renderer, sp_eval, cells)
        passk, divs, raw = [], [], []
        for q, it in enumerate(test_items):
            rec = data[(0, q)]
            c = int(sum(rec["rewards"]))
            passk.append(pass_at_k_for_range(args.test_group_size, c))
            divs.append(group_diversity(rec["comps"], rec["preds"], rec["logprobs"]))
            for slot, (g, pred, ok) in enumerate(zip(rec["comps"], rec["preds"], rec["rewards"])):
                raw.append({"step": step, "item_idx": it["idx"], "slot": slot, "gold": it["gold"],
                            "pred": pred, "correct": int(ok), "gen_len": len(rec["tokens"][slot]), "generation": g})
        with (run_dir / "raw" / f"eval_step{step:04d}.jsonl").open("w") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")
        passk_mean = np.mean(passk, axis=0).tolist()  # index k-1
        return {"pass@1": passk_mean[0], "pass@k": passk_mean, "div_group": mean_dicts(divs)}

    for step in range(max(args.n_steps, 1)):
        t0 = time.time()
        # refresh on-policy sampler from current weights (avoid stale-sampler desync)
        spath = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = svc.create_sampling_client(model_path=spath)

        # held-out neutral eval (internalization): at step 0, every eval_every, and final
        eval_stats = None
        if step == 0 or step % args.eval_every == 0 or step == args.n_steps - 1:
            eval_stats = eval_neutral(sampling_client, step)
            print(f"[eval s{step}] pass@1={eval_stats['pass@1']:.3f} "
                  f"pass@{args.test_group_size}={eval_stats['pass@k'][-1]:.3f} "
                  f"d4={eval_stats['div_group'].get('distinct4', 0):.3f}", flush=True)

        # build this step's batch + cells
        start = (step * B) % len(order)
        batch = [train_items[order[(start + i) % len(order)]] for i in range(B)]
        if args.arm == "within":
            P, R = args.n_distinct, G // args.n_distinct
            cells = [(p, q, prompt_set[p], R, it) for p in range(P) for q, it in enumerate(batch)]
        else:  # across: one prompt per question, varied across questions
            P = 1
            cells = [(0, q, prompt_set[q % len(prompt_set)], G, it) for q, it in enumerate(batch)]
        data = sample_cells(sampling_client, renderer, sp, cells)

        # reward tensor [P, Q, *] -> advantage (marginalize for within, normal for across)
        rew = torch.tensor([[data[(p, q)]["rewards"] for q in range(B)] for p in range(P)])
        if args.arm == "within":
            base = rew.mean(dim=0, keepdim=True).mean(dim=-1, keepdim=True)
        else:
            base = rew.mean(dim=-1, keepdim=True)
        adv = (rew - base).tolist()

        # datums (skip groups whose advantages are all ~0, as E-SPL does)
        datums, skipped = [], 0
        for p in range(P):
            for q in range(B):
                a = adv[p][q]
                if all(abs(x) < 1e-9 for x in a):
                    skipped += 1
                    continue
                rec = data[(p, q)]
                for tokens, ob, lp, av in zip(rec["tokens"], rec["ob_lens"], rec["logprobs"], a):
                    inp = [int(t) for t in tokens[:-1]]
                    tgt = tokens[1:]
                    all_lp = [0.0] * ob + lp
                    all_av = [0.0] * ob + [av] * (len(inp) - ob)
                    assert len(inp) == len(tgt) == len(all_lp) == len(all_av)
                    datums.append(types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=inp),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(tgt)),
                            "logprobs": TensorData.from_torch(torch.tensor(all_lp)),
                            "advantages": TensorData.from_torch(torch.tensor(all_av)),
                        }))

        # train-rollout metrics: per-question grouping (all P prompts for a question)
        grp_divs, q_pass1, q_passG = [], [], []
        pool_comps, pool_preds = [], []
        for q in range(B):
            comps, preds, lps, rews = [], [], [], []
            for p in range(P):
                rec = data[(p, q)]
                comps += rec["comps"]; preds += rec["preds"]; lps += rec["logprobs"]; rews += rec["rewards"]
            grp_divs.append(group_diversity(comps, preds, lps))
            q_pass1.append(float(np.mean(rews)))
            q_passG.append(pass_at_k(len(rews), int(sum(rews)), len(rews)))
            pool_comps += comps; pool_preds += preds
        train_stats = {
            "avg_reward": float(rew.mean().item()),
            "pass@1": float(np.mean(q_pass1)),
            f"pass@{G}_union": float(np.mean(q_passG)),
            "n_groups": P * B, "n_skipped": skipped, "n_datums": len(datums),
            "div_group": mean_dicts(grp_divs),
            "div_batch": {"distinct4": distinct_n(pool_comps, 4), "answer_entropy": answer_entropy(pool_preds)},
        }

        # raw train rollouts
        with (run_dir / "raw" / f"train_step{step:04d}.jsonl").open("w") as f:
            for p in range(P):
                for q in range(B):
                    rec = data[(p, q)]
                    for slot, (g, pred, ok) in enumerate(zip(rec["comps"], rec["preds"], rec["rewards"])):
                        f.write(json.dumps({"step": step, "prompt_idx": p, "item_idx": batch[q]["idx"],
                                            "slot": slot, "gold": batch[q]["gold"], "pred": pred,
                                            "correct": int(ok), "adv": adv[p][q][slot],
                                            "gen_len": len(rec["tokens"][slot]), "generation": g}) + "\n")

        # weight update
        did_update = False
        if not args.no_update and datums:
            fbf = training_client.forward_backward(datums, loss_fn=args.rl_loss)
            osf = training_client.optim_step(adam)
            fbf.result(); osf.result()
            did_update = True

        rec_line = {"step": step, "arm": args.arm, "prompt_source": args.prompt_source,
                    "n_distinct": args.n_distinct, "temp": args.temp, "lr": lr, "updated": did_update,
                    "secs": round(time.time() - t0, 1), "train": train_stats}
        if eval_stats is not None:
            rec_line["eval"] = eval_stats
        metrics_f.write(json.dumps(rec_line) + "\n"); metrics_f.flush()
        d = train_stats["div_group"]
        print(f"[step {step}] r={train_stats['avg_reward']:.3f} p@1={train_stats['pass@1']:.3f} "
              f"p@{G}={train_stats[f'pass@{G}_union']:.3f} d4={d['distinct4']:.3f} sbleu={d['self_bleu']:.3f} "
              f"ans_H={d['answer_entropy']:.3f} datums={len(datums)} skip={skipped} upd={did_update} "
              f"{rec_line['secs']}s", flush=True)

    metrics_f.close()
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
