#!/usr/bin/env python3
"""F1-venue probe: base-model pass@1 vs pass@k across candidate benchmarks (SAMPLING ONLY).

Measures exploration headroom (gap = pass@K - pass@1) of base Qwen3-8B (non-thinking) on:
  aime_amc | math_l5 | humaneval | countdown
to pick the venue with the most room for diversity-RL. No training. Reuses Tinker sampling.

  set -a; . ./.tinker_env; set +a
  HF_HOME=/scratch/yirenl2/.cache/huggingface .venv/bin/python scripts/11_venue_probe.py --unit
  ... --smoke ;  ... (full: --n 40 --k 16)
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeppersona.config import RunConfig
from deeppersona.manifest import write_manifest
from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
from deeppersona.passk import pass_at_k_for_range

MATH_SYS = "Solve the problem. Put your final answer inside \\boxed{}."
CODE_SYS = ("You are an expert Python programmer. Complete the function. Return one complete, "
            "self-contained solution inside a single ```python code block.")
CD_SYS = ("You solve countdown number puzzles. Reason briefly, then END your reply with the final "
          "arithmetic expression inside \\boxed{}, e.g. \\boxed{(3+4)*5}. The \\boxed{} is required.")
MAX_TOK = {"aime_amc": 1024, "math_l5": 1024, "humaneval": 768, "countdown": 768}


# ── loaders → [{idx, ...}] ──────────────────────────────────────────────────
def load_aime_amc(n, seed):
    from deeppersona.data_espl import load_raw
    raw = load_raw("aime_and_amc")  # already seed-42 shuffled
    return [{"idx": i, "problem": r["problem"], "gold": r["groundtruth"]} for i, r in enumerate(raw[:n])]


def load_math_l5(n, seed):
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
    pieces = [load_dataset("EleutherAI/hendrycks_math", name=c, split=s)
              for c in get_dataset_config_names("EleutherAI/hendrycks_math") for s in ("train", "test")]
    full = concatenate_datasets(pieces)
    items = []
    for r in full:
        if "".join(ch for ch in str(r.get("level", "")) if ch.isdigit()) != "5":
            continue
        try:
            items.append({"problem": r["problem"], "gold": extract_boxed(r["solution"])})
        except ValueError:
            continue
    random.Random(seed).shuffle(items)
    items = items[:n]
    for i, it in enumerate(items):
        it["idx"] = i
    return items


def load_humaneval(n, seed):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    items = [{"idx": i, "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"],
              "canonical": r["canonical_solution"]} for i, r in enumerate(ds)]
    random.Random(seed).shuffle(items)
    return items[:n]


# ── countdown: generate solvable instances + AST-safe verifier ──────────────
def _reach_exprs(nums):
    vals = [(Fraction(x), str(x)) for x in nums]

    def rec(vs):
        if len(vs) == 1:
            return {vs[0][0]: vs[0][1]}
        out = {}
        for i in range(len(vs)):
            for j in range(len(vs)):
                if i == j:
                    continue
                rest = [vs[k] for k in range(len(vs)) if k != i and k != j]
                (a, ae), (b, be) = vs[i], vs[j]
                cand = [(a + b, f"({ae}+{be})"), (a - b, f"({ae}-{be})"), (a * b, f"({ae}*{be})")]
                if b != 0:
                    cand.append((a / b, f"({ae}/{be})"))
                for v, e in cand:
                    for vv, ee in rec(rest + [(v, e)]).items():
                        out.setdefault(vv, ee)
        return out
    return rec(vals)


def countdown_gen(n_items, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n_items:
        nums = [rng.randint(1, 9) for _ in range(4)]
        reach = {int(v): e for v, e in _reach_exprs(nums).items() if v.denominator == 1 and 0 < v <= 100}
        if len(reach) < 4:
            continue
        target = rng.choice(sorted(reach))
        out.append({"idx": len(out), "nums": nums, "target": target, "witness": reach[target]})
    return out


def _eval_frac(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        l, r = _eval_frac(node.left), _eval_frac(node.right)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if r == 0:
            raise ValueError("div0")
        return l / r
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_frac(node.operand)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return Fraction(node.value)
    raise ValueError("bad node")


def _collect_ints(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return [node.value]
    if isinstance(node, ast.BinOp):
        return _collect_ints(node.left) + _collect_ints(node.right)
    if isinstance(node, ast.UnaryOp):
        return _collect_ints(node.operand)
    return []


def countdown_verify(expr, nums, target):
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        val = _eval_frac(tree.body)
        used = _collect_ints(tree.body)
    except Exception:
        return 0
    return int(sorted(used) == sorted(nums) and val == Fraction(target))


# ── graders ─────────────────────────────────────────────────────────────────
def grade_math(gen, it):
    try:
        pred = extract_boxed(gen)
    except ValueError:
        return 0
    return int(bool(run_with_timeout_signal(grade_answer, args=(pred, it["gold"]), timeout_seconds=2)))


def grade_countdown(gen, it):
    try:
        expr = extract_boxed(gen)
    except ValueError:
        return 0
    return countdown_verify(expr, it["nums"], it["target"])


_GUARD = textwrap.dedent("""
    import os, builtins, resource
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_AS, (2**31, 2**31))
    os.system = lambda *a, **k: (_ for _ in ()).throw(OSError("blocked"))
    import subprocess as _sp
    _sp.Popen = _sp.run = _sp.call = lambda *a, **k: (_ for _ in ()).throw(OSError("blocked"))
    import shutil; shutil.rmtree = lambda *a, **k: None
    import socket; socket.socket = lambda *a, **k: (_ for _ in ()).throw(OSError("blocked"))
    _open = builtins.open
    def _safe_open(f, mode="r", *a, **k):
        if any(m in mode for m in ("w", "a", "x", "+")):
            raise OSError("write blocked")
        return _open(f, mode, *a, **k)
    builtins.open = _safe_open
""")


def _extract_code(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def _parseable(venue, gen):
    """Did the model emit the venue's required answer format (vs a capability miss)?"""
    if venue == "humaneval":
        return _extract_code(gen) is not None
    try:
        extract_boxed(gen)
        return True
    except ValueError:
        return False


def grade_humaneval(gen, it):
    code = _extract_code(gen) or (it["prompt"] + gen)
    program = _GUARD + "\n" + code + "\n" + it["test"] + f"\ncheck({it['entry_point']})\n"
    try:
        r = subprocess.run([sys.executable, "-c", program], capture_output=True,
                           timeout=10, cwd=tempfile.gettempdir())
        return int(r.returncode == 0)
    except Exception:
        return 0


VENUES = {
    "aime_amc": (load_aime_amc, grade_math, MATH_SYS, lambda it: it["problem"]),
    "math_l5": (load_math_l5, grade_math, MATH_SYS, lambda it: it["problem"]),
    "humaneval": (load_humaneval, grade_humaneval, CODE_SYS, lambda it: it["prompt"]),
    "countdown": (countdown_gen, grade_countdown, CD_SYS,
                  lambda it: f"Using each of these numbers exactly once: {it['nums']}. "
                             f"Combine them with + - * / and parentheses to make {it['target']}."),
}


# ── unit checks (no Tinker) ─────────────────────────────────────────────────
def run_unit():
    print("[unit] HumanEval grader (canonical should pass, stub should fail):")
    for it in load_humaneval(3, 0):
        good = grade_humaneval("```python\n" + it["prompt"] + it["canonical"] + "\n```", it)
        bad = grade_humaneval(f"```python\ndef {it['entry_point']}(*a, **k):\n    return None\n```", it)
        print(f"  {it['entry_point']:<22} canonical={good} stub={bad}", "OK" if good == 1 and bad == 0 else "!! FAIL")
    print("[unit] Countdown gen+verify (witness should pass, perturbations fail):")
    for it in countdown_gen(5, 0):
        w = it["witness"]
        ok = countdown_verify(w, it["nums"], it["target"])
        wrong_t = countdown_verify(w, it["nums"], it["target"] + 1)
        wrong_n = countdown_verify(w, it["nums"] + [1], it["target"])
        print(f"  nums={it['nums']} -> {it['target']}  witness={w}  verify={ok} wrongT={wrong_t} wrongN={wrong_n}",
              "OK" if ok == 1 and wrong_t == 0 and wrong_n == 0 else "!! FAIL")


# ── Tinker sampling fan-out ──────────────────────────────────────────────────
def sample_items(sc, renderer, tok, items, sysmsg, usermsg, K, temp, top_p, max_tokens, concurrency):
    import tinker
    sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=temp, top_p=top_p,
                               stop=renderer.get_stop_sequences())
    mis = [renderer.build_generation_prompt(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": usermsg(it)}]) for it in items]
    gens = [None] * len(items)
    for i in range(0, len(items), concurrency):
        chunk = list(range(i, min(i + concurrency, len(items))))
        futs = [sc.sample(prompt=mis[j], num_samples=K, sampling_params=sp) for j in chunk]
        for j, f in zip(chunk, futs):
            gens[j] = [tok.decode(s.tokens) for s in f.result().sequences]
        print(f"[sample] {min(i + concurrency, len(items))}/{len(items)}", flush=True)
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venues", default="aime_amc,math_l5,humaneval,countdown")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default="qwen3_disable_thinking")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.unit:
        run_unit()
        return

    if args.smoke:
        args.n, args.k = 2, 4

    import tinker
    from tinker_cookbook import renderers
    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(base_model=args.model)
    tok = sc.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tok)
    print(f"[setup] model={args.model} renderer={args.renderer} N={args.n} K={args.k} T={args.temp}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = REPO_ROOT / "runs" / (f"venue-probe-{ts}" + ("-smoke" if args.smoke else ""))
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, RunConfig(model_id=args.model, model_revision="tinker", task="venue-probe",
                                      condition="passk", seed=args.seed, split="probe", n_items=args.n,
                                      run_name="venue-probe", smoke=args.smoke), REPO_ROOT)

    results = {}
    for venue in args.venues.split(","):
        load, grade, sysmsg, usermsg = VENUES[venue]
        items = load(args.n, args.seed)
        mt = 256 if args.smoke else MAX_TOK[venue]
        print(f"\n[{venue}] {len(items)} items, max_tokens={mt}", flush=True)
        t0 = time.time()
        gens = sample_items(sc, renderer, tok, items, sysmsg, usermsg, args.k,
                            args.temp, args.top_p, mt, args.concurrency)
        counts, gen_lens, n_fmt, n_samp = [], [], 0, 0
        with (run_dir / "raw" / f"{venue}.jsonl").open("w") as f:
            for it, gl in zip(items, gens):
                c = 0
                for s, g in enumerate(gl):
                    ok = grade(g, it)
                    c += ok
                    n_fmt += int(_parseable(venue, g))
                    n_samp += 1
                    gen_lens.append(len(g))
                    f.write(json.dumps({"venue": venue, "idx": it["idx"], "sample": s, "correct": ok,
                                        "generation": g}) + "\n")
                counts.append(c)
        curve = np.mean([pass_at_k_for_range(args.k, c) for c in counts], axis=0).tolist()
        gk = lambda k: curve[min(k, args.k) - 1]
        p1, pK = curve[0], curve[-1]
        band = 0.10 <= p1 <= 0.60
        results[venue] = {"n": len(items), "K": args.k, "pass1": p1, "pass4": gk(4), "pass8": gk(8),
                          f"pass{args.k}": pK, "gap": pK - p1, "pass1_trainable_band": band,
                          "format_rate": n_fmt / max(n_samp, 1),
                          "mean_gen_chars": float(np.mean(gen_lens)), "secs": round(time.time() - t0, 1),
                          "curve": curve}
        r = results[venue]
        print(f"[{venue}] pass@1={p1:.3f} pass@4={gk(4):.3f} pass@8={gk(8):.3f} pass@{args.k}={pK:.3f} "
              f"GAP={r['gap']:.3f} band_ok={band} fmt={r['format_rate']:.2f} ({r['secs']}s)", flush=True)

    (run_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "venues": results}, indent=2))
    # recommendation: largest gap among trainable-band venues (else largest gap overall, flagged)
    band_v = {v: r for v, r in results.items() if r["pass1_trainable_band"]}
    pool = band_v or results
    best = max(pool, key=lambda v: pool[v]["gap"])
    print(f"\n=== venue probe summary (gap = pass@{args.k} - pass@1) ===")
    print(f"{'venue':>11} {'pass@1':>7} {'pass@8':>7} {'pass@'+str(args.k):>8} {'GAP':>7} {'band':>5} {'fmt':>5}")
    for v, r in results.items():
        print(f"{v:>11} {r['pass1']:>7.3f} {r['pass8']:>7.3f} {r[f'pass{args.k}']:>8.3f} {r['gap']:>7.3f} "
              f"{str(r['pass1_trainable_band']):>5} {r['format_rate']:>5.2f}")
    print(f"\nRECOMMEND: {best} (gap={pool[best]['gap']:.3f}, pass@1={pool[best]['pass1']:.3f})"
          + ("" if band_v else "  [NB: no venue in 0.10-0.60 band; picked max-gap overall]"))
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
