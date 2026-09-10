#!/usr/bin/env python3
"""P0: can method-conditioned prompting manufacture a pool of CORRECT solutions that use
genuinely different methods? Gate for P1 (diversity-curated SFT).

Why: natural sampling can't. On the same problems at K=8 only 18% have >=2 distinct-method
correct solutions and 0% have >=3 (measured 2026-08-24 from the regime probe) — which is why
the N4 diversity reward had no support. P1 needs an externally-sourced diverse pool.

Same regime/problems as the regime probe (Qwen3-8B non-thinking, AIME+AMC, 2048 tok, seed 0),
first 20 problems, so hinted rollouts are directly comparable to the natural K=8 baseline.

Gate: >=50% of problems reach >=3 distinct-method CORRECT solutions in the pooled
(natural + hinted) set, without a big accuracy hit. Secondary and equally important: does
raw_val (free) pick the judge-diverse subset as well as the judge would?

  CUDA_VISIBLE_DEVICES=3 HF_HOME=/scratch/yirenl2/.cache/huggingface \
    PYTHONPATH=/scratch/yirenl2/DeepPersona .venv-vllm/bin/python scripts/36_p0_method_pool.py --generate
  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/36_p0_method_pool.py --judge      # ~$1.5
  .venv/bin/python scripts/36_p0_method_pool.py --analyze
"""
import argparse, collections, importlib.util, json, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


OUT = Path("runs/p0-method-pool")
RP = Path("runs/regime-probe")
MODEL, REV = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
NPROB, NHINT_SAMP, SEED, MAXJUDGE = 20, 2, 0, 12
JUDGE = "gpt-5.5-2026-04-23"

HINTS = {
    "algebra": "direct algebraic manipulation — set up equations in the unknowns and solve symbolically",
    "casework": "exhaustive casework — split the problem into cases and handle each separately",
    "counting": "a counting/combinatorial argument — count configurations directly or by complement",
    "coords": "coordinates or an explicit geometric construction",
    "invariant": "a symmetry, parity, or invariant argument",
    "recursion": "recursion or a generating function",
    "bounding": "bounding/extremal reasoning — bound the quantity, then pin down the exact value",
    "brute": "brute-force computation — enumerate the possibilities and compute numerically",
}


def generate():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
    from deeppersona.personas import system_message
    s28 = _load("scripts/28_regime_probe.py", "s28")
    OUT.mkdir(exist_ok=True)
    items = s28.load_problems("aime_and_amc")[:NPROB]  # same shuffle/seed as regime probe
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    sysmsg = system_message("math", -1, 0, "basic")
    jobs, prompts = [], []
    for q, it in enumerate(items):
        for hk, hv in HINTS.items():
            user = (f"{it['problem']}\n\nSolve this using {hv}. If that approach genuinely "
                    f"cannot solve this problem, solve it whichever way works.")
            p = tok.apply_chat_template(
                [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for s in range(NHINT_SAMP):
                jobs.append((q, hk, s)); prompts.append(p)
    llm = LLM(model=MODEL, revision=REV, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=8192)
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048,
                         seed=SEED * 7919 + i) for i in range(len(jobs))]
    outs = llm.generate(prompts, sp)
    rows = []
    for (q, hk, s), o in zip(jobs, outs):
        text = o.outputs[0].text
        try:
            ext = extract_boxed(text)
        except ValueError:
            ext = None
        ok = bool(run_with_timeout_signal(grade_answer, args=(ext, items[q]["gold"]),
                                          timeout_seconds=2)) if ext else False
        rows.append({"qidx": q, "hint": hk, "samp": s, "text": text, "extracted": ext,
                     "correct": int(ok), "gold": items[q]["gold"],
                     "problem": items[q]["problem"]})
    with open(OUT / "hinted.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acc = np.mean([r["correct"] for r in rows])
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qidx"]].append(r["correct"])
    print(f"hinted: {len(rows)} gens, acc {acc:.3f}, "
          f"problems with >=1 correct {np.mean([any(v) for v in byq.values()]):.2f}, "
          f"mean #correct/problem {np.mean([sum(v) for v in byq.values()]):.1f}")
    per = collections.defaultdict(list)
    for r in rows:
        per[r["hint"]].append(r["correct"])
    print("  acc by hint: " + "  ".join(f"{k}:{np.mean(v):.2f}" for k, v in per.items()))


def pool(qmax=NPROB):
    """union of CORRECT solutions per problem: natural (regime probe K=8) + hinted."""
    nat = [json.loads(l) for l in open(RP / "traces_nonthink-hard.jsonl")]
    hin = [json.loads(l) for l in open(OUT / "hinted.jsonl")]
    byq = collections.defaultdict(list)
    for r in nat:
        if r["qidx"] < qmax and r["correct"]:
            byq[r["qidx"]].append({"id": f"nat{r['samp']}", "text": r["text"],
                                   "src": "natural", "extracted": r["extracted"]})
    for r in hin:
        if r["correct"]:
            byq[r["qidx"]].append({"id": f"h{r['hint']}_{r['samp']}", "text": r["text"],
                                   "src": r["hint"], "extracted": r["extracted"]})
    return byq


def judge():
    from openai import OpenAI
    s28 = _load("scripts/28_regime_probe.py", "s28")
    client = OpenAI()
    byq = pool()
    jobs = []
    for q, sols in sorted(byq.items()):
        if len(sols) < 2:
            continue
        # cap cost: round-robin over source so the sample isn't all one hint
        bysrc = collections.defaultdict(list)
        for s in sols:
            bysrc[s["src"]].append(s)
        picked, srcs = [], sorted(bysrc)
        while len(picked) < min(MAXJUDGE, len(sols)):
            for k in srcs:
                if bysrc[k]:
                    picked.append(bysrc[k].pop(0))
                    if len(picked) >= min(MAXJUDGE, len(sols)):
                        break
        jobs.append({"qidx": q, "problem": [r for r in
                                            (json.loads(l) for l in open(OUT / "hinted.jsonl"))
                                            if r["qidx"] == q][0]["problem"], "sols": picked})
    est = sum(sum(len(s["text"][:s28.JUDGE_CAP]) for s in j["sols"]) // 3 for j in jobs) * 2.5 / 1e6 \
        + len(jobs) * 2000 * 25 / 1e6
    print(f"judge jobs={len(jobs)} (mean {np.mean([len(j['sols']) for j in jobs]):.1f} sols) "
          f"est ~ ${est:.2f}")

    def one(j):
        body = f"PROBLEM:\n{j['problem']}\n\n" + "\n\n".join(
            f"--- SOLUTION {i + 1} ---\n{s['text'][:s28.JUDGE_CAP]}"
            for i, s in enumerate(j["sols"]))
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=JUDGE, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": s28.JUDGE_SYS},
                              {"role": "user", "content": body}],
                    max_completion_tokens=8000, reasoning_effort="low")
                out = json.loads(r.choices[0].message.content)
                asg = out.get("assignment", [])
                assert len(asg) == len(j["sols"])
                u = r.usage
                return {"qidx": j["qidx"], "ids": [s["id"] for s in j["sols"]],
                        "srcs": [s["src"] for s in j["sols"]], "assignment": asg,
                        "methods": out.get("methods", []), "prompt_tok": u.prompt_tokens,
                        "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as p:
        rows = list(p.map(one, jobs))
    with open(OUT / "judge_pool.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pt, ct = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote {len(rows)} rows, cost ~ ${(pt * 2.5 + ct * 25.0) / 1e6:.2f}")


def raw_vals(t, minv=2):
    out = set()
    for m in re.findall(r"-?\d+(?:/\d+)?", t or ""):
        try:
            v = Fraction(m)
        except Exception:
            continue
        if abs(v) > minv:
            out.add(v)
    return out


def analyze():
    s6 = _load("scripts/23_logdist_seqot.py", "s6")
    byq = pool()
    rows = [json.loads(l) for l in open(OUT / "judge_pool.jsonl")]
    txt = {(q, s["id"]): s["text"] for q, sols in byq.items() for s in sols}
    nm, n3, n4, cover_div, cover_rand, cover_best = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    print(f"{'q':>3s} {'|pool|':>6s} {'judged':>6s} {'methods':>7s}  "
          f"{'raw_val-4':>9s} {'random-4':>8s}   methods present")
    for r in rows:
        q, ids, asg = r["qidx"], r["ids"], r["assignment"]
        k = len(set(asg))
        nm.append(k); n3.append(int(k >= 3)); n4.append(int(k >= 4))
        V = {i: raw_vals(txt[(q, i)]) for i in ids}

        def d(a, b):
            A, B = V[a], V[b]
            return 1 - len(A & B) / len(A | B) if A and B else 1.0
        # greedy max-min-distance selection of 4 by raw_val (free curation)
        sel = [ids[0]]
        while len(sel) < min(4, len(ids)):
            best = max((i for i in ids if i not in sel),
                       key=lambda i: min(d(i, j) for j in sel))
            sel.append(best)
        cd = len({asg[ids.index(i)] for i in sel})
        rd = np.mean([len({asg[ids.index(i)] for i in
                           rng.choice(ids, min(4, len(ids)), replace=False)})
                      for _ in range(200)])
        cover_div.append(cd); cover_rand.append(rd); cover_best.append(min(4, k))
        print(f"{q:3d} {len(byq[q]):6d} {len(ids):6d} {k:7d}  {cd:9d} {rd:8.2f}   "
              f"{'; '.join(m[:28] for m in r['methods'][:3])}")
    nm = np.array(nm)
    print(f"\nPOOL (natural + hinted, {len(rows)} problems with >=2 correct):")
    print(f"  mean distinct methods among correct: {nm.mean():.2f}")
    print(f"  P(>=2 methods) = {np.mean(nm >= 2):.2f}   P(>=3) = {np.mean(n3):.2f}   "
          f"P(>=4) = {np.mean(n4):.2f}")
    print(f"  BASELINE natural K=8 (same regime, 50 problems): P(>=2)=0.18, P(>=3)=0.00")
    print(f"\nCURATION (does free raw_val pick the judge-diverse 4?):")
    print(f"  raw_val-4 covers {np.mean(cover_div):.2f} methods vs random-4 "
          f"{np.mean(cover_rand):.2f} vs oracle-max {np.mean(cover_best):.2f}")
    gate = np.mean(n3)
    print(f"\nGATE (>=50% of problems reach >=3 distinct-method correct): "
          f"{gate:.2f} -> {'PASS' if gate >= 0.5 else 'FAIL'}")
    json.dump({"n_problems": len(rows), "mean_methods": float(nm.mean()),
               "p_ge2": float(np.mean(nm >= 2)), "p_ge3": float(np.mean(n3)),
               "p_ge4": float(np.mean(n4)), "cover_rawval4": float(np.mean(cover_div)),
               "cover_random4": float(np.mean(cover_rand)),
               "cover_oracle4": float(np.mean(cover_best))},
              open(OUT / "p0_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    for f in ("generate", "judge", "analyze"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "hints": HINTS, "seed": SEED,
               "time": time.strftime("%FT%TZ", time.gmtime())},
              open(OUT / "manifest.json", "w"), indent=1)
    if a.generate:
        generate()
    if a.judge:
        judge()
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
