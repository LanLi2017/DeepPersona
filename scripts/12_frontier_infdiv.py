#!/usr/bin/env python3
"""F2: inference-only diversity at frontier scale (no training).

Does manually injecting system-prompt diversity lift pass@k / coverage@k for a frontier
reasoning model? Budget-matched arms at K samples/problem:
  neutral  — same neutral prompt xK (internal-sampling baseline; gpt-5.x rejects temperature,
             so this is the only "vanilla" diversity available)
  static   — K distinct STRATEGY_AXES directives, one sample each (input diversity up front)
  adaptive — sequential: an orchestrator call reads prior attempts' visible solutions/answers
             and writes the next system prompt steering to an unexplored approach
             ("spawn sub-agents during inference", API-feasible version)
Venue must be unsaturated at frontier scale (BeyondAIME / hmmt_nov_2025); run --arm neutral
as the probe first. Same fixed problem subset across arms -> paired comparison.
"""
import argparse, json, math, random, sys, time

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deeppersona.data_espl import load_raw
from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
from deeppersona.personas import STRATEGY_AXES, ANSWER_INSTRUCTION
from deeppersona.passk import pass_at_k

ANS = ANSWER_INSTRUCTION["math"]
NEUTRAL = "Solve the math problem. Show your key steps." + ANS

ORCH_TMPL = """A solver is making multiple attempts at a competition math problem. Previous attempts (approach excerpt + final answer) are listed below. Write ONE short directive (<=40 words, imperative, like "Solve the problem by ...") steering the next attempt toward a genuinely DIFFERENT solution strategy than all previous ones. Output only the directive.

PROBLEM:
{problem}

PREVIOUS ATTEMPTS:
{prev}"""


def chat(client, model, sys_prompt, user, effort, max_tok, retries=4):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                reasoning_effort=effort, max_completion_tokens=max_tok)
            u = r.usage
            return (r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens)
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(2 ** a * 3)


def grade(text, gold):
    # signal-based timeout -> MAIN THREAD ONLY (call after sampling threads finish)
    ext = extract_boxed(text or "")
    if ext is None:
        return None, 0
    ok = run_with_timeout_signal(grade_answer, (ext, str(gold)), timeout_seconds=5)
    return ext, int(bool(ok))


def one_sample(client, args, item, sys_prompt, sp_id, samp_idx):
    text, ptok, ctok = chat(client, args.model, sys_prompt, item["problem"], args.effort, args.max_tokens)
    return {"qidx": item["qidx"], "arm": args.arm, "samp": samp_idx, "sp_id": sp_id,
            "sys": sys_prompt, "text": text, "gold": str(item["groundtruth"]),
            "prompt_tok": ptok, "compl_tok": ctok}


def adaptive_problem(client, args, item):
    rows, prev = [], []
    directive = NEUTRAL  # attempt 0 is neutral; later attempts get orchestrated directives
    for s in range(args.k):
        rows.append(one_sample(client, args, item, directive, f"adap{s}", s))
        r = rows[-1]
        # orchestrator sees answers/excerpts only, no oracle correctness (extract_boxed is thread-safe)
        prev.append(f"[attempt {s}] answer={extract_boxed(r['text'] or '')}\nexcerpt: {(r['text'] or '')[:600]}")
        if s < args.k - 1:
            d, ptok, ctok = chat(client, args.orch_model, "You direct a math solver.",
                                 ORCH_TMPL.format(problem=item["problem"], prev="\n\n".join(prev)),
                                 "low", 2000)
            rows[-1]["orch_tok"] = ptok + ctok
            directive = (d.strip() or NEUTRAL) + ANS
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["neutral", "static", "adaptive"], required=True)
    ap.add_argument("--dataset", default="BeyondAIME")
    ap.add_argument("--model", default="gpt-5.5-2026-04-23")
    ap.add_argument("--orch-model", default="gpt-5-mini-2025-08-07")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--tag", default="f2")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()
    random.seed(args.seed)

    data = load_raw(args.dataset)
    order = list(range(len(data)))
    random.Random(20260705).shuffle(order)  # FIXED subset -> paired across arms
    items = [{**data[i], "qidx": i} for i in order[: args.n]]

    if args.arm == "static":
        sps = random.Random(args.seed).sample(STRATEGY_AXES, min(args.k, len(STRATEGY_AXES)))
        sps = [(f"strat{STRATEGY_AXES.index(s)}", s + ANS) for s in (sps * math.ceil(args.k / len(sps)))[: args.k]]
    else:
        sps = [("neutral", NEUTRAL)] * args.k

    out = Path(f"runs/infdiv-{args.tag}-{args.dataset}-{args.arm}-k{args.k}-s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(vars(args), indent=1))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        if args.arm == "adaptive":
            rows = [r for lst in pool.map(lambda it: adaptive_problem(client, args, it), items) for r in lst]
        else:
            jobs = [(it, sp, spid, s) for it in items for s, (spid, sp) in enumerate(sps)]
            rows = list(pool.map(lambda j: one_sample(client, args, j[0], j[1], j[2], j[3]), jobs))

    for r in rows:  # grade in main thread (signal-based sympy timeout)
        r["extracted"], r["correct"] = grade(r["text"], r["gold"])

    with (out / "raw.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_q = {}
    for r in rows:
        by_q.setdefault(r["qidx"], []).append(r["correct"])
    n_corr = {q: sum(v) for q, v in by_q.items()}
    pass1 = sum(n_corr.values()) / sum(len(v) for v in by_q.values())
    cov = sum(c > 0 for c in n_corr.values()) / len(n_corr)
    pk = {k: sum(pass_at_k(len(v), n_corr[q], k) for q, v in by_q.items()) / len(by_q)
          for k in (1, 2, 4, 8) if k <= args.k}
    fmt = sum(r["extracted"] is not None for r in rows) / len(rows)
    ans_per_q = [len(set(x["extracted"] for x in rows if x["qidx"] == q and x["extracted"])) for q in by_q]
    ctok = sum(r["compl_tok"] for r in rows)
    ptok = sum(r["prompt_tok"] for r in rows)
    metrics = {"arm": args.arm, "n": len(by_q), "k": args.k, "pass1": round(pass1, 4),
               "coverage": round(cov, 4), "pass_at_k": {k: round(v, 4) for k, v in pk.items()},
               "format_rate": round(fmt, 4), "mean_distinct_answers": round(sum(ans_per_q) / len(ans_per_q), 2),
               "prompt_tok": ptok, "compl_tok": ctok, "wall_s": round(time.time() - t0)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
