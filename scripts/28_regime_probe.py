#!/usr/bin/env python3
"""Regime probe: does genuine method diversity exist in candidate GRPO regimes?

N-methods showed the current regime (Qwen3-8B non-thinking MATH L4-5 @1024tok) has
V_method ~ 1.4, 38% of problems >=2 methods -- too thin to reward. Before N4 can be
considered we need a regime where natural k=8 rollouts actually contain method switches.
Candidates (all Qwen3-8B, temp 0.7 top_p 0.95, seed 0):
  think-l45      thinking, same 50 MATH L4-5 problems as logdist-qwen (mode-only contrast)
  think-hard     thinking, 50 AIME+AMC problems, 8192 tok (hardness + mode)
  nonthink-hard  non-thinking, same 50 AIME+AMC problems, 2048 tok (hardness only)

Judge = method-clustering prompt from scripts/27 (validity of the prompt established there
with gpt-4.1: self-agreement 92%, style-blindness 91%), run on gpt-5.5 (user choice,
2026-08-12) over the post-</think> solution writeup. Labeling includes a 4th pseudo-regime
`nonthink-l45` = re-label of the existing logdist-qwen traces with the SAME judge, so the
cross-regime table is judge-controlled (and gives 5.5-vs-4.1 judge agreement for free).

  # generation runs in .venv-vllm (vLLM + sympy/pylatexenc/datasets); labeling in .venv
  CUDA_VISIBLE_DEVICES=3 .venv-vllm/bin/python scripts/28_regime_probe.py --generate think-l45 --smoke
  CUDA_VISIBLE_DEVICES=3 .venv-vllm/bin/python scripts/28_regime_probe.py --generate all
  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/28_regime_probe.py --label all        # est <$25 (approved 2026-08-12)
  .venv/bin/python scripts/28_regime_probe.py --analyze
"""
import argparse, collections, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from deeppersona.data_espl import load_raw
from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
from deeppersona.personas import system_message

OUT = Path("runs/regime-probe")
QT = Path("runs/logdist-qwen")
MODEL = "Qwen/Qwen3-8B"
REV = "b968826d9c46dd6066d109eabc6255188de91218"
K, NPROB, NREPEAT, SEED = 8, 50, 10, 0
JUDGE_CAP = 6000  # chars per solution shown to the judge

REGIMES = {
    "think-l45": dict(source="qwen-testbed", thinking=True, max_new=12288),
    "think-hard": dict(source="aime_and_amc", thinking=True, max_new=16384),
    "nonthink-hard": dict(source="aime_and_amc", thinking=False, max_new=2048),
}
LABEL_SETS = ["nonthink-l45"] + list(REGIMES)  # nonthink-l45 = baseline re-label from QT
PRICE = {"gpt-4.1": (2.0, 8.0), "gpt-5.5-2026-04-23": (2.5, 25.0)}  # $/M in, out (5.5 assumed in)

JUDGE_SYS = """You are given several solutions to the same math problem. Cluster them by SOLUTION METHOD.

Two solutions use the SAME method if they rely on the same key idea and overall strategy (same setup, same kind of decomposition/theorem/technique), even if they differ in wording, verbosity, notation, variable names, order of presentation, or contain arithmetic slips leading to different final answers.
Two solutions use DIFFERENT methods only if the mathematical approach itself differs (e.g. casework vs generating functions; coordinate geometry vs synthetic; direct counting vs complementary counting; induction vs closed-form derivation).

Output JSON:
{"methods": ["short description of method 1", ...],
 "assignment": [m, m, ...]}   // for each solution in the order given, the 1-based index of its method
Every solution must be assigned. Do not create a new method for presentation differences."""


def boxed_or_none(text):
    try:
        return extract_boxed(text or "")
    except ValueError:
        return None


def load_problems(source):
    if source == "qwen-testbed":
        rows = [json.loads(l) for l in open(QT / "traces.jsonl")]
        byq = {r["qidx"]: r for r in rows if r["samp"] == 0}
        return [{"problem": byq[q]["problem"], "gold": byq[q]["gold"]} for q in sorted(byq)[:NPROB]]
    data = load_raw(source)
    random.Random(SEED).shuffle(data)
    return [{"problem": d["problem"], "gold": d["groundtruth"]} for d in data[:NPROB]]


def generate(llm, regime, smoke):
    # vLLM engine passed in (HF generate OOMs/too slow at 8-16k tokens; see log 2026-08-12)
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    cfg = REGIMES[regime]
    OUT.mkdir(exist_ok=True)
    items = load_problems(cfg["source"])
    k = K
    if smoke:
        items, k = items[:2], 2
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    sysmsg = system_message("math", -1, 0, "basic")
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": it["problem"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=cfg["thinking"])
        for it in items]
    jobs = [(q, s) for q in range(len(items)) for s in range(k)]
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=cfg["max_new"],
                         seed=SEED * 1000003 + q * K + s) for q, s in jobs]
    t0 = time.time()
    outs = llm.generate([prompts[q] for q, _ in jobs], sp)
    print(f"[{regime}] {len(jobs)} gens in {time.time() - t0:.0f}s", flush=True)
    texts = {(q, s): o.outputs[0].text for (q, s), o in zip(jobs, outs)}
    rows = []
    for (q, s), text in sorted(texts.items()):
        # thinking mode: solution = post-</think> writeup; truncated-in-think -> no solution
        sol = text.split("</think>")[-1].strip() if "</think>" in text \
            else ("" if cfg["thinking"] else text)
        truncated = cfg["thinking"] and "</think>" not in text
        ext = boxed_or_none(sol)
        ok = bool(run_with_timeout_signal(grade_answer, args=(ext, items[q]["gold"]),
                                          timeout_seconds=2)) if ext else False
        rows.append({"qidx": q, "samp": s, "text": text, "solution": sol, "truncated": int(truncated),
                     "extracted": ext, "gold": items[q]["gold"], "correct": int(ok),
                     "problem": items[q]["problem"]})
    fn = OUT / f"traces_{regime}{'_smoke' if smoke else ''}.jsonl"
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acc = np.mean([r["correct"] for r in rows])
    trunc = np.mean([r["truncated"] for r in rows])
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qidx"]].append(r["correct"])
    p8 = np.mean([any(v) for v in byq.values()])
    print(f"[{regime}] wrote {len(rows)} traces: acc {acc:.3f}  pass@{k} {p8:.2f}  "
          f"truncated {trunc:.1%}")


def label(regime, smoke, model_name):
    from openai import OpenAI
    client = OpenAI()
    if regime == "nonthink-l45":
        rows = [r for l in open(QT / "traces.jsonl") if (r := json.loads(l))["qidx"] < NPROB]
        for r in rows:
            r["solution"] = r["text"]
    else:
        rows = [json.loads(l) for l in open(OUT / f"traces_{regime}.jsonl")]
    byq = collections.defaultdict(dict)
    for r in rows:
        byq[r["qidx"]][r["samp"]] = r
    jobs = []
    for q in sorted(byq):
        # judge the solution writeup; fall back to tail of thinking for truncated traces
        texts = [(byq[q][s]["solution"] or byq[q][s]["text"][-JUDGE_CAP:])[:JUDGE_CAP]
                 for s in range(K)]
        jobs.append({"qidx": q, "problem": byq[q][0]["problem"], "texts": texts})
    runs = [(j, 0) for j in jobs] + [(j, 1) for j in jobs[:NREPEAT]]
    if smoke:
        runs = runs[:2]
    pin, pout = PRICE[model_name]
    is_g5 = model_name.startswith("gpt-5")
    est = sum(sum(len(t) for t in j["texts"]) // 3 + 500 for j, _ in runs) * pin / 1e6 \
        + len(runs) * (2000 if is_g5 else 400) * pout / 1e6
    print(f"[{regime}] jobs={len(runs)}  est cost ~ ${est:.2f} ({model_name})")

    def one(run):
        j, rep = run
        body = f"PROBLEM:\n{j['problem']}\n\n" + "\n\n".join(
            f"--- SOLUTION {i + 1} ---\n{t}" for i, t in enumerate(j["texts"]))
        for attempt in range(4):
            try:
                kw = dict(model=model_name, response_format={"type": "json_object"},
                          messages=[{"role": "system", "content": JUDGE_SYS},
                                    {"role": "user", "content": body}])
                if is_g5:  # reasoning models: fixed temp, thinking tokens bill as output
                    kw.update(max_completion_tokens=8000, reasoning_effort="low")
                else:
                    kw.update(max_completion_tokens=1500,
                              temperature=0.0 if rep == 0 else 0.7)
                r = client.chat.completions.create(**kw)
                out = json.loads(r.choices[0].message.content)
                asg = out.get("assignment", [])
                assert len(asg) == len(j["texts"]), f"len {len(asg)} != {len(j['texts'])}"
                u = r.usage
                return {"qidx": j["qidx"], "rep": rep, "methods": out.get("methods", []),
                        "assignment": asg, "judge": model_name,
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        out_rows = list(pool.map(one, runs))
    fn = OUT / f"method_labels_{regime}{'_smoke' if smoke else ''}.jsonl"
    with open(fn, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in out_rows), sum(r["compl_tok"] for r in out_rows)
    print(f"[{regime}] wrote {fn}: {len(out_rows)} rows, "
          f"cost ~ ${(ptok * pin + ctok * pout) / 1e6:.2f}")


def vendi(assignment):
    c = np.array(list(collections.Counter(assignment).values()), float)
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def summarize(name, main, rep, corr):
    nm = [len(set(r["assignment"][:K])) for r in main.values()]
    vm = [vendi(r["assignment"][:K]) for r in main.values()]
    agree = tot = 0
    for q, r1 in rep.items():
        a1, a2 = main[q]["assignment"][:K], r1["assignment"][:K]
        for i, j in combinations(range(K), 2):
            agree += (a1[i] == a1[j]) == (a2[i] == a2[j]); tot += 1
    with_sw = [q for q, r in main.items() if len(set(r["assignment"][:K])) >= 2]
    p8_sw = np.mean([any(corr[q].values()) for q in with_sw]) if with_sw else float("nan")
    no_sw = [q for q in main if q not in with_sw]
    p8_no = np.mean([any(corr[q].values()) for q in no_sw]) if no_sw else float("nan")
    acc = np.mean([v for q in corr for v in corr[q].values()])
    p8 = np.mean([any(corr[q].values()) for q in corr])
    print(f"{name:14s} acc {acc:.3f}  pass@8 {p8:.2f} | methods/prob mean {np.mean(nm):.2f}  "
          f"V_method {np.mean(vm):.2f}  >=2m {np.mean([x >= 2 for x in nm]):.0%}  "
          f">=4m {np.mean([x >= 4 for x in nm]):.0%} | "
          f"p8 multi {p8_sw:.2f} (n={len(with_sw)}) vs single {p8_no:.2f} | "
          f"judge-agree {agree / max(tot, 1):.0%}")
    return dict(name=name, acc=float(acc), pass8=float(p8), mean_methods=float(np.mean(nm)),
                v_method=float(np.mean(vm)), pct_ge2=float(np.mean([x >= 2 for x in nm])),
                pct_ge4=float(np.mean([x >= 4 for x in nm])), agree=agree / max(tot, 1),
                p8_multi=float(p8_sw), p8_single=float(p8_no), n_multi=len(with_sw))


def analyze():
    res = []
    for regime in LABEL_SETS:
        fn = OUT / f"method_labels_{regime}.jsonl"
        if not fn.exists():
            print(f"{regime}: no labels yet"); continue
        rows = [json.loads(l) for l in open(fn)]
        main = {r["qidx"]: r for r in rows if r["rep"] == 0}
        rep = {r["qidx"]: r for r in rows if r["rep"] == 1}
        tr_path = QT / "traces.jsonl" if regime == "nonthink-l45" \
            else OUT / f"traces_{regime}.jsonl"
        tr = [r for l in open(tr_path) if (r := json.loads(l))["qidx"] < NPROB]
        corr = collections.defaultdict(dict)
        for r in tr:
            corr[r["qidx"]][r["samp"]] = r["correct"]
        d = summarize(regime, main, rep, corr)
        if regime != "nonthink-l45":
            d["truncated"] = float(np.mean([r["truncated"] for r in tr]))
        res.append(d)

    # judge check: gpt-5.5 vs the original gpt-4.1 labels on the shared baseline traces
    fn = OUT / "method_labels_nonthink-l45.jsonl"
    if fn.exists():
        new = {r["qidx"]: r for l in open(fn)
               if (r := json.loads(l))["rep"] == 0}
        old = {r["qidx"]: r for l in open(QT / "method_labels.jsonl")
               if (r := json.loads(l))["rep"] == 0 and r["qidx"] < NPROB}
        agree = tot = 0
        for q in new:
            a1, a2 = new[q]["assignment"][:K], old[q]["assignment"][:K]
            for i, j in combinations(range(K), 2):
                agree += (a1[i] == a1[j]) == (a2[i] == a2[j]); tot += 1
        nm_old = [len(set(r["assignment"][:K])) for r in old.values()]
        print(f"\ncross-judge (5.5 vs 4.1, same traces): pairwise agreement "
              f"{agree / tot:.1%};  gpt-4.1 methods/prob {np.mean(nm_old):.2f}")
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", metavar="REGIME")
    ap.add_argument("--label", metavar="REGIME")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--model", default="gpt-5.5-2026-04-23")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "model": MODEL, "rev": REV, "seed": SEED,
               "sampling": "temp0.7 top_p0.95", "time": time.strftime("%FT%TZ", time.gmtime())},
              open(OUT / f"manifest_{int(time.time())}.json", "w"), indent=1)
    if a.generate:
        from vllm import LLM
        llm = LLM(model=MODEL, revision=REV, dtype="bfloat16", gpu_memory_utilization=0.9,
                  max_model_len=20480, enforce_eager=False)
        for r in (REGIMES if a.generate == "all" else [a.generate]):
            generate(llm, r, a.smoke)
    if a.label:
        for r in (LABEL_SETS if a.label == "all" else [a.label]):
            label(r, a.smoke, a.model)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
