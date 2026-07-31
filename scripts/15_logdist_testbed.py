"""Stage 0 of logical-distance construct: assemble labeled testbed + generate paraphrase pairs.

  .venv/bin/python scripts/15_logdist_testbed.py --stage assemble
  .venv/bin/python scripts/15_logdist_testbed.py --stage paraphrase --smoke
  .venv/bin/python scripts/15_logdist_testbed.py --stage paraphrase
"""
import argparse, collections, itertools, json, math, os, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal


def boxed_match(para_text, orig_extracted):
    ext = extract_boxed(para_text or "")
    if ext is None or orig_extracted is None:
        return ext == orig_extracted
    if ext == orig_extracted:
        return True
    # tolerate formatting-only drift (thousands separators, \text spacing)
    return bool(run_with_timeout_signal(grade_answer, (ext, orig_extracted), timeout_seconds=5)) or \
        ext.replace(",", "").replace(" ", "") == orig_extracted.replace(",", "").replace(" ", "")

RUNS = Path("runs")
NEUTRAL = RUNS / "infdiv-f2main-BeyondAIME-neutral-k8-s0/raw.jsonl"
STYLE_ARMS = {a: RUNS / f"infdiv-f2main-BeyondAIME-{a}-k8-s0/raw.jsonl" for a in ("static", "adaptive")}
CGE = RUNS / "cge-BeyondAIME/raw.jsonl"

TEMPLATE_BASIN = {28, 39, 62, 70}
SCATTERED = {7, 26, 49, 58, 74, 81, 91}

PARA_SYS = """You rewrite mathematical solutions. Rewrite the given solution in a different style while preserving its logic EXACTLY: every mathematical step, case split, computed quantity, and the final \\boxed{} answer must be kept, with no steps added, dropped, merged, or reordered unless the style instruction says otherwise. Change only wording, verbosity, formatting, and notation presentation. Output ONLY the rewritten solution."""

STYLES = {
    "A_concise": "Style: maximally terse and formal. Strip all prose to the minimum; keep every equation and case.",
    "B_pedagogical": "Style: verbose, friendly teacher explaining to a student. Expand the prose around each step (motivate it, restate what was found) without adding any new mathematical content.",
    "C_casual": "Style: casual first-person student voice with hedges ('okay so', 'I think', 'let me check'). Same math, informal register.",
    "D_restructured": "Style: restructure the presentation — state the final answer and key claims first, then justify; rename all bound variables (e.g. n->m, a_i->b_i) consistently; convert display math to inline prose where natural. Logic and all steps must remain identical.",
}


def jload(p):
    return [json.loads(l) for l in open(p)]


def answers_equal(a, b):
    if a == b or a.replace(",", "").replace(" ", "") == b.replace(",", "").replace(" ", ""):
        return True
    if run_with_timeout_signal(grade_answer, (a, b), timeout_seconds=5):
        return True
    return bool(run_with_timeout_signal(grade_answer, (b, a), timeout_seconds=5))


def canon_classes(answers):
    # equivalence-class id per answer under normalized comparison (None stays None)
    reps, cls = [], []
    for a in answers:
        if a is None:
            cls.append(None)
            continue
        for k, r in enumerate(reps):
            if answers_equal(a, r):
                cls.append(k)
                break
        else:
            reps.append(a)
            cls.append(len(reps) - 1)
    return cls


def entropy(counts):
    n = sum(counts)
    return -sum(c / n * math.log(c / n) for c in counts if c)


def assemble(out):
    neutral = jload(NEUTRAL)
    traces = [{**r, "source": "neutral-k8"} for r in neutral]
    for arm, p in STYLE_ARMS.items():
        if p.exists():
            traces += [{**r, "source": f"{arm}-k8"} for r in jload(p)]
    for r in jload(CGE):
        traces.append({"qidx": r["qidx"], "arm": "cge-esc", "samp": r["role"], "sp_id": r["role"],
                       "sys": "", "text": r["text"], "gold": r["gold"], "prompt_tok": r["prompt_tok"],
                       "compl_tok": r["compl_tok"], "extracted": r["extracted"], "correct": r["correct"],
                       "source": "cge-esc"})
    with open(out / "traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    byq = collections.defaultdict(list)
    for r in neutral:
        byq[r["qidx"]].append(r)
    labels, neg_pairs, same_pairs = {}, [], []
    for q, rs in sorted(byq.items()):
        rs.sort(key=lambda r: r["samp"])
        answers = [r["extracted"] for r in rs]
        cls = canon_classes(answers)
        counts = collections.Counter(c for c in cls if c is not None)
        modal = counts.most_common(1)[0][0] if counts else None
        flip_class = ("template_basin" if q in TEMPLATE_BASIN else
                      "scattered" if q in SCATTERED else None)
        labels[q] = {"gold": rs[0]["gold"], "answers": answers, "answer_class": cls,
                     "coverage8": int(any(r["correct"] for r in rs)),
                     "maj_correct": int(modal is not None
                                        and any(r["correct"] and c == modal for r, c in zip(rs, cls))),
                     "n_unique": len(counts), "answer_entropy": entropy(list(counts.values())),
                     "flip_class": flip_class}
        for i, j in itertools.combinations(range(len(rs)), 2):
            pair = {"qidx": q, "samp_i": rs[i]["samp"], "samp_j": rs[j]["samp"]}
            if cls[i] is not None and cls[j] is not None and cls[i] != cls[j]:
                neg_pairs.append(pair)
            elif cls[i] is not None and cls[i] == cls[j]:
                same_pairs.append(pair)
    json.dump(labels, open(out / "problem_labels.json", "w"), indent=1)
    with open(out / "pairs_distinct_answer.jsonl", "w") as f:
        for p in neg_pairs:
            f.write(json.dumps(p) + "\n")
    with open(out / "pairs_same_answer.jsonl", "w") as f:
        for p in same_pairs:
            f.write(json.dumps(p) + "\n")
    cov = sum(l["coverage8"] for l in labels.values())
    print(f"traces={len(traces)}  problems={len(labels)}  coverage@8={cov}/100  "
          f"neg_pairs={len(neg_pairs)}  same_answer_pairs={len(same_pairs)}")
    print("flip-class counts:", collections.Counter(l["flip_class"] for l in labels.values()))


def pick_para_sources(neutral, n_problems, seed):
    # prefer a correct trace per problem (paraphrasing correct logic); else the modal-answer trace
    rng = random.Random(seed)
    byq = collections.defaultdict(list)
    for r in neutral:
        byq[r["qidx"]].append(r)
    qs = sorted(byq)
    rng.shuffle(qs)
    picked = []
    for q in qs[:n_problems]:
        rs = byq[q]
        correct = [r for r in rs if r["correct"]]
        pool = correct or [r for r in rs if r["extracted"] is not None] or rs
        pool.sort(key=lambda r: abs(r["compl_tok"] - 1900))  # mid-length, avoid truncated/degenerate
        picked.append(pool[0])
    return picked


def paraphrase(out, args):
    from openai import OpenAI
    client = OpenAI()
    neutral = jload(NEUTRAL)
    srcs = pick_para_sources(neutral, 2 if args.smoke else args.n_problems, args.seed)
    styles = dict(list(STYLES.items())[:2]) if args.smoke else STYLES
    jobs = [(r, sk, sv) for r in srcs for sk, sv in styles.items()]
    est_out_tok = sum(r["compl_tok"] for r, _, _ in jobs)
    est = (sum(r["compl_tok"] + 250 for r, _, _ in jobs) * 0.4 + est_out_tok * 1.6) / 1e6
    print(f"jobs={len(jobs)}  est cost ≈ ${est:.2f} ({args.model})")

    def one(job):
        r, sk, sv = job
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": PARA_SYS},
                              {"role": "user", "content": f"{sv}\n\nSOLUTION TO REWRITE:\n{r['text']}"}],
                    max_completion_tokens=8000, temperature=0.7)
                u = resp.usage
                text = resp.choices[0].message.content or ""
                return {"qidx": r["qidx"], "samp": r["samp"], "style": sk, "text": text,
                        "orig_extracted": r["extracted"], "para_extracted": extract_boxed(text),
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    for r in rows:  # signal-based grading -> main thread only
        r["boxed_ok"] = int(boxed_match(r["text"], r["orig_extracted"]))
    fn = out / ("paraphrases_smoke.jsonl" if args.smoke else "paraphrases.jsonl")
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    cost = (ptok * 0.4 + ctok * 1.6) / 1e6
    ok = sum(r["boxed_ok"] for r in rows)
    print(f"wrote {fn}: {len(rows)} rewrites, boxed_ok={ok}/{len(rows)}, "
          f"tok in/out={ptok}/{ctok}, actual cost ≈ ${cost:.2f}")
    by_style = collections.Counter((r["style"], r["boxed_ok"]) for r in rows)
    print("per-style (style, boxed_ok):", dict(by_style))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["assemble", "paraphrase"], required=True)
    ap.add_argument("--out", default="runs/logdist-testbed")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(args), "time": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
              open(out / f"manifest_{args.stage}{'_smoke' if args.smoke else ''}.json", "w"), indent=1)
    if args.stage == "assemble":
        assemble(out)
    else:
        paraphrase(out, args)


if __name__ == "__main__":
    main()
