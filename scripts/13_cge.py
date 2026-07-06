#!/usr/bin/env python3
"""F2b: Consensus-Gated Escalation (CGE) — orchestration designed against the F2 failure modes.

Round 1 = first 4 neutral low-effort samples per problem, REUSED from the existing F2 neutral run
(paired round-1 across methods). Gate on the answer multiset only: unanimous 4/4 -> accept modal,
stop (never re-open a winner, kills M2). Else escalate with 4 blind medium-effort calls:
2x neutral, 1x re-interpretation, 1x verifier that sees the candidate ANSWER multiset only (no
derivations anywhere in the system, kills M1; no strategy directives, kills M3).
Selection (pre-registered): weighted vote — round-1 weight 1, escalated weight 2; ties broken by
the verifier's answer. Primary metric: final-answer accuracy vs neutral maj@8 (paired bootstrap).

Modes: --simulate (Phase A, no API), --smoke (Phase B, 3 split problems), full (Phase C).
"""
import argparse, json, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
from deeppersona.personas import ANSWER_INSTRUCTION

ANS = ANSWER_INSTRUCTION["math"]
NEUTRAL = "Solve the math problem. Show your key steps." + ANS
REINTERPRET = ("First restate the problem formally from scratch in your own words, explicitly "
               "flagging any subtle conditions or possible misreadings. Then solve it." + ANS)
VERIFY_TMPL = ("Several independent attempts at this problem produced these candidate final "
               "answers (answer: count): <<cands>>. Solve the problem independently from scratch, "
               "then check your result against the candidates and reconcile any disagreement."
               + ANS)

NEUTRAL_RUN = "runs/infdiv-f2main-BeyondAIME-neutral-k8-s0/raw.jsonl"


def chat(client, model, sys_prompt, user, effort, max_tok=48000, retries=4):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                reasoning_effort=effort, max_completion_tokens=max_tok)
            return (r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens)
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 ** a * 3)


def boxed(text):
    try:
        return extract_boxed(text or "")
    except ValueError:
        return None


def grade(ext, gold):  # main thread only (signal-based sympy timeout)
    if ext is None:
        return 0
    return int(bool(run_with_timeout_signal(grade_answer, (ext, str(gold)), timeout_seconds=5)))


def vote(attempts):
    """attempts: list of {extracted, weight, role}. Returns selected answer."""
    w = Counter()
    for a in attempts:
        if a["extracted"]:
            w[a["extracted"]] += a["weight"]
    if not w:
        return None
    top = w.most_common()
    best = [ans for ans, c in top if c == top[0][1]]
    if len(best) > 1:  # tie-break: verifier's answer if among tied, else first
        ver = [a["extracted"] for a in attempts if a["role"] == "verify" and a["extracted"] in best]
        return ver[0] if ver else best[0]
    return best[0]


def escalate_problem(client, args, item):
    calls = [("med0", NEUTRAL, item["problem"]), ("med1", NEUTRAL, item["problem"]),
             ("reinterp", REINTERPRET, item["problem"]),
             ("verify", NEUTRAL,  # .replace not .format: ANS contains \boxed{} braces
              VERIFY_TMPL.replace("<<cands>>", str(item["cands"])) + "\n\nPROBLEM:\n" + item["problem"])]
    out = []
    for role, sp, user in calls:
        text, ptok, ctok = chat(client, args.model, sp, user, "medium")
        out.append({"qidx": item["qidx"], "role": role, "text": text, "gold": item["gold"],
                    "extracted": boxed(text), "weight": 2,
                    "prompt_tok": ptok, "compl_tok": ctok})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="gpt-5.5-2026-04-23")
    ap.add_argument("--tag", default="cge")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(NEUTRAL_RUN)]
    by_q = defaultdict(dict)
    for r in rows:
        by_q[r["qidx"]][r["samp"]] = r
    qs = sorted(by_q)

    # gate + baselines from existing raw
    probs = {}
    for q in qs:
        r4 = [by_q[q][s] for s in range(4)]
        cnt = Counter(x["extracted"] for x in r4 if x["extracted"])
        top, n = cnt.most_common(1)[0] if cnt else (None, 0)
        a8 = [by_q[q][s]["extracted"] for s in range(8) if by_q[q][s]["extracted"]]
        top8 = Counter(a8).most_common(1)[0][0] if a8 else None
        maj8_ok = int(any(by_q[q][s]["correct"] and by_q[q][s]["extracted"] == top8 for s in range(8)))
        probs[q] = {"qidx": q,
                    "gold": by_q[q][0]["gold"], "bin": "unan" if n >= 4 else ("weak" if n == 3 else "split"),
                    "modal4": top, "cands": dict(cnt), "maj8_ok": maj8_ok,
                    "r1": [{"extracted": x["extracted"], "correct": x["correct"], "weight": 1,
                            "role": "r1", "compl_tok": x["compl_tok"]} for x in r4]}
    # need actual problem text for escalation: pull from dataset (raw.jsonl doesn't store it)
    from deeppersona.data_espl import load_raw
    data = load_raw("BeyondAIME")
    for q in qs:
        probs[q]["problem"] = data[q]["problem"]

    bins = Counter(p["bin"] for p in probs.values())
    maj8 = sum(p["maj8_ok"] for p in probs.values()) / len(qs)
    print(f"gate bins: {dict(bins)} | neutral maj@8 = {maj8:.3f}")

    esc_q = [probs[q] for q in qs if probs[q]["bin"] != "unan"]
    if args.smoke:
        esc_q = [p for p in esc_q if p["bin"] == "split"][:3]

    esc_rows = []
    if not args.simulate:
        from openai import OpenAI
        client = OpenAI()
        with ThreadPoolExecutor(max_workers=16) as pool:
            esc_rows = [r for lst in pool.map(lambda p: escalate_problem(client, args, p), esc_q) for r in lst]
        for r in esc_rows:  # grade in main thread
            r["correct"] = grade(r["extracted"], r["gold"])

    esc_by_q = defaultdict(list)
    for r in esc_rows:
        esc_by_q[r["qidx"]].append(r)

    # selection + metrics
    out = Path(f"runs/{args.tag}-BeyondAIME" + ("-smoke" if args.smoke else "") + ("-sim" if args.simulate else ""))
    out.mkdir(parents=True, exist_ok=True)
    per_q, esc_tok = [], sum(r["compl_tok"] for r in esc_rows)
    for q in qs:
        p = probs[q]
        if p["bin"] == "unan":
            sel, attempts = p["modal4"], p["r1"]
        elif q in esc_by_q:
            attempts = p["r1"] + esc_by_q[q]
            sel = vote(attempts)
        else:  # simulate, or non-escalated problem in smoke: round-1-only vote
            sel, attempts = vote(p["r1"]), p["r1"]
        sel_ok = grade(sel, p["gold"])
        cov = int(any(a.get("correct") for a in attempts))
        tok = sum(a["compl_tok"] for a in p["r1"]) + sum(r["compl_tok"] for r in esc_by_q[q])
        per_q.append({"qidx": q, "bin": p["bin"], "sel": sel, "sel_ok": sel_ok, "cov": cov,
                      "maj8_ok": p["maj8_ok"], "tok": tok})

    with (out / "raw.jsonl").open("w") as f:
        for r in esc_rows:
            f.write(json.dumps(r) + "\n")
    (out / "per_q.json").write_text(json.dumps(per_q, indent=1))
    (out / "manifest.json").write_text(json.dumps(vars(args), indent=1))

    acc = sum(r["sel_ok"] for r in per_q) / len(per_q)
    cov = sum(r["cov"] for r in per_q) / len(per_q)
    diffs = [r["sel_ok"] - r["maj8_ok"] for r in per_q]
    rng = random.Random(0)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10000))
    print(f"CGE final-acc={acc:.3f} vs maj@8={maj8:.3f}  d={sum(diffs)/len(diffs):+.3f} "
          f"[{boots[250]:+.3f},{boots[9750]:+.3f}]  cov={cov:.3f}  esc_tok={esc_tok/1e6:.2f}M "
          f"mean_tok/q={sum(r['tok'] for r in per_q)/len(per_q)/1e3:.1f}k")
    for b in ("unan", "weak", "split"):
        rs = [r for r in per_q if r["bin"] == b]
        print(f"  {b}: n={len(rs)} sel_acc={sum(r['sel_ok'] for r in rs)/len(rs):.3f} "
              f"maj8={sum(r['maj8_ok'] for r in rs)/len(rs):.3f}")


if __name__ == "__main__":
    main()
