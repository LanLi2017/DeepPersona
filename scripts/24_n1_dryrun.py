#!/usr/bin/env python3
"""N1 dry run ($0): does logical diversity (ENS[cl-dtw] Vendi) of the existing k=8 neutral
gpt-5.5 packs predict correctness on the 100 BeyondAIME problems?

Circularity caveat, stated upfront: the metric was tuned on distinct-ANSWER pairs, so raw
correlation with answer diversity (n_unique / answer entropy) is baked in. The informative
numbers are the ones CONDITIONAL on answer diversity:
  (a) across problems: vendi -> pass@8 within n_unique strata;
  (b) within problems: vendi of a 4-subset -> cov@4 within (problem x subset-answer-entropy)
      strata -- the V5 deflationary design, now with the full upgraded ensemble.
"""
import collections, importlib.util, json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TB = Path("runs/logdist-testbed")
RNG = np.random.default_rng(20260805)


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s6 = _load("scripts/23_logdist_seqot.py", "s6")


def main():
    m = s6.build_metrics(with_pad=False)
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    npair = [(p["qidx"], f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    s6.add_ensembles(m, npair)
    f = m["ENS[cl-dtw]"]
    tau = np.mean([f(x, y) for _, x, y in npair])

    labels = {int(k): v for k, v in json.load(open(TB / "problem_labels.json")).items()}
    neutral = [r for r in (json.loads(l) for l in open(TB / "traces.jsonl"))
               if r["source"] == "neutral-k8"]
    corr = collections.defaultdict(dict)
    for r in neutral:
        corr[r["qidx"]][r["samp"]] = r["correct"]

    # ---- per-problem 8x8 distance matrix, pack Vendi -----------------------------
    rows, D8 = [], {}
    for q in sorted(corr):
        tids = [f"n{q}_{s}" for s in range(8)]
        D = np.zeros((8, 8))
        for i, j in combinations(range(8), 2):
            D[i, j] = D[j, i] = f(tids[i], tids[j])
        D8[q] = D
        c = [corr[q][s] for s in range(8)]
        rows.append({"q": q, "vendi": s6.vendi(D, tau), "ncorr": sum(c), "pass8": int(any(c)),
                     "nuniq": labels[q]["n_unique"], "aent": labels[q]["answer_entropy"]})
    v = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"n={len(rows)} problems  vendi8: mean {v['vendi'].mean():.2f} sd {v['vendi'].std():.2f}"
          f"  pass@8 {v['pass8'].mean():.2f}  mean ncorr {v['ncorr'].mean():.2f}")
    print("\nacross-problem spearman (raw, confounded by difficulty + circularity):")
    for a, b in (("vendi", "pass8"), ("vendi", "ncorr"), ("vendi", "nuniq"),
                 ("nuniq", "pass8"), ("aent", "pass8")):
        r = spearmanr(v[a], v[b])
        print(f"  {a:6s} vs {b:6s}: rho {r.statistic:+.3f}  p {r.pvalue:.3f}")
    print("\nvendi -> pass@8 within n_unique strata (the circularity control):")
    cs = []
    for u in sorted(set(v["nuniq"])):
        sel = v["nuniq"] == u
        if sel.sum() < 5 or len(set(v["pass8"][sel])) < 2 or len(set(v["vendi"][sel].round(6))) < 2:
            continue
        r = spearmanr(v["vendi"][sel], v["pass8"][sel])
        cs.append((r.statistic, sel.sum()))
        print(f"  n_unique={u}: rho {r.statistic:+.3f}  (n={sel.sum()}, p {r.pvalue:.3f})")
    if cs:
        w = np.array([n for _, n in cs])
        print(f"  weighted mean rho: {np.average([c for c, _ in cs], weights=w):+.3f}")

    # ---- within-problem subset test: vendi(4-subset) -> cov@4, stratified --------
    sub_rows = []
    for q in sorted(corr):
        c = [corr[q][s] for s in range(8)]
        if not 0 < sum(c) < 8:
            continue
        cls = labels[q]["answer_class"]
        for sub in combinations(range(8), 4):
            D = D8[q][np.ix_(sub, sub)]
            cnt = collections.Counter(cls[i] for i in sub)
            ae = -sum(k / 4 * np.log(k / 4) for k in cnt.values())
            sub_rows.append({"q": q, "vendi": s6.vendi(D, tau), "aent": round(ae, 6),
                             "cov": int(any(c[i] for i in sub))})
    qs = sorted(set(r["q"] for r in sub_rows))
    print(f"\nwithin-problem 4-subset test ({len(qs)} problems with 0<ncorr<8, "
          f"{len(sub_rows)} subsets):")
    raw = []
    for q in qs:
        sub = [r for r in sub_rows if r["q"] == q]
        if len(set(r["cov"] for r in sub)) < 2:
            continue
        raw.append(spearmanr([r["vendi"] for r in sub], [r["cov"] for r in sub]).statistic)
    print(f"  raw within-problem spearman(vendi, cov@4): mean {np.mean(raw):+.3f} "
          f"(n={len(raw)} problems, {np.mean(np.array(raw) > 0):.0%} positive)")
    strat = []
    for q in qs:
        sub = [r for r in sub_rows if r["q"] == q]
        for a in sorted(set(r["aent"] for r in sub)):
            s2 = [r for r in sub if r["aent"] == a]
            if len(set(r["cov"] for r in s2)) < 2 or len(set(round(r["vendi"], 8) for r in s2)) < 2:
                continue
            rho = spearmanr([r["vendi"] for r in s2], [r["cov"] for r in s2]).statistic
            if not np.isnan(rho):
                strat.append(rho)
    strat = np.array(strat)
    boot = [RNG.choice(strat, len(strat)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  within (problem x answer-entropy) strata: mean {strat.mean():+.3f} "
          f"n={len(strat)} 95% CI [{lo:+.3f},{hi:+.3f}]"
          f"  -- beats circularity iff CI excludes 0")

    out = {"per_problem": rows,
           "within_strata": {"mean": float(strat.mean()), "ci": [float(lo), float(hi)],
                             "n": int(len(strat))}}
    (TB / "n1_dryrun.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
