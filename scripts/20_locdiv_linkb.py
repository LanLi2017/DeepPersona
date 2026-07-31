#!/usr/bin/env python3
"""E3 Link B: does output diversity predict pooled recall on SWE-bench localization?

V5 re-run in the no-discrete-answer regime. On final-answer math the answer multiset fully
mediated diversity->coverage (stratified Vendi effect ~0). Here the "answer" is a SET of files
with graded overlap, so the cheap discrete mediator does not exist by construction.

Design: within each (problem x arm) stratum, enumerate all C(8,4)=70 4-subsets; regress
union recall on subset diversity (Vendi score over the Jaccard kernel of prediction sets),
controlling for mean per-sample recall and mean prediction-set size. Cluster-bootstrap by
problem. Secondary spec adds |union| as a control -- deliberate over-control that asks whether
diversity buys anything beyond mechanically proposing more distinct files.

NOTE: runs/locdiv-* stored only the last 500 chars of each completion (the file list), so the
logical-distance kernel cannot be applied here -- this is an outcome-level analysis only.
"""
import json, re
from itertools import combinations
from pathlib import Path

import numpy as np

RUN = Path("runs/locdiv-swebv-n50-k8")
K_SUB = 4
NBOOT = 2000
RNG = np.random.default_rng(20260730)


def gold_files(patch):
    return sorted({m.group(1) for m in re.finditer(r"^diff --git a/\S+ b/(\S+)", patch, re.M)
                   if m.group(1).endswith(".py")})


def vendi(sets):
    n = len(sets)
    K = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = len(sets[i] | sets[j])
            K[i, j] = K[j, i] = (len(sets[i] & sets[j]) / u) if u else 1.0
    lam = np.clip(np.linalg.eigvalsh(K / n), 1e-12, None)
    return float(np.exp(-(lam * np.log(lam)).sum()))


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    rows = [json.loads(l) for l in (RUN / "raw.jsonl").open()]
    man = json.loads((RUN / "manifest.json").read_text())
    iids = set(man["iids"])

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    gold = {r["instance_id"]: set(gold_files(r["patch"])) for r in ds if r["instance_id"] in iids}
    assert len(gold) == len(iids)

    by = {}
    for r in rows:
        by.setdefault((r["iid"], r["arm"]), {})[r["samp"]] = r
    # stored per-sample recall must reproduce from recomputed gold, else the join is wrong
    for (iid, _), d in by.items():
        for r in d.values():
            assert abs(len(gold[iid] & set(r["preds"])) / len(gold[iid]) - r["recall"]) < 1e-9

    recs = []
    for (iid, arm), d in sorted(by.items()):
        g = gold[iid]
        preds = [set(d[s]["preds"]) for s in range(8)]
        for idx in combinations(range(8), K_SUB):
            sub = [preds[i] for i in idx]
            u = set().union(*sub)
            recs.append(dict(
                iid=iid, arm=arm, stratum=f"{iid}|{arm}",
                y=len(g & u) / len(g),
                div=vendi(sub),
                acc=float(np.mean([len(g & p) / len(g) for p in sub])),
                accsd=float(np.std([len(g & p) / len(g) for p in sub])),
                accmax=float(np.max([len(g & p) / len(g) for p in sub])),
                size=float(np.mean([len(p) for p in sub])),
                usize=float(len(u)),
            ))

    strata = sorted({r["stratum"] for r in recs})
    print(f"n_subsets={len(recs)}  strata={len(strata)}  (C(8,{K_SUB})={len(list(combinations(range(8), K_SUB)))} per stratum)")

    y = np.array([r["y"] for r in recs])
    prob = np.array([r["iid"] for r in recs])
    strat = np.array([r["stratum"] for r in recs])
    cols = {k: np.array([r[k] for r in recs]) for k in ("div", "acc", "size", "usize", "accsd", "accmax")}

    # descriptive: how much room is there to move?
    var_ok = sum(1 for s in strata if np.std(y[strat == s]) > 1e-9)
    print(f"strata with within-stratum variance in union recall: {var_ok}/{len(strata)}")
    print(f"union recall@{K_SUB}: mean={y.mean():.3f}  at ceiling (=1.0): {100*np.mean(y==1.0):.0f}%  at floor (=0.0): {100*np.mean(y==0.0):.0f}%")
    print(f"Vendi (effective distinct predictions, max {K_SUB}): mean={cols['div'].mean():.2f}  sd={cols['div'].std():.2f}")

    def demean(v, keys):
        out = v.astype(float).copy()
        for s in np.unique(keys):
            m = keys == s
            out[m] -= out[m].mean()
        return out

    yd = demean(y, strat)
    # standardize predictors on the pooled within-stratum scale -> coefficients are per-SD
    def design(names):
        Z = []
        for n in names:
            d = demean(cols[n], strat)
            sd = d.std()
            Z.append(d / sd if sd > 1e-12 else d)
        return np.column_stack(Z)

    for label, names in [("primary  ", ["div", "acc", "size"]),
                         ("+|union| ", ["div", "acc", "size", "usize"]),
                         ("div only ", ["div"]),
                         ("+sd(acc) ", ["div", "acc", "size", "accsd"]),
                         ("+max(acc)", ["div", "acc", "size", "accsd", "accmax"])]:
        X = design(names)
        b = ols(X, yd)
        boots = []
        uprob = np.unique(prob)
        for _ in range(NBOOT):
            pick = RNG.choice(uprob, len(uprob), replace=True)
            m = np.concatenate([np.flatnonzero(prob == p) for p in pick])
            boots.append(ols(X[m], yd[m])[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        terms = "  ".join(f"{n}={c:+.4f}" for n, c in zip(names, b))
        print(f"{label} d(union recall)/SD: div={b[0]:+.4f} [{lo:+.4f},{hi:+.4f}]   ({terms})")

    # per-arm split: is the relation the same where the intervention did/didn't apply?
    for arm in ("neutral", "lens"):
        m = np.array([r["arm"] == arm for r in recs])
        X = design(["div", "acc", "size"])[m]
        b = ols(X, yd[m])
        print(f"  arm={arm:8s} div={b[0]:+.4f}  acc={b[1]:+.4f}  (n={m.sum()})")

    # how many problem clusters actually carry information (bootstrap is clustered on problem)
    infp = sorted({r["iid"] for r in recs if np.std(y[strat == r["stratum"]]) > 1e-9})
    print(f"\ninformative problems (>=1 arm with non-constant union recall): {len(infp)}/50")

    # exact-|union| matching: compare subsets that propose the SAME number of distinct files
    m = np.zeros(len(recs), bool)
    cell = np.array([f"{r['stratum']}|{r['usize']:.0f}" for r in recs])
    for c in np.unique(cell):
        s = cell == c
        if s.sum() >= 4 and np.std(y[s]) > 1e-9 and np.std(cols["div"][s]) > 1e-9:
            m |= s
    if m.sum():
        yc = demean(y[m], cell[m])
        dv = demean(cols["div"][m], cell[m]); dv /= dv.std()
        ac = demean(cols["acc"][m], cell[m]); ac /= ac.std()
        b = ols(np.column_stack([dv, ac]), yc)
        print(f"exact-|union| matched: div={b[0]:+.4f}  acc={b[1]:+.4f}  (n={m.sum()}, cells={len(np.unique(cell[m]))})")

    # placebo: permute diversity within stratum -> coefficient must collapse to ~0
    X = design(["div", "acc", "size"])
    perm = X.copy()
    for s in np.unique(strat):
        i = np.flatnonzero(strat == s)
        perm[i, 0] = perm[RNG.permutation(i), 0]
    print(f"placebo (div permuted within stratum): div={ols(perm, yd)[0]:+.4f}")

    Path("runs/locdiv-swebv-n50-k8/linkb_subsets.json").write_text(json.dumps(recs))


if __name__ == "__main__":
    main()
