#!/usr/bin/env python3
"""E3 Link B, moderator analysis: for WHICH problems does output diversity buy pooled recall?

Follow-up to scripts/20. Only 17/50 problems had within-stratum variance in union recall; this
asks (a) why the other 33 are constant, and (b) whether the diversity effect is heterogeneous
across problem types -- i.e. is there a regime where pooling genuinely pays?

Moderators are problem-level (constant within stratum), so they are identified as interactions
with the within-stratum-demeaned diversity term under problem x arm fixed effects.
"""
import json, re
from pathlib import Path

import numpy as np

RUN = Path("runs/locdiv-swebv-n50-k8")
RNG = np.random.default_rng(20260730)


def gold_files(patch):
    return sorted({m.group(1) for m in re.finditer(r"^diff --git a/\S+ b/(\S+)", patch, re.M)
                   if m.group(1).endswith(".py")})


def demean(v, keys):
    out = np.asarray(v, float).copy()
    for s in np.unique(keys):
        m = keys == s
        out[m] -= out[m].mean()
    return out


def fit(recs, names, boot_by=None, nboot=2000):
    """Within-stratum FE regression; returns coef on names[0] plus cluster-bootstrap CI."""
    strat = np.array([r["stratum"] for r in recs])
    y = demean([r["y"] for r in recs], strat)
    Z = []
    for n in names:
        d = demean([r[n] for r in recs], strat) if n != "divXz" else np.array([r["divXz"] for r in recs])
        sd = d.std()
        Z.append(d / sd if sd > 1e-12 else d)
    X = np.column_stack(Z)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    if boot_by is None:
        return b, None
    p = np.array([r[boot_by] for r in recs])
    up = np.unique(p)
    bs = []
    for _ in range(nboot):
        idx = np.concatenate([np.flatnonzero(p == q) for q in RNG.choice(up, len(up), replace=True)])
        if np.linalg.matrix_rank(X[idx]) < X.shape[1]:
            continue
        bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][0])
    return b, np.percentile(bs, [2.5, 97.5])


def main():
    recs = json.loads((RUN / "linkb_subsets.json").read_text())
    rows = [json.loads(l) for l in (RUN / "raw.jsonl").open()]
    man = json.loads((RUN / "manifest.json").read_text())
    iids = set(man["iids"])

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    gold = {r["instance_id"]: set(gold_files(r["patch"])) for r in ds if r["instance_id"] in iids}

    by = {}
    for r in rows:
        by.setdefault((r["iid"], r["arm"]), {})[r["samp"]] = set(r["preds"])

    # ---- problem-level features -------------------------------------------------
    feat = {}
    for iid in sorted(iids):
        g = gold[iid]
        allp = [p for arm in ("neutral", "lens") for p in by[(iid, arm)].values()]
        found = set().union(*allp) & g
        persamp = [len(g & p) / len(g) for p in allp]
        J = [len(a & b) / len(a | b) if (a | b) else 1.0
             for i, a in enumerate(allp) for b in allp[i + 1:]]
        # p_ prefix: these are problem-level and must not collide with the subset-level columns
        feat[iid] = dict(
            p_ngold=len(g),
            p_acc=float(np.mean(persamp)),               # per-sample difficulty
            p_ceil=len(found) / len(g),                  # best achievable by pooling all 16
            p_blind=1 - len(found) / len(g),             # gold no sample ever finds
            p_headroom=len(found) / len(g) - float(np.mean(persamp)),  # what pooling could add
            p_selfsim=float(np.mean(J)),                 # how alike the model's guesses are
        )

    # ---- why are 33 problems uninformative? -------------------------------------
    strat_y = {}
    for r in recs:
        strat_y.setdefault(r["stratum"], []).append(r["y"])
    cls = {"ceiling (all subsets =1.0)": 0, "floor (all subsets =0.0)": 0,
           "constant, intermediate": 0, "variable": 0}
    for s, ys in strat_y.items():
        ys = np.array(ys)
        if ys.std() > 1e-9:
            cls["variable"] += 1
        elif ys[0] == 1.0:
            cls["ceiling (all subsets =1.0)"] += 1
        elif ys[0] == 0.0:
            cls["floor (all subsets =0.0)"] += 1
        else:
            cls["constant, intermediate"] += 1
    print("strata (problem x arm) by regime:")
    for k, v in cls.items():
        print(f"  {k:28s} {v:3d}/100")

    infp = sorted({r["iid"] for r in recs if np.std(strat_y[r["stratum"]]) > 1e-9})
    print(f"\ninformative problems: {len(infp)}/50")
    print("                         n_gold   per-samp acc   pool ceiling   headroom   self-sim")
    for grp, ids in (("informative ", infp), ("uninformative", [i for i in sorted(iids) if i not in infp])):
        f = [feat[i] for i in ids]
        print(f"  {grp}  n={len(ids):2d}   {np.mean([x['p_ngold'] for x in f]):.2f}      "
              f"{np.mean([x['p_acc'] for x in f]):.3f}         {np.mean([x['p_ceil'] for x in f]):.3f}"
              f"        {np.mean([x['p_headroom'] for x in f]):+.3f}      {np.mean([x['p_selfsim'] for x in f]):.3f}")

    # ---- is the diversity effect heterogeneous across problem types? ------------
    for r in recs:
        r.update(feat[r["iid"]])

    print("\ndiversity effect by problem stratification  (d union recall per SD of Vendi)")
    print("                                    n_prob   div [95% CI]            +sd(acc) spec")
    def show(label, sel):
        sub = [r for r in recs if sel(feat[r["iid"]])]
        nprob = len({r["iid"] for r in sub})
        if nprob < 2 or np.std([r["y"] for r in sub]) < 1e-9:
            print(f"  {label:32s} {nprob:3d}     (no variance)")
            return
        b, ci = fit(sub, ["div", "acc", "size"], boot_by="iid")
        b2, _ = fit(sub, ["div", "acc", "size", "accsd"])
        print(f"  {label:32s} {nprob:3d}     {b[0]:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]   {b2[0]:+.4f}")

    show("all problems", lambda f: True)
    show("n_gold == 2", lambda f: f["p_ngold"] == 2)
    show("n_gold >= 3", lambda f: f["p_ngold"] >= 3)
    show("per-sample acc < 0.5 (hard)", lambda f: f["p_acc"] < 0.5)
    show("per-sample acc 0.5-0.85", lambda f: 0.5 <= f["p_acc"] < 0.85)
    show("per-sample acc >= 0.85 (easy)", lambda f: f["p_acc"] >= 0.85)
    show("headroom > 0.15 (pooling can pay)", lambda f: f["p_headroom"] > 0.15)
    show("headroom <= 0.15", lambda f: f["p_headroom"] <= 0.15)
    show("has blind spot (blind > 0)", lambda f: f["p_blind"] > 0)
    show("no blind spot", lambda f: f["p_blind"] == 0)

    # ---- formal interaction test on the full sample ------------------------------
    print("\ninteraction tests (div x moderator, problem x arm FE, full 7000 rows)")
    strat = np.array([r["stratum"] for r in recs])
    dv = demean([r["div"] for r in recs], strat)
    dv = dv / dv.std()
    for z in ("p_headroom", "p_acc", "p_ngold", "p_blind", "p_selfsim"):
        zv = np.array([r[z] for r in recs], float)
        zc = (zv - zv.mean()) / zv.std()
        for r, v in zip(recs, dv * zc):
            r["divXz"] = v
        b, ci = fit(recs, ["divXz", "div", "acc", "size"], boot_by="iid")
        star = "  *" if (ci[0] > 0) or (ci[1] < 0) else ""
        print(f"  div x {z:9s} = {b[0]:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]{star}")


if __name__ == "__main__":
    main()
