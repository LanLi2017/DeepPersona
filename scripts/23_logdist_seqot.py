#!/usr/bin/env python3
"""V6b: alternative pairwise kernels over the SAME precomputed banks -- DTW (order-aware),
Wasserstein/EMD (mass-conserving point cloud), Gromov-Wasserstein (internal geometry only) --
each run through the V1/V2 AUC battery, the V6 gameability probe, and the padding attack.
$0, CPU-only.

Predictions (stated before running, 2026-08-05):
- DTW: better AUC than Chamfer iff method identity lives partly in step ORDER; but MORE
  hackable vs D_restructured, which deliberately reorders presentation.
- OT: ~Chamfer AUC or slightly better (mass conservation forbids one-chunk-matches-many);
  duplication-invariant after normalization, but mass shifts under appended-new-content.
- GW: weak AUC -- coordinate invariance solves a problem we don't have (shared encoder),
  and it discards absolute semantics.
"""
import argparse, collections, importlib.util, json
from itertools import combinations
from pathlib import Path

import numpy as np
import ot as pot
from sklearn.metrics import roc_auc_score

TB = Path("runs/logdist-testbed")
STYLES = ("A_concise", "B_pedagogical", "C_casual", "D_restructured")
RNG = np.random.default_rng(20260805)
NBOOT = 2000


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s2 = _load("scripts/17_logdist_stage2.py", "s2")


def bank(npz, pad=None):
    z = np.load(TB / npz)
    E, off = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    if pad:  # merge padding-attack bank (P* ids) onto the original, as in scripts/22
        zp = np.load(TB / pad)
        n0 = len(E)
        E = np.concatenate([E, zp["E"]])
        off.update({k: (a + n0, b + n0) for k, (a, b) in json.loads(str(zp["offsets"])).items()})
    return E, off


def d_chamfer(X, Y):
    if not len(X) or not len(Y):
        return 1.0
    S = X @ Y.T
    return float(1 - 0.5 * (S.max(1).mean() + S.max(0).mean()))


def d_dtw(X, Y):
    if not len(X) or not len(Y):
        return 1.0
    C = 1 - X @ Y.T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def d_ot(X, Y):
    if not len(X) or not len(Y):
        return 1.0
    C = np.ascontiguousarray(1 - X @ Y.T, dtype=np.float64)
    return float(pot.emd2(np.ones(len(X)) / len(X), np.ones(len(Y)) / len(Y), C))


def d_gw(X, Y):
    if len(X) < 2 or len(Y) < 2:
        return 1.0
    C1 = np.ascontiguousarray(1 - X @ X.T, dtype=np.float64)
    C2 = np.ascontiguousarray(1 - Y @ Y.T, dtype=np.float64)
    return float(pot.gromov.gromov_wasserstein2(
        C1, C2, np.ones(len(X)) / len(X), np.ones(len(Y)) / len(Y),
        loss_fun="square_loss", symmetric=True))


FUNCS = {"chamfer": d_chamfer, "dtw": d_dtw, "ot": d_ot, "gw": d_gw}


def memo(f):
    c = {}

    def g(a, b):
        k = (a, b) if a <= b else (b, a)
        if k not in c:
            c[k] = f(a, b)
        return c[k]
    return g


def build_metrics(with_pad=False):
    banks = {
        "claim": bank("claim_embs.npz"),
        "l4head": bank("chunk_l4_L18_head.npz",
                       "chunk_pad_l4_L18_head.npz" if with_pad else None),
        "l4raw": bank("chunk_l4_L18.npz",
                      "chunk_pad_l4_L18.npz" if with_pad else None),
        "mpnet": bank("chunk_mpnet.npz"),
    }
    m = {}
    for bn, (E, off) in banks.items():
        for fn, f in FUNCS.items():
            m[f"{bn}:{fn}"] = memo(
                lambda a, b, E=E, off=off, f=f: f(E[slice(*off[a])], E[slice(*off[b])]))
    F15 = np.load(TB / "emb_l15_l4_L18.npz")["l15_l4_L18"]
    ids, _ = s2.load_texts()
    idx = {s: i for i, s in enumerate(ids)}
    m["l15_trace"] = memo(lambda a, b: float(1 - F15[idx[a]] @ F15[idx[b]]))
    return m


def add_ensembles(m, npair):
    # adopted 3-way ensemble with the L4 slot swapped for each new kernel; components scaled
    # by NEGATIVE-pair sd only (no centering: V6 ratios need a real zero at d(a,a)=0)
    variants = {f"ENS[{fn}]": ["claim:chamfer", "l15_trace", f"l4head:{fn}"] for fn in FUNCS}
    variants["ENS[cl-dtw]"] = ["claim:dtw", "l15_trace", "l4head:chamfer"]
    variants["ENS[cl-dtw+]"] = ["claim:dtw", "claim:chamfer", "l15_trace", "l4head:chamfer"]
    for name, comps in variants.items():
        zs = {k: float(np.std([m[k](a, b) for _, a, b in npair])) for k in comps}
        m[name] = memo(
            lambda a, b, zs=zs: float(np.mean([m[k](a, b) / z for k, z in zs.items()])))


def vendi(D, tau):
    K = np.exp(-D / tau)
    lam = np.clip(np.linalg.eigvalsh(K / len(K)), 1e-12, None)
    lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    nboot = 100 if a.smoke else NBOOT

    paras, train_q, test_q = s2.split_para_problems()
    base = {p["qidx"]: p["samp"] for p in paras}
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    npair = [(p["qidx"], f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    ppair = [(p["qidx"], f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}", p["style"])
             for p in paras]
    if a.smoke:
        npair, ppair = npair[:25], ppair[:25]

    m = build_metrics(with_pad=True)
    add_ensembles(m, npair)
    res = {}

    # ---- V1/V2 AUC: positives (label 0) = paraphrase pairs, negatives (label 1) = distinct-answer
    print(f"{'metric':16s} AUC_full  AUC_ho   AUC_D_ho")
    for mn, f in m.items():
        dpos = [(q, f(x, y), st) for q, x, y, st in ppair]
        dneg = [(q, f(x, y)) for q, x, y in npair]

        def auc(pos, neg):
            if not pos or not neg:
                return float("nan")
            return roc_auc_score([0] * len(pos) + [1] * len(neg), pos + neg)
        full = auc([d for _, d, _ in dpos], [d for _, d in dneg])
        ho = auc([d for q, d, _ in dpos if q in test_q], [d for q, d in dneg if q in test_q])
        dho = auc([d for q, d, st in dpos if q in test_q and st == "D_restructured"],
                  [d for q, d in dneg if q in test_q])
        res[mn] = {"auc_full": full, "auc_ho": ho, "auc_D_ho": dho}
        print(f"{mn:16s}  {full:.3f}    {ho:.3f}    {dho:.3f}")

    # ---- V6 set-level: k=4 style pack (4 paraphrases of ONE solution) vs method pack
    traces = [json.loads(l) for l in open(TB / "traces.jsonl")]
    ans = {(t["qidx"], t["samp"]): t["extracted"] for t in traces if t["source"] == "neutral-k8"}
    matched = sorted({q for q, *_ in npair} & set(base))
    packs = {}
    for q in matched:
        by_a = collections.defaultdict(list)
        for s in range(8):
            by_a[ans.get((q, s))].append(s)
        reps = [v[0] for v in by_a.values()]
        if len(reps) < 4:
            continue
        packs[q] = ([f"p{q}_{st}" for st in STYLES], [f"n{q}_{s}" for s in sorted(reps)[:4]])
    ho_i = [i for i, q in enumerate(sorted(packs)) if q in test_q]
    print(f"\nset-level V6: {len(packs)} packs ({len(ho_i)} held-out)")
    print(f"{'metric':16s} V_style  V_meth   R_set [95% CI]      hack%   R_ho")
    for mn, f in m.items():
        tau = np.mean([f(x, y) for _, x, y in npair])
        tau = abs(tau) if abs(tau) > 1e-9 else 1.0
        vs, vm = [], []
        for q in sorted(packs):
            for pack, acc in zip(packs[q], (vs, vm)):
                D = np.zeros((4, 4))
                for i, j in combinations(range(4), 2):
                    D[i, j] = D[j, i] = f(pack[i], pack[j])
                acc.append(vendi(D, tau))
        vs, vm = np.array(vs), np.array(vm)
        r = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9)
        bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
              for i in RNG.integers(0, len(vs), (nboot, len(vs)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rh = (vm[ho_i].mean() - 1) / max(vs[ho_i].mean() - 1, 1e-9) if ho_i else float("nan")
        res[mn].update({"V_style": vs.mean(), "V_meth": vm.mean(), "R_set": r,
                        "R_ci": [lo, hi], "R_ho": rh})
        print(f"{mn:16s}  {vs.mean():.3f}   {vm.mean():.3f}   {r:5.2f} [{lo:5.2f},{hi:5.2f}]"
              f"   {100 / max(r, 1e-9):4.0f}%   {rh:.2f}")

    # ---- padding attack (l4 banks only; P* ids exist only there)
    pad_pairs = [(q, f"n{q}_{base[q]}", f"P{q}_{t}") for q in sorted(base)
                 for t in ("self", "A", "B", "C")]
    style_pairs = [(q, x, y) for q, x, y, _ in ppair]
    meth_pairs = [(q, x, y) for q, x, y in npair if q in set(matched)]
    if a.smoke:
        pad_pairs = pad_pairs[:25]
    print(f"\npadding attack (base ++ dup/paraphrase): mean d per arm, R_pad = meth/pad")
    print(f"{'metric':16s} d_pad    d_style  d_meth   R_pad")
    for mn in [k for k in m if k.startswith(("l4head:", "l4raw:")) or k.startswith("ENS[")]:
        f = m[mn]
        try:
            dp = np.mean([f(x, y) for _, x, y in pad_pairs])
        except KeyError:  # ENS components (claims) have no P* entries
            continue
        ds = np.mean([f(x, y) for _, x, y in style_pairs])
        dm = np.mean([f(x, y) for _, x, y in meth_pairs])
        res[mn]["pad"] = {"d_pad": dp, "d_style": ds, "d_meth": dm, "R_pad": dm / max(dp, 1e-9)}
        print(f"{mn:16s}  {dp:.4f}   {ds:.4f}   {dm:.4f}   {dm / max(dp, 1e-9):5.2f}")

    if not a.smoke:
        (TB / "v6b_seqot.json").write_text(json.dumps(res, indent=1))
        import subprocess, time
        (TB / "manifest_v6b.json").write_text(json.dumps({
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip(),
            "seed": 20260805, "nboot": NBOOT, "time": time.strftime("%F %T"),
            "kernels": list(FUNCS), "banks": ["claim", "l4head", "l4raw", "mpnet"]}, indent=1))


if __name__ == "__main__":
    main()
