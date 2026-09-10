#!/usr/bin/env python3
"""G4: can claim-Chamfer's domain-generality be combined with raw_val's exactness?

Diagnosis (2026-08-24): claim-Chamfer is hackable because soft embedding matching has no
TRUE ZERO (0% of style pairs at d=0 vs raw_val's 39%) and a compressed dynamic range
(style/method distance ratio 0.46 vs raw_val's 0.28). Vendi turns that nonzero style floor
into apparent diversity.

Variants tested (all $0, reusing extracted claims):
  claim_cham   baseline (soft Chamfer over claim embeddings)
  claim_dead   Chamfer with a deadband: d' = max(0, (d-delta)/(1-delta)) — restores a true zero
  claim_ent    Jaccard over CONTENT TOKENS of the canonicalized claims — raw_val's exact
               set-matching applied to claims instead of raw text (the synthesis)
  claim_bin    thresholded bipartite match: cosine>=theta counts as the same unit, then Jaccard
  hybrid       union of exact artifacts: numbers (raw_val) + claim content tokens

delta/theta are calibrated on NON-PACK problems only; R_set is evaluated on pack problems
(clean). AUC pairs span all problems (calibration leak noted in the write-up).

  .venv/bin/python scripts/35_hybrid_metric.py --domain csr --domain code
"""
import argparse, collections, importlib.util, json, re
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s34 = _load("scripts/34_domain_transfer.py", "s34")
s6 = _load("scripts/23_logdist_seqot.py", "s6")
OUT = Path("runs/domain-transfer")

STOP = s34.STOP | set("also would could there their these those which what when where "
                      "given using used uses value values result results step steps".split())


def content_tokens(claims):
    out = set()
    for c in claims:
        for w in re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", c.lower()):
            if w.isdigit() or (len(w) >= 4 and w not in STOP):
                out.add(w)
    return out


def numbers(text, minv=2):
    out = set()
    for m in re.findall(r"-?\d+(?:/\d+)?", text or ""):
        try:
            v = Fraction(m)
        except Exception:
            continue
        if abs(v) > minv:
            out.add(("num", v))
    return out


def d_jac(A, B):
    if not A or not B:
        return float("nan")
    return 1 - len(A & B) / len(A | B)


def run(domain):
    from sklearn.metrics import roc_auc_score
    ids, texts, pa = s34.ids_texts(domain)
    txt = dict(zip(ids, texts))
    if domain == "code":
        txt = {k: s34.code_block(v) for k, v in txt.items()}
    cl = {r["id"]: r["claims"] for r in
          (json.loads(l) for l in open(OUT / f"claims_{domain}.jsonl"))}
    z = np.load(OUT / f"claimemb_{domain}.npz")
    E = z["E"]
    off = {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    lab = {r["qidx"]: r["assignment"] for r in
           (json.loads(l) for l in open(OUT / f"labels_{domain}.jsonl"))}
    base = {r["qidx"]: r["samp"] for r in pa}
    styles = list(s34.CODE_STYLES if domain == "code" else s34.CSR_STYLES)
    verified = {(r["qidx"], r["style"]) for r in pa if domain == "csr" or r.get("verified")}

    CT = {t: content_tokens(cl.get(t, [])) for t in ids}
    NU = {t: numbers(txt.get(t, "")) for t in ids}

    def cham_raw(a, b):
        (a0, a1), (b0, b1) = off.get(a, (0, 0)), off.get(b, (0, 0))
        if a1 <= a0 or b1 <= b0:
            return float("nan")
        S = E[a0:a1] @ E[b0:b1].T
        return float(1 - 0.5 * (S.max(1).mean() + S.max(0).mean()))

    def bin_match(a, b, theta):
        (a0, a1), (b0, b1) = off.get(a, (0, 0)), off.get(b, (0, 0))
        if a1 <= a0 or b1 <= b0:
            return float("nan")
        S = E[a0:a1] @ E[b0:b1].T
        na, nb = a1 - a0, b1 - b0
        inter = 0.5 * ((S.max(1) >= theta).sum() + (S.max(0) >= theta).sum())
        return float(1 - inter / (na + nb - inter)) if (na + nb - inter) > 0 else 0.0

    # packs (>=3 judge methods, size 3) — the R_set eval set; calibrate on the rest
    packs = {}
    for q, a in lab.items():
        rep = {}
        for s in range(s34.K):
            rep.setdefault(a[s], s)
        if len(rep) >= 3:
            packs[q] = ["n%d_%d" % (q, s) for s in sorted(rep.values())[:3]]
    packq = [q for q in sorted(packs) if sum((q, st) in verified for st in styles) >= 3]
    calib_q = [q for q in lab if q not in packq]
    cal_pairs = [("n%d_%d" % (q, base[q]), "p%d_%s" % (q, st))
                 for q in calib_q for st in styles
                 if (q, st) in verified and "p%d_%s" % (q, st) in off]
    cal_d = [x for x in (cham_raw(a, b) for a, b in cal_pairs) if not np.isnan(x)]
    delta = float(np.percentile(cal_d, 75))
    cal_sims = []
    for a, b in cal_pairs[:200]:
        (a0, a1), (b0, b1) = off.get(a, (0, 0)), off.get(b, (0, 0))
        if a1 > a0 and b1 > b0:
            cal_sims += list((E[a0:a1] @ E[b0:b1].T).max(1))
    theta = float(np.percentile(cal_sims, 25)) if cal_sims else 0.9
    print(f"\n[{domain}] calibrated on {len(calib_q)} non-pack problems "
          f"({len(cal_d)} style pairs): deadband delta={delta:.3f}, match theta={theta:.3f}")

    m = {
        "claim_cham": cham_raw,
        "claim_dead": lambda a, b: (lambda d: float("nan") if np.isnan(d)
                                    else max(0.0, (d - delta) / (1 - delta)))(cham_raw(a, b)),
        "claim_bin": lambda a, b: bin_match(a, b, theta),
        "claim_ent": lambda a, b: d_jac(CT.get(a), CT.get(b)),
        "raw_val": lambda a, b: d_jac(NU.get(a), NU.get(b)),
        "hybrid": lambda a, b: d_jac(CT.get(a) | NU.get(a), CT.get(b) | NU.get(b)),
    }
    diff, same = [], []
    for q, a in lab.items():
        for i, j in combinations(range(s34.K), 2):
            (diff if a[i] != a[j] else same).append(("n%d_%d" % (q, i), "n%d_%d" % (q, j)))
    rng = np.random.default_rng(20260824)
    print(f"{'metric':11s} {'AUC_D':>6s} {'ALLst':>6s} {'d_sty':>6s} {'d_meth':>6s} {'0-frac':>6s}  "
          f"{'V_sty':>5s} {'V_met':>5s}  R_set [95% CI]  hack%")
    res = {}
    for mn, f in m.items():
        dd = [x for x in (f(a, b) for a, b in diff) if not np.isnan(x)]
        ds = [x for x in (f(a, b) for a, b in same) if not np.isnan(x)]
        aucd = roc_auc_score([0] * len(ds) + [1] * len(dd), ds + dd)
        allpos = []
        for st in styles:
            v = [f("n%d_%d" % (q, base[q]), "p%d_%s" % (q, st)) for q in sorted(lab)
                 if (q, st) in verified and "p%d_%s" % (q, st) in txt]
            allpos += [x for x in v if not np.isnan(x)]
        aucg = roc_auc_score([0] * len(allpos) + [1] * len(dd), allpos + dd)
        zfrac = float(np.mean(np.array(allpos) == 0))
        tau = abs(np.nanmean(dd)) or 1.0
        vs, vm = [], []
        for q in packq:
            sp = ["p%d_%s" % (q, st) for st in styles if (q, st) in verified][:3]
            for pack, acc in zip((sp, packs[q]), (vs, vm)):
                D = np.zeros((3, 3))
                for i, j in combinations(range(3), 2):
                    D[i, j] = D[j, i] = np.nan_to_num(f(pack[i], pack[j]), nan=tau)
                acc.append(s6.vendi(D, tau))
        vs, vm = np.array(vs), np.array(vm)
        R = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9)
        bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
              for i in rng.integers(0, len(vs), (2000, len(vs)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[mn] = {"auc_d": aucd, "auc_allst": aucg, "zero_frac": zfrac,
                   "d_style": float(np.mean(allpos)), "d_meth": float(np.mean(dd)),
                   "R": float(R), "ci": [float(lo), float(hi)],
                   "V_style": float(vs.mean()), "V_meth": float(vm.mean())}
        print(f"{mn:11s} {aucd:6.3f} {aucg:6.3f} {np.mean(allpos):6.3f} {np.mean(dd):6.3f} "
              f"{zfrac:6.2f}  {vs.mean():5.2f} {vm.mean():5.2f}  {R:5.2f} [{lo:4.2f},{min(hi,99):5.2f}]"
              f"  {100 / max(R, 1e-9):4.0f}%")
    (OUT / f"g4_hybrid_{domain}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", action="append", default=[], choices=["code", "csr"])
    a = ap.parse_args()
    for d in a.domain or ["csr", "code"]:
        run(d)


if __name__ == "__main__":
    main()
