#!/usr/bin/env python3
"""V6 gameability probe: is the cheapest way to move the metric a style edit or a real method switch?

V1/V2 asked whether the metric can *classify* paraphrase vs distinct-solution pairs (AUC 0.93-0.97).
V6 asks a different question: under RL, the policy maximizes the reward the cheapest way available.
So we measure Delta-metric per unit of surface change for (a) a free style rewrite, (b) a genuine
method switch, (c) a padding attack (inflate the trace without changing the logic -- the failure
mode that appeared spontaneously on SWE-bench localization, scripts/20).

  reward-worthiness R = Delta(method switch) / Delta(style attack)     -- want >> 1

Surface change is measured as token-set Jaccard distance, which doubles as the pure-surface
baseline metric -- so that baseline has R_edit == 1.0 by construction (a built-in calibration).

  .venv/bin/python scripts/22_logdist_v6.py --core
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/22_logdist_v6.py --pad-encode
  .venv/bin/python scripts/22_logdist_v6.py --pad

NOTE: styles A/B/C were seen in metric training; D_restructured was held out and also renames
variables, so it is the strongest free attack and is reported separately throughout.
"""
import argparse, collections, importlib.util, json, re, subprocess, time
from itertools import combinations
from pathlib import Path

import numpy as np

TB = Path("runs/logdist-testbed")
spec = importlib.util.spec_from_file_location("s2", "scripts/17_logdist_stage2.py")
s2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2)
spec4 = importlib.util.spec_from_file_location("s4", "scripts/19_logdist_l4chunk.py")
s4 = importlib.util.module_from_spec(spec4); spec4.loader.exec_module(s4)

STYLES = ("A_concise", "B_pedagogical", "C_casual", "D_restructured")
RNG = np.random.default_rng(20260731)
NBOOT = 2000


def tokset(t):
    return set(re.findall(r"[A-Za-z0-9\\]+", t.lower()))


def build_metrics(texts_by_id):
    """dict name -> pairwise distance fn on trace ids. All from precomputed embeddings ($0)."""
    ids, _ = s2.load_texts()
    idx = {s: i for i, s in enumerate(ids)}
    F15 = np.load(TB / "emb_l15_l4_L18.npz")["l15_l4_L18"]
    R4 = np.load(TB / "emb_l4.npz")["l4_L18"]
    R4 = R4 / np.linalg.norm(R4, axis=1, keepdims=True)
    ts = {k: tokset(v) for k, v in texts_by_id.items()}
    m = {
        "l2_claim": s2.chamfer_dists(),
        "l15_trace": lambda a, b: float(1 - F15[idx[a]] @ F15[idx[b]]),
        "l4_chunk": s4.chamfer_of("chunk_l4_L18_head.npz"),
        "raw_qwen": lambda a, b: float(1 - R4[idx[a]] @ R4[idx[b]]),
        "raw_chunk": s4.chamfer_of("chunk_l4_L18.npz"),
        "mpnet_chunk": s4.chamfer_of("chunk_mpnet.npz"),
        "tok_jaccard": lambda a, b: 1 - len(ts[a] & ts[b]) / max(len(ts[a] | ts[b]), 1),
    }
    # ensemble as adopted in scripts/19 (components scaled by their NEGATIVE-pair sd), but without
    # the mean-centering: V6 needs ratios, which require a real zero (identical traces -> d=0).
    # This is an affine transform of the adopted ensemble, so every AUC is unchanged.
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    npair = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    zs = {k: float(np.std([m[k](*p) for p in npair])) for k in ("l2_claim", "l15_trace", "l4_chunk")}
    m["ENSEMBLE"] = lambda a, b: float(np.mean([m[k](a, b) / zs[k] for k in zs]))
    return m, zs


def boot_ratio(num, den, prob, nboot=NBOOT):
    """CI on mean(num)/mean(den) where the two arms are separate pair lists; cluster by problem."""
    (nv, npq), (dv, dpq) = num, den
    up = np.unique(np.concatenate([npq, dpq]))
    out = []
    for _ in range(nboot):
        pick = RNG.choice(up, len(up), replace=True)
        ni = np.concatenate([np.flatnonzero(npq == p) for p in pick])
        di = np.concatenate([np.flatnonzero(dpq == p) for p in pick])
        if len(ni) == 0 or len(di) == 0 or np.mean(dv[di]) <= 0:
            continue
        out.append(np.mean(nv[ni]) / np.mean(dv[di]))
    return np.percentile(out, [2.5, 97.5])


def vendi(D, tau):
    K = np.exp(-D / tau)
    lam = np.clip(np.linalg.eigvalsh(K / len(K)), 1e-12, None)
    lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))


def core():
    texts = dict(zip(*[list(x) for x in s2.load_texts()]))
    paras = [json.loads(l) for l in open(TB / "paraphrases.jsonl")]
    _, _, test_q = s2.split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    sames = [json.loads(l) for l in open(TB / "pairs_same_answer.jsonl")]
    base = {p["qidx"]: p["samp"] for p in paras}
    metrics, _ = build_metrics(texts)

    pq, nq = set(base), {p["qidx"] for p in negs}
    matched = sorted(pq & nq)
    print(f"problems: paraphrased={len(pq)}  with distinct-answer pairs={len(nq)}  matched={len(matched)}")
    print(f"(matched = both arms observable on the same problem; primary analysis)\n")

    def arms(qs):
        a = {}
        for st in STYLES:
            a[f"style:{st[0]}"] = [(q, f"n{q}_{base[q]}", f"p{q}_{st}") for q in qs if q in base]
        a["style:ALL"] = [x for st in STYLES for x in a[f"style:{st[0]}"]]
        a["method:distinct-ans"] = [(p["qidx"], f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
                                    for p in negs if p["qidx"] in qs]
        a["same-ans (ambiguous)"] = [(p["qidx"], f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
                                     for p in sames if p["qidx"] in qs]
        return a

    for tag, qs in (("MATCHED problems", set(matched)),
                    ("HELD-OUT problems (metric never trained on)", set(matched) & test_q)):
        A = arms(qs)
        surf = {k: np.array([metrics["tok_jaccard"](a, b) for _, a, b in v]) for k, v in A.items()}
        print(f"=== {tag}  (n_prob={len(qs)}) ===")
        print("arm                     n     surface-change   " + "  ".join(f"{k:>11s}" for k in metrics))
        for k, v in A.items():
            if not v:
                continue
            row = "  ".join(f"{np.mean([metrics[mn](a, b) for _, a, b in v]):11.4f}" for mn in metrics)
            print(f"  {k:22s}{len(v):4d}   {surf[k].mean():.3f}            {row}")

        print("\n  reward-worthiness R = d(method) / d(style)      [>1 = method moves it more]")
        print("  metric        R(vs all styles)         R(vs D held-out)        R per unit surface change")
        for mn in metrics:
            out = []
            for stk in ("style:ALL", "style:D"):
                num = A["method:distinct-ans"]; den = A[stk]
                nv = np.array([metrics[mn](a, b) for _, a, b in num])
                dv = np.array([metrics[mn](a, b) for _, a, b in den])
                npq = np.array([q for q, _, _ in num]); dpq = np.array([q for q, _, _ in den])
                r = nv.mean() / dv.mean()
                lo, hi = boot_ratio((nv, npq), (dv, dpq), None)
                out.append((r, lo, hi))
            # edit-cost-matched: Delta per unit of token-set change
            ns = surf["method:distinct-ans"].mean(); ds = surf["style:ALL"].mean()
            nv = np.array([metrics[mn](a, b) for _, a, b in A["method:distinct-ans"]])
            dv = np.array([metrics[mn](a, b) for _, a, b in A["style:ALL"]])
            re_ = (nv.mean() / ns) / (dv.mean() / ds)
            print(f"  {mn:12s}  {out[0][0]:6.2f} [{out[0][1]:5.2f},{out[0][2]:5.2f}]   "
                  f"{out[1][0]:6.2f} [{out[1][1]:5.2f},{out[1][2]:5.2f}]   {re_:8.2f}")
        print()

    # ---- set-level: what a GRPO diversity reward would actually see -------------------
    print("=== set-level reward surface (k=4 packs, Vendi over exp(-d/tau)) ===")
    print("tau = mean distinct-answer-pair distance per metric; clone pack (4 copies) = 1.00 exactly")
    traces = [json.loads(l) for l in open(TB / "traces.jsonl")]
    ans = {(t["qidx"], t["samp"]): t["extracted"] for t in traces if t["source"] == "neutral-k8"}
    packs = {}
    for q in matched:
        style_pack = [f"p{q}_{st}" for st in STYLES]                        # pure style attack
        by_a = collections.defaultdict(list)
        for s in range(8):
            by_a[ans.get((q, s))].append(s)
        reps = [v[0] for v in by_a.values()]
        if len(reps) < 4:
            continue
        method_pack = [f"n{q}_{s}" for s in sorted(reps)[:4]]               # 4 distinct answers
        packs[q] = (style_pack, method_pack)
    ho = [i for i, q in enumerate(sorted(packs)) if q in test_q]
    print(f"problems with >=4 distinct answers among 8 samples: {len(packs)}  (of which held-out: {len(ho)})\n")
    print("  metric          V(style pack)   V(method pack)    R_set = (Vm-1)/(Vs-1)   hackable %   held-out R")
    for mn in metrics:
        tau = np.mean([metrics[mn](f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs])
        tau = abs(tau) if abs(tau) > 1e-9 else 1.0
        vs, vm = [], []
        for q, (sp, mp) in sorted(packs.items()):
            for pack, acc in ((sp, vs), (mp, vm)):
                D = np.zeros((4, 4))
                for i, j in combinations(range(4), 2):
                    D[i, j] = D[j, i] = metrics[mn](pack[i], pack[j])
                acc.append(vendi(D, tau))
        vs, vm = np.array(vs), np.array(vm)
        r = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9)
        bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
              for i in (RNG.integers(0, len(vs), (NBOOT, len(vs))))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rh = (vm[ho].mean() - 1) / max(vs[ho].mean() - 1, 1e-9) if ho else float("nan")
        print(f"  {mn:12s}    {vs.mean():.3f}           {vm.mean():.3f}            {r:6.2f} [{lo:5.2f},{hi:5.2f}]"
              f"     {100/max(r,1e-9):5.0f}%      {rh:6.2f}")

    Path(TB / "v6_core.json").write_text(json.dumps({"matched": matched, "n_packs": len(packs)}))


def pad_encode():
    """Padding attack: A' = A ++ paraphrase(A). Logic identical, length ~2x. Needs a GPU re-encode."""
    texts = dict(zip(*[list(x) for x in s2.load_texts()]))
    paras = [json.loads(l) for l in open(TB / "paraphrases.jsonl")]
    base = {p["qidx"]: p["samp"] for p in paras}
    ids, out = [], []
    for q, s in sorted(base.items()):
        b = texts[f"n{q}_{s}"]
        ids.append(f"P{q}_self"); out.append(b + "\n\n" + b)              # pure duplication
        for st in ("A_concise", "B_pedagogical", "C_casual"):
            ids.append(f"P{q}_{st[0]}"); out.append(b + "\n\n" + texts[f"p{q}_{st}"])
    print(f"encoding {len(ids)} padded traces (mean {np.mean([len(t.split()) for t in out]):.0f} words)")
    chunks = s4.encode_l4(out, layer=18)
    s4.save_chunks("pad_l4_L18", ids, chunks)

    import torch
    z = np.load(TB / "chunk_pad_l4_L18.npz")
    W = torch.nn.Linear(z["E"].shape[1], 256, bias=False)
    W.load_state_dict(torch.load(TB / "head_l4_L18.pt", map_location="cpu"))
    with torch.no_grad():
        F = torch.nn.functional.normalize(W(torch.tensor(z["E"])), dim=-1).numpy()
    np.savez(TB / "chunk_pad_l4_L18_head.npz", E=F, offsets=json.dumps(json.loads(str(z["offsets"]))))
    Path(TB / "pad_texts.json").write_text(json.dumps(dict(zip(ids, out))))
    print("wrote chunk_pad_l4_L18_head.npz")


def pad():
    """R for the padding arm. L4-chunk head is the reward-relevant metric (LLM-free, differentiable)."""
    texts = dict(zip(*[list(x) for x in s2.load_texts()]))
    ptexts = json.loads((TB / "pad_texts.json").read_text())
    paras = [json.loads(l) for l in open(TB / "paraphrases.jsonl")]
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    base = {p["qidx"]: p["samp"] for p in paras}
    matched = sorted(set(base) & {p["qidx"] for p in negs})

    # merge the padded chunk banks with the originals so one Chamfer fn covers both
    dfun = {}
    for name, orig, padz in (("l4_chunk", "chunk_l4_L18_head.npz", "chunk_pad_l4_L18_head.npz"),
                             ("raw_chunk", "chunk_l4_L18.npz", "chunk_pad_l4_L18.npz")):
        za, zb = np.load(TB / orig), np.load(TB / padz)
        E = np.concatenate([za["E"], zb["E"]])
        off = json.loads(str(za["offsets"]))
        n0 = len(za["E"])
        off.update({k: (a + n0, b + n0) for k, (a, b) in json.loads(str(zb["offsets"])).items()})
        def mk(E=E, off=off):
            def d(a, b):
                (a0, a1), (b0, b1) = off[a], off[b]
                S = E[a0:a1] @ E[b0:b1].T
                return float(1 - 0.5 * (S.max(1).mean() + S.max(0).mean()))
            return d
        dfun[name] = mk()
    allt = dict(texts); allt.update(ptexts)
    ts = {k: tokset(v) for k, v in allt.items()}
    dfun["tok_jaccard"] = lambda a, b: 1 - len(ts[a] & ts[b]) / max(len(ts[a] | ts[b]), 1)
    dfun["len_ratio"] = lambda a, b: abs(len(ts[a]) - len(ts[b])) / max(len(ts[a] | ts[b]), 1)

    pad_pairs = [(q, f"n{q}_{base[q]}", f"P{q}_{t}") for q in matched for t in ("self", "A", "B", "C")]
    style_pairs = [(q, f"n{q}_{base[q]}", f"p{q}_{st}") for q in matched for st in STYLES]
    meth_pairs = [(p["qidx"], f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
                  for p in negs if p["qidx"] in set(matched)]
    print(f"padding arm: {len(pad_pairs)} pairs over {len(matched)} problems\n")
    print("  metric        d(pad)   d(style)  d(method)    R=d(meth)/d(pad)")
    for mn, f in dfun.items():
        dp = np.array([f(a, b) for _, a, b in pad_pairs])
        ds = np.array([f(a, b) for _, a, b in style_pairs])
        dm = np.array([f(a, b) for _, a, b in meth_pairs])
        pq = np.array([q for q, _, _ in pad_pairs]); mq = np.array([q for q, _, _ in meth_pairs])
        lo, hi = boot_ratio((dm, mq), (dp, pq), None)
        print(f"  {mn:12s} {dp.mean():7.4f}  {ds.mean():7.4f}  {dm.mean():7.4f}   "
              f"{dm.mean()/max(dp.mean(),1e-9):6.2f} [{lo:5.2f},{hi:5.2f}]")

    # set-level: 4 padded variants of ONE trace vs 4 genuinely distinct traces
    traces = [json.loads(l) for l in open(TB / "traces.jsonl")]
    ans = {(t["qidx"], t["samp"]): t["extracted"] for t in traces if t["source"] == "neutral-k8"}
    print("\n  set-level (k=4): pack of 4 padded clones vs 4 distinct-answer traces")
    print("  metric          V(pad pack)   V(method pack)   R_set")
    for mn, f in dfun.items():
        tau = abs(np.mean([f(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs
                           if p["qidx"] in set(matched)])) or 1.0
        vp, vm = [], []
        for q in matched:
            by_a = collections.defaultdict(list)
            for s in range(8):
                by_a[ans.get((q, s))].append(s)
            reps = sorted(v[0] for v in by_a.values())
            if len(reps) < 4:
                continue
            for pack, acc in (([f"P{q}_{t}" for t in ("self", "A", "B", "C")], vp),
                              ([f"n{q}_{s}" for s in reps[:4]], vm)):
                D = np.zeros((4, 4))
                for i, j in combinations(range(4), 2):
                    D[i, j] = D[j, i] = f(pack[i], pack[j])
                acc.append(vendi(D, tau))
        vp, vm = np.array(vp), np.array(vm)
        print(f"  {mn:12s}    {vp.mean():.3f}         {vm.mean():.3f}          "
              f"{(vm.mean()-1)/max(vp.mean()-1,1e-9):6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--pad-encode", action="store_true")
    ap.add_argument("--pad", action="store_true")
    a = ap.parse_args()
    (TB / "manifest_v6.json").write_text(json.dumps({
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "args": vars(a), "seed": 20260731, "time": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}, indent=1))
    if a.core:
        core()
    if a.pad_encode:
        pad_encode()
    if a.pad:
        pad()
