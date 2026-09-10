#!/usr/bin/env python3
"""V1 step 1: wide rollout pools for the split-half diversity -> pass@k validity test.

Why wide: the group-level estimator (diversity of a group vs that group's pass@k) is
degenerate -- pass@k = 1{n_correct >= 1} is a deterministic function of the group's correct
count, so the correlation is fully mediated by correctness composition (measured 2026-08-26:
r = -0.112 marginal, undefined within strata). The non-degenerate version measures diversity
on one half of a problem's rollouts and pass@k on the DISJOINT half, then asks for the
incremental predictive value of diversity over pass@1. On the existing K=8 data that flips
the sign: rho(D, pass@4) = -0.455 marginal, but +0.350 partial on accuracy (p=0.01, n=50).
K=8 -> 4/4 splits is too thin to push further, hence N=64.

  CUDA_VISIBLE_DEVICES=3 HF_HOME=/scratch/yirenl2/.cache/huggingface \
    PYTHONPATH=/scratch/yirenl2/DeepPersona .venv-vllm/bin/python \
    scripts/37_passk_validity.py --generate all
  .venv/bin/python scripts/37_passk_validity.py --summary
  .venv/bin/python scripts/37_passk_validity.py --p0        # free, no GPU
"""
import argparse, collections, importlib.util, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path("runs/passk-validity")
MODEL, REV = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
N, SEED = 64, 0

BENCH = {
    "math-hard": dict(kind="math", source="aime_and_amc", nprob=150, max_new=2048),
    "math-l45": dict(kind="math", source="qwen-testbed", nprob=100, max_new=2048),
    "code": dict(kind="code", nprob=150, max_new=1024),
    "csr": dict(kind="csr", nprob=150, max_new=1024),
}


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def items_for(bench):
    cfg = BENCH[bench]
    if cfg["kind"] == "math":
        if cfg["source"] == "qwen-testbed":  # MATH L4-5 problem set from the logdist testbed
            rows = [json.loads(l) for l in open("runs/logdist-qwen/traces.jsonl")]
            byq = {r["qidx"]: r for r in rows if r["samp"] == 0}
            return [{"problem": byq[q]["problem"], "gold": byq[q]["gold"], "tests": None,
                     "imports": None} for q in sorted(byq)[:cfg["nprob"]]]
        from deeppersona.data_espl import load_raw
        data = load_raw(cfg["source"])
        random.Random(SEED).shuffle(data)  # same shuffle as scripts/28
        return [{"problem": d["problem"], "gold": d["groundtruth"], "tests": None,
                 "imports": None} for d in data[:cfg["nprob"]]]
    s34 = _load("scripts/34_domain_transfer.py", "s34")
    return s34.load_problems(cfg["kind"], cfg["nprob"])


def generate(llm, bench, smoke, force):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
    from deeppersona.personas import system_message
    s34 = _load("scripts/34_domain_transfer.py", "s34")
    cfg = BENCH[bench]
    fn = OUT / f"traces_{bench}{'_smoke' if smoke else ''}.jsonl"
    if fn.exists() and not force:
        print(f"[{bench}] exists, skipping ({sum(1 for _ in open(fn))} rows)", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    items = items_for(bench)
    n = N
    if smoke:
        items, n = items[:3], 4
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    if cfg["kind"] == "math":
        msgs = [[{"role": "system", "content": system_message("math", -1, 0, "basic")},
                 {"role": "user", "content": it["problem"]}] for it in items]
    else:
        msgs = [[{"role": "user", "content": s34.user_prompt(cfg["kind"], it)}] for it in items]
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for m in msgs]
    jobs = [(q, s) for q in range(len(items)) for s in range(n)]
    sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=cfg["max_new"],
                         seed=SEED * 7_000_003 + q * N + s) for q, s in jobs]
    t0 = time.time()
    outs = llm.generate([prompts[q] for q, _ in jobs], sp)
    print(f"[{bench}] {len(jobs)} gens in {time.time() - t0:.0f}s", flush=True)
    texts = [o.outputs[0].text for o in outs]

    t0 = time.time()
    if cfg["kind"] == "math":
        # signal-based sympy timeout: main thread only
        graded = []
        for text, (q, _) in zip(texts, jobs):
            try:
                ext = extract_boxed(text or "")
            except ValueError:
                ext = None
            ok = bool(run_with_timeout_signal(grade_answer, args=(ext, items[q]["gold"]),
                                              timeout_seconds=2)) if ext else False
            graded.append((int(ok), ext))
    elif cfg["kind"] == "code":
        with ThreadPoolExecutor(16) as ex:  # each test runs in its own subprocess
            graded = list(ex.map(lambda a: s34.grade("code", a[0], items[a[1]]),
                                 [(t, q) for t, (q, _) in zip(texts, jobs)]))
    else:
        graded = [s34.grade("csr", t, items[q]) for t, (q, _) in zip(texts, jobs)]
    print(f"[{bench}] graded in {time.time() - t0:.0f}s", flush=True)

    with open(fn, "w") as f:
        for (q, s), text, (ok, ext) in zip(jobs, texts, graded):
            f.write(json.dumps({"qidx": q, "samp": s, "text": text, "extracted": ext,
                                "correct": ok, "gold": items[q]["gold"],
                                "problem": items[q]["problem"]}) + "\n")
    C = collections.defaultdict(list)
    for (q, _), (ok, _) in zip(jobs, graded):
        C[q].append(ok)
    nc = np.array([sum(C[q]) for q in sorted(C)])
    print(f"[{bench}] acc {np.mean([g[0] for g in graded]):.3f}  "
          f"informative(0<c<{n}) {np.mean((nc > 0) & (nc < n)):.2f}  "
          f"all-wrong {np.mean(nc == 0):.2f}  all-correct {np.mean(nc == n):.2f}", flush=True)
    (OUT / f"manifest_{bench}.json").write_text(json.dumps({
        "bench": bench, "cfg": cfg, "n_rollouts": n, "n_problems": len(items), "seed": SEED,
        "seed_formula": "SEED*7000003 + qidx*64 + samp", "model": MODEL, "revision": REV,
        "temperature": 0.7, "top_p": 0.95, "thinking": False,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip(),
        "ts": time.time()}, indent=1))


def passk(c, n, k):
    """Unbiased pass@k (Chen et al. 2021) for c correct out of n samples."""
    if n - c < k:
        return 1.0
    return float(1 - np.prod([(n - c - i) / (n - i) for i in range(k)]))


def summary():
    print(f"{'bench':11s} {'probs':>5s} {'N':>3s} {'acc':>6s} {'info':>5s} {'0':>5s} {'all':>5s}"
          f"   pass@1  @2   @4   @8   @16  @32")
    for b in BENCH:
        fn = OUT / f"traces_{b}.jsonl"
        if not fn.exists():
            continue
        C = collections.defaultdict(list)
        for l in open(fn):
            r = json.loads(l)
            C[r["qidx"]].append(r["correct"])
        n = min(len(v) for v in C.values())
        nc = np.array([sum(C[q][:n]) for q in sorted(C)])
        pk = [np.mean([passk(c, n, k) for c in nc]) for k in (1, 2, 4, 8, 16, 32)]
        print(f"{b:11s} {len(nc):5d} {n:3d} {nc.mean() / n:6.3f} "
              f"{np.mean((nc > 0) & (nc < n)):5.2f} {np.mean(nc == 0):5.2f} {np.mean(nc == n):5.2f}"
              f"   " + " ".join(f"{v:.3f}" for v in pk))


def _rawval(t, minv=2):
    from fractions import Fraction
    import re
    out = set()
    for m in re.findall(r"-?\d+(?:/\d+)?", t or ""):
        try:
            v = Fraction(m)
        except Exception:
            continue
        if abs(v) > minv:
            out.add(v)
    return out


def _words(t):
    import re
    return set(re.findall(r"[a-z]{4,}", (t or "").lower()))


def validity(bench, nsplit=200, ks=(2, 4, 8, 16), info_only=False):
    """Split-half: diversity on half A, pass@k on disjoint half B, controlling pass@1(A)."""
    from scipy import stats
    s6 = _load("scripts/23_logdist_seqot.py", "s6")
    rows = [json.loads(l) for l in open(OUT / f"traces_{bench}.jsonl")]
    P = collections.defaultdict(dict)
    for r in rows:
        P[r["qidx"]][r["samp"]] = r
    n = min(len(v) for v in P.values())
    h, TAU = n // 2, 0.85
    feats = {"raw_val": _rawval, "tok_jac": _words}
    qs = sorted(P)
    DM = {}
    for nm, f in feats.items():
        for q in qs:
            S = [f(P[q][s]["text"]) for s in range(n)]
            M = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(i + 1, n):
                    d = 1 - len(S[i] & S[j]) / len(S[i] | S[j]) if (S[i] and S[j]) else TAU
                    M[i, j] = M[j, i] = d
            np.fill_diagonal(M, 0.0)
            DM[(nm, q)] = np.nan_to_num(M, nan=TAU)
    C = {q: np.array([P[q][s]["correct"] for s in range(n)], float) for q in qs}
    if info_only:  # all-wrong/all-correct problems have zero pass@k variance
        qs = [q for q in qs if 0 < C[q].sum() < n]
    rng = np.random.default_rng(20260826)
    splits = [rng.permutation(n) for _ in range(nsplit)]
    print(f"\n[{bench}{'/info' if info_only else ''}] split-half validity: {len(qs)} problems, N={n} ({h}/{h} splits), "
          f"{nsplit} splits/problem")
    print(f"{'metric':9s} {'k':>3s}  {'rho(D,pk)':>9s} {'rho(D,acc)':>10s}  "
          f"{'partial rho(D,pk|acc)':>21s}  {'[95% CI]':>16s}  {'dR2':>6s}")
    res = {}
    for nm in feats:
        per_q = {k: [] for k in ks}
        accA = []
        for q in qs:
            M, c = DM[(nm, q)], C[q]
            da, p1, pk = [], [], {k: [] for k in ks}
            for perm in splits:
                A, B = perm[:h], perm[h:]
                da.append(s6.vendi(M[np.ix_(A, A)], TAU))
                p1.append(c[A].mean())
                cb = int(c[B].sum())
                for k in ks:
                    pk[k].append(passk(cb, h, k))
            accA.append((np.mean(da), np.mean(p1)))
            for k in ks:
                per_q[k].append(np.mean(pk[k]))
        D = np.array([a for a, _ in accA]); A1 = np.array([b for _, b in accA])
        for k in ks:
            B = np.array(per_q[k])
            rk = stats.rankdata
            X = np.c_[np.ones(len(qs)), rk(A1)]
            rd = rk(D) - X @ np.linalg.lstsq(X, rk(D), rcond=None)[0]
            rb = rk(B) - X @ np.linalg.lstsq(X, rk(B), rcond=None)[0]
            pr = stats.pearsonr(rd, rb)
            X2 = np.c_[X, rk(D)]
            r2 = lambda M_: 1 - np.sum((rk(B) - M_ @ np.linalg.lstsq(M_, rk(B), rcond=None)[0]) ** 2) \
                / np.sum((rk(B) - rk(B).mean()) ** 2)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(qs), len(qs))
                Xb = np.c_[np.ones(len(i)), rk(A1[i])]
                a_ = rk(D[i]) - Xb @ np.linalg.lstsq(Xb, rk(D[i]), rcond=None)[0]
                b_ = rk(B[i]) - Xb @ np.linalg.lstsq(Xb, rk(B[i]), rcond=None)[0]
                sd = a_.std() * b_.std()
                bs.append(float((a_ * b_).mean() / sd) if sd > 0 else 0.0)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[f"{nm}@{k}"] = {"rho_pk": float(stats.spearmanr(D, B).statistic),
                                "rho_acc": float(stats.spearmanr(D, A1).statistic),
                                "partial": float(pr.statistic), "p": float(pr.pvalue),
                                "ci": [float(lo), float(hi)], "dR2": float(r2(X2) - r2(X))}
            r = res[f"{nm}@{k}"]
            print(f"{nm:9s} {k:3d}  {r['rho_pk']:+9.3f} {r['rho_acc']:+10.3f}  "
                  f"{r['partial']:+13.3f} (p={r['p']:.3f})  [{lo:+.2f},{hi:+.2f}]  {r['dR2']:+6.3f}")
    (OUT / f"validity_{bench}{'_info' if info_only else ''}.json").write_text(json.dumps(res, indent=1))


def p0_passk():
    """Interventional preview: does method-conditioned prompting buy pass@k despite lower acc?"""
    nat = collections.defaultdict(list)
    for l in open("runs/regime-probe/traces_nonthink-hard.jsonl"):
        r = json.loads(l)
        if r["qidx"] < 20:
            nat[r["qidx"]].append(r["correct"])
    hint = collections.defaultdict(list)
    hint_by = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open("runs/p0-method-pool/hinted.jsonl"):
        r = json.loads(l)
        hint[r["qidx"]].append(r["correct"])
        hint_by[r["hint"]][r["qidx"]].append(r["correct"])
    qs = sorted(set(nat) & set(hint))
    nn, nh = min(len(nat[q]) for q in qs), min(len(hint[q]) for q in qs)
    pools = {"natural": (nat, nn), "hinted": (hint, nh),
             "pooled": ({q: nat[q][:nn] + hint[q][:nh] for q in qs}, nn + nh)}
    rng = np.random.default_rng(0)
    print(f"P0 interventional preview: {len(qs)} problems (AIME+AMC, Qwen3-8B non-thinking)")
    print(f"{'pool':9s} {'n':>3s} {'acc':>6s}   " + "  ".join(f"@{k:<5d}" for k in (1, 2, 4, 8, 16)))
    for nm, (pool, n) in pools.items():
        nc = np.array([sum(pool[q][:n]) for q in qs])
        row, cis = [], []
        for k in (1, 2, 4, 8, 16):
            if k > n:
                row.append(None); cis.append(None); continue
            v = np.array([passk(c, n, k) for c in nc])
            bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
            row.append(v.mean()); cis.append(np.percentile(bs, [2.5, 97.5]))
        cells = "  ".join("  --  " if r is None else f"{r:.3f} " for r in row)
        print(f"{nm:9s} {n:3d} {nc.mean() / n:6.3f}   {cells}")
        print(f"{'':9s} {'':3s} {'':6s}   " + "  ".join(
            "      " if c is None else f"[{c[0]:.2f},{c[1]:.2f}]" for c in cis))
    # matched-budget head-to-head at k=8: natural(8 of 8) vs hinted(8 of 16 subsampled)
    ncn = np.array([sum(nat[q][:nn]) for q in qs])
    d = []
    for _ in range(4000):  # paired bootstrap over PROBLEMS, fresh hint subsample each draw
        idx = rng.integers(0, len(qs), len(qs))
        sub = np.array([sum(rng.permutation(hint[qs[i]][:nh])[:8]) for i in idx])
        d.append(np.mean([passk(c, 8, 8) for c in sub])
                 - np.mean([passk(ncn[i], nn, 8) for i in idx]))
    d = np.array(d)
    print(f"\nmatched budget k=8 (hinted 8-subsample - natural 8): "
          f"{d.mean():+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]")
    print(f"per-hint accuracy: " + "  ".join(
        f"{h}={np.mean([c for q in hint_by[h] for c in hint_by[h][q]]):.2f}"
        for h in sorted(hint_by)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="append", default=[])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--p0", action="store_true")
    ap.add_argument("--validity", action="append", default=[])
    ap.add_argument("--info-only", action="store_true", dest="info_only")
    a = ap.parse_args()
    if a.generate:
        from vllm import LLM
        bs = list(BENCH) if "all" in a.generate else a.generate
        llm = LLM(model=MODEL, revision=REV, dtype="bfloat16", gpu_memory_utilization=0.9,
                  max_model_len=8192)
        for b in bs:
            generate(llm, b, a.smoke, a.force)
    if a.summary:
        summary()
    for b in (list(BENCH) if 'all' in a.validity else a.validity):
        validity(b, info_only=a.info_only)
    if a.p0:
        p0_passk()


if __name__ == "__main__":
    main()
