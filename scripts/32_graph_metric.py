#!/usr/bin/env python3
"""G1: training-free reasoning-graph diversity metric, in the target regime.

Revisits L3 (scripts/18, closed 2026-07-23 under the broken distinct-answer proxy) with
three changes: (1) eval target = gpt-5.5 judge method labels (N3b pairs/packs), (2) nodes
carry a fixed operation-type taxonomy so topology can be compared text-free (WL kernel),
(3) full N3b battery incl. style-pack gameability R_set — the axis every training-free
metric failed (81-101% hackable) and the reason the (test-contaminated) heads exist.
No heads, no contrastive supervision, nothing fit to eval problems.

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/32_graph_metric.py --extract --smoke
  .venv/bin/python scripts/32_graph_metric.py --extract            # ~$3, temp 0
  .venv/bin/python scripts/32_graph_metric.py --extract --pass2    # reliability subset, temp 0.7
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/32_graph_metric.py --encode
  .venv/bin/python scripts/32_graph_metric.py --eval
"""
import argparse, collections, importlib.util, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s6 = _load("scripts/23_logdist_seqot.py", "s6")
s9 = _load("scripts/29_n3b_ingame.py", "s9")

RP = Path("runs/regime-probe")
TYPES = ["setup", "recall", "substitute", "algebra", "compute", "equation", "solve",
         "case", "count", "bound", "symmetry", "verify", "conclude", "other"]

GRAPH_SYS = f"""You are given a mathematical solution. Extract its reasoning as a typed dependency graph.

Steps: break the solution into atomic reasoning steps (5-25). For each step give:
- "t": the operation type, EXACTLY one of {json.dumps(TYPES)}
  (setup=restate problem/define variables; recall=cite a theorem/formula/known identity;
   substitute=plug one expression into another; algebra=symbolic manipulation/simplification;
   compute=numeric arithmetic; equation=set up an equation/system; solve=solve an
   equation/inequality for an unknown; case=split into or handle one case of a case analysis;
   count=combinatorial counting/enumeration; bound=inequality/estimate/extremal argument;
   symmetry=exploit symmetry/invariant/parity; verify=check a candidate/consistency;
   conclude=state the final answer)
- "s": the step's mathematical content in one short canonical clause (<=15 words, symbolic
  where possible, NO narrative or stylistic wording from the solution)

Edges: directed [i, j] (0-based) meaning step j is derived DIRECTLY using step i.
- Only direct dependencies the reasoning actually uses, not transitive closures.
- No self-loops. Every non-setup step should have >=1 incoming edge.

Describe the UNDERLYING mathematical reasoning, ignoring presentation order and phrasing:
identical math presented in different words/order must yield the same graph.
Output JSON: {{"steps": [{{"t": ..., "s": ...}}, ...], "edges": [[i, j], ...]}}"""


def graphs_file(tag):
    return RP / f"graphs_{tag}.jsonl"


def pass2_ids():
    packs = s9.method_packs()
    ids = {t for q in packs for t in packs[q]}
    ids |= {f"p{q}_{st}" for q in packs for st in s9.PACK_STYLES}
    for q in sorted(packs)[:3]:
        ids |= {f"n{q}_{s}" for s in range(s9.K)}
    return ids


def extract(args):
    from openai import OpenAI
    client = OpenAI()
    ids, texts, _ = s9.ids_texts()
    tag = ("b" if args.pass2 else "a") + args.suffix
    jobs = list(zip(ids, texts))
    if args.pass2 or args.subset:
        keep = pass2_ids()
        jobs = [j for j in jobs if j[0] in keep]
    if args.smoke:
        jobs = jobs[:3]
    is_g5 = args.model.startswith("gpt-5")
    pi, po = (2.5, 25.0) if is_g5 else (2.0, 8.0) if args.model == "gpt-4.1" else (0.4, 1.6)
    est = (sum(len(t) // 3 + 700 for _, t in jobs) * pi +
           len(jobs) * (1300 if is_g5 else 700) * po) / 1e6
    print(f"graph jobs={len(jobs)} tag={tag}  est cost ~ ${est:.2f} ({args.model})")

    def one(job):
        tid, text = job
        kw = dict(max_completion_tokens=6000, reasoning_effort="low") if is_g5 else \
            dict(max_completion_tokens=2500, temperature=0.0 if tag.startswith("a") else 0.7)
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": GRAPH_SYS},
                              {"role": "user", "content": f"SOLUTION:\n{text}"}], **kw)
                out = json.loads(r.choices[0].message.content)
                steps = [{"t": s.get("t") if s.get("t") in TYPES else "other",
                          "s": str(s.get("s", ""))[:200]} for s in out.get("steps", [])][:30]
                n = len(steps)
                edges = sorted({(i, j) for i, j in out.get("edges", [])
                                if isinstance(i, int) and isinstance(j, int)
                                and 0 <= i < n and 0 <= j < n and i != j})
                u = r.usage
                return {"id": tid, "steps": steps, "edges": [list(e) for e in edges],
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    fn = RP / f"graphs_{tag}_smoke.jsonl" if args.smoke else graphs_file(tag)
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ns, ne = [len(r["steps"]) for r in rows], [len(r["edges"]) for r in rows]
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote {fn}: {len(rows)} graphs, steps mean={np.mean(ns):.1f} "
          f"edges mean={np.mean(ne):.1f}, cost ~ ${(ptok * pi + ctok * po) / 1e6:.2f}")
    if args.smoke:
        for r in rows:
            print(f"\n-- {r['id']} edges={r['edges']}")
            for i, s in enumerate(r["steps"]):
                print(f"   {i}. [{s['t']}] {s['s'][:80]}")


def encode():
    import torch
    from transformers import AutoModel, AutoTokenizer
    rows = [json.loads(l) for l in open(graphs_file("a"))]
    flat = [(r["id"], s["s"]) for r in rows for s in r["steps"]]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B",
                                      dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    embs = []
    with torch.no_grad():
        for i in range(0, len(flat), 64):
            b = tok([s for _, s in flat[i:i + 64]], padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state[:, -1]
            embs.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    E, off, k = np.concatenate(embs), {}, 0
    for r in rows:
        off[r["id"]] = (k, k + len(r["steps"])); k += len(r["steps"])
    np.savez(RP / "graph_stmt_embs.npz", E=E, offsets=json.dumps(off))
    print(f"encoded {len(rows)} graphs, {len(E)} step statements")


def load_graphs(tag):
    return {r["id"]: r for r in (json.loads(l) for l in open(graphs_file(tag)))}


def wl_counter(g, h=2):
    labels = [s["t"] for s in g["steps"]]
    pred = collections.defaultdict(list); succ = collections.defaultdict(list)
    for i, j in g["edges"]:
        succ[i].append(j); pred[j].append(i)
    feats = collections.Counter(labels)
    for _ in range(h):
        labels = [labels[i] + "<" + ",".join(sorted(labels[p] for p in pred[i])) +
                  ">" + ",".join(sorted(labels[c] for c in succ[i]))
                  for i in range(len(labels))]
        feats.update(labels)
    return feats


def reach_counter(g):
    n = len(g["steps"])
    adj = collections.defaultdict(set)
    for i, j in g["edges"]:
        adj[i].add(j)
    feats = collections.Counter()
    for s in range(n):
        seen, stack = set(), [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        for v in seen:
            feats[(g["steps"][s]["t"], g["steps"][v]["t"])] += 1
    return feats


def edge_counter(g):
    return collections.Counter((g["steps"][i]["t"], g["steps"][j]["t"])
                               for i, j in g["edges"])


def soft_nodes(g):
    ti = {t: k for k, t in enumerate(TYPES)}
    n = len(g["steps"])
    oh = np.zeros((n, len(TYPES)))
    for i, s in enumerate(g["steps"]):
        oh[i, ti[s["t"]]] = 1
    P, S = np.zeros_like(oh), np.zeros_like(oh)
    cp, cs = np.zeros(n), np.zeros(n)
    for i, j in g["edges"]:
        P[j] += oh[i]; cp[j] += 1
        S[i] += oh[j]; cs[i] += 1
    V = np.concatenate([oh, P / np.maximum(cp, 1)[:, None],
                        S / np.maximum(cs, 1)[:, None]], 1)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def cos_counter(ca, cb):
    if not ca or not cb:
        return 1.0
    num = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
    na, nb = sum(v * v for v in ca.values()) ** 0.5, sum(v * v for v in cb.values()) ** 0.5
    return float(1 - num / (na * nb + 1e-9))


def sp_matrix(g):
    n = len(g["steps"])
    A = np.full((n, n), np.inf)
    np.fill_diagonal(A, 0)
    for i, j in g["edges"]:
        A[i, j] = A[j, i] = 1
    for k in range(n):
        A = np.minimum(A, A[:, [k]] + A[[k], :])
    A[np.isinf(A)] = n
    return A / max(n, 1)


def build_metrics(tag="a"):
    import ot
    G = load_graphs(tag)
    z = np.load(RP / "graph_stmt_embs.npz")
    E, off = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    wl = {t: wl_counter(g) for t, g in G.items()}
    wl1 = {t: wl_counter(g, h=1) for t, g in G.items()}
    ty = {t: collections.Counter(s["t"] for s in g["steps"]) for t, g in G.items()}
    rc = {t: reach_counter(g) for t, g in G.items()}
    ec = {t: edge_counter(g) for t, g in G.items()}
    sn = {t: soft_nodes(g) for t, g in G.items() if g["steps"]}
    sp = {}

    def d_fgw(a, b):
        if a not in G or b not in G or not G[a]["steps"] or not G[b]["steps"]:
            return float("nan")
        (a0, a1), (b0, b1) = off[a], off[b]
        for t in (a, b):
            if t not in sp:
                sp[t] = sp_matrix(G[t])
        M = 1 - E[a0:a1] @ E[b0:b1].T
        p = np.ones(a1 - a0) / (a1 - a0); q = np.ones(b1 - b0) / (b1 - b0)
        return float(ot.gromov.fused_gromov_wasserstein2(
            M, sp[a], sp[b], p, q, loss_fun="square_loss", alpha=0.5))

    def d_stmt(a, b):
        if a not in off or b not in off:
            return float("nan")
        (a0, a1), (b0, b1) = off[a], off[b]
        if a1 == a0 or b1 == b0:
            return float("nan")
        return s6.d_chamfer(E[a0:a1], E[b0:b1])

    m = {
        "g:types": s6.memo(lambda a, b: cos_counter(ty.get(a), ty.get(b))),
        "g:wl": s6.memo(lambda a, b: cos_counter(wl.get(a), wl.get(b))),
        "g:wl1": s6.memo(lambda a, b: cos_counter(wl1.get(a), wl1.get(b))),
        "g:edges": s6.memo(lambda a, b: cos_counter(ec.get(a), ec.get(b))),
        "g:soft": s6.memo(lambda a, b: s6.d_chamfer(sn[a], sn[b])
                          if a in sn and b in sn else float("nan")),
        "g:reach": s6.memo(lambda a, b: cos_counter(rc.get(a), rc.get(b))),
        "g:stmt": s6.memo(d_stmt),
        "g:fgw": s6.memo(d_fgw),
    }
    return m, G


def evaluate():
    from sklearn.metrics import roc_auc_score
    m, G = build_metrics()
    diff, same = s9.judge_pairs()
    train_q, test_q = s9.qsplit()
    pa = [json.loads(l) for l in open(RP / "paraphrases.jsonl")]
    base = {r["qidx"]: r["samp"] for r in pa}
    for tag, ks in (("gENS:wl+st", ["g:wl", "g:stmt"]),
                    ("gENS:w+r+s", ["g:wl", "g:reach", "g:stmt"])):
        fs = [(m[k], float(np.nanstd([m[k](a, b) for a, b in diff]))) for k in ks]
        m[tag] = s6.memo(lambda a, b, fs=fs: float(np.nanmean([f(a, b) / z for f, z in fs])))

    def teq(p):
        return int(p[0][1:].split("_")[0]) in test_q
    diff_t = [p for p in diff if teq(p)]; same_t = [p for p in same if teq(p)]
    styles = list(s9.ALL_STYLES)
    print(f"training-free graph metrics; test problems n={len(test_q)} "
          f"({len(diff_t)} diff / {len(same_t)} same judge pairs); ALL-problem AUC_D in ()\n")
    print(f"{'':12s}   AUC_D  (all) " + "".join(f"{st[:5]:>7s}" for st in styles) + "    ALLst")
    for mn, f in m.items():
        dd = [x for x in (f(a, b) for a, b in diff_t) if not np.isnan(x)]
        ds = [x for x in (f(a, b) for a, b in same_t) if not np.isnan(x)]
        dda = [x for x in (f(a, b) for a, b in diff) if not np.isnan(x)]
        dsa = [x for x in (f(a, b) for a, b in same) if not np.isnan(x)]
        aucd = roc_auc_score([0] * len(ds) + [1] * len(dd), ds + dd)
        auca = roc_auc_score([0] * len(dsa) + [1] * len(dda), dsa + dda)
        row, allpos = [], []
        for st in styles:
            v = [f(f"n{q}_{base[q]}", f"p{q}_{st}") for q in sorted(test_q)]
            v = [x for x in v if not np.isnan(x)]
            allpos += v
            row.append(roc_auc_score([0] * len(v) + [1] * len(dd), v + dd))
        overall = roc_auc_score([0] * len(allpos) + [1] * len(dd), allpos + dd)
        print(f"{mn:12s}  {aucd:.3f} ({auca:.3f})" +
              "".join(f"{x:7.3f}" for x in row) + f"{overall:9.3f}")

    packs = s9.method_packs()
    print(f"\nset-level R_set: {len(packs)} method packs vs style packs "
          f"({'/'.join(s[0] for s in s9.PACK_STYLES)}); tau from judge diff pairs:")
    print(f"{'metric':12s}  V_style  V_meth   R_set [95% CI]   hack%")
    rng = np.random.default_rng(20260818)
    res = {}
    for mn, f in m.items():
        pr = [f(a, b) for a, b in diff]
        tau = abs(np.nanmean(pr)) or 1.0
        vs, vm = [], []
        for q in sorted(packs):
            spack = [f"p{q}_{st}" for st in s9.PACK_STYLES]
            for pack, acc in zip((spack, packs[q]), (vs, vm)):
                D = np.zeros((4, 4))
                for i, j in combinations(range(4), 2):
                    D[i, j] = D[j, i] = np.nan_to_num(f(pack[i], pack[j]), nan=tau)
                acc.append(s6.vendi(D, tau))
        vs, vm = np.array(vs), np.array(vm)
        r = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9)
        bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
              for i in rng.integers(0, len(vs), (2000, len(vs)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[mn] = {"R": r, "ci": [lo, hi], "V_style": float(vs.mean()),
                   "V_meth": float(vm.mean())}
        print(f"{mn:12s}   {vs.mean():.3f}   {vm.mean():.3f}   {r:5.2f} [{lo:4.2f},{hi:4.2f}]"
              f"   {100 / max(r, 1e-9):4.0f}%")

    if graphs_file("b").exists():
        Gb = load_graphs("b")
        common = sorted(set(G) & set(Gb))
        feats = {"wl": lambda g: wl_counter(g), "wl1": lambda g: wl_counter(g, h=1),
                 "edges": edge_counter,
                 "types": lambda g: collections.Counter(s["t"] for s in g["steps"])}
        print(f"\nV3 reliability (n={len(common)} re-extractions, temp 0 vs 0.7): "
              f"self-dist vs diff-method dist")
        res["v3"] = {}
        for fn, ff in feats.items():
            fa = {t: ff(G[t]) for t in common}
            fb = {t: ff(Gb[t]) for t in common}
            selfd = [cos_counter(fa[t], fb[t]) for t in common]
            fad = {t: ff(G[t]) for t in G}
            dd = [cos_counter(fad[a], fad[b]) for a, b in diff]
            auc = roc_auc_score([0] * len(selfd) + [1] * len(dd), selfd + dd)
            res["v3"][fn] = {"self": float(np.mean(selfd)), "diff": float(np.mean(dd)),
                             "auc": float(auc)}
            print(f"  {fn:6s} self {np.mean(selfd):.3f}  diff {np.mean(dd):.3f}  "
                  f"ratio {np.mean(dd) / max(np.mean(selfd), 1e-9):.1f}x  AUC {auc:.3f}")
        jac = []
        for t in common:
            ta = collections.Counter(s["t"] for s in G[t]["steps"])
            tb = collections.Counter(s["t"] for s in Gb[t]["steps"])
            union = sum((ta | tb).values())
            jac.append(sum((ta & tb).values()) / union if union else 1.0)
        res["v3"]["type_jac"] = float(np.mean(jac))
        print(f"  type-multiset jaccard mean {np.mean(jac):.3f}")
    (RP / "g1_eval.json").write_text(json.dumps(res, indent=1))


def v3_only(suffix):
    from sklearn.metrics import roc_auc_score
    Ga, Gb = load_graphs("a" + suffix), load_graphs("b" + suffix)
    common = sorted(set(Ga) & set(Gb))
    diff, _ = s9.judge_pairs()
    diff = [(a, b) for a, b in diff if a in Ga and b in Ga]
    feats = {"wl": lambda g: wl_counter(g), "wl1": lambda g: wl_counter(g, h=1),
             "edges": edge_counter,
             "types": lambda g: collections.Counter(s["t"] for s in g["steps"])}
    print(f"V3 shootout suffix='{suffix}': n={len(common)} re-extractions, "
          f"{len(diff)} in-subset diff-method pairs")
    for fn, ff in feats.items():
        fa = {t: ff(Ga[t]) for t in Ga}
        fb = {t: ff(Gb[t]) for t in common}
        selfd = [cos_counter(fa[t], fb[t]) for t in common]
        dd = [cos_counter(fa[a], fa[b]) for a, b in diff]
        auc = roc_auc_score([0] * len(selfd) + [1] * len(dd), selfd + dd)
        print(f"  {fn:6s} self {np.mean(selfd):.3f}  diff {np.mean(dd):.3f}  "
              f"ratio {np.mean(dd) / max(np.mean(selfd), 1e-9):.1f}x  AUC {auc:.3f}")
    jac = []
    for t in common:
        ta = collections.Counter(s["t"] for s in Ga[t]["steps"])
        tb = collections.Counter(s["t"] for s in Gb[t]["steps"])
        union = sum((ta | tb).values())
        jac.append(sum((ta & tb).values()) / union if union else 1.0)
    print(f"  type-multiset jaccard mean {np.mean(jac):.3f}")


def main():
    ap = argparse.ArgumentParser()
    for f in ("extract", "encode", "eval", "pass2", "smoke", "subset", "v3"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "types": TYPES,
               "time": time.strftime("%FT%TZ", time.gmtime())},
              open(RP / "manifest_g1.json", "w"), indent=1)
    if a.extract:
        extract(a)
    if a.encode:
        encode()
    if a.v3:
        v3_only(a.suffix)
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
