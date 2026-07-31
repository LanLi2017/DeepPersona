"""Stage 3 of logical-distance: L3 reasoning-graph distance (edges over frozen L2 claims).

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/18_logdist_l3.py --extract-edges --tag a --smoke
  .venv/bin/python scripts/18_logdist_l3.py --extract-edges --tag a          # temp 0
  .venv/bin/python scripts/18_logdist_l3.py --extract-edges --tag b          # temp 0.7 (V3 reparse)
  .venv/bin/python scripts/18_logdist_l3.py --eval --tag a
  .venv/bin/python scripts/18_logdist_l3.py --reliability
"""
import argparse, collections, importlib.util, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TB = Path("runs/logdist-testbed")
spec = importlib.util.spec_from_file_location("s2", "scripts/17_logdist_stage2.py")
s2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2)

EDGE_SYS = """You are given a mathematical solution and the numbered list of atomic claims extracted from it.
Output the direct derivation dependencies among the claims as directed edges [i, j], meaning claim j is derived directly using claim i.

Rules:
- Indices are 0-based positions in the given claim list.
- Include only DIRECT dependencies the solution's reasoning actually uses (not transitive closures).
- Premises/definitions have no incoming edges; every derived claim should have >=1 incoming edge.
- No self-loops, no cycles.
Output JSON: {"edges": [[i, j], ...]}"""


def extract_edges(args):
    from openai import OpenAI
    client = OpenAI()
    ids, texts = s2.load_texts()
    claims = {r["id"]: r["claims"] for r in (json.loads(l) for l in open(TB / "claims.jsonl"))}
    jobs = list(zip(ids, texts))
    if args.smoke:
        jobs = [j for j in jobs if j[0] in ("n23_0", "p23_D_restructured", "n58_6")]
    est = (sum(len(t) // 3 + 600 for _, t in jobs) * 0.4 + len(jobs) * 200 * 1.6) / 1e6
    print(f"edge jobs={len(jobs)}  est cost ≈ ${est:.2f}  temp={0.0 if args.tag == 'a' else 0.7}")

    def one(job):
        tid, text = job
        cl = claims[tid]
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(cl))
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": EDGE_SYS},
                              {"role": "user", "content": f"SOLUTION:\n{text}\n\nCLAIMS:\n{numbered}"}],
                    max_completion_tokens=1500, temperature=0.0 if args.tag == "a" else 0.7)
                edges = json.loads(r.choices[0].message.content).get("edges", [])
                edges = [[i, j] for i, j in edges
                         if isinstance(i, int) and isinstance(j, int)
                         and 0 <= i < len(cl) and 0 <= j < len(cl) and i != j]
                u = r.usage
                return {"id": tid, "edges": edges, "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    fn = TB / (f"edges_{args.tag}_smoke.jsonl" if args.smoke else f"edges_{args.tag}.jsonl")
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    ne = [len(r["edges"]) for r in rows]
    print(f"wrote {fn}: {len(rows)} graphs, edges/graph mean={np.mean(ne):.1f} min={min(ne)} max={max(ne)}, "
          f"cost ≈ ${(ptok*0.4+ctok*1.6)/1e6:.2f}")
    if args.smoke:
        for r in rows:
            print(f"\n-- {r['id']}: {r['edges']}")
            for i, c in enumerate(claims[r["id"]]):
                print(f"   {i}. {c[:90]}")


def graph_dists(tag):
    import ot
    z = np.load(TB / "claim_embs.npz")
    E, offsets = z["E"], json.loads(str(z["offsets"]))
    edges = {r["id"]: r["edges"] for r in (json.loads(l) for l in open(TB / f"edges_{tag}.jsonl"))}

    def edge_vecs(tid):
        a0, a1 = offsets[tid]
        ev = [np.concatenate([E[a0 + i], E[a0 + j]]) for i, j in edges.get(tid, []) if a0 + j < a1 and a0 + i < a1]
        if not ev:
            return None
        V = np.stack(ev)
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    def sp_matrix(tid):
        a0, a1 = offsets[tid]
        n = a1 - a0
        A = np.full((n, n), np.inf)
        np.fill_diagonal(A, 0)
        for i, j in edges.get(tid, []):
            if i < n and j < n:
                A[i, j] = A[j, i] = 1
        for k in range(n):  # floyd-warshall, n<=31
            A = np.minimum(A, A[:, [k]] + A[[k], :])
        A[np.isinf(A)] = n
        return A / max(n, 1)

    cache_v, cache_c = {}, {}
    def d_edge(a, b):
        for t in (a, b):
            if t not in cache_v:
                cache_v[t] = edge_vecs(t)
        Va, Vb = cache_v[a], cache_v[b]
        if Va is None or Vb is None:
            return 1.0
        S = Va @ Vb.T
        return float(1 - 0.5 * (S.max(1).mean() + S.max(0).mean()))

    def d_fgw(a, b):
        (a0, a1), (b0, b1) = offsets[a], offsets[b]
        if a1 == a0 or b1 == b0:
            return 1.0
        for t in (a, b):
            if t not in cache_c:
                cache_c[t] = sp_matrix(t)
        M = 1 - E[a0:a1] @ E[b0:b1].T
        p, q = np.ones(a1 - a0) / (a1 - a0), np.ones(b1 - b0) / (b1 - b0)
        return float(ot.gromov.fused_gromov_wasserstein2(
            M, cache_c[a], cache_c[b], p, q, loss_fun="square_loss", alpha=0.5))

    return d_edge, d_fgw


def battery(dfun, name):
    from sklearn.metrics import roc_auc_score
    paras, _, test_q = s2.split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    d_neg = [dfun(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    d_neg_t = [dfun(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs if p["qidx"] in test_q]
    print(f"== {name} ==")
    allpos = []
    for st in ("A_concise", "B_pedagogical", "C_casual", "D_restructured"):
        v = [dfun(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in paras if p["style"] == st]
        vt = [dfun(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in paras
              if p["style"] == st and p["qidx"] in test_q]
        allpos += v
        print(f"  {st[0]}: full {roc_auc_score([0]*len(v)+[1]*len(d_neg), v+d_neg):.3f}"
              f"  heldout {roc_auc_score([0]*len(vt)+[1]*len(d_neg_t), vt+d_neg_t):.3f}")
    print(f"  overall full: {roc_auc_score([0]*len(allpos)+[1]*len(d_neg), allpos+d_neg):.3f}")


def reliability():
    from scipy.stats import pearsonr, spearmanr
    da_e, da_f = graph_dists("a")
    db_e, db_f = graph_dists("b")
    paras, _, _ = s2.split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    pairs = [(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in paras] + \
            [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    for nm, da, db in (("edge", da_e, db_e), ("fgw", da_f, db_f)):
        xa = np.array([da(*p) for p in pairs]); xb = np.array([db(*p) for p in pairs])
        print(f"V3 {nm}: pearson {pearsonr(xa, xb).statistic:.3f}  spearman {spearmanr(xa, xb).statistic:.3f}")
    ea = {r["id"]: set(map(tuple, r["edges"])) for r in (json.loads(l) for l in open(TB / "edges_a.jsonl"))}
    eb = {r["id"]: set(map(tuple, r["edges"])) for r in (json.loads(l) for l in open(TB / "edges_b.jsonl"))}
    jac = [len(ea[i] & eb[i]) / len(ea[i] | eb[i]) for i in ea if ea[i] | eb[i]]
    print(f"V3 raw edge-set jaccard(parse a, parse b): mean {np.mean(jac):.3f}  median {np.median(jac):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-edges", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--reliability", action="store_true")
    ap.add_argument("--tag", default="a")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.extract_edges:
        extract_edges(args)
    if args.eval:
        d_edge, d_fgw = graph_dists(args.tag)
        battery(d_edge, f"l3edge_{args.tag}")
        battery(d_fgw, f"l3fgw_{args.tag}")
    if args.reliability:
        reliability()


if __name__ == "__main__":
    main()
