"""Stage 2 of logical-distance: L2 claim-set distance + L1.5 contrastive metric.

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/17_logdist_stage2.py --extract --smoke
  .venv/bin/python scripts/17_logdist_stage2.py --extract
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/17_logdist_stage2.py --embed-claims
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/17_logdist_stage2.py --train-l15
  .venv/bin/python scripts/17_logdist_stage2.py --eval
"""
import argparse, collections, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TB = Path("runs/logdist-testbed")

EXTRACT_SYS = """You extract the logical skeleton of a mathematical solution as a list of atomic claims.

Rules:
- One claim per list item: a short, self-contained mathematical statement the solution asserts and uses (definitions it introduces, case splits, derived equations/bounds, computed intermediate values, the final answer).
- Express claims in terms of the problem's original quantities. If the solution introduces auxiliary variables, describe them by their defining property, not their letter name (e.g. "the number of even-position beads" not "k").
- Canonical form: plain digits (no thousands separators), simplified expressions, no LaTeX decoration beyond what is needed.
- EXCLUDE: prose, motivation, restatements of the problem, verification chatter, dead ends that the solution abandons.
- Preserve the solution's actual logic — do not correct errors or add missing steps.
Output JSON: {"claims": ["...", "..."]}"""


def load_texts():
    ids, texts = [], []
    for r in (json.loads(l) for l in open(TB / "traces.jsonl")):
        if r["source"] == "neutral-k8":
            ids.append(f"n{r['qidx']}_{r['samp']}"); texts.append(r["text"])
    for r in (json.loads(l) for l in open(TB / "paraphrases.jsonl")):
        ids.append(f"p{r['qidx']}_{r['style']}"); texts.append(r["text"])
    return ids, texts


def extract(args):
    from openai import OpenAI
    client = OpenAI()
    ids, texts = load_texts()
    if args.smoke:
        keep = ["n23_0", "p23_A_concise", "p23_D_restructured"]
        ids, texts = zip(*[(i, t) for i, t in zip(ids, texts) if i in keep])
    est = (sum(len(t) // 3 + 200 for t in texts) * 0.4 + len(texts) * 400 * 1.6) / 1e6
    print(f"extract jobs={len(ids)}  est cost ≈ ${est:.2f}")

    def one(job):
        tid, text = job
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": EXTRACT_SYS},
                              {"role": "user", "content": text}],
                    max_completion_tokens=2000, temperature=0.0)
                claims = json.loads(r.choices[0].message.content).get("claims", [])
                u = r.usage
                return {"id": tid, "claims": claims, "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, zip(ids, texts)))
    fn = TB / ("claims_smoke.jsonl" if args.smoke else "claims.jsonl")
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    n_claims = [len(r["claims"]) for r in rows]
    print(f"wrote {fn}: {len(rows)} traces, claims/trace mean={np.mean(n_claims):.1f} "
          f"min={min(n_claims)} max={max(n_claims)}, cost ≈ ${(ptok*0.4+ctok*1.6)/1e6:.2f}")
    if args.smoke:
        for r in rows:
            print(f"\n-- {r['id']} ({len(r['claims'])} claims)")
            for c in r["claims"][:12]:
                print("  ", c)


def embed_claims():
    import torch
    from transformers import AutoModel, AutoTokenizer
    rows = [json.loads(l) for l in open(TB / "claims.jsonl")]
    flat = [(r["id"], c) for r in rows for c in r["claims"]]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(flat), 64):
            b = tok([c for _, c in flat[i:i + 64]], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state[:, -1]
            embs.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    E = np.concatenate(embs)
    offsets, k = {}, 0
    for r in rows:
        offsets[r["id"]] = (k, k + len(r["claims"])); k += len(r["claims"])
    np.savez(TB / "claim_embs.npz", E=E, offsets=json.dumps(offsets))
    # bag-of-claims mean vector per trace -> PSD kernel for Vendi (V4/V5), auto-picked up by 16 --eval
    ids, _ = load_texts()
    bags = np.stack([E[offsets[i][0]:offsets[i][1]].mean(0) if offsets[i][1] > offsets[i][0]
                     else np.zeros(E.shape[1]) for i in ids])
    bags /= np.linalg.norm(bags, axis=1, keepdims=True) + 1e-9
    np.savez(TB / "emb_l2bag.npz", l2bag=bags)
    print(f"embedded {len(flat)} claims from {len(rows)} traces; wrote claim_embs.npz + emb_l2bag.npz")


def chamfer_dists():
    z = np.load(TB / "claim_embs.npz")
    E, offsets = z["E"], json.loads(str(z["offsets"]))
    def d(a, b):
        (a0, a1), (b0, b1) = offsets[a], offsets[b]
        if a1 == a0 or b1 == b0:
            return 1.0
        S = E[a0:a1] @ E[b0:b1].T
        return float(1 - 0.5 * (S.max(1).mean() + S.max(0).mean()))
    return d


def split_para_problems():
    paras = [json.loads(l) for l in open(TB / "paraphrases.jsonl")]
    pqs = sorted({p["qidx"] for p in paras})
    rng = random.Random(0)
    rng.shuffle(pqs)
    return paras, set(pqs[:25]), set(pqs[25:])


def eval_l2():
    from sklearn.metrics import roc_auc_score
    paras, _, test_q = split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    d = chamfer_dists()
    d_neg_all = [d(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    print("== l2chamfer ==")
    by_style = collections.defaultdict(list)
    for p in paras:
        by_style[p["style"]].append(d(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}"))
    all_pos = [x for v in by_style.values() for x in v]
    print(f"  V1V2_auc (all {len(all_pos)} pos vs {len(d_neg_all)} neg): "
          f"{roc_auc_score([0]*len(all_pos)+[1]*len(d_neg_all), all_pos+d_neg_all):.3f}")
    for s, v in sorted(by_style.items()):
        print(f"  {s}: {roc_auc_score([0]*len(v)+[1]*len(d_neg_all), v+d_neg_all):.3f}", end="")
    print(f"\n  medians: para={np.median(all_pos):.3f} distinct={np.median(d_neg_all):.3f}")
    # held-out-problems view for comparability with eval_proj / l15
    neg_t = [d(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs if p["qidx"] in test_q]
    for s in sorted(by_style):
        v = [d(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in paras
             if p["qidx"] in test_q and p["style"] == s]
        print(f"  heldout {s[0]}: {roc_auc_score([0]*len(v)+[1]*len(neg_t), v+neg_t):.3f}", end="")
    print()


def train_l15():
    import torch
    ids, _ = load_texts()
    idx = {s: i for i, s in enumerate(ids)}
    paras, train_q, test_q = split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    from sklearn.metrics import roc_auc_score

    for base, npz in (("l1", "emb_l1.npz"), ("l4_L18", "emb_l4.npz")):
        torch.manual_seed(0)
        E = torch.tensor(np.load(TB / npz)[base]).float().cuda()
        by_q = collections.defaultdict(dict)
        for p in paras:
            if p["style"] != "D_restructured":
                by_q[p["qidx"]][p["style"]] = idx[f"p{p['qidx']}_{p['style']}"]
        pos = []
        for q in train_q:
            mem = [idx[f"n{q}_{[p['samp'] for p in paras if p['qidx']==q][0]}"]] + list(by_q[q].values())
            pos += [(a, b) for i, a in enumerate(mem) for b in mem[i + 1:]]
        neg_tr = [(idx[f"n{p['qidx']}_{p['samp_i']}"], idx[f"n{p['qidx']}_{p['samp_j']}"])
                  for p in negs if p["qidx"] in train_q]
        W = torch.nn.Linear(E.shape[1], 256, bias=False).cuda()
        opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)
        pi = torch.tensor(pos).cuda(); ni = torch.tensor(neg_tr).cuda()
        for step in range(400):
            f = torch.nn.functional.normalize(W(E), dim=-1)
            cp = (f[pi[:, 0]] * f[pi[:, 1]]).sum(-1)
            cn = (f[ni[:, 0]] * f[ni[:, 1]]).sum(-1)
            loss = (1 - cp).mean() + torch.relu(cn - 0.4).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            F = torch.nn.functional.normalize(W(E), dim=-1).cpu().numpy()
        np.savez(TB / f"emb_l15_{base}.npz", **{f"l15_{base}": F})
        torch.save(W.state_dict(), TB / f"head_l15_{base}.pt")  # else new texts can't be scored
        d = lambda a, b: float(1 - F[idx[a]] @ F[idx[b]])
        neg_t = [d(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs if p["qidx"] in test_q]
        print(f"== l15_{base} ==  (train pos={len(pos)} neg={len(neg_tr)}; final loss {loss.item():.3f})")
        for st in ("A_concise", "B_pedagogical", "C_casual", "D_restructured"):
            v = [d(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in paras
                 if p["qidx"] in test_q and p["style"] == st]
            print(f"  heldout {st[0]}: {roc_auc_score([0]*len(v)+[1]*len(neg_t), v+neg_t):.3f}", end="")
        print()


def eval_extra():
    # (1) ensemble kernel, (2) chamfer-vendi V4/V5, (3) within-problem subset test:
    # does subset Vendi predict coverage@4 beyond subset answer entropy? (stratified)
    import itertools
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    ids, _ = load_texts()
    idx = {s: i for i, s in enumerate(ids)}
    labels = {int(k): v for k, v in json.load(open(TB / "problem_labels.json")).items()}
    paras, _, test_q = split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    dch = chamfer_dists()
    F = np.load(TB / "emb_l15_l4_L18.npz")["l15_l4_L18"]
    dl15 = lambda a, b: float(1 - F[idx[a]] @ F[idx[b]])
    neg_pairs = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    zc = np.array([dch(*p) for p in neg_pairs]); zl = np.array([dl15(*p) for p in neg_pairs])
    dens = lambda a, b: 0.5 * ((dch(a, b) - zc.mean()) / zc.std() + (dl15(a, b) - zl.mean()) / zl.std())
    dn = [dens(*p) for p in neg_pairs]
    allpos = []
    print("== ensemble (l2chamfer + l15_l4_L18), full-set AUC ==")
    for st in ("A_concise", "B_pedagogical", "C_casual", "D_restructured"):
        v = [dens(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in paras if p["style"] == st]
        allpos += v
        print(f"  {st[0]}: {roc_auc_score([0]*len(v)+[1]*len(dn), v+dn):.3f}", end="")
    print(f"\n  overall: {roc_auc_score([0]*len(allpos)+[1]*len(dn), allpos+dn):.3f}")

    neutral = [r for r in (json.loads(l) for l in open(TB / "traces.jsonl")) if r["source"] == "neutral-k8"]
    byq = collections.defaultdict(list)
    for r in neutral:
        byq[r["qidx"]].append(r)
    for q in byq:
        byq[q].sort(key=lambda r: r["samp"])

    def ventropy(vals):
        c = collections.Counter(v for v in vals if v is not None); n = sum(c.values())
        return -sum(k / n * np.log(k / n) for k in c.values()) if n else 0.0

    rows = []
    for q, rs in byq.items():
        corr = [r["correct"] for r in rs]
        if not 0 < sum(corr) < 8:
            continue
        tids = [f"n{q}_{s}" for s in range(8)]
        S = np.eye(8)
        for i in range(8):
            for j in range(i + 1, 8):
                S[i, j] = S[j, i] = 1 - dch(tids[i], tids[j])
        cls = labels[q]["answer_class"]
        for sub in itertools.combinations(range(8), 4):
            K = S[np.ix_(sub, sub)] / 4
            lam = np.clip(np.linalg.eigvalsh(K), 0, None); lam = lam[lam > 1e-12]
            rows.append({"q": q, "vendi": float(np.exp(-(lam * np.log(lam)).sum())),
                         "aent": ventropy([cls[i] for i in sub]),
                         "cov": int(any(corr[i] for i in sub))})
    qs = sorted(set(r["q"] for r in rows))
    for key in ("vendi", "aent"):
        cs = []
        for q in qs:
            sub = [r for r in rows if r["q"] == q]
            if len(set(r["cov"] for r in sub)) < 2 or len(set(round(r[key], 6) for r in sub)) < 2:
                continue
            cs.append(spearmanr([r[key] for r in sub], [r["cov"] for r in sub]).statistic)
        print(f"within-problem spearman({key}, cov@4): mean {np.mean(cs):.3f} (n={len(cs)}, {np.mean(np.array(cs)>0):.0%} pos)")
    diffs = []
    for q in qs:
        sub = [r for r in rows if r["q"] == q]
        for a in sorted(set(round(r["aent"], 6) for r in sub)):
            s2 = [r for r in sub if round(r["aent"], 6) == a]
            if len(set(r["cov"] for r in s2)) < 2:
                continue
            rho = spearmanr([r["vendi"] for r in s2], [r["cov"] for r in s2]).statistic
            if not np.isnan(rho):
                diffs.append(rho)
    diffs = np.array(diffs)
    boot = [np.random.default_rng(i).choice(diffs, len(diffs)).mean() for i in range(2000)]
    print(f"vendi->cov within (problem x answer-entropy) strata: mean {diffs.mean():.3f} "
          f"n={len(diffs)} 95% CI [{np.percentile(boot, 2.5):.3f},{np.percentile(boot, 97.5):.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--embed-claims", action="store_true")
    ap.add_argument("--train-l15", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-extra", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(args), "time": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
              open(TB / "manifest_stage2.json", "w"), indent=1)
    if args.extract:
        extract(args)
    if args.embed_claims:
        embed_claims()
    if args.train_l15:
        train_l15()
    if args.eval:
        eval_l2()
    if args.eval_extra:
        eval_extra()


if __name__ == "__main__":
    main()
