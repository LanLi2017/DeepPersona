"""Stage 1 of logical-distance: L1/L4 kernels + battery (V1/V2 AUC, V4 known-groups, V5 regression).

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/16_logdist_stage1.py --encode l1
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/16_logdist_stage1.py --encode l1chunk
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/16_logdist_stage1.py --encode l4
  .venv/bin/python scripts/16_logdist_stage1.py --eval
"""
import argparse, collections, json, subprocess, time
from pathlib import Path

import numpy as np

TB = Path("runs/logdist-testbed")
L4_LAYERS = [9, 18, 27, 36]


def load_texts():
    ids, texts = [], []
    for r in (json.loads(l) for l in open(TB / "traces.jsonl")):
        if r["source"] == "neutral-k8":
            ids.append(f"n{r['qidx']}_{r['samp']}"); texts.append(r["text"])
    for r in (json.loads(l) for l in open(TB / "paraphrases.jsonl")):
        ids.append(f"p{r['qidx']}_{r['style']}"); texts.append(r["text"])
    return ids, texts


def encode_l1(texts, bs=8, maxlen=8192):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", torch_dtype=torch.bfloat16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state[:, -1]  # left pad -> last token
            out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    return {"l1": np.concatenate(out)}


def encode_l1chunk(texts, bs=32, chunk=384):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").cuda().eval()
    embs = []
    with torch.no_grad():
        for t in texts:
            toks = tok(t, truncation=False)["input_ids"][1:-1]
            chunks = [tok.decode(toks[j:j + chunk]) for j in range(0, max(len(toks), 1), chunk)]
            b = tok(chunks, padding=True, truncation=True, max_length=chunk + 2, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            e = (h * m).sum(1) / m.sum(1)
            embs.append(torch.nn.functional.normalize(e.mean(0), dim=-1).float().cpu().numpy())
    return {"l1chunk": np.stack(embs)}


def encode_l4(texts, bs=2, maxlen=6144):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    outs = {f"l4_L{k}": [] for k in L4_LAYERS}
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to("cuda")
            hs = model(**b, output_hidden_states=True).hidden_states
            m = b["attention_mask"].unsqueeze(-1)
            for k in L4_LAYERS:
                e = (hs[k] * m).sum(1) / m.sum(1)
                outs[f"l4_L{k}"].append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
            if i % 100 == 0:
                print(f"l4 {i}/{len(texts)}", flush=True)
    return {k: np.concatenate(v) for k, v in outs.items()}


def vendi(K):
    lam = np.linalg.eigvalsh(K / K.shape[0])
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def evaluate():
    from scipy.stats import mannwhitneyu, spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import chi2

    ids, _ = load_texts()
    idx = {s: i for i, s in enumerate(ids)}
    labels = {int(k): v for k, v in json.load(open(TB / "problem_labels.json")).items()}
    paras = [json.loads(l) for l in open(TB / "paraphrases.jsonl")]
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    sames = [json.loads(l) for l in open(TB / "pairs_same_answer.jsonl")]

    results = {}
    for f in sorted(TB.glob("emb_*.npz")):
        for name, E in np.load(f).items():
            d = lambda a, b: float(1 - E[idx[a]] @ E[idx[b]])
            d_pos = {p["style"]: [] for p in paras}
            for p in paras:
                d_pos[p["style"]].append(d(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}"))
            d_neg = [d(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
            d_same = [d(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in sames]
            all_pos = [x for v in d_pos.values() for x in v]
            auc = roc_auc_score([0] * len(all_pos) + [1] * len(d_neg), all_pos + d_neg)
            auc_style = {s: roc_auc_score([0] * len(v) + [1] * len(d_neg), v + d_neg) for s, v in d_pos.items()}

            vend = {}
            for q, l in labels.items():
                sub = E[[idx[f"n{q}_{s}"] for s in range(8)]]
                vend[q] = vendi(sub @ sub.T)
            v_basin = [vend[q] for q, l in labels.items() if l["flip_class"] == "template_basin"]
            v_scat = [vend[q] for q, l in labels.items() if l["flip_class"] == "scattered"]
            mwu = mannwhitneyu(v_basin, v_scat, alternative="less")

            qs = sorted(labels)
            y = np.array([labels[q]["coverage8"] for q in qs])
            ent = np.array([labels[q]["answer_entropy"] for q in qs])
            vv = np.array([vend[q] for q in qs])
            z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
            X0, X1 = z(ent).reshape(-1, 1), np.c_[z(ent), z(vv)]
            ll = {}
            for tag, X in (("base", X0), ("full", X1)):
                m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
                p = m.predict_proba(X)[:, 1]
                ll[tag] = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
            lr_p = float(chi2.sf(2 * (ll["full"] - ll["base"]), 1))

            results[name] = {
                "V1V2_auc": round(auc, 3),
                "V1V2_auc_by_style": {s: round(a, 3) for s, a in auc_style.items()},
                "d_median_para": round(float(np.median(all_pos)), 4),
                "d_median_same_answer": round(float(np.median(d_same)), 4),
                "d_median_distinct_answer": round(float(np.median(d_neg)), 4),
                "V4_vendi_basin_mean": round(float(np.mean(v_basin)), 3),
                "V4_vendi_scattered_mean": round(float(np.mean(v_scat)), 3),
                "V4_mwu_p": round(float(mwu.pvalue), 3),
                "V5_lr_p_vendi_given_entropy": round(lr_p, 4),
                "V5_spearman_vendi_entropy": round(float(spearmanr(vv, ent).statistic), 3),
                "V5_spearman_vendi_coverage": round(float(spearmanr(vv, y).statistic), 3),
            }
            print(f"\n== {name} ==")
            for k, v in results[name].items():
                print(f"  {k}: {v}")
    json.dump(results, open(TB / "stage1_metrics.json", "w"), indent=1)


def eval_proj():
    # style-subspace projection (L1.5 design a): fit on styles A/B/C x train problems,
    # eval on held-out style D x held-out problems
    import random
    from sklearn.metrics import roc_auc_score
    ids, _ = load_texts()
    idx = {s: i for i, s in enumerate(ids)}
    paras = [json.loads(l) for l in open(TB / "paraphrases.jsonl")]
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    pqs = sorted({p["qidx"] for p in paras})
    rng = random.Random(0)
    rng.shuffle(pqs)
    train_q, test_q = set(pqs[:25]), set(pqs[25:])

    for f in sorted(TB.glob("emb_*.npz")):
        for name, E in np.load(f).items():
            diffs = np.stack([E[idx[f"n{p['qidx']}_{p['samp']}"]] - E[idx[f"p{p['qidx']}_{p['style']}"]]
                              for p in paras if p["qidx"] in train_q and p["style"] != "D_restructured"])
            _, _, Vt = np.linalg.svd(diffs - diffs.mean(0), full_matrices=False)
            neg_test = [p for p in negs if p["qidx"] in test_q]
            d_neg_raw = None
            row = {}
            for k in (0, 5, 10, 20):
                P = E - (E @ Vt[:k].T) @ Vt[:k] if k else E
                P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
                d = lambda a, b: float(1 - P[idx[a]] @ P[idx[b]])
                d_neg = [d(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in neg_test]
                aucs = {}
                for st in ("A_concise", "B_pedagogical", "C_casual", "D_restructured"):
                    pos = [d(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}")
                           for p in paras if p["qidx"] in test_q and p["style"] == st]
                    aucs[st[0]] = round(roc_auc_score([0] * len(pos) + [1] * len(d_neg), pos + d_neg), 3)
                row[k] = aucs
            print(f"== {name} ==  (test: {len(test_q)} problems, {len(neg_test)} neg pairs)")
            for k, aucs in row.items():
                print(f"  k={k:>2}: {aucs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", choices=["l1", "l1chunk", "l4"])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-proj", action="store_true")
    args = ap.parse_args()
    if args.encode:
        ids, texts = load_texts()
        json.dump(ids, open(TB / "texts_index.json", "w"))
        t0 = time.time()
        embs = {"l1": encode_l1, "l1chunk": encode_l1chunk, "l4": encode_l4}[args.encode](texts)
        np.savez(TB / f"emb_{args.encode}.npz", **embs)
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        print(f"encoded {args.encode}: {len(texts)} texts in {time.time()-t0:.0f}s (git {sha})")
    if args.eval:
        evaluate()
    if args.eval_proj:
        eval_proj()


if __name__ == "__main__":
    main()
