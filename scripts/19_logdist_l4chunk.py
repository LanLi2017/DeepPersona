"""Stage 4 of logical-distance: L4 completed — LLM-free chunked hidden-state Chamfer kernel.

Question: is L2's win the LLM claim extraction, or just set-of-parts + Chamfer? If a trained
chunk kernel over Qwen3-8B hidden states matches L2, we have a $0 differentiable GRPO reward.

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/19_logdist_l4chunk.py --encode l4
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/19_logdist_l4chunk.py --encode mpnet
  .venv/bin/python scripts/19_logdist_l4chunk.py --eval            # raw chunk Chamfer battery
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/19_logdist_l4chunk.py --train    # contrastive head on chunk-Chamfer
  .venv/bin/python scripts/19_logdist_l4chunk.py --eval-ensemble
"""
import argparse, collections, importlib.util, json, subprocess, time
from pathlib import Path

import numpy as np

TB = Path("runs/logdist-testbed")
spec = importlib.util.spec_from_file_location("s2", "scripts/17_logdist_stage2.py")
s2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2)
CHUNK = 128  # visible text is only ~550 emb-tokens (gpt-5.5 hides reasoning tokens) -> ~4-5 chunks/trace


def encode_l4(texts, layer=18, bs=2, maxlen=6144):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to("cuda")
            h = model(**b, output_hidden_states=True).hidden_states[layer]
            for r in range(h.shape[0]):
                v = h[r][b["attention_mask"][r].bool()]
                cs = torch.stack([v[j:j + CHUNK].mean(0) for j in range(0, v.shape[0], CHUNK)])
                chunks.append(torch.nn.functional.normalize(cs, dim=-1).float().cpu().numpy())
            if i % 100 == 0:
                print(f"l4chunk {i}/{len(texts)}", flush=True)
    return chunks


def encode_mpnet(texts, bs=32):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").cuda().eval()
    chunks = []
    with torch.no_grad():
        for t in texts:
            toks = tok(t, truncation=False)["input_ids"][1:-1]
            parts = [tok.decode(toks[j:j + CHUNK]) for j in range(0, max(len(toks), 1), CHUNK)]
            b = tok(parts, padding=True, truncation=True, max_length=CHUNK + 2, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            e = (h * m).sum(1) / m.sum(1)
            chunks.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    return chunks


def save_chunks(name, ids, chunks):
    offsets, k = {}, 0
    for i, c in zip(ids, chunks):
        offsets[i] = (k, k + len(c)); k += len(c)
    np.savez(TB / f"chunk_{name}.npz", E=np.concatenate(chunks), offsets=json.dumps(offsets))
    n = [len(c) for c in chunks]
    print(f"wrote chunk_{name}.npz: {len(ids)} traces, chunks/trace mean={np.mean(n):.1f} max={max(n)}")


def chamfer_of(npz):
    z = np.load(TB / npz)
    E, offsets = z["E"], json.loads(str(z["offsets"]))
    def d(a, b):
        (a0, a1), (b0, b1) = offsets[a], offsets[b]
        S = E[a0:a1] @ E[b0:b1].T
        return float(1 - 0.5 * (S.max(1).mean() + S.max(0).mean()))
    return d


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


def train(base):
    import torch
    z = np.load(TB / f"chunk_{base}.npz")
    E = torch.tensor(z["E"]).cuda()
    offsets = json.loads(str(z["offsets"]))
    paras, train_q, test_q = s2.split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]

    import random as pyrandom
    from sklearn.metrics import roc_auc_score
    tq = sorted(train_q)
    pyrandom.Random(1).shuffle(tq)
    fit_q, val_q = set(tq[5:]), set(tq[:5])

    def clique_pairs(qs):
        by_q = collections.defaultdict(dict)
        for p in paras:
            if p["style"] != "D_restructured":
                by_q[p["qidx"]][p["style"]] = f"p{p['qidx']}_{p['style']}"
        pos = []
        for q in qs:
            mem = [f"n{q}_{[p['samp'] for p in paras if p['qidx']==q][0]}"] + list(by_q[q].values())
            pos += [(a, b) for i, a in enumerate(mem) for b in mem[i + 1:]]
        return pos

    pos = clique_pairs(fit_q)
    pos_val = clique_pairs(val_q)
    neg_tr = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
              for p in negs if p["qidx"] in fit_q]
    neg_val = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
               for p in negs if p["qidx"] in val_q]

    W = torch.nn.Linear(E.shape[1], 256, bias=False).cuda()
    opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)

    def cham_sim(F, a, b):
        (a0, a1), (b0, b1) = offsets[a], offsets[b]
        S = F[a0:a1] @ F[b0:b1].T
        return 0.5 * (S.max(1).values.mean() + S.max(0).values.mean())

    best = (-1, None)
    for step in range(601):
        F = torch.nn.functional.normalize(W(E), dim=-1)
        if step % 25 == 0:
            with torch.no_grad():
                dv = [1 - cham_sim(F, *p).item() for p in pos_val] + \
                     [1 - cham_sim(F, *p).item() for p in neg_val]
                auc = roc_auc_score([0] * len(pos_val) + [1] * len(neg_val), dv)
            if auc > best[0]:
                best = (auc, step, {k: v.clone() for k, v in W.state_dict().items()})
            if step % 100 == 0:
                print(f"step {step} val_auc {auc:.3f}", flush=True)
        bp = [pos[i] for i in rng.choice(len(pos), 48)]
        bn = [neg_tr[i] for i in rng.choice(len(neg_tr), 48)]
        sp = torch.stack([cham_sim(F, *p) for p in bp])
        sn = torch.stack([cham_sim(F, *p) for p in bn])
        loss = (1 - sp).mean() + torch.relu(sn - 0.4).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    W.load_state_dict(best[2])
    print(f"early stop: step {best[1]} val_auc {best[0]:.3f}")
    with torch.no_grad():
        F = torch.nn.functional.normalize(W(E), dim=-1).cpu().numpy()
    np.savez(TB / f"chunk_{base}_head.npz", E=F, offsets=json.dumps(offsets))
    torch.save(W.state_dict(), TB / f"head_{base}.pt")
    print(f"== l4chunk-head[{base}] ==  (train pos={len(pos)} neg={len(neg_tr)}; final loss {loss.item():.3f})")
    d = chamfer_of(f"chunk_{base}_head.npz")
    from sklearn.metrics import roc_auc_score
    neg_t = [d(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs if p["qidx"] in test_q]
    for st in ("A_concise", "B_pedagogical", "C_casual", "D_restructured"):
        v = [d(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in paras
             if p["qidx"] in test_q and p["style"] == st]
        print(f"  heldout {st[0]}: {roc_auc_score([0]*len(v)+[1]*len(neg_t), v+neg_t):.3f}", end="")
    print()


def eval_ensemble():
    # z-normed (on neg pairs) combos: adopted 2-way reference vs $0 LLM-free candidates
    from sklearn.metrics import roc_auc_score
    ids, _ = s2.load_texts()
    idx = {s: i for i, s in enumerate(ids)}
    paras, _, test_q = s2.split_para_problems()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    Fl15 = np.load(TB / "emb_l15_l4_L18.npz")["l15_l4_L18"]
    parts = {"l2": s2.chamfer_dists(),
             "l15": lambda a, b: float(1 - Fl15[idx[a]] @ Fl15[idx[b]]),
             "l4ch": chamfer_of("chunk_l4_L18_head.npz")}
    neg_pairs = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    zs = {k: (np.array([f(*p) for p in neg_pairs]).mean(), np.array([f(*p) for p in neg_pairs]).std())
          for k, f in parts.items()}
    neg_pairs_t = [(a, b) for (a, b), p in zip(neg_pairs, negs) if p["qidx"] in test_q]
    for combo in (("l2", "l15"), ("l15", "l4ch"), ("l2", "l4ch"), ("l2", "l15", "l4ch"), ("l4ch",)):
        dens = lambda a, b: np.mean([(parts[k](a, b) - zs[k][0]) / zs[k][1] for k in combo])
        for tag, ps, nps in (("full", paras, neg_pairs),
                             ("heldout", [p for p in paras if p["qidx"] in test_q], neg_pairs_t)):
            dn = [dens(*p) for p in nps]
            allpos, line = [], []
            for st in ("A_concise", "B_pedagogical", "C_casual", "D_restructured"):
                v = [dens(f"n{p['qidx']}_{p['samp']}", f"p{p['qidx']}_{p['style']}") for p in ps if p["style"] == st]
                allpos += v
                line.append(f"{st[0]} {roc_auc_score([0]*len(v)+[1]*len(dn), v+dn):.3f}")
            print(f"{'+'.join(combo):<14} {tag:<7} overall "
                  f"{roc_auc_score([0]*len(allpos)+[1]*len(dn), allpos+dn):.3f}  " + "  ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", choices=["l4", "mpnet"])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-ensemble", action="store_true")
    args = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(args), "chunk": CHUNK,
               "time": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
              open(TB / "manifest_stage4.json", "w"), indent=1)
    if args.encode:
        ids, texts = s2.load_texts()
        t0 = time.time()
        if args.encode == "l4":
            save_chunks("l4_L18", ids, encode_l4(texts))
        else:
            save_chunks("mpnet", ids, encode_mpnet(texts))
        print(f"encode {args.encode}: {time.time()-t0:.0f}s")
    if args.eval:
        for name in ("l4_L18", "mpnet"):
            if (TB / f"chunk_{name}.npz").exists():
                battery(chamfer_of(f"chunk_{name}.npz"), f"l4chunk_{name}_raw")
    if args.train:
        for base in ("l4_L18", "mpnet"):
            if (TB / f"chunk_{base}.npz").exists():
                train(base)
    if args.eval_ensemble:
        eval_ensemble()


if __name__ == "__main__":
    main()
