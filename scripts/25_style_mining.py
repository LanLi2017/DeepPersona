#!/usr/bin/env python3
"""N2: adversarial style mining. 9 new rewrite styles of the same 50 base traces --
6 train styles (targeting the V6b weak spots: order, naming, register, structure) and
3 held-out attack styles never seen in training. Retrain both heads on the expanded
positive cliques, then re-run the V6 acceptance test with UNSEEN-style packs {D,K,L,M},
comparing the v1 (adopted) vs v2 (hardened) metric. Target: R_set >= 4-5 on unseen styles.

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/25_style_mining.py --generate --smoke
  .venv/bin/python scripts/25_style_mining.py --generate          # ~$1.8
  .venv/bin/python scripts/25_style_mining.py --extract           # ~$0.7
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/25_style_mining.py --encode
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/25_style_mining.py --retrain
  .venv/bin/python scripts/25_style_mining.py --eval
"""
import argparse, collections, importlib.util, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

TB = Path("runs/logdist-testbed")
CHUNK = 128


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s1 = _load("scripts/15_logdist_testbed.py", "s1")
s2 = _load("scripts/17_logdist_stage2.py", "s2")
s6 = _load("scripts/23_logdist_seqot.py", "s6")

# train styles isolate single attack axes; held-out styles compound them
V2_STYLES = {
    "E_reordered": "Style: reorder the presentation of the steps as freely as coherence allows — present later steps first with forward references, interleave case discussions, state key intermediate results before their derivations. Every step, computed quantity, and the final \\boxed{} answer must still appear. Keep the wording of each individual step mostly intact.",
    "F_renamed": "Style: rename every variable and symbol consistently (e.g. n->k, x->t, a_i->c_i), and switch notational conventions where exactly equivalent (summation notation <-> expanded sums, fractions <-> ratios). Keep step order, prose structure, and logic identical.",
    "G_formal": "Style: rigid formal write-up — Lemma/Claim/Proof structure with numbered steps and minimal prose. Same derivation, every step kept.",
    "H_dialogue": "Style: a dialogue between a tutor asking short guiding questions and a student answering; all mathematical content appears in the student's answers, nothing added or dropped.",
    "I_answerfirst": "Style: state the final answer in the first sentence; then justify it by presenting the derivation in roughly reverse order (goal, what it required, then the base computations). All steps and values preserved.",
    "J_symbolic": "Style: compress all prose into terse symbolic chains (=>, therefore, :=) with minimal connective words; every equation, case, and the final answer kept.",
    "K_compound": "Style: simultaneously restructure the presentation (answer and key claims first), rename all variables consistently, and use a casual first-person register. All steps and computed quantities preserved.",
    "L_maxlex": "Style: rewrite so that no sentence shares more than two content words with the corresponding original sentence; reorder the presentation freely; but the mathematical method, every intermediate value, and the final \\boxed{} answer must be exactly preserved.",
    "M_narrative": "Style: retell the derivation as a first-person narrative of discovery ('at first I computed..., which gave..., so then...'), reordering freely and paraphrasing heavily, while preserving every mathematical step and value actually used.",
}
TRAIN_V2 = ("E_reordered", "F_renamed", "G_formal", "H_dialogue", "I_answerfirst", "J_symbolic")
HELDOUT_V2 = ("K_compound", "L_maxlex", "M_narrative")
HELDOUT_PACK = ("D_restructured", "K_compound", "L_maxlex", "M_narrative")  # unseen-style attack pack
TRAIN_V1 = ("A_concise", "B_pedagogical", "C_casual")


def v2_rows():
    return [json.loads(l) for l in open(TB / "paraphrases_v2.jsonl")]


def v2_ids_texts():
    rows = v2_rows()
    return [f"p{r['qidx']}_{r['style']}" for r in rows], [r["text"] for r in rows]


def generate(args):
    from openai import OpenAI
    client = OpenAI()
    paras = [json.loads(l) for l in open(TB / "paraphrases.jsonl")]
    base = {p["qidx"]: p["samp"] for p in paras}
    neutral = {(r["qidx"], r["samp"]): r for r in (json.loads(l) for l in open(TB / "traces.jsonl"))
               if r["source"] == "neutral-k8"}
    srcs = [neutral[(q, s)] for q, s in sorted(base.items())]
    styles = dict(list(V2_STYLES.items())[:2]) if args.smoke else V2_STYLES
    if args.smoke:
        srcs = srcs[:2]
    jobs = [(r, sk, sv) for r in srcs for sk, sv in styles.items()]
    est = (sum(r["compl_tok"] + 300 for r, _, _ in jobs) * 0.4
           + sum(r["compl_tok"] for r, _, _ in jobs) * 1.6) / 1e6
    print(f"jobs={len(jobs)}  est cost ≈ ${est:.2f} ({args.model})")

    def one(job):
        r, sk, sv = job
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": s1.PARA_SYS},
                              {"role": "user", "content": f"{sv}\n\nSOLUTION TO REWRITE:\n{r['text']}"}],
                    max_completion_tokens=8000, temperature=0.7)
                u = resp.usage
                text = resp.choices[0].message.content or ""
                return {"qidx": r["qidx"], "samp": r["samp"], "style": sk, "text": text,
                        "orig_extracted": r["extracted"],
                        "para_extracted": s1.extract_boxed(text),
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    for r in rows:  # signal-based grading -> main thread only
        r["boxed_ok"] = int(s1.boxed_match(r["text"], r["orig_extracted"]))
    fn = TB / ("paraphrases_v2_smoke.jsonl" if args.smoke else "paraphrases_v2.jsonl")
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote {fn}: {len(rows)} rewrites, boxed_ok={sum(r['boxed_ok'] for r in rows)}/{len(rows)}, "
          f"actual cost ≈ ${(ptok * 0.4 + ctok * 1.6) / 1e6:.2f}")
    print("per-style boxed_ok:", dict(collections.Counter((r["style"], r["boxed_ok"]) for r in rows)))
    if args.smoke:
        for r in rows:
            print(f"\n-- p{r['qidx']}_{r['style']} --\n{r['text'][:500]}")


def extract(args):
    from openai import OpenAI
    client = OpenAI()
    ids, texts = v2_ids_texts()
    est = (sum(len(t) // 3 + 200 for t in texts) * 0.4 + len(texts) * 400 * 1.6) / 1e6
    print(f"extract jobs={len(ids)}  est cost ≈ ${est:.2f}")

    def one(job):
        tid, text = job
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": s2.EXTRACT_SYS},
                              {"role": "user", "content": text}],
                    max_completion_tokens=2000, temperature=0.0)
                claims = json.loads(r.choices[0].message.content).get("claims", [])
                u = r.usage
                return {"id": tid, "claims": claims, "prompt_tok": u.prompt_tokens,
                        "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, zip(ids, texts)))
    with open(TB / "claims_v2.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote claims_v2.jsonl: {len(rows)} traces, "
          f"claims/trace mean={np.mean([len(r['claims']) for r in rows]):.1f}, "
          f"cost ≈ ${(ptok * 0.4 + ctok * 1.6) / 1e6:.2f}")


def encode():
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    ids, texts = v2_ids_texts()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    chunks, trace = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 2):
            b = tok(texts[i:i + 2], padding=True, truncation=True, max_length=6144,
                    return_tensors="pt").to("cuda")
            hs = model(**b, output_hidden_states=True).hidden_states[18]
            for r in range(hs.shape[0]):
                v = hs[r][b["attention_mask"][r].bool()]
                cs = torch.stack([v[j:j + CHUNK].mean(0) for j in range(0, v.shape[0], CHUNK)])
                chunks.append(torch.nn.functional.normalize(cs, dim=-1).float().cpu().numpy())
                trace.append(torch.nn.functional.normalize(v.mean(0), dim=-1).float().cpu().numpy())
            if i % 50 == 0:
                print(f"l4 {i}/{len(texts)}", flush=True)
    E, offsets, k = np.concatenate(chunks), {}, 0
    for tid, c in zip(ids, chunks):
        offsets[tid] = (k, k + len(c)); k += len(c)
    np.savez(TB / "chunk_v2_l4_L18.npz", E=E, offsets=json.dumps(offsets))
    np.savez(TB / "emb_v2_l4_L18.npz", l4_L18=np.stack(trace))
    del model
    torch.cuda.empty_cache()

    rows = [json.loads(l) for l in open(TB / "claims_v2.jsonl")]
    flat = [(r["id"], c) for r in rows for c in r["claims"]]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
    emodel = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(flat), 64):
            b = tok([c for _, c in flat[i:i + 64]], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to("cuda")
            h = emodel(**b).last_hidden_state[:, -1]
            embs.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    CE, coff, k = np.concatenate(embs), {}, 0
    for r in rows:
        coff[r["id"]] = (k, k + len(r["claims"])); k += len(r["claims"])
    np.savez(TB / "claim_v2_embs.npz", E=CE, offsets=json.dumps(coff))
    print(f"encoded {len(ids)} v2 texts: {len(E)} chunks, {len(CE)} claims")


def all_ids_and_banks():
    ids1, _ = s2.load_texts()
    ids2, _ = v2_ids_texts()
    E1 = np.load(TB / "emb_l4.npz")["l4_L18"]
    E1 = E1 / np.linalg.norm(E1, axis=1, keepdims=True)
    E2 = np.load(TB / "emb_v2_l4_L18.npz")["l4_L18"]
    return ids1 + ids2, np.concatenate([E1, E2])


def clique_pairs(paras_all, base, qs, styles):
    pos = []
    for q in qs:
        mem = [f"n{q}_{base[q]}"] + [f"p{q}_{st}" for st in styles
                                     if any(p["qidx"] == q and p["style"] == st for p in paras_all)]
        pos += [(a, b) for i, a in enumerate(mem) for b in mem[i + 1:]]
    return pos


def retrain():
    import torch
    from sklearn.metrics import roc_auc_score
    ids_all, E_tr = all_ids_and_banks()
    idx = {s: i for i, s in enumerate(ids_all)}
    paras, train_q, test_q = s2.split_para_problems()
    base = {p["qidx"]: p["samp"] for p in paras}
    paras_all = paras + v2_rows()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    styles_tr = TRAIN_V1 + TRAIN_V2

    # ---- l15 trace head (mirror scripts/17 train_l15, expanded positives) --------
    torch.manual_seed(0)
    E = torch.tensor(E_tr).float().cuda()
    pos = [(idx[a], idx[b]) for a, b in clique_pairs(paras_all, base, train_q, styles_tr)]
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
    np.savez(TB / "emb_l15_l4_L18_v2.npz", l15_l4_L18=F)
    torch.save(W.state_dict(), TB / "head_l15_l4_L18_v2.pt")
    print(f"l15-v2: pos={len(pos)} neg={len(neg_tr)} final loss {loss.item():.3f}")

    # ---- l4 chunk head (mirror scripts/19 train, expanded positives) -------------
    za, zb = np.load(TB / "chunk_l4_L18.npz"), np.load(TB / "chunk_v2_l4_L18.npz")
    offsets = json.loads(str(za["offsets"]))
    n0 = len(za["E"])
    offsets.update({k: (a + n0, b + n0) for k, (a, b) in json.loads(str(zb["offsets"])).items()})
    offsets = {k: tuple(v) for k, v in offsets.items()}
    EC = torch.tensor(np.concatenate([za["E"], zb["E"]])).cuda()
    import random as pyrandom
    tq = sorted(train_q)
    pyrandom.Random(1).shuffle(tq)
    fit_q, val_q = set(tq[5:]), set(tq[:5])
    pos = clique_pairs(paras_all, base, fit_q, styles_tr)
    pos_val = clique_pairs(paras_all, base, val_q, styles_tr)
    neg_fit = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
               for p in negs if p["qidx"] in fit_q]
    neg_val = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
               for p in negs if p["qidx"] in val_q]
    torch.manual_seed(0)
    W = torch.nn.Linear(EC.shape[1], 256, bias=False).cuda()
    opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)

    def cham_sim(F, a, b):
        (a0, a1), (b0, b1) = offsets[a], offsets[b]
        S = F[a0:a1] @ F[b0:b1].T
        return 0.5 * (S.max(1).values.mean() + S.max(0).values.mean())

    best = (-1, None, None)
    for step in range(601):
        F = torch.nn.functional.normalize(W(EC.float()), dim=-1)
        if step % 25 == 0:
            with torch.no_grad():
                dv = [1 - cham_sim(F, *p).item() for p in pos_val] + \
                     [1 - cham_sim(F, *p).item() for p in neg_val]
                auc = roc_auc_score([0] * len(pos_val) + [1] * len(neg_val), dv)
            if auc > best[0]:
                best = (auc, step, {k: v.clone() for k, v in W.state_dict().items()})
        bp = [pos[i] for i in rng.choice(len(pos), 48)]
        bn = [neg_fit[i] for i in rng.choice(len(neg_fit), 48)]
        sp = torch.stack([cham_sim(F, *p) for p in bp])
        sn = torch.stack([cham_sim(F, *p) for p in bn])
        loss = (1 - sp).mean() + torch.relu(sn - 0.4).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    W.load_state_dict(best[2])
    print(f"l4chunk-v2: early stop step {best[1]} val_auc {best[0]:.3f} "
          f"(pos={len(pos)} neg={len(neg_fit)})")
    with torch.no_grad():
        F = torch.nn.functional.normalize(W(EC.float()), dim=-1).cpu().numpy()
    np.savez(TB / "chunk_l4_L18_v2_head.npz", E=F, offsets=json.dumps(
        {k: list(v) for k, v in offsets.items()}))
    torch.save(W.state_dict(), TB / "head_l4_L18_v2.pt")


def evaluate():
    import torch
    from sklearn.metrics import roc_auc_score
    ids_all, E_tr = all_ids_and_banks()
    idx = {s: i for i, s in enumerate(ids_all)}
    paras, train_q, test_q = s2.split_para_problems()
    base = {p["qidx"]: p["samp"] for p in paras}
    paras_all = paras + v2_rows()
    negs = [json.loads(l) for l in open(TB / "pairs_distinct_answer.jsonl")]
    npair = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]

    # v1 metric applied to ALL texts via the saved v1 heads (task #1 payoff)
    W1 = torch.nn.Linear(4096, 256, bias=False)
    W1.load_state_dict(torch.load(TB / "head_l15_l4_L18.pt", map_location="cpu"))
    with torch.no_grad():
        F1 = torch.nn.functional.normalize(W1(torch.tensor(E_tr).float()), dim=-1).numpy()
    F2 = np.load(TB / "emb_l15_l4_L18_v2.npz")["l15_l4_L18"]

    Wc = torch.nn.Linear(4096, 256, bias=False)
    Wc.load_state_dict(torch.load(TB / "head_l4_L18.pt", map_location="cpu"))
    za, zb = np.load(TB / "chunk_l4_L18.npz"), np.load(TB / "chunk_v2_l4_L18.npz")
    coff = json.loads(str(za["offsets"]))
    n0 = len(za["E"])
    coff.update({k: (a + n0, b + n0) for k, (a, b) in json.loads(str(zb["offsets"])).items()})
    coff = {k: tuple(v) for k, v in coff.items()}
    with torch.no_grad():
        FC1 = torch.nn.functional.normalize(
            Wc(torch.tensor(np.concatenate([za["E"], zb["E"]])).float()), dim=-1).numpy()
    z2 = np.load(TB / "chunk_l4_L18_v2_head.npz")
    FC2, coff2 = z2["E"], {k: tuple(v) for k, v in json.loads(str(z2["offsets"])).items()}
    CE, cloff = s6.bank("claim_embs.npz", None)
    zc2 = np.load(TB / "claim_v2_embs.npz")
    m0 = len(CE)
    CE = np.concatenate([CE, zc2["E"]])
    cloff.update({k: (a + m0, b + m0) for k, (a, b) in json.loads(str(zc2["offsets"])).items()})

    def mk(F, off=None, fun=None, tracevec=False):
        if tracevec:
            return s6.memo(lambda a, b: float(1 - F[idx[a]] @ F[idx[b]]))
        return s6.memo(lambda a, b: fun(F[slice(*off[a])], F[slice(*off[b])]))

    comps = {
        "claim:dtw": mk(CE, cloff, s6.d_dtw),
        "l15_v1": mk(F1, tracevec=True), "l15_v2": mk(F2, tracevec=True),
        "l4head_v1": mk(FC1, coff, s6.d_chamfer), "l4head_v2": mk(FC2, coff2, s6.d_chamfer),
    }
    m = {}
    for tag, ks in (("ENSv1", ["claim:dtw", "l15_v1", "l4head_v1"]),
                    ("ENSv2", ["claim:dtw", "l15_v2", "l4head_v2"]),
                    ("ENSv2-2way", ["l15_v2", "l4head_v2"])):
        zs = {k: float(np.std([comps[k](a, b) for a, b in npair])) for k in ks}
        m[tag] = s6.memo(lambda a, b, zs=zs: float(np.mean(
            [comps[k](a, b) / z for k, z in zs.items()])))
    m.update(comps)

    # ---- AUC per style, held-out problems ----------------------------------------
    neg_ho = [(a, b) for (a, b), p in zip(npair, negs) if p["qidx"] in test_q]
    all_styles = TRAIN_V1 + ("D_restructured",) + TRAIN_V2 + HELDOUT_V2
    print(f"{'':14s}" + "".join(f"{st[:6]:>8s}" for st in all_styles) + "   (heldout-problem AUC; * = unseen style)")
    for mn, f in m.items():
        dn = [f(a, b) for a, b in neg_ho]
        row = []
        for st in all_styles:
            v = [f(f"n{p['qidx']}_{base[p['qidx']]}", f"p{p['qidx']}_{st}") for p in paras_all
                 if p["qidx"] in test_q and p["style"] == st]
            row.append(roc_auc_score([0] * len(v) + [1] * len(dn), v + dn) if v else float("nan"))
        print(f"{mn:14s}" + "".join(f"{x:8.3f}" for x in row))
    print("unseen styles: D_restructured (v1 heldout), K_compound, L_maxlex, M_narrative")

    # ---- V6 set-level with UNSEEN-style packs ------------------------------------
    traces = [json.loads(l) for l in open(TB / "traces.jsonl")]
    ans = {(t["qidx"], t["samp"]): t["extracted"] for t in traces if t["source"] == "neutral-k8"}
    matched = sorted({p["qidx"] for p in negs} & set(base))
    packs = {}
    for q in matched:
        by_a = collections.defaultdict(list)
        for s in range(8):
            by_a[ans.get((q, s))].append(s)
        reps = [v[0] for v in by_a.values()]
        if len(reps) < 4:
            continue
        packs[q] = {"style_v1": [f"p{q}_{st}" for st in s6.STYLES],
                    "style_unseen": [f"p{q}_{st}" for st in HELDOUT_PACK],
                    "method": [f"n{q}_{s}" for s in sorted(reps)[:4]]}
    print(f"\nset-level, {len(packs)} packs: R_set = (V_method-1)/(V_stylepack-1); hack% = 100/R")
    print(f"{'metric':10s} {'pack':12s}  V_style  V_meth   R_set   hack%")
    res = {}
    rng = np.random.default_rng(20260805)
    for mn in ("ENSv1", "ENSv2", "ENSv2-2way", "claim:dtw", "l15_v1", "l15_v2", "l4head_v1", "l4head_v2"):
        f = m[mn]
        tau = abs(np.mean([f(a, b) for a, b in npair])) or 1.0
        for pk in ("style_v1", "style_unseen"):
            vs, vm = [], []
            for q in sorted(packs):
                for key, acc in ((pk, vs), ("method", vm)):
                    ids4 = packs[q][key]
                    D = np.zeros((4, 4))
                    for i, j in combinations(range(4), 2):
                        D[i, j] = D[j, i] = f(ids4[i], ids4[j])
                    acc.append(s6.vendi(D, tau))
            vs, vm = np.array(vs), np.array(vm)
            r = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9)
            bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
                  for i in rng.integers(0, len(vs), (2000, len(vs)))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[f"{mn}|{pk}"] = {"R": r, "ci": [lo, hi]}
            print(f"{mn:10s} {pk:12s}   {vs.mean():.3f}   {vm.mean():.3f}   "
                  f"{r:5.2f} [{lo:4.2f},{hi:4.2f}]   {100 / max(r, 1e-9):4.0f}%")
    (TB / "n2_eval.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "time": time.strftime("%FT%TZ", time.gmtime()),
               "train_styles": TRAIN_V1 + TRAIN_V2, "heldout_styles": HELDOUT_PACK},
              open(TB / "manifest_n2.json", "w"), indent=1)
    if a.generate:
        generate(a)
    if a.extract:
        extract(a)
    if a.encode:
        encode()
    if a.retrain:
        retrain()
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
