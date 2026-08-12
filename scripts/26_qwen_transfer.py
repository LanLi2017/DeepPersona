#!/usr/bin/env python3
"""Transfer testbed: do the v2 heads (trained on gpt-5.5 BeyondAIME traces) still separate
method from style, ZERO-SHOT, on the trace distribution N4/GRPO would actually produce --
Qwen3-8B non-thinking, MATH levels 4-5, 1024-token cap (the F1 regime)?

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/26_qwen_transfer.py --generate
  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/26_qwen_transfer.py --paraphrase   # ~$0.6
  .venv/bin/python scripts/26_qwen_transfer.py --claims       # ~$0.8
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/26_qwen_transfer.py --encode
  .venv/bin/python scripts/26_qwen_transfer.py --eval
"""
import argparse, collections, importlib.util, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from deeppersona.math_grade import extract_boxed, grade_answer, run_with_timeout_signal
from deeppersona.personas import system_message

QT = Path("runs/logdist-qwen")
CHUNK = 128
K = 8
NPROB = 100
NPARA = 50
SEED = 0


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s1 = _load("scripts/15_logdist_testbed.py", "s1")
s5 = _load("scripts/25_style_mining.py", "s5")
s6 = _load("scripts/23_logdist_seqot.py", "s6")

# 7 eval-only styles: v1 four (incl. unseen D) + the three N2 held-out compound attacks
STYLES7 = dict(list(s1.STYLES.items()) + [(k, s5.V2_STYLES[k]) for k in s5.HELDOUT_V2])
PACK_STYLES = s5.HELDOUT_PACK  # {D, K, L, M} -- all unseen by the v2 heads


def boxed_or_none(text):
    try:
        return extract_boxed(text or "")
    except ValueError:
        return None


def load_problems():
    from datasets import load_dataset
    test_problems = {r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    pieces = []
    for cfg in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"):
        for split in ("train", "test"):
            pieces.append(load_dataset("EleutherAI/hendrycks_math", name=cfg, split=split))
    items = []
    for ds in pieces:
        for r in ds:
            lvl = "".join(ch for ch in str(r.get("level", "")) if ch.isdigit())
            if lvl not in ("4", "5") or r["problem"] in test_problems:
                continue
            gold = boxed_or_none(r["solution"])
            if gold is None:
                continue
            items.append({"problem": r["problem"], "gold": gold, "level": lvl})
    random.Random(SEED).shuffle(items)
    return items[:NPROB]


def generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    QT.mkdir(exist_ok=True)
    items = load_problems()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    sysmsg = system_message("math", -1, 0, "basic")
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": it["problem"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for it in items]
    jobs = [(q, s) for q in range(len(items)) for s in range(K)]
    texts = {}
    BS = 16
    torch.manual_seed(SEED)
    t0 = time.time()
    for i in range(0, len(jobs), BS):
        batch = jobs[i:i + BS]
        b = tok([prompts[q] for q, _ in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**b, max_new_tokens=1024, do_sample=True, temperature=0.7,
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
        for (q, s), seq, plen in zip(batch, out, b["attention_mask"].sum(1)):
            texts[(q, s)] = tok.decode(seq[b["input_ids"].shape[1]:], skip_special_tokens=True)
        if i % (BS * 5) == 0:
            print(f"gen {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    rows = []
    for (q, s), text in sorted(texts.items()):
        ext = boxed_or_none(text)
        ok = bool(run_with_timeout_signal(grade_answer, args=(ext, items[q]["gold"]),
                                          timeout_seconds=2)) if ext else False
        rows.append({"qidx": q, "samp": s, "text": text, "extracted": ext,
                     "gold": items[q]["gold"], "correct": int(ok), "level": items[q]["level"],
                     "problem": items[q]["problem"]})
    with open(QT / "traces.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # distinct-answer pairs + labels, mirroring the gpt-5.5 testbed schema
    pairs, labels = [], {}
    for q in range(len(items)):
        rs = [r for r in rows if r["qidx"] == q]
        cls = s1.canon_classes([r["extracted"] for r in rs])
        for i, j in combinations(range(K), 2):
            if cls[i] != cls[j] and rs[i]["extracted"] is not None and rs[j]["extracted"] is not None:
                pairs.append({"qidx": q, "samp_i": i, "samp_j": j})
        labels[q] = {"gold": items[q]["gold"], "answers": [r["extracted"] for r in rs],
                     "answer_class": cls, "n_unique": len(set(cls)),
                     "ncorr": sum(r["correct"] for r in rs)}
    with open(QT / "pairs_distinct_answer.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    json.dump(labels, open(QT / "problem_labels.json", "w"))
    acc = np.mean([r["correct"] for r in rows])
    nu = [labels[q]["n_unique"] for q in labels]
    print(f"wrote {len(rows)} traces: acc {acc:.3f}, mean n_unique {np.mean(nu):.2f}, "
          f"{len(pairs)} distinct-answer pairs over {len({p['qidx'] for p in pairs})} problems, "
          f"{sum(1 for q in labels if labels[q]['n_unique'] >= 4)} problems with >=4 unique")


def pick_bases(rows):
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qidx"]].append(r)
    rng = random.Random(SEED)
    labels = {int(k): v for k, v in json.load(open(QT / "problem_labels.json")).items()}
    qs = sorted(byq)
    rng.shuffle(qs)
    # set-level packs need >=4 distinct answers AND paraphrases -- force those problems in
    qs.sort(key=lambda q: -(labels[q]["n_unique"] >= 4))
    picked = []
    for q in qs[:NPARA]:
        pool = [r for r in byq[q] if r["correct"]] or \
               [r for r in byq[q] if r["extracted"] is not None] or byq[q]
        pool.sort(key=lambda r: abs(len(r["text"].split()) - 500))  # mid-length
        picked.append(pool[0])
    return picked


def paraphrase(args):
    from openai import OpenAI
    client = OpenAI()
    rows = [json.loads(l) for l in open(QT / "traces.jsonl")]
    srcs = pick_bases(rows)
    jobs = [(r, sk, sv) for r in srcs for sk, sv in STYLES7.items()]
    est = sum(len(r["text"]) // 3 for r, _, _ in jobs) * (0.4 + 1.6) / 1e6
    print(f"jobs={len(jobs)}  est cost ≈ ${est:.2f} ({args.model})")

    def one(job):
        r, sk, sv = job
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": s1.PARA_SYS},
                              {"role": "user", "content": f"{sv}\n\nSOLUTION TO REWRITE:\n{r['text']}"}],
                    max_completion_tokens=4000, temperature=0.7)
                u = resp.usage
                text = resp.choices[0].message.content or ""
                return {"qidx": r["qidx"], "samp": r["samp"], "style": sk, "text": text,
                        "orig_extracted": r["extracted"], "para_extracted": boxed_or_none(text),
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(one, jobs))
    with open(QT / "paraphrases_raw.jsonl", "w") as f:  # persist BEFORE grading can crash
        for r in out:
            f.write(json.dumps(r) + "\n")
    for r in out:  # signal-based grading -> main thread only
        try:
            r["boxed_ok"] = int(s1.boxed_match(r["text"], r["orig_extracted"]))
        except ValueError:
            r["boxed_ok"] = 0
    with open(QT / "paraphrases.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in out), sum(r["compl_tok"] for r in out)
    print(f"wrote {len(out)} rewrites, boxed_ok={sum(r['boxed_ok'] for r in out)}/{len(out)}, "
          f"actual cost ≈ ${(ptok * 0.4 + ctok * 1.6) / 1e6:.2f}")


def qids_texts():
    tr = [json.loads(l) for l in open(QT / "traces.jsonl")]
    pa = [json.loads(l) for l in open(QT / "paraphrases.jsonl")]
    ids = [f"n{r['qidx']}_{r['samp']}" for r in tr] + [f"p{r['qidx']}_{r['style']}" for r in pa]
    return ids, [r["text"] for r in tr + pa], pa


def claims(args):
    from openai import OpenAI
    client = OpenAI()
    ids, texts, pa = qids_texts()
    para_q = {r["qidx"] for r in pa}
    keep = [(i, t) for i, t in zip(ids, texts)
            if i.startswith("p") or int(i[1:].split("_")[0]) in para_q]
    est = (sum(len(t) // 3 + 200 for _, t in keep) * 0.4 + len(keep) * 400 * 1.6) / 1e6
    print(f"claims jobs={len(keep)}  est cost ≈ ${est:.2f}")

    def one(job):
        tid, text = job
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": s5.s2.EXTRACT_SYS},
                              {"role": "user", "content": text}],
                    max_completion_tokens=2000, temperature=0.0)
                cl = json.loads(r.choices[0].message.content).get("claims", [])
                u = r.usage
                return {"id": tid, "claims": cl, "prompt_tok": u.prompt_tokens,
                        "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, keep))
    with open(QT / "claims.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote claims.jsonl: {len(rows)}, mean {np.mean([len(r['claims']) for r in rows]):.1f}/trace, "
          f"cost ≈ ${(ptok * 0.4 + ctok * 1.6) / 1e6:.2f}")


def encode():
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    ids, texts, _ = qids_texts()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    chunks, trace = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 4):
            b = tok(texts[i:i + 4], padding=True, truncation=True, max_length=6144,
                    return_tensors="pt").to("cuda")
            hs = model(**b, output_hidden_states=True).hidden_states[18]
            for r in range(hs.shape[0]):
                v = hs[r][b["attention_mask"][r].bool()]
                cs = torch.stack([v[j:j + CHUNK].mean(0) for j in range(0, v.shape[0], CHUNK)])
                chunks.append(torch.nn.functional.normalize(cs, dim=-1).float().cpu().numpy())
                trace.append(torch.nn.functional.normalize(v.mean(0), dim=-1).float().cpu().numpy())
            if i % 100 == 0:
                print(f"l4 {i}/{len(texts)}", flush=True)
    E, offsets, k = np.concatenate(chunks), {}, 0
    for tid, c in zip(ids, chunks):
        offsets[tid] = (k, k + len(c)); k += len(c)
    np.savez(QT / "chunk_l4_L18.npz", E=E, offsets=json.dumps(offsets))
    np.savez(QT / "emb_l4_L18.npz", l4_L18=np.stack(trace), ids=np.array(ids))
    del model
    torch.cuda.empty_cache()

    rows = [json.loads(l) for l in open(QT / "claims.jsonl")]
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
    np.savez(QT / "claim_embs.npz", E=CE, offsets=json.dumps(coff))
    print(f"encoded {len(ids)} texts: {len(E)} chunks, {len(CE)} claims")


def evaluate():
    import torch
    from sklearn.metrics import roc_auc_score
    ids, _, pa = qids_texts()
    idx = {s: i for i, s in enumerate(ids)}
    ET = np.load(QT / "emb_l4_L18.npz")["l4_L18"]
    z = np.load(QT / "chunk_l4_L18.npz")
    EC, coff = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    zc = np.load(QT / "claim_embs.npz")
    CE, cloff = zc["E"], {k: tuple(v) for k, v in json.loads(str(zc["offsets"])).items()}

    def head(F, pt):
        W = torch.nn.Linear(4096, 256, bias=False)
        W.load_state_dict(torch.load(Path("runs/logdist-testbed") / pt, map_location="cpu"))
        with torch.no_grad():
            return torch.nn.functional.normalize(W(torch.tensor(F).float()), dim=-1).numpy()

    comps = {
        "l15_v1": s6.memo(lambda a, b, F=head(ET, "head_l15_l4_L18.pt"):
                          float(1 - F[idx[a]] @ F[idx[b]])),
        "l15_v2": s6.memo(lambda a, b, F=head(ET, "head_l15_l4_L18_v2.pt"):
                          float(1 - F[idx[a]] @ F[idx[b]])),
        "l4head_v1": s6.memo(lambda a, b, F=head(EC, "head_l4_L18.pt"):
                             s6.d_chamfer(F[slice(*coff[a])], F[slice(*coff[b])])),
        "l4head_v2": s6.memo(lambda a, b, F=head(EC, "head_l4_L18_v2.pt"):
                             s6.d_chamfer(F[slice(*coff[a])], F[slice(*coff[b])])),
        "claim:dtw": s6.memo(lambda a, b: s6.d_dtw(CE[slice(*cloff[a])], CE[slice(*cloff[b])])
                             if a in cloff and b in cloff else float("nan")),
        "raw_chunk": s6.memo(lambda a, b: s6.d_chamfer(EC[slice(*coff[a])], EC[slice(*coff[b])])),
    }
    negs = [json.loads(l) for l in open(QT / "pairs_distinct_answer.jsonl")]
    npair = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    m = dict(comps)
    for tag, ks in (("ENSv2-2way", ["l15_v2", "l4head_v2"]),
                    ("ENSv2-3way", ["claim:dtw", "l15_v2", "l4head_v2"]),
                    ("ENSv1-2way", ["l15_v1", "l4head_v1"])):
        para_q = {r["qidx"] for r in pa}
        np_ok = [(a, b) for a, b in npair
                 if "claim:dtw" not in ks or int(a[1:].split("_")[0]) in para_q]
        zs = {k: float(np.std([comps[k](a, b) for a, b in np_ok])) for k in ks}
        m[tag] = s6.memo(lambda a, b, zs=zs: float(np.mean(
            [comps[k](a, b) / z for k, z in zs.items()])))

    base = {r["qidx"]: r["samp"] for r in pa}
    para_q = sorted(base)
    neg_pq = [(a, b) for (a, b), p in zip(npair, negs) if p["qidx"] in base]
    styles = list(STYLES7)
    print(f"{'':12s}" + "".join(f"{st[:6]:>8s}" for st in styles) + "     ALL   (AUC on Qwen traces, zero-shot)")
    for mn, f in m.items():
        dn = [f(a, b) for a, b in (neg_pq if "claim" in mn or "3way" in mn else npair)]
        dn = [x for x in dn if not np.isnan(x)]
        row, allpos = [], []
        for st in styles:
            v = [f(f"n{q}_{base[q]}", f"p{q}_{st}") for q in para_q]
            v = [x for x in v if not np.isnan(x)]
            allpos += v
            row.append(roc_auc_score([0] * len(v) + [1] * len(dn), v + dn))
        overall = roc_auc_score([0] * len(allpos) + [1] * len(dn), allpos + dn)
        print(f"{mn:12s}" + "".join(f"{x:8.3f}" for x in row) + f"{overall:8.3f}")

    labels = {int(k): v for k, v in json.load(open(QT / "problem_labels.json")).items()}
    packs = {}
    for q in para_q:
        cls = labels[q]["answer_class"]
        by_a = collections.defaultdict(list)
        for s in range(K):
            by_a[cls[s]].append(s)
        reps = sorted(v[0] for v in by_a.values())
        if len(reps) < 4:
            continue
        packs[q] = ([f"p{q}_{st}" for st in PACK_STYLES], [f"n{q}_{s}" for s in reps[:4]])
    print(f"\nset-level, {len(packs)} packs (style pack = 4 UNSEEN styles D/K/L/M):")
    print(f"{'metric':12s}  V_style  V_meth   R_set [95% CI]   hack%")
    rng = np.random.default_rng(20260806)
    res = {}
    for mn in ("ENSv1-2way", "ENSv2-2way", "ENSv2-3way", "l15_v2", "l4head_v2", "raw_chunk"):
        f = m[mn]
        pr = [f(a, b) for a, b in (neg_pq if "3way" in mn else npair)]
        tau = abs(np.nanmean(pr)) or 1.0
        vs, vm = [], []
        for q in sorted(packs):
            for pack, acc in zip(packs[q], (vs, vm)):
                D = np.zeros((4, 4))
                for i, j in combinations(range(4), 2):
                    D[i, j] = D[j, i] = f(pack[i], pack[j])
                acc.append(s6.vendi(D, tau))
        vs, vm = np.array(vs), np.array(vm)
        r = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9)
        bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
              for i in rng.integers(0, len(vs), (2000, len(vs)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[mn] = {"R": r, "ci": [lo, hi], "V_style": vs.mean(), "V_meth": vm.mean()}
        print(f"{mn:12s}   {vs.mean():.3f}   {vm.mean():.3f}   {r:5.2f} [{lo:4.2f},{hi:4.2f}]"
              f"   {100 / max(r, 1e-9):4.0f}%")
    (QT / "transfer_eval.json").write_text(json.dumps(res, indent=1))


def qsplit():
    # 25/25 split of the para problems; all >=4-unique (pack) problems forced into TEST so
    # the set-level probe stays clean of head training
    pa = [json.loads(l) for l in open(QT / "paraphrases.jsonl")]
    labels = {int(k): v for k, v in json.load(open(QT / "problem_labels.json")).items()}
    pqs = sorted({r["qidx"] for r in pa})
    packq = [q for q in pqs if labels[q]["n_unique"] >= 4]
    rest = [q for q in pqs if q not in packq]
    random.Random(SEED).shuffle(rest)
    test_q = set(packq) | set(rest[:25 - len(packq)])
    return pa, set(pqs) - test_q, test_q


def retrain():
    # in-domain heads: train positives = {base, A, B, C} cliques on train problems only;
    # D/K/L/M styles and all pack problems stay unseen
    import torch
    from sklearn.metrics import roc_auc_score
    pa, train_q, test_q = qsplit()
    base = {r["qidx"]: r["samp"] for r in pa}
    negs = [json.loads(l) for l in open(QT / "pairs_distinct_answer.jsonl")]
    ids, _, _ = qids_texts()
    idx = {s: i for i, s in enumerate(ids)}
    ET = np.load(QT / "emb_l4_L18.npz")["l4_L18"]

    def cliques(qs):
        pos = []
        for q in qs:
            mem = [f"n{q}_{base[q]}"] + [f"p{q}_{st}" for st in
                                         ("A_concise", "B_pedagogical", "C_casual")]
            pos += [(a, b) for i, a in enumerate(mem) for b in mem[i + 1:]]
        return pos

    torch.manual_seed(SEED)
    E = torch.tensor(ET).float().cuda()
    pos = [(idx[a], idx[b]) for a, b in cliques(train_q)]
    neg_tr = [(idx[f"n{p['qidx']}_{p['samp_i']}"], idx[f"n{p['qidx']}_{p['samp_j']}"])
              for p in negs if p["qidx"] in train_q]
    print(f"in-domain l15: pos={len(pos)} neg={len(neg_tr)} (train_q={len(train_q)})")
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
    np.savez(QT / "emb_l15_indom.npz", l15=F)
    torch.save(W.state_dict(), QT / "head_l15_indom.pt")

    z = np.load(QT / "chunk_l4_L18.npz")
    offsets = {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    EC = torch.tensor(z["E"]).float().cuda()
    torch.manual_seed(SEED)
    W = torch.nn.Linear(EC.shape[1], 256, bias=False).cuda()
    opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)
    posn = cliques(train_q)
    rng = np.random.default_rng(SEED)

    def cham_sim(F, a, b):
        (a0, a1), (b0, b1) = offsets[a], offsets[b]
        S = F[a0:a1] @ F[b0:b1].T
        return 0.5 * (S.max(1).values.mean() + S.max(0).values.mean())

    negn = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}")
            for p in negs if p["qidx"] in train_q]
    for step in range(300):
        F = torch.nn.functional.normalize(W(EC), dim=-1)
        bp = [posn[i] for i in rng.choice(len(posn), 48)]
        bn = [negn[i] for i in rng.choice(len(negn), min(48, len(negn)))]
        sp = torch.stack([cham_sim(F, *p) for p in bp])
        sn = torch.stack([cham_sim(F, *p) for p in bn])
        loss = (1 - sp).mean() + torch.relu(sn - 0.4).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        F = torch.nn.functional.normalize(W(EC), dim=-1).cpu().numpy()
    np.savez(QT / "chunk_l4_indom_head.npz", E=F, offsets=json.dumps(
        {k: list(v) for k, v in offsets.items()}))
    torch.save(W.state_dict(), QT / "head_l4_indom.pt")
    print(f"in-domain l4chunk trained ({len(posn)} pos cliques)")


def eval_indom():
    from sklearn.metrics import roc_auc_score
    pa, train_q, test_q = qsplit()
    base = {r["qidx"]: r["samp"] for r in pa}
    negs = [json.loads(l) for l in open(QT / "pairs_distinct_answer.jsonl")]
    ids, _, _ = qids_texts()
    idx = {s: i for i, s in enumerate(ids)}
    F15 = np.load(QT / "emb_l15_indom.npz")["l15"]
    z = np.load(QT / "chunk_l4_indom_head.npz")
    FC, coff = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    comps = {
        "l15_in": s6.memo(lambda a, b: float(1 - F15[idx[a]] @ F15[idx[b]])),
        "l4_in": s6.memo(lambda a, b: s6.d_chamfer(FC[slice(*coff[a])], FC[slice(*coff[b])])),
    }
    npair = [(f"n{p['qidx']}_{p['samp_i']}", f"n{p['qidx']}_{p['samp_j']}") for p in negs]
    zs = {k: float(np.std([comps[k](a, b) for a, b in npair])) for k in comps}
    m = dict(comps)
    m["ENS-in"] = s6.memo(lambda a, b: float(np.mean([comps[k](a, b) / zs[k] for k in zs])))

    neg_ho = [(a, b) for (a, b), p in zip(npair, negs) if p["qidx"] in test_q]
    styles = list(STYLES7)
    print(f"{'':10s}" + "".join(f"{st[:6]:>8s}" for st in styles)
          + "     ALL   (held-out-problem AUC, in-domain heads)")
    for mn, f in m.items():
        dn = [f(a, b) for a, b in neg_ho]
        row, allpos = [], []
        for st in styles:
            v = [f(f"n{q}_{base[q]}", f"p{q}_{st}") for q in sorted(test_q)]
            allpos += v
            row.append(roc_auc_score([0] * len(v) + [1] * len(dn), v + dn))
        overall = roc_auc_score([0] * len(allpos) + [1] * len(dn), allpos + dn)
        print(f"{mn:10s}" + "".join(f"{x:8.3f}" for x in row) + f"{overall:8.3f}")

    labels = {int(k): v for k, v in json.load(open(QT / "problem_labels.json")).items()}
    packs = {}
    for q in sorted(base):
        if labels[q]["n_unique"] < 4:
            continue
        by_a = collections.defaultdict(list)
        for s in range(K):
            by_a[labels[q]["answer_class"][s]].append(s)
        reps = sorted(v[0] for v in by_a.values())
        packs[q] = ([f"p{q}_{st}" for st in PACK_STYLES], [f"n{q}_{s}" for s in reps[:4]])
    print(f"\nset-level, {len(packs)} packs (unseen styles D/K/L/M; pack problems unseen in training):")
    rng = np.random.default_rng(20260806)
    for mn, f in m.items():
        tau = abs(np.mean([f(a, b) for a, b in npair])) or 1.0
        vs, vm = [], []
        for q in sorted(packs):
            for pack, acc in zip(packs[q], (vs, vm)):
                D = np.zeros((4, 4))
                for i, j in combinations(range(4), 2):
                    D[i, j] = D[j, i] = f(pack[i], pack[j])
                acc.append(s6.vendi(D, tau))
        vs, vm = np.array(vs), np.array(vm)
        r = (vm.mean() - 1) / max(vs.mean() - 1, 1e-9)
        bs = [(vm[i].mean() - 1) / max(vs[i].mean() - 1, 1e-9)
              for i in rng.integers(0, len(vs), (2000, len(vs)))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{mn:10s}  V_style {vs.mean():.3f}  V_meth {vm.mean():.3f}  "
              f"R {r:5.2f} [{lo:4.2f},{hi:4.2f}]  hack {100 / max(r, 1e-9):4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    for f in ("generate", "paraphrase", "claims", "encode", "eval", "retrain", "eval-indom"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    a = ap.parse_args()
    QT.mkdir(exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "time": time.strftime("%FT%TZ", time.gmtime()),
               "regime": "qwen3-8b nonthinking MATH L4-5 1024tok temp0.7 top_p0.95 (F1)",
               "seed": SEED}, open(QT / "manifest.json", "w"), indent=1)
    if a.generate:
        generate()
    if a.paraphrase:
        paraphrase(a)
    if a.claims:
        claims(a)
    if a.encode:
        encode()
    if a.eval:
        evaluate()
    if a.retrain:
        retrain()
    if getattr(a, "eval_indom"):
        eval_indom()


if __name__ == "__main__":
    main()
