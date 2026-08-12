#!/usr/bin/env python3
"""N3b: gameability (V6) in the TARGET regime — Qwen3-8B non-thinking, AIME+AMC, 2048 tok
(chosen by the 2026-08-12 regime probe: V_method 2.02, 72% >=2 methods, 18% >=4).

Differences vs the failed transfer test (scripts/26): method arm = gpt-5.5 JUDGE labels
(runs/regime-probe/method_labels_nonthink-hard.jsonl), never the broken distinct-answer
proxy — for eval pairs, set-level packs, AND head-training negatives. Styles: 13 rewrite
styles; train heads on A/B/C + E..J cliques, hold out D/K/L/M as the unseen attack pack.
Pack problems (>=4 judge methods) are excluded from head training.

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/29_n3b_ingame.py --paraphrase --smoke
  .venv/bin/python scripts/29_n3b_ingame.py --paraphrase   # ~$1.2
  .venv/bin/python scripts/29_n3b_ingame.py --claims       # ~$1.3
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/29_n3b_ingame.py --encode
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/29_n3b_ingame.py --retrain
  .venv/bin/python scripts/29_n3b_ingame.py --eval
"""
import argparse, collections, importlib.util, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s1 = _load("scripts/15_logdist_testbed.py", "s1")
s5 = _load("scripts/25_style_mining.py", "s5")
s6 = _load("scripts/23_logdist_seqot.py", "s6")

RP = Path("runs/regime-probe")
TB = Path("runs/logdist-testbed")
K, SEED, CHUNK = 8, 0, 128
ALL_STYLES = dict(list(s1.STYLES.items()) + list(s5.V2_STYLES.items()))  # A..M, 13
TRAIN_STYLES = s5.TRAIN_V1 + s5.TRAIN_V2                                 # A,B,C,E..J
PACK_STYLES = s5.HELDOUT_PACK                                            # D,K,L,M unseen


def traces():
    return [json.loads(l) for l in open(RP / "traces_nonthink-hard.jsonl")]


def judge():
    rows = [json.loads(l) for l in open(RP / "method_labels_nonthink-hard.jsonl")]
    return {r["qidx"]: r["assignment"][:K] for r in rows if r["rep"] == 0}


def judge_pairs():
    asg = judge()
    diff, same = [], []
    for q, a in asg.items():
        for i, j in combinations(range(K), 2):
            (diff if a[i] != a[j] else same).append((f"n{q}_{i}", f"n{q}_{j}"))
    return diff, same


def pick_bases(rows):
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qidx"]].append(r)
    picked = []
    for q in sorted(byq):
        pool = [r for r in byq[q] if r["correct"]] or \
               [r for r in byq[q] if r["extracted"] is not None] or byq[q]
        pool.sort(key=lambda r: abs(len(r["text"].split()) - 500))
        picked.append(pool[0])
    return picked


def method_packs():
    # 4 traces with 4 distinct judge methods (cluster reps = lowest samp per method)
    asg = judge()
    packs = {}
    for q, a in asg.items():
        rep = {}
        for s in range(K):
            rep.setdefault(a[s], s)
        if len(rep) >= 4:
            packs[q] = [f"n{q}_{s}" for s in sorted(rep.values())[:4]]
    return packs


def qsplit():
    # pack problems (>=4 judge methods) forced into TEST; rest split ~half
    packq = sorted(method_packs())
    rest = [q for q in judge() if q not in packq]
    random.Random(SEED).shuffle(rest)
    test_q = set(packq) | set(rest[:max(0, 25 - len(packq))])
    return set(judge()) - test_q, test_q


def paraphrase(args):
    from openai import OpenAI
    client = OpenAI()
    srcs = pick_bases(traces())
    styles = dict(list(ALL_STYLES.items())[:2]) if args.smoke else ALL_STYLES
    if args.smoke:
        srcs = srcs[:2]
    jobs = [(r, sk, sv) for r in srcs for sk, sv in styles.items()]
    est = sum(len(r["text"]) // 3 for r, _, _ in jobs) * 2 * (0.4 + 1.6) / 2 / 1e6
    print(f"jobs={len(jobs)}  est cost ~ ${est:.2f} ({args.model})")

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
                return {"qidx": r["qidx"], "samp": r["samp"], "style": sk,
                        "text": resp.choices[0].message.content or "",
                        "orig_extracted": r["extracted"],
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(one, jobs))
    with open(RP / ("paraphrases_smoke.jsonl" if args.smoke else "paraphrases.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    for r in out:  # signal-based grading -> main thread only
        try:
            r["boxed_ok"] = int(s1.boxed_match(r["text"], r["orig_extracted"]))
        except ValueError:
            r["boxed_ok"] = 0
    ptok, ctok = sum(r["prompt_tok"] for r in out), sum(r["compl_tok"] for r in out)
    print(f"wrote {len(out)} rewrites, boxed_ok={sum(r['boxed_ok'] for r in out)}/{len(out)}, "
          f"cost ~ ${(ptok * 0.4 + ctok * 1.6) / 1e6:.2f}")


def ids_texts():
    tr = traces()
    pa = [json.loads(l) for l in open(RP / "paraphrases.jsonl")]
    ids = [f"n{r['qidx']}_{r['samp']}" for r in tr] + [f"p{r['qidx']}_{r['style']}" for r in pa]
    return ids, [r["text"] for r in tr + pa], pa


def claims(args):
    from openai import OpenAI
    client = OpenAI()
    ids, texts, _ = ids_texts()
    est = (sum(len(t) // 3 + 200 for t in texts) * 0.4 + len(texts) * 400 * 1.6) / 1e6
    print(f"claims jobs={len(texts)}  est cost ~ ${est:.2f}")

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
        rows = list(pool.map(one, list(zip(ids, texts))))
    with open(RP / "claims.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote claims.jsonl: {len(rows)}, mean {np.mean([len(r['claims']) for r in rows]):.1f}/trace, "
          f"cost ~ ${(ptok * 0.4 + ctok * 1.6) / 1e6:.2f}")


def encode():
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    ids, texts, _ = ids_texts()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",
                                                 torch_dtype=torch.bfloat16).cuda().eval()
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
    np.savez(RP / "chunk_l4_L18.npz", E=E, offsets=json.dumps(offsets))
    np.savez(RP / "emb_l4_L18.npz", l4_L18=np.stack(trace), ids=np.array(ids))
    del model
    torch.cuda.empty_cache()

    rows = [json.loads(l) for l in open(RP / "claims.jsonl")]
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
    np.savez(RP / "claim_embs.npz", E=CE, offsets=json.dumps(coff))
    print(f"encoded {len(ids)} texts: {len(E)} chunks, {len(CE)} claims")


def retrain():
    # in-regime heads: positives = style cliques {base + 9 train styles} on TRAIN problems;
    # negatives = judge diff-method pairs (never the answer proxy). D/K/L/M + pack problems unseen.
    import torch
    train_q, test_q = qsplit()
    _, pa = pick_bases(traces()), None
    pa = [json.loads(l) for l in open(RP / "paraphrases.jsonl")]
    base = {r["qidx"]: r["samp"] for r in pa}
    diff, _ = judge_pairs()
    ids, _, _ = ids_texts()
    idx = {s: i for i, s in enumerate(ids)}

    def cliques(qs):
        pos = []
        for q in qs:
            mem = [f"n{q}_{base[q]}"] + [f"p{q}_{st}" for st in TRAIN_STYLES]
            pos += [(a, b) for i, a in enumerate(mem) for b in mem[i + 1:]]
        return pos

    pos = [(idx[a], idx[b]) for a, b in cliques(train_q)]
    neg = [(idx[a], idx[b]) for a, b in diff if int(a[1:].split("_")[0]) in train_q]
    print(f"in-regime heads: pos={len(pos)} neg={len(neg)} train_q={len(train_q)} "
          f"test_q={len(test_q)} (packs={len(method_packs())})")

    ET = np.load(RP / "emb_l4_L18.npz")["l4_L18"]
    torch.manual_seed(SEED)
    E = torch.tensor(ET).float().cuda()
    W = torch.nn.Linear(E.shape[1], 256, bias=False).cuda()
    opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)
    pi = torch.tensor(pos).cuda(); ni = torch.tensor(neg).cuda()
    for step in range(400):
        f = torch.nn.functional.normalize(W(E), dim=-1)
        cp = (f[pi[:, 0]] * f[pi[:, 1]]).sum(-1)
        cn = (f[ni[:, 0]] * f[ni[:, 1]]).sum(-1)
        loss = (1 - cp).mean() + torch.relu(cn - 0.4).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        F = torch.nn.functional.normalize(W(E), dim=-1).cpu().numpy()
    np.savez(RP / "emb_l15_inreg.npz", l15=F)
    torch.save(W.state_dict(), RP / "head_l15_inreg.pt")

    z = np.load(RP / "chunk_l4_L18.npz")
    offsets = {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    EC = torch.tensor(z["E"]).float().cuda()
    torch.manual_seed(SEED)
    W = torch.nn.Linear(EC.shape[1], 256, bias=False).cuda()
    opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)
    posn, negn = cliques(train_q), [(a, b) for a, b in diff
                                    if int(a[1:].split("_")[0]) in train_q]
    rng = np.random.default_rng(SEED)

    def cham_sim(F, a, b):
        (a0, a1), (b0, b1) = offsets[a], offsets[b]
        S = F[a0:a1] @ F[b0:b1].T
        return 0.5 * (S.max(1).values.mean() + S.max(0).values.mean())

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
    np.savez(RP / "chunk_l4_inreg_head.npz", E=F, offsets=json.dumps(
        {k: list(v) for k, v in offsets.items()}))
    torch.save(W.state_dict(), RP / "head_l4_inreg.pt")
    print("in-regime heads trained")


def evaluate():
    import torch
    from sklearn.metrics import roc_auc_score
    ids, _, pa = ids_texts()
    idx = {s: i for i, s in enumerate(ids)}
    base = {r["qidx"]: r["samp"] for r in pa}
    diff, same = judge_pairs()
    train_q, test_q = qsplit()
    ET = np.load(RP / "emb_l4_L18.npz")["l4_L18"]
    z = np.load(RP / "chunk_l4_L18.npz")
    EC, coff = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    zc = np.load(RP / "claim_embs.npz")
    CE, cloff = zc["E"], {k: tuple(v) for k, v in json.loads(str(zc["offsets"])).items()}

    def head(F, d, pt):
        W = torch.nn.Linear(4096, 256, bias=False)
        W.load_state_dict(torch.load(Path(d) / pt, map_location="cpu"))
        with torch.no_grad():
            return torch.nn.functional.normalize(W(torch.tensor(F).float()), dim=-1).numpy()

    comps = {
        "l15_v2": s6.memo(lambda a, b, F=head(ET, TB, "head_l15_l4_L18_v2.pt"):
                          float(1 - F[idx[a]] @ F[idx[b]])),
        "l4head_v2": s6.memo(lambda a, b, F=head(EC, TB, "head_l4_L18_v2.pt"):
                             s6.d_chamfer(F[slice(*coff[a])], F[slice(*coff[b])])),
        "claim:dtw": s6.memo(lambda a, b: s6.d_dtw(CE[slice(*cloff[a])], CE[slice(*cloff[b])])
                             if a in cloff and b in cloff else float("nan")),
        "raw_chunk": s6.memo(lambda a, b: s6.d_chamfer(EC[slice(*coff[a])], EC[slice(*coff[b])])),
    }
    if (RP / "emb_l15_inreg.npz").exists():
        F15r = np.load(RP / "emb_l15_inreg.npz")["l15"]
        zr = np.load(RP / "chunk_l4_inreg_head.npz")
        FCr, coffr = zr["E"], {k: tuple(v) for k, v in json.loads(str(zr["offsets"])).items()}
        comps["l15_inreg"] = s6.memo(lambda a, b: float(1 - F15r[idx[a]] @ F15r[idx[b]]))
        comps["l4_inreg"] = s6.memo(lambda a, b: s6.d_chamfer(FCr[slice(*coffr[a])],
                                                              FCr[slice(*coffr[b])]))
    m = dict(comps)
    for tag, ks in (("ENSv2-2way", ["l15_v2", "l4head_v2"]),
                    ("ENSv2-3way", ["claim:dtw", "l15_v2", "l4head_v2"]),
                    ("ENSinreg", ["l15_inreg", "l4_inreg"]),
                    ("ENSinreg3", ["claim:dtw", "l15_inreg", "l4_inreg"])):
        if not all(k in comps for k in ks):
            continue
        zs = {k: float(np.nanstd([comps[k](a, b) for a, b in diff])) for k in ks}
        m[tag] = s6.memo(lambda a, b, zs=zs: float(np.mean(
            [comps[k](a, b) / z for k, z in zs.items()])))

    # eval pairs restricted to TEST problems (train problems saw the in-regime heads)
    def teq(p):
        return int(p[0][1:].split("_")[0]) in test_q
    diff_t, same_t = [p for p in diff if teq(p)], [p for p in same if teq(p)]
    styles = list(ALL_STYLES)
    print(f"test problems n={len(test_q)}: {len(diff_t)} diff-method, {len(same_t)} "
          f"same-method judge pairs\n")
    print("AUC_D (same-vs-diff method) and per-style AUC_G (style-vs-diff-method); "
          "test problems only:")
    print(f"{'':11s}   AUC_D " + "".join(f"{st[:5]:>7s}" for st in styles) + "    ALLst")
    for mn, f in m.items():
        dd = [f(a, b) for a, b in diff_t]; dd = [x for x in dd if not np.isnan(x)]
        ds = [f(a, b) for a, b in same_t]; ds = [x for x in ds if not np.isnan(x)]
        aucd = roc_auc_score([0] * len(ds) + [1] * len(dd), ds + dd)
        row, allpos = [], []
        for st in styles:
            v = [f(f"n{q}_{base[q]}", f"p{q}_{st}") for q in sorted(test_q)]
            v = [x for x in v if not np.isnan(x)]
            allpos += v
            row.append(roc_auc_score([0] * len(v) + [1] * len(dd), v + dd))
        overall = roc_auc_score([0] * len(allpos) + [1] * len(dd), allpos + dd)
        print(f"{mn:11s}  {aucd:.3f} " + "".join(f"{x:7.3f}" for x in row) + f"{overall:9.3f}")

    packs = method_packs()
    print(f"\nset-level: {len(packs)} method packs (4 distinct judge methods) vs style packs "
          f"(4 unseen styles {'/'.join(s[0] for s in PACK_STYLES)}); tau/sd from judge "
          f"diff-method pairs:")
    print(f"{'metric':11s}  V_style  V_meth   R_set [95% CI]   hack%")
    rng = np.random.default_rng(20260812)
    res = {}
    for mn, f in m.items():
        pr = [f(a, b) for a, b in diff]
        tau = abs(np.nanmean(pr)) or 1.0
        vs, vm = [], []
        for q in sorted(packs):
            spack = [f"p{q}_{st}" for st in PACK_STYLES]
            for pack, acc in zip((spack, packs[q]), (vs, vm)):
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
        print(f"{mn:11s}   {vs.mean():.3f}   {vm.mean():.3f}   {r:5.2f} [{lo:4.2f},{hi:4.2f}]"
              f"   {100 / max(r, 1e-9):4.0f}%")
    (RP / "n3b_eval.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for f in ("paraphrase", "claims", "encode", "retrain", "eval"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "seed": SEED,
               "regime": "qwen3-8b nonthinking aime_and_amc 2048tok temp0.7 top_p0.95",
               "time": time.strftime("%FT%TZ", time.gmtime())},
              open(RP / "manifest_n3b.json", "w"), indent=1)
    if a.paraphrase:
        paraphrase(a)
    if a.claims:
        claims(a)
    if a.encode:
        encode()
    if a.retrain:
        retrain()
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
