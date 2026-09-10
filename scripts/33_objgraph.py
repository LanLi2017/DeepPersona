#!/usr/bin/env python3
"""G2: verifiable math-object derivation DAG (approach C from 2026-08-18 brainstorm).

Nodes = canonicalized mathematical objects the trace actually derives on its path to the
final answer (variables renamed by first appearance, sympy-normalized locally); edges =
direct derivation dependencies. Style invariance by SEMANTICS: meaning-preserving rewrites
preserve the math objects. Falsifier-first (--probe): on 12 problems, do judge-distinct
methods separate from same-method pairs in object space, and do style rewrites stay put?
Kill if AUC_D <= ~0.65 or style pairs ~= diff pairs.

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/33_objgraph.py --extract --probe --smoke
  .venv/bin/python scripts/33_objgraph.py --extract --probe          # ~$0.5
  .venv/bin/python scripts/33_objgraph.py --extract --probe --pass2  # reliability
  .venv/bin/python scripts/33_objgraph.py --analyze --probe
"""
import argparse, collections, importlib.util, json, re, signal, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import numpy as np


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s9 = _load("scripts/29_n3b_ingame.py", "s9")
RP = Path("runs/regime-probe")

OBJ_SYS = """You are given a mathematical solution. Extract the mathematical OBJECTS it derives on its logical path to the final answer.

An object is a concrete mathematical statement the solution establishes: an equation, an expression with its value, a defined quantity, a counted set size, a bound. NOT narrative steps, NOT restatements of the problem, NOT abandoned attempts.

Canonical form rules (CRITICAL — identical math in different words must give identical objects):
- Rename solution-introduced variables in order of first appearance: a, b, c, ... Keep variables named by the PROBLEM as-is.
- Write each object as one short equation/expression, sympy-parseable where possible (use *, **, /, ==). No prose. <= 60 chars.
- Exact values (fractions, not decimals). One object per distinct fact; no duplicates.
- 4-15 objects covering the essential derivation; the LAST object must be the final answer as `ans == <value>`.

Edges: [i, j] (0-based) when object j is derived DIRECTLY using object i. Only direct dependencies.
Output JSON: {"objects": ["...", ...], "edges": [[i, j], ...]}"""


def probe_qs():
    packs = sorted(s9.method_packs())
    asg = s9.judge()
    two = [q for q in sorted(asg) if q not in packs and 2 <= len(set(asg[q])) <= 3]
    return packs + two[:3]


def probe_ids(qs):
    ids = {f"n{q}_{s}" for q in qs for s in range(s9.K)}
    ids |= {f"p{q}_{st}" for q in qs for st in s9.PACK_STYLES}
    return ids


def objs_file(tag):
    return RP / f"objgraphs_{tag}.jsonl"


def extract(args):
    from openai import OpenAI
    client = OpenAI()
    ids, texts, _ = s9.ids_texts()
    tag = "b" if args.pass2 else "a"
    jobs = list(zip(ids, texts))
    if args.probe:
        keep = probe_ids(probe_qs())
        if args.pass2:  # reliability: traces only, 3 problems' worth
            keep = {f"n{q}_{s}" for q in probe_qs()[:3] for s in range(s9.K)}
        jobs = [j for j in jobs if j[0] in keep]
    if args.smoke:
        jobs = jobs[:3]
    fn = RP / f"objgraphs_{tag}_smoke.jsonl" if args.smoke else objs_file(tag)
    if fn.exists() and not args.smoke:  # incremental: skip ids already extracted
        done = {json.loads(l)["id"] for l in open(fn)}
        jobs = [j for j in jobs if j[0] not in done]
    est = (sum(len(t) // 3 + 600 for _, t in jobs) * 0.4 + len(jobs) * 400 * 1.6) / 1e6
    print(f"obj jobs={len(jobs)} tag={tag}  est cost ~ ${est:.2f} ({args.model})")

    def one(job):
        tid, text = job
        for attempt in range(4):
            try:
                # temp bumps on retry so a deterministic mid-JSON truncation can't loop
                r = client.chat.completions.create(
                    model=args.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": OBJ_SYS},
                              {"role": "user", "content": f"SOLUTION:\n{text}"}],
                    max_completion_tokens=2500,
                    temperature=(0.0 if tag == "a" else 0.7) + 0.3 * min(attempt, 1))
                out = json.loads(r.choices[0].message.content)
                objs = [str(o)[:80] for o in out.get("objects", [])][:20]
                n = len(objs)
                edges = sorted({(i, j) for i, j in out.get("edges", [])
                                if isinstance(i, int) and isinstance(j, int)
                                and 0 <= i < n and 0 <= j < n and i != j})
                u = r.usage
                return {"id": tid, "objects": objs, "edges": [list(e) for e in edges],
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception as e:
                if attempt == 3:
                    print(f"FAILED {tid}: {e}", flush=True)
                    return {"id": tid, "objects": [], "edges": [], "failed": True,
                            "prompt_tok": 0, "compl_tok": 0}
                time.sleep(2 ** attempt * 3)

    rows = []
    with ThreadPoolExecutor(8) as pool, open(fn, "a") as f:
        for r in pool.map(one, jobs):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    no = [len(r["objects"]) for r in rows]
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote {fn}: {len(rows)} graphs, objects mean={np.mean(no):.1f}, "
          f"cost ~ ${(ptok * 0.4 + ctok * 1.6) / 1e6:.2f}")
    if args.smoke:
        for r in rows:
            print(f"\n-- {r['id']} edges={r['edges']}")
            for i, o in enumerate(r["objects"]):
                print(f"   {i}. {o}")


class _TO(Exception):
    pass


def _alarm(*a):
    raise _TO()


def canon(o):
    # tier 2: sympy canonicalization (main thread only, alarm timeout); tier 1 fallback
    s = re.sub(r"\s+", "", o).lower().replace("==", "=")
    try:
        import sympy
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(2)
        try:
            if "=" in s:
                l, r = s.split("=", 1)
                e = sympy.expand(sympy.sympify(l) - sympy.sympify(r))
                out = "eq0:" + sympy.srepr(e)
            else:
                out = sympy.srepr(sympy.expand(sympy.sympify(s)))
        finally:
            signal.alarm(0)
        return out
    except Exception:
        return s


def load_objs(tag):
    rows = {r["id"]: r for r in (json.loads(l) for l in open(objs_file(tag)))}
    for r in rows.values():
        r["canon"] = [canon(o) for o in r["objects"]]
        # ancestor filter: objects with a path to the last object (final answer)
        n = len(r["objects"])
        if n and r["edges"]:
            anc, stack = {n - 1}, [n - 1]
            pred = collections.defaultdict(list)
            for i, j in r["edges"]:
                pred[j].append(i)
            while stack:
                u = stack.pop()
                for v in pred[u]:
                    if v not in anc:
                        anc.add(v); stack.append(v)
            r["anc"] = sorted(anc) if len(anc) > 1 else list(range(n))
        else:
            r["anc"] = list(range(n))
    return rows


def d_jac(ra, rb, anc=True):
    A = set(np.array(ra["canon"])[ra["anc"]]) if anc else set(ra["canon"])
    B = set(np.array(rb["canon"])[rb["anc"]]) if anc else set(rb["canon"])
    if not A or not B:
        return float("nan")
    return 1 - len(A & B) / len(A | B)


def d_soft(ra, rb, anc=True):
    A = [ra["objects"][i] for i in (ra["anc"] if anc else range(len(ra["objects"])))]
    B = [rb["objects"][i] for i in (rb["anc"] if anc else range(len(rb["objects"])))]
    if not A or not B:
        return float("nan")
    S = np.array([[SequenceMatcher(None, a, b).ratio() for b in B] for a in A])
    return 1 - 0.5 * (S.max(1).mean() + S.max(0).mean())


def analyze(args):
    from sklearn.metrics import roc_auc_score
    O = load_objs("a")
    qs = probe_qs()
    asg = s9.judge()
    diff, same = [], []
    for q in qs:
        for i, j in combinations(range(s9.K), 2):
            a, b = f"n{q}_{i}", f"n{q}_{j}"
            if a in O and b in O:
                (diff if asg[q][i] != asg[q][j] else same).append((a, b))
    pa = [json.loads(l) for l in open(RP / "paraphrases.jsonl")]
    base = {r["qidx"]: r["samp"] for r in pa}
    stypairs = [(f"n{q}_{base[q]}", f"p{q}_{st}") for q in qs for st in s9.PACK_STYLES
                if f"p{q}_{st}" in O]
    ms = {"jac": d_jac, "soft": d_soft,
          "jac_nof": lambda a, b: d_jac(a, b, anc=False),
          "soft_nof": lambda a, b: d_soft(a, b, anc=False)}
    print(f"probe: {len(qs)} problems, {len(diff)} diff / {len(same)} same judge pairs, "
          f"{len(stypairs)} style pairs")
    print(f"{'metric':9s}  {'d_same':>6s} {'d_diff':>6s} {'d_style':>7s}  "
          f"{'AUC_D':>6s} {'AUC_G':>6s}")
    for mn, f in ms.items():
        dd = [f(O[a], O[b]) for a, b in diff]
        ds = [f(O[a], O[b]) for a, b in same]
        dg = [f(O[a], O[b]) for a, b in stypairs]
        dd, ds, dg = [[x for x in v if not np.isnan(x)] for v in (dd, ds, dg)]
        aucd = roc_auc_score([0] * len(ds) + [1] * len(dd), ds + dd)
        aucg = roc_auc_score([0] * len(dg) + [1] * len(dd), dg + dd)
        print(f"{mn:9s}  {np.mean(ds):6.3f} {np.mean(dd):6.3f} {np.mean(dg):7.3f}  "
              f"{aucd:6.3f} {aucg:6.3f}")
    if objs_file("b").exists():
        Ob = load_objs("b")
        common = sorted(set(O) & set(Ob))
        for mn, f in (("jac", d_jac), ("soft", d_soft)):
            selfd = [f(O[t], Ob[t]) for t in common]
            selfd = [x for x in selfd if not np.isnan(x)]
            dd = [x for x in (f(O[a], O[b]) for a, b in diff) if not np.isnan(x)]
            auc = roc_auc_score([0] * len(selfd) + [1] * len(dd), selfd + dd)
            print(f"V3 {mn}: self {np.mean(selfd):.3f} vs diff {np.mean(dd):.3f} "
                  f"(ratio {np.mean(dd) / max(np.mean(selfd), 1e-9):.1f}x)  AUC {auc:.3f}")
    # canonicalization health
    ok = sum(c.startswith(("eq0:", "Integer", "Rational", "Mul", "Add", "Pow", "Symbol"))
             for r in O.values() for c in r["canon"])
    tot = sum(len(r["canon"]) for r in O.values())
    print(f"sympy-canonicalized {ok}/{tot} objects ({100 * ok / tot:.0f}%)")


def _vals(strings, minv=2):
    from fractions import Fraction
    out = set()
    for s in strings:
        for m in re.findall(r"-?\d+(?:/\d+)?", s):
            try:
                v = Fraction(m)
            except Exception:
                continue
            if abs(v) > minv:
                out.add(v)
    return out


def encode():
    import torch
    from transformers import AutoModel, AutoTokenizer
    rows = [json.loads(l) for l in open(objs_file("a"))]
    flat = [(r["id"], o) for r in rows for o in r["objects"]]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B",
                                      dtype=torch.bfloat16).cuda().eval()
    assert next(model.parameters()).is_cuda
    embs = []
    with torch.no_grad():
        for i in range(0, len(flat), 128):
            b = tok([o for _, o in flat[i:i + 128]], padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state[:, -1]
            embs.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
    E, off, k = np.concatenate(embs), {}, 0
    for r in rows:
        off[r["id"]] = (k, k + len(r["objects"])); k += len(r["objects"])
    np.savez(RP / "obj_embs.npz", E=E, offsets=json.dumps(off))
    print(f"encoded {len(rows)} object graphs, {len(E)} objects")


def analyze_full():
    from sklearn.metrics import roc_auc_score
    s6 = _load("scripts/23_logdist_seqot.py", "s6")
    O = {r["id"]: r for r in (json.loads(l) for l in open(objs_file("a")))}
    ids, texts, pa = s9.ids_texts()
    txt = dict(zip(ids, texts))
    tr = s9.traces()
    ext = {f"n{r['qidx']}_{r['samp']}": r["extracted"] for r in tr}
    z = np.load(RP / "obj_embs.npz")
    E, off = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    RVals = {t: _vals([txt[t]]) for t in ids}
    OVals = {t: _vals(r["objects"]) for t, r in O.items()}
    VEdges = {t: {(a, b) for i, j in r["edges"]
                  for a in _vals([r["objects"][i]]) for b in _vals([r["objects"][j]])}
              for t, r in O.items()}

    def d_set(S, a, b):
        A, B = S.get(a), S.get(b)
        if not A or not B:
            return float("nan")
        return 1 - len(A & B) / len(A | B)

    def d_emb(a, b):
        if a not in off or b not in off:
            return float("nan")
        (a0, a1), (b0, b1) = off[a], off[b]
        if a1 == a0 or b1 == b0:
            return float("nan")
        S = E[a0:a1] @ E[b0:b1].T
        return float(1 - 0.5 * (S.max(1).mean() + S.max(0).mean()))

    m = {"raw_val": s6.memo(lambda a, b: d_set(RVals, a, b)),
         "obj_val": s6.memo(lambda a, b: d_set(OVals, a, b)),
         "obj_emb": s6.memo(d_emb),
         "val_edge": s6.memo(lambda a, b: d_set(VEdges, a, b))}
    diff, same = s9.judge_pairs()
    for tag, ks in (("ENS:raw+emb", ["raw_val", "obj_emb"]),
                    ("ENS:r+v+e", ["raw_val", "obj_val", "obj_emb"])):
        fs = [(m[k], float(np.nanstd([m[k](a, b) for a, b in diff]))) for k in ks]
        m[tag] = s6.memo(lambda a, b, fs=fs: float(np.nanmean([f(a, b) / z for f, z in fs])))

    def ansmatch(p):
        ea, eb = ext.get(p[0]), ext.get(p[1])
        return ea is not None and eb is not None and str(ea) == str(eb)

    base = {r["qidx"]: r["samp"] for r in pa}
    packs = s9.method_packs()
    diff_m = [p for p in diff if ansmatch(p)]; same_m = [p for p in same if ansmatch(p)]
    diff_x = [p for p in diff if not ansmatch(p)]; same_x = [p for p in same if not ansmatch(p)]
    print(f"G2 full battery: {len(diff)} diff / {len(same)} same pairs "
          f"(ans-matched {len(diff_m)}/{len(same_m)}); {len(packs)} packs\n")
    print(f"{'metric':11s} {'AUC_D':>6s} {'ansM':>6s} {'ansX':>6s} {'ALLst':>6s}   "
          f"{'V_sty':>5s} {'V_met':>5s}   R_set [95% CI]  hack%")
    rng = np.random.default_rng(20260818)
    res = {}
    for mn, f in m.items():
        def auc(dl, sl):
            dd = [x for x in (f(a, b) for a, b in dl) if not np.isnan(x)]
            ds = [x for x in (f(a, b) for a, b in sl) if not np.isnan(x)]
            return roc_auc_score([0] * len(ds) + [1] * len(dd), ds + dd), dd
        aucd, dd = auc(diff, same)
        aucm, _ = auc(diff_m, same_m)
        aucx, _ = auc(diff_x, same_x)
        allpos = []
        for st in s9.ALL_STYLES:
            v = [f(f"n{q}_{base[q]}", f"p{q}_{st}") for q in sorted(s9.judge())]
            allpos += [x for x in v if not np.isnan(x)]
        aucg = roc_auc_score([0] * len(allpos) + [1] * len(dd), allpos + dd)
        tau = abs(np.nanmean([f(a, b) for a, b in diff])) or 1.0
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
        res[mn] = {"auc_d": aucd, "auc_ansmatch": aucm, "auc_ansdiff": aucx,
                   "auc_allst": aucg, "R": r, "ci": [lo, hi],
                   "V_style": float(vs.mean()), "V_meth": float(vm.mean())}
        print(f"{mn:11s} {aucd:6.3f} {aucm:6.3f} {aucx:6.3f} {aucg:6.3f}   "
              f"{vs.mean():5.2f} {vm.mean():5.2f}   {r:5.2f} [{lo:4.2f},{hi:4.2f}]  "
              f"{100 / max(r, 1e-9):3.0f}%")
    (RP / "g2_eval.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for f in ("extract", "analyze", "probe", "pass2", "smoke", "encode", "full"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "time": time.strftime("%FT%TZ", time.gmtime())},
              open(RP / "manifest_g2.json", "w"), indent=1)
    if a.extract:
        extract(a)
    if a.encode:
        encode()
    if a.full:
        analyze_full()
    elif a.analyze:
        analyze(a)


if __name__ == "__main__":
    main()
