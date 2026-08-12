#!/usr/bin/env python3
"""Method-labeled ground truth for the Qwen testbed: LLM judge clusters each problem's 8
traces by SOLUTION METHOD, replacing the distinct-answer proxy. Decides whether the F1/N4
regime lacks method diversity (construct absent) or the proxy just can't see it (mislabeled).

Validity checks baked in:
- hidden paraphrase controls: for para problems, 2 paraphrases of a base trace are appended
  (positions 8,9) -- a style-blind judge must put them in the base's cluster;
- 20-problem repeat pass -> pairwise self-agreement.

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/27_method_labels.py --label --smoke
  .venv/bin/python scripts/27_method_labels.py --label      # ~$2.5, gpt-4.1
  .venv/bin/python scripts/27_method_labels.py --analyze    # $0
"""
import argparse, collections, importlib.util, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

QT = Path("runs/logdist-qwen")
K = 8
NREPEAT = 20
CTRL_STYLES = ("D_restructured", "L_maxlex")

JUDGE_SYS = """You are given several solutions to the same math problem. Cluster them by SOLUTION METHOD.

Two solutions use the SAME method if they rely on the same key idea and overall strategy (same setup, same kind of decomposition/theorem/technique), even if they differ in wording, verbosity, notation, variable names, order of presentation, or contain arithmetic slips leading to different final answers.
Two solutions use DIFFERENT methods only if the mathematical approach itself differs (e.g. casework vs generating functions; coordinate geometry vs synthetic; direct counting vs complementary counting; induction vs closed-form derivation).

Output JSON:
{"methods": ["short description of method 1", ...],
 "assignment": [m, m, ...]}   // for each solution in the order given, the 1-based index of its method
Every solution must be assigned. Do not create a new method for presentation differences."""


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_jobs():
    tr = [json.loads(l) for l in open(QT / "traces.jsonl")]
    pa = [json.loads(l) for l in open(QT / "paraphrases.jsonl")]
    byq = collections.defaultdict(dict)
    for r in tr:
        byq[r["qidx"]][r["samp"]] = r
    pmap = {(r["qidx"], r["style"]): r for r in pa}
    base = {r["qidx"]: r["samp"] for r in pa}
    jobs = []
    for q in sorted(byq):
        texts = [byq[q][s]["text"] for s in range(K)]
        ctrl = []
        if q in base:
            for st in CTRL_STYLES:
                if (q, st) in pmap:
                    ctrl.append(st)
                    texts.append(pmap[(q, st)]["text"])
        jobs.append({"qidx": q, "problem": byq[q][0]["problem"], "texts": texts,
                     "ctrl_styles": ctrl, "base_samp": base.get(q)})
    return jobs


def label(args):
    from openai import OpenAI
    client = OpenAI()
    jobs = build_jobs()
    runs = [(j, 0) for j in jobs] + [(j, 1) for j in jobs[:NREPEAT]]
    if args.smoke:
        runs = runs[:2]
    est = sum(sum(len(t) for t in j["texts"]) // 3 + 500 for j, _ in runs) * 2.0 / 1e6 \
        + len(runs) * 400 * 8.0 / 1e6
    print(f"jobs={len(runs)} ({len(jobs)} problems + {min(NREPEAT, len(jobs))} repeats)  "
          f"est cost ≈ ${est:.2f} ({args.model})")

    def one(run):
        j, rep = run
        body = f"PROBLEM:\n{j['problem']}\n\n" + "\n\n".join(
            f"--- SOLUTION {i + 1} ---\n{t}" for i, t in enumerate(j["texts"]))
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": JUDGE_SYS},
                              {"role": "user", "content": body}],
                    max_completion_tokens=1500, temperature=0.0 if rep == 0 else 0.7)
                out = json.loads(r.choices[0].message.content)
                asg = out.get("assignment", [])
                assert len(asg) == len(j["texts"]), f"len {len(asg)} != {len(j['texts'])}"
                u = r.usage
                return {"qidx": j["qidx"], "rep": rep, "methods": out.get("methods", []),
                        "assignment": asg, "ctrl_styles": j["ctrl_styles"],
                        "base_samp": j["base_samp"],
                        "prompt_tok": u.prompt_tokens, "compl_tok": u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, runs))
    fn = QT / ("method_labels_smoke.jsonl" if args.smoke else "method_labels.jsonl")
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote {fn}: {len(rows)} rows, cost ≈ ${(ptok * 2.0 + ctok * 8.0) / 1e6:.2f}")
    if args.smoke:
        for r in rows:
            print(json.dumps({k: r[k] for k in ("qidx", "methods", "assignment")}, indent=1)[:600])


def analyze():
    s6 = _load("scripts/23_logdist_seqot.py", "s6")
    from sklearn.metrics import roc_auc_score
    rows = [json.loads(l) for l in open(QT / "method_labels.jsonl")]
    labels = {int(k): v for k, v in json.load(open(QT / "problem_labels.json")).items()}
    main = {r["qidx"]: r for r in rows if r["rep"] == 0}
    rep = {r["qidx"]: r for r in rows if r["rep"] == 1}

    # ---- judge validity ----------------------------------------------------------
    agree = tot = 0
    for q, r1 in rep.items():
        a1, a2 = main[q]["assignment"][:K], r1["assignment"][:K]
        for i, j in combinations(range(K), 2):
            agree += (a1[i] == a1[j]) == (a2[i] == a2[j]); tot += 1
    print(f"repeat self-agreement (same/diff-method on pairs): {agree / tot:.1%} (n={tot})")
    ok = nc = 0
    for q, r in main.items():
        for pos, st in enumerate(r["ctrl_styles"]):
            nc += 1
            ok += r["assignment"][K + pos] == r["assignment"][r["base_samp"]]
    print(f"paraphrase controls in base's cluster: {ok}/{nc} = {ok / max(nc, 1):.1%} "
          f"(style-blindness; styles {CTRL_STYLES})")

    # ---- construct presence ------------------------------------------------------
    nm = [len(set(r["assignment"][:K])) for r in main.values()]
    nu = [labels[q]["n_unique"] for q in main]
    print(f"\nmethods/problem: mean {np.mean(nm):.2f}  "
          f"dist {dict(sorted(collections.Counter(nm).items()))}")
    print(f"problems with >=2 methods: {np.mean([x >= 2 for x in nm]):.0%}   "
          f">=4 methods: {np.mean([x >= 4 for x in nm]):.0%}   "
          f"(answers: n_unique mean {np.mean(nu):.2f})")

    # ---- proxy validity: answer-based pairing vs method truth --------------------
    ct = collections.Counter()
    for q, r in main.items():
        a = r["assignment"][:K]
        cls = labels[q]["answer_class"]
        for i, j in combinations(range(K), 2):
            ct[("diff" if cls[i] != cls[j] else "same") + "-ans",
               "diff" if a[i] != a[j] else "same"] += 1
    for ans in ("diff-ans", "same-ans"):
        d, s = ct[(ans, "diff")], ct[(ans, "same")]
        print(f"P(diff-method | {ans}) = {d / max(d + s, 1):.2f}   (n={d + s})")

    # ---- metric vs judge labels (proxy-free AUC), within-problem pairs -----------
    tr = [json.loads(l) for l in open(QT / "traces.jsonl")]
    pa = [json.loads(l) for l in open(QT / "paraphrases.jsonl")]
    ids = [f"n{r['qidx']}_{r['samp']}" for r in tr] + [f"p{r['qidx']}_{r['style']}" for r in pa]
    idx = {s: i for i, s in enumerate(ids)}
    import torch
    F15i = np.load(QT / "emb_l15_indom.npz")["l15"]
    zi = np.load(QT / "chunk_l4_indom_head.npz")
    FCi, coffi = zi["E"], {k: tuple(v) for k, v in json.loads(str(zi["offsets"])).items()}
    ET = np.load(QT / "emb_l4_L18.npz")["l4_L18"]
    z = np.load(QT / "chunk_l4_L18.npz")
    ECr, coffr = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}

    def head(F, pt):
        W = torch.nn.Linear(4096, 256, bias=False)
        W.load_state_dict(torch.load(Path("runs/logdist-testbed") / pt, map_location="cpu"))
        with torch.no_grad():
            return torch.nn.functional.normalize(W(torch.tensor(F).float()), dim=-1).numpy()

    F15v2, FCv2 = head(ET, "head_l15_l4_L18_v2.pt"), head(ECr, "head_l4_L18_v2.pt")
    mets = {
        "l15_indom": lambda a, b: float(1 - F15i[idx[a]] @ F15i[idx[b]]),
        "l4_indom": lambda a, b: s6.d_chamfer(FCi[slice(*coffi[a])], FCi[slice(*coffi[b])]),
        "l15_v2": lambda a, b: float(1 - F15v2[idx[a]] @ F15v2[idx[b]]),
        "l4_v2": lambda a, b: s6.d_chamfer(FCv2[slice(*coffr[a])], FCv2[slice(*coffr[b])]),
        "raw_chunk": lambda a, b: s6.d_chamfer(ECr[slice(*coffr[a])], ECr[slice(*coffr[b])]),
    }
    print("\nmetric AUC vs judge labels (same-method pairs = 0, diff-method = 1):")
    same_c, diff_c = [], []
    for q, r in main.items():
        a = r["assignment"][:K]
        for i, j in combinations(range(K), 2):
            (diff_c if a[i] != a[j] else same_c).append((f"n{q}_{i}", f"n{q}_{j}"))
    print(f"  pairs: {len(same_c)} same-method, {len(diff_c)} diff-method")
    for mn, f in mets.items():
        ds = [f(*p) for p in same_c]
        dd = [f(*p) for p in diff_c]
        print(f"  {mn:10s}: AUC {roc_auc_score([0] * len(ds) + [1] * len(dd), ds + dd):.3f}")

    # ---- method switches vs correctness ------------------------------------------
    corr = collections.defaultdict(dict)
    for r in tr:
        corr[r["qidx"]][r["samp"]] = r["correct"]
    with_sw = [q for q, r in main.items() if len(set(r["assignment"][:K])) >= 2]
    p8_sw = np.mean([any(corr[q].values()) for q in with_sw]) if with_sw else float("nan")
    p8_no = np.mean([any(corr[q].values()) for q in main if q not in with_sw])
    print(f"\npass@8: problems with >=2 methods {p8_sw:.2f} (n={len(with_sw)})  "
          f"vs single-method {p8_no:.2f} (n={len(main) - len(with_sw)})")
    json.dump({"n_methods": nm, "agree": agree / tot, "ctrl_ok": ok / max(nc, 1)},
              open(QT / "method_labels_summary.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    json.dump({"git": sha, "args": vars(a), "time": time.strftime("%FT%TZ", time.gmtime())},
              open(QT / "manifest_methods.json", "w"), indent=1)
    if a.label:
        label(a)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
