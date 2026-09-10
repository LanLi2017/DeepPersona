#!/usr/bin/env python3
"""N4 style-drift audit: gpt-5.5 judge method-clusters the eval rollouts of both GRPO arms
at first/last eval step. Primary outcome = judge methods among CORRECT rollouts (what the
gated reward targets). Failure signature = ENSinreg V_corr up but judge methods flat.

  set -a; . ./.tinker_env; set +a
  .venv/bin/python scripts/31_n4_audit.py --label   # ~$4
  .venv/bin/python scripts/31_n4_audit.py --analyze
"""
import argparse, collections, importlib.util, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s8 = _load("scripts/28_regime_probe.py", "s8")

RUNS = {
    "lam0": Path("runs/n4grpo-20260813-001508Z-lam0.0-s0"),
    "lam0.5": Path("runs/n4grpo-20260813-004753Z-lam0.5-s0"),
    "lam5-d2": Path("runs/n4grpo-20260813-012513Z-lam5.0-d2-s0"),
}
STEPS = {"lam0": (0, 19), "lam0.5": (0, 19), "lam5-d2": (0, 39)}
OUT = Path("runs/regime-probe")  # keep logdist-side artifacts together
JUDGE = "gpt-5.5-2026-04-23"


def eval_rows(run, step):
    byq = collections.defaultdict(dict)
    for l in open(RUNS[run] / "raw" / f"eval_step{step:04d}.jsonl"):
        r = json.loads(l)
        byq[r["item_idx"]][r["slot"]] = r
    return byq


def label(smoke):
    from openai import OpenAI
    client = OpenAI()
    done = set()
    fn_full = OUT / "n4_audit_labels.jsonl"
    if fn_full.exists():  # incremental: only label runs/steps not already judged
        done = {(r["run"], r["step"]) for l in open(fn_full) if (r := json.loads(l))}
    jobs = []
    for run in RUNS:
        for step in STEPS[run]:
            if (run, step) in done:
                continue
            for q, slots in sorted(eval_rows(run, step).items()):
                texts = [slots[s]["generation"][:s8.JUDGE_CAP] for s in sorted(slots)]
                jobs.append({"run": run, "step": step, "qidx": q,
                             "problem": slots[0]["problem"], "texts": texts})
    if smoke:
        jobs = jobs[:2]
    est = sum(sum(len(t) for t in j["texts"]) // 3 + 500 for j in jobs) * 2.5 / 1e6
    print(f"jobs={len(jobs)}  est cost ~ ${est:.2f} ({JUDGE})")

    def one(j):
        body = f"PROBLEM:\n{j['problem']}\n\n" + "\n\n".join(
            f"--- SOLUTION {i + 1} ---\n{t}" for i, t in enumerate(j["texts"]))
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=JUDGE, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": s8.JUDGE_SYS},
                              {"role": "user", "content": body}],
                    max_completion_tokens=8000, reasoning_effort="low")
                out = json.loads(r.choices[0].message.content)
                asg = out.get("assignment", [])
                assert len(asg) == len(j["texts"])
                u = r.usage
                return {**{k: j[k] for k in ("run", "step", "qidx")}, "assignment": asg,
                        "methods": out.get("methods", []), "prompt_tok": u.prompt_tokens,
                        "compl_tok": u.compl_tok if hasattr(u, "compl_tok") else u.completion_tokens}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 3)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    fn = OUT / ("n4_audit_labels_smoke.jsonl" if smoke else "n4_audit_labels.jsonl")
    with open(fn, "a" if fn.exists() and not smoke else "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ptok, ctok = sum(r["prompt_tok"] for r in rows), sum(r["compl_tok"] for r in rows)
    print(f"wrote {fn}: {len(rows)} rows, cost ~ ${(ptok * 2.5 + ctok * 25.0) / 1e6:.2f}")


def analyze():
    rows = [json.loads(l) for l in open(OUT / "n4_audit_labels.jsonl")]
    res = {}
    print(f"{'arm':7s} {'step':>4s} {'p@1':>6s} {'p@8':>6s} {'meth/prob':>9s} {'V_meth':>7s} "
          f"{'meth-corr':>9s} {'V_meth_corr':>11s}")
    for r in rows:
        key = (r["run"], r["step"])
        res.setdefault(key, []).append(r)
    summary = {}
    for (run, step) in sorted(res):
        byq = eval_rows(run, step)
        nm, vm, nmc, vmc, p1, p8 = [], [], [], [], [], []
        for r in res[(run, step)]:
            slots = byq[r["qidx"]]
            corr = [slots[s]["correct"] for s in sorted(slots)]
            a = r["assignment"]
            nm.append(len(set(a)))
            vm.append(s8.vendi(a))
            ac = [x for x, c in zip(a, corr) if c]
            nmc.append(len(set(ac)) if ac else 0)
            vmc.append(s8.vendi(ac) if ac else 1.0)
            p1.append(np.mean(corr))
            p8.append(int(any(corr)))
        summary[f"{run}_s{step}"] = {
            "pass1": float(np.mean(p1)), "pass8": float(np.mean(p8)),
            "methods": float(np.mean(nm)), "v_method": float(np.mean(vm)),
            "methods_correct": float(np.mean(nmc)), "v_method_correct": float(np.mean(vmc))}
        s = summary[f"{run}_s{step}"]
        print(f"{run:7s} {step:4d} {s['pass1']:6.3f} {s['pass8']:6.2f} {s['methods']:9.2f} "
              f"{s['v_method']:7.2f} {s['methods_correct']:9.2f} {s['v_method_correct']:11.2f}")
    json.dump(summary, open(OUT / "n4_audit_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.label:
        label(a.smoke)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
