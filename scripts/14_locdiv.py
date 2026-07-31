#!/usr/bin/env python3
"""F3-E3: does lens-diversity in subagent prompts raise UNION recall on fault localization?

Task: issue text + repo .py file list at base commit -> up to 5 files that must change.
Gold = .py files touched by the SWE-bench Verified gold patch (verifiable, no execution).
Arms (K samples each, budget-matched, paired per instance): neutral (same suffix x K) vs
lens (K distinct diagnostic-lens suffixes). Aggregation is POOLING (union over samples),
so a wrong sample costs budget only - the M2 channel from F2 cannot exist by construction.
Primary metric: union recall@K, paired bootstrap. Secondary: per-sample recall/precision
(directive-tax check), union-recall growth curves, distinct files proposed.
"""
import argparse, json, random, re, subprocess, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_DIR = Path("runs/swebench_repos")
SEED = 20260706

SYS = ("You are a software engineer localizing a bug. Given a GitHub issue and the repository's "
       "Python file list, identify the non-test files that must be edited to fix the issue. "
       "Think briefly, then output ONLY a fenced code block containing up to 5 repo-relative "
       "file paths from the list, one per line, most likely first.")

NEUTRAL_SUFFIX = "Approach: use your standard debugging judgment."
LENSES = [
    "Approach: trace the error message / traceback strings in the issue to where they are raised or formatted.",
    "Approach: follow the data flow - from the user-facing API mentioned in the issue down through the call chain to where the value is computed.",
    "Approach: suspect API misuse or a recently changed interface - look for the module owning the public API named in the issue.",
    "Approach: start from the test files exercising this behavior and infer which source files they target (report the source files, not the tests).",
    "Approach: suspect configuration, defaults, or option handling - look for settings, registry, or default-value code paths.",
    "Approach: suspect state, caching, or lifecycle - look for files managing shared state, caches, or object initialization order.",
    "Approach: suspect a boundary or edge case - look for files doing validation, type dispatch, or special-casing near the reported behavior.",
    "Approach: suspect the documentation/behavior mismatch - find the file implementing the documented feature named in the issue.",
]


def chat(client, model, user, effort, max_tok=20000, retries=4):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                reasoning_effort=effort, max_completion_tokens=max_tok)
            u = r.usage
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            return (r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, cached)
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 ** a * 3)


def gold_files(patch):
    return sorted({m.group(1) for m in re.finditer(r"^diff --git a/\S+ b/(\S+)", patch, re.M)
                   if m.group(1).endswith(".py")})


def py_tree(repo, commit):
    d = REPO_DIR / repo.replace("/", "__")
    out = subprocess.run(["git", "-C", str(d), "ls-tree", "-r", "--name-only", commit],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.endswith(".py")]


def parse_pred(text, tree_set):
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text or "", re.S)
    lines = (blocks[-1] if blocks else (text or "")).splitlines()
    preds = []
    for l in lines:
        p = l.strip().strip("`").lstrip("./")
        if p.startswith(("a/", "b/")):
            p = p[2:]
        if p.endswith(".py") and p in tree_set and p not in preds:
            preds.append(p)
    return preds[:5]


def sample_call(client, args, item, arm, k):
    suffix = NEUTRAL_SUFFIX if arm == "neutral" else LENSES[k]
    user = (f"ISSUE:\n{item['problem']}\n\nREPOSITORY PYTHON FILES:\n{item['tree_text']}\n\n{suffix}")
    text, ptok, ctok, cached = chat(client, args.model, user, args.effort)
    preds = parse_pred(text, item["tree_set"])
    g = set(item["gold"])
    return {"iid": item["iid"], "arm": arm, "samp": k, "preds": preds,
            "recall": len(g & set(preds)) / len(g), "prec": len(g & set(preds)) / max(len(preds), 1),
            "prompt_tok": ptok, "compl_tok": ctok, "cached_tok": cached, "text_tail": (text or "")[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.5-2026-04-23")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tag", default="locdiv")
    # single-file gold (430/500) reduces union@K to plain coverage; the pooling claim needs
    # multiple gold items per instance, so multi-file gold is the default venue
    ap.add_argument("--min-gold", type=int, default=2)
    ap.add_argument("--arms", default="neutral,lens")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.k = 3, 2

    from datasets import load_dataset
    ds = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
    rng = random.Random(SEED)
    rng.shuffle(ds)
    items = []
    for r in ds:
        g = gold_files(r["patch"])
        if len(g) < args.min_gold:
            continue
        tree = py_tree(r["repo"], r["base_commit"])
        if not all(x in set(tree) for x in g):  # skip patch-creates-file instances (unreachable gold)
            continue
        items.append({"iid": r["instance_id"], "problem": r["problem_statement"][:8000],
                      "gold": g, "tree_text": "\n".join(tree), "tree_set": set(tree)})
        if len(items) == args.n:
            break

    from openai import OpenAI
    client = OpenAI()
    arms = args.arms.split(",")
    calls = [(it, arm, k) for it in items for arm in arms for k in range(args.k)]
    # warm-up pass: one call per instance populates the prompt cache for the shared
    # issue+tree prefix; the remaining calls then hit it (suffix-only variation)
    warm = [c for c in calls if c[1] == arms[0] and c[2] == 0]
    rest = [c for c in calls if not (c[1] == arms[0] and c[2] == 0)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(lambda c: sample_call(client, args, *c), warm))
        rows += list(pool.map(lambda c: sample_call(client, args, *c), rest))

    out = Path(f"runs/{args.tag}-swebv-n{args.n}-k{args.k}" + ("-smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "raw.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "manifest.json").write_text(json.dumps({**vars(args), "seed": SEED,
        "iids": [it["iid"] for it in items], "n_gold": {it["iid"]: len(it["gold"]) for it in items}}, indent=1))

    by = defaultdict(list)
    for r in rows:
        by[(r["iid"], r["arm"])].append(r)
    gold = {it["iid"]: set(it["gold"]) for it in items}

    def union_recall(iid, arm, k):
        u = set().union(*[set(r["preds"]) for r in sorted(by[(iid, arm)], key=lambda x: x["samp"])[:k]])
        return len(gold[iid] & u) / len(gold[iid])

    print(f"n={len(items)} K={args.k}  ptok={sum(r['prompt_tok'] for r in rows)/1e6:.2f}M "
          f"(cached {sum(r['cached_tok'] for r in rows)/1e6:.2f}M)  ctok={sum(r['compl_tok'] for r in rows)/1e6:.2f}M")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        ur = {k: sum(union_recall(it["iid"], arm, k) for it in items) / len(items)
              for k in (1, 2, 4, 8) if k <= args.k}
        distinct = sum(len(set().union(*[set(r["preds"]) for r in by[(it['iid'], arm)]])) for it in items) / len(items)
        print(f"{arm:8s} per-sample recall={sum(r['recall'] for r in rs)/len(rs):.3f} "
              f"prec={sum(r['prec'] for r in rs)/len(rs):.3f}  union@k={ {k: round(v,3) for k,v in ur.items()} } "
              f"distinct-files/inst={distinct:.1f}")
    if args.k >= 2 and len(arms) == 2:
        diffs = [union_recall(it["iid"], "lens", args.k) - union_recall(it["iid"], "neutral", args.k) for it in items]
        boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10000))
        print(f"paired d(union@{args.k}) = {sum(diffs)/len(diffs):+.3f} [{boots[250]:+.3f},{boots[9750]:+.3f}]")


if __name__ == "__main__":
    main()
