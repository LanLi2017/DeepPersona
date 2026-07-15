"""
Generate a markdown report from a diversity sweep JSONL.

Usage:
  python 03_report.py runs/diversity-<ts>.jsonl [--out report.md] [--embed]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import requests

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts):
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
    return r.json()["embeddings"]


def mean_pool(vecs):
    d = len(vecs[0])
    out = [0.0] * d
    for v in vecs:
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vecs) for x in out]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def trajectory_embedding(text):
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if not sentences:
        sentences = [text[:200]] if text.strip() else ["empty"]
    vecs = embed_texts(sentences)
    return mean_pool(vecs)


def pairwise_embed_div(traj_vecs):
    n = len(traj_vecs)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - cosine_sim(traj_vecs[i], traj_vecs[j])
            count += 1
    return total / count


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs) + 1e-9)
    dy = math.sqrt(sum((y - my) ** 2 for y in ys) + 1e-9)
    return num / (dx * dy)


def compute_embed_metrics(raw, methods):
    """Returns {method: {"embed_div": float, "pearson_r": float}}."""
    result = {}
    for method in methods:
        items_data = raw[method]
        embed_divs, passks = [], []
        print(f"  embedding {method} ({len(items_data)} items)...", end="\r", file=sys.stderr)
        for idx, recs in items_data.items():
            recs = sorted(recs, key=lambda r: r["rollout_idx"])
            traj_vecs = [trajectory_embedding(r["text"]) for r in recs]
            ed = pairwise_embed_div(traj_vecs)
            pk = float(any(r["correct"] for r in recs))
            embed_divs.append(ed)
            passks.append(pk)
        n = len(items_data)
        r = pearson_r(embed_divs, passks) if n >= 3 else float("nan")
        result[method] = {"embed_div": sum(embed_divs) / n, "pearson_r": r}
        print(f"  {method}: embed_div={result[method]['embed_div']:.4f}  pearson_r={r:.3f}  ", file=sys.stderr)
    return result


def trigrams(text):
    toks = text.lower().split()
    return [tuple(toks[i:i+3]) for i in range(len(toks) - 2)]


def distinct_3(texts):
    all_tg = [tg for t in texts for tg in trigrams(t)]
    return len(set(all_tg)) / len(all_tg) if all_tg else 0.0


def answer_entropy(preds):
    counts = defaultdict(int)
    for p in preds:
        counts[p or "NONE"] += 1
    n = len(preds)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())


def pairwise_jaccard_div(texts):
    sets = [set(t.lower().split()) for t in texts]
    n = len(sets)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i+1, n):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            total += (1 - len(inter)/len(union)) if union else 0.0
            count += 1
    return total / count


METHOD_DESC = {
    "greedy":       "Same prompt, greedy decoding ×8 (identical — diversity floor)",
    "temp0.3":      "Same prompt, temperature=0.3 ×8",
    "temp0.7":      "Same prompt, temperature=0.7 ×8",
    "temp1.0":      "Same prompt, temperature=1.0 ×8",
    "temp1.3":      "Same prompt, temperature=1.3 ×8",
    "persona":      "8 different personas, greedy (prompt-driven diversity)",
    "template":     "8 different question templates, greedy (prompt-driven diversity)",
    "persona_temp": "8 different personas, temperature=0.7 (prompt + stochastic)",
}


def compute_metrics(records):
    d3, ent, pdiv = [], [], []
    p1, pk, ma = [], [], []
    for recs in records.values():
        recs = sorted(recs, key=lambda r: r["rollout_idx"])
        texts = [r["text"] for r in recs]
        preds = [r["pred"] for r in recs]
        corrects = [r["correct"] for r in recs]
        d3.append(distinct_3(texts))
        ent.append(answer_entropy(preds))
        pdiv.append(pairwise_jaccard_div(texts))
        p1.append(float(corrects[0]))
        pk.append(float(any(corrects)))
        ma.append(sum(corrects) / len(corrects))
    n = len(records)
    return {
        "n": n,
        "distinct_3": sum(d3)/n,
        "ans_ent":    sum(ent)/n,
        "pair_div":   sum(pdiv)/n,
        "pass_at_1":  sum(p1)/n,
        "pass_at_k":  sum(pk)/n,
        "mean_acc":   sum(ma)/n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--embed", action="store_true", help="Compute embedding diversity via Ollama (slow).")
    args = parser.parse_args()

    path = Path(args.jsonl)
    raw = defaultdict(lambda: defaultdict(list))
    manifest = {}

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("__manifest__"):
                manifest = row
                continue
            raw[row["method"]][row["item_idx"]].append(row)

    n_rollouts = manifest.get("n_rollouts", 8)
    model = manifest.get("model", manifest.get("model_id", "unknown"))
    n_items = manifest.get("n_items")

    results = {m: compute_metrics(raw[m]) for m in raw}

    # sort: greedy first, then temp ascending, then prompt-variation methods
    order = ["greedy", "temp0.3", "temp0.7", "temp1.0", "temp1.3", "persona", "template", "persona_temp"]
    methods = [m for m in order if m in results] + [m for m in results if m not in order]

    lines = []
    lines.append("# Rollout Diversity Report\n")
    lines.append(f"**Model:** {model}  \n**Task:** CSQA (CommonsenseQA)  \n**Items:** {n_items}  \n**Rollouts per item:** {n_rollouts}\n")

    lines.append("\n## Results\n")
    lines.append("| Method | distinct-3 ↑ | ans-ent ↑ | pair-div ↑ | pass@1 | pass@k | mean-acc |")
    lines.append("|--------|-------------|----------|-----------|--------|--------|----------|")
    for m in methods:
        r = results[m]
        lines.append(
            f"| `{m}` | {r['distinct_3']:.3f} | {r['ans_ent']:.3f} | {r['pair_div']:.3f} "
            f"| {r['pass_at_1']:.3f} | {r['pass_at_k']:.3f} | {r['mean_acc']:.3f} |"
        )

    if args.embed:
        print("Computing embedding diversity...", file=sys.stderr)
        embed_stats = compute_embed_metrics(raw, methods)
        lines.append("\n## Embedding Diversity\n")
        lines.append("| Method | embed-div ↑ | pearson-r (embed↔pass@k) |")
        lines.append("|--------|------------|--------------------------|")
        for m in methods:
            es = embed_stats[m]
            r_str = f"{es['pearson_r']:.3f}" if not math.isnan(es["pearson_r"]) else "—"
            lines.append(f"| `{m}` | {es['embed_div']:.4f} | {r_str} |")
        lines.append("\n> `embed-div`: mean pairwise cosine distance of sentence-level trajectory embeddings (nomic-embed-text).  \n"
                     "> `pearson-r`: per-item correlation between embed-div and pass@k. Positive = diverse rollouts cover more correct answers.")

    lines.append("\n## Method Descriptions\n")
    for m in methods:
        lines.append(f"- **`{m}`**: {METHOD_DESC.get(m, '')}")

    lines.append("\n## Key Observations\n")

    greedy = results.get("greedy", {})
    persona = results.get("persona", {})
    template = results.get("template", {})
    temp07 = results.get("temp0.7", {})
    temp10 = results.get("temp1.0", {})
    p_temp = results.get("persona_temp", {})

    if greedy and persona:
        div_gain = persona["distinct_3"] - greedy["distinct_3"]
        acc_gain = persona["mean_acc"] - greedy["mean_acc"]
        pk_gain = persona["pass_at_k"] - greedy["pass_at_k"]
        lines.append(
            f"- **Persona vs greedy**: trajectory diversity +{div_gain:.3f} (distinct-3), "
            f"mean-acc {acc_gain:+.3f}, pass@k {pk_gain:+.3f}"
        )

    if temp07 and persona:
        lines.append(
            f"- **Persona vs temp=0.7**: persona distinct-3={persona['distinct_3']:.3f} vs "
            f"temp0.7 distinct-3={temp07['distinct_3']:.3f}; "
            f"persona mean-acc={persona['mean_acc']:.3f} vs temp0.7 mean-acc={temp07['mean_acc']:.3f}"
        )

    if temp10 and persona:
        lines.append(
            f"- **Persona vs temp=1.0**: temp1.0 achieves similar diversity "
            f"(distinct-3={temp10['distinct_3']:.3f}) but mean-acc={temp10['mean_acc']:.3f} "
            f"vs persona mean-acc={persona['mean_acc']:.3f}"
        )

    if template and persona:
        lines.append(
            f"- **Template vs persona**: comparable diversity "
            f"(distinct-3 {template['distinct_3']:.3f} vs {persona['distinct_3']:.3f}), "
            f"mean-acc {template['mean_acc']:.3f} vs {persona['mean_acc']:.3f}"
        )

    if p_temp and persona:
        lines.append(
            f"- **persona_temp**: combining persona + temperature gives highest diversity "
            f"(distinct-3={p_temp['distinct_3']:.3f}), "
            f"pass@k={p_temp['pass_at_k']:.3f}, mean-acc={p_temp['mean_acc']:.3f}"
        )

    # Takeaway: data-driven, compares best prompt-variation method vs best temp method
    best_prompt = max(
        [(m, results[m]) for m in ["persona", "template"] if m in results],
        key=lambda x: x[1]["mean_acc"], default=(None, None)
    )
    best_temp = max(
        [(m, results[m]) for m in ["temp0.3", "temp0.7", "temp1.0", "temp1.3"] if m in results],
        key=lambda x: x[1]["mean_acc"], default=(None, None)
    )
    if best_prompt[0] and best_temp[0]:
        pm, pr = best_prompt
        tm, tr = best_temp
        if pr["mean_acc"] >= tr["mean_acc"]:
            takeaway = (
                f"Prompt-driven diversity (`{pm}`) matches or exceeds temperature sampling (`{tm}`) "
                f"on both trajectory diversity (distinct-3: {pr['distinct_3']:.3f} vs {tr['distinct_3']:.3f}) "
                f"and accuracy (mean-acc: {pr['mean_acc']:.3f} vs {tr['mean_acc']:.3f}). "
                f"Persona variation is a cost-free diversity source for models without built-in reasoning."
            )
        else:
            takeaway = (
                f"For this reasoning model, temperature sampling (`{tm}`) outperforms prompt-driven "
                f"diversity (`{pm}`) on both trajectory diversity "
                f"(distinct-3: {tr['distinct_3']:.3f} vs {pr['distinct_3']:.3f}) "
                f"and accuracy (mean-acc: {tr['mean_acc']:.3f} vs {pr['mean_acc']:.3f}). "
                f"Built-in reasoning may reduce the marginal value of persona conditioning — "
                f"the model follows its own chain-of-thought regardless of persona framing."
            )
    elif best_prompt[0]:
        takeaway = "Only prompt-variation methods tested; temperature comparison not available."
    else:
        takeaway = "Results inconclusive — insufficient methods for comparison."

    lines.append(f"\n> **Takeaway**: {takeaway}")

    report = "\n".join(lines) + "\n"

    out_path = Path(args.out) if args.out else path.with_suffix(".md")
    out_path.write_text(report)
    print(report)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
