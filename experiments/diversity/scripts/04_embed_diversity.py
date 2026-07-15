"""
Embedding-based trajectory diversity analysis.

For each rollout, splits reasoning into sentences, embeds each with
nomic-embed-text via Ollama, mean-pools to get a trajectory vector,
then computes pairwise cosine distance across rollouts per item.

Also prints per-item correlation: embed_div vs. whether any rollout
is correct (pass@k signal).

Usage:
  python 04_embed_diversity.py runs/diversity-<ts>.jsonl
"""
import json
import math
import sys
from collections import defaultdict

import requests

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts):
    """Batch embed a list of strings. Returns list of vectors."""
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
    """Split into sentences, embed each, mean-pool."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if not sentences:
        sentences = [text[:200]] if text.strip() else ["empty"]
    vecs = embed_texts(sentences)
    return mean_pool(vecs)


def pairwise_embed_div(traj_vecs):
    """Mean pairwise cosine distance (1 - sim) across trajectory embeddings."""
    n = len(traj_vecs)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - cosine_sim(traj_vecs[i], traj_vecs[j])
            count += 1
    return total / count


def main():
    path = sys.argv[1]
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
    print(f"Model: {manifest.get('model', '?')}  n_items={manifest.get('n_items')}  n_rollouts={n_rollouts}\n")

    order = ["greedy", "temp0.3", "temp0.7", "temp1.0", "temp1.3", "persona", "template", "persona_temp"]
    methods = [m for m in order if m in raw] + [m for m in raw if m not in order]

    # ── per-method aggregate ──────────────────────────────────────────────────
    print(f"{'method':<16}  {'embed_div':>10}  {'pass@1':>7}  {'pass@k':>7}  {'mean_acc':>9}")
    print("-" * 55)

    method_item_stats = {}  # method -> list of (embed_div, pass_k, mean_acc) per item

    for method in methods:
        items_data = raw[method]
        embed_divs, pass1s, passks, mean_accs = [], [], [], []
        item_stats = []

        print(f"  embedding {method} ({len(items_data)} items)...", end="\r")

        for idx, recs in items_data.items():
            recs = sorted(recs, key=lambda r: r["rollout_idx"])
            texts = [r["text"] for r in recs]
            corrects = [r["correct"] for r in recs]

            traj_vecs = [trajectory_embedding(t) for t in texts]
            ed = pairwise_embed_div(traj_vecs)
            pk = float(any(corrects))
            p1 = float(corrects[0])
            ma = sum(corrects) / len(corrects)

            embed_divs.append(ed)
            pass1s.append(p1)
            passks.append(pk)
            mean_accs.append(ma)
            item_stats.append((ed, pk, ma))

        method_item_stats[method] = item_stats
        n = len(items_data)
        print(
            f"{method:<16}  "
            f"{sum(embed_divs)/n:>10.4f}  "
            f"{sum(pass1s)/n:>7.3f}  "
            f"{sum(passks)/n:>7.3f}  "
            f"{sum(mean_accs)/n:>9.3f}"
        )

    # ── per-item correlation: embed_div vs pass@k ─────────────────────────────
    print("\n## Per-item correlation: embed_div vs pass@k\n")
    print(f"{'method':<16}  {'pearson_r':>10}  {'n_items':>8}")
    print("-" * 38)

    for method in methods:
        stats = method_item_stats[method]
        if len(stats) < 3:
            continue
        xs = [s[0] for s in stats]   # embed_div
        ys = [s[1] for s in stats]   # pass@k
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx)**2 for x in xs) + 1e-9)
        dy = math.sqrt(sum((y - my)**2 for y in ys) + 1e-9)
        r = num / (dx * dy)
        print(f"{method:<16}  {r:>10.3f}  {n:>8}")

    print("\n> Positive r: higher embedding diversity → more likely at least one rollout correct.")


if __name__ == "__main__":
    main()
