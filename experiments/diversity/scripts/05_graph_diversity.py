"""
Graph-based trajectory diversity analysis.

For each rollout:
  1. LLM parses reasoning text -> JSON graph {nodes, edges}
  2. Node texts embedded with nomic-embed-text
  3. Mean-pool node embeddings -> graph vector
  4. Pairwise cosine distance across rollouts = graph_embed_div

Hypothesis: graph_embed_div captures structural reasoning diversity better than
sentence-level embedding, and should correlate more strongly with pass@k.

Compares graph_embed_div vs sentence_embed_div (04_embed_diversity approach).

Usage:
  python 05_graph_diversity.py runs/diversity-<ts>.jsonl [--parser qwen3:1.7b]
"""
import json
import math
import re
import sys
from collections import defaultdict

import ollama
import requests

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

PARSE_PROMPT = """Extract a reasoning graph from the text below.
Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{"nodes": ["step or concept"], "edges": [["node_from", "node_to"]]}}

Rules:
- 3 to 6 nodes maximum
- Nodes are key reasoning steps, concepts, or conclusions (short phrases)
- Edges show logical dependencies or flow
- node_from and node_to must both appear in the nodes list

Text:
{text}"""


def parse_graph(text, parser_model):
    """Call LLM to extract a reasoning graph. Returns (nodes, edges)."""
    prompt = PARSE_PROMPT.format(text=text[:1200])  # cap input length
    try:
        resp = ollama.chat(
            model=parser_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 256},
        )
        raw = resp["message"]["content"]
        # strip any <think>...</think> block from reasoning models
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # extract first JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("no JSON found")
        obj = json.loads(m.group())
        nodes = obj.get("nodes", [])
        edges = obj.get("edges", [])
        if not nodes:
            raise ValueError("empty nodes")
        return nodes, edges
    except Exception as e:
        # fallback: treat truncated text as single node
        return [text[:150]], []


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


def graph_embedding(nodes):
    """Embed node texts, mean-pool -> graph vector."""
    vecs = embed_texts(nodes)
    return mean_pool(vecs)


def sentence_embedding(text):
    """Sentence-level baseline: split on '.', embed, mean-pool."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if not sentences:
        sentences = [text[:200]] if text.strip() else ["empty"]
    vecs = embed_texts(sentences)
    return mean_pool(vecs)


def pairwise_div(vecs):
    n = len(vecs)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - cosine_sim(vecs[i], vecs[j])
            count += 1
    return total / count


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs) + 1e-9)
    dy = math.sqrt(sum((y - my) ** 2 for y in ys) + 1e-9)
    return num / (dx * dy)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--parser", default="qwen2.5:1.5b", help="Ollama model for graph parsing (avoid reasoning models — they consume token budget on think blocks)")
    args = parser.parse_args()

    raw = defaultdict(lambda: defaultdict(list))
    manifest = {}
    with open(args.jsonl) as f:
        for line in f:
            row = json.loads(line)
            if row.get("__manifest__"):
                manifest = row
                continue
            raw[row["method"]][row["item_idx"]].append(row)

    print(f"Model: {manifest.get('model', '?')}  "
          f"n_items={manifest.get('n_items')}  n_rollouts={manifest.get('n_rollouts')}")
    print(f"Parser: {args.parser}\n")

    order = ["greedy", "temp0.3", "temp0.7", "temp1.0", "temp1.3", "persona", "template", "persona_temp"]
    methods = [m for m in order if m in raw] + [m for m in raw if m not in order]

    print(f"{'method':<16}  {'graph_div':>10}  {'sent_div':>9}  {'pass@k':>7}  {'g_r':>7}  {'s_r':>7}")
    print("-" * 62)

    for method in methods:
        items_data = raw[method]
        graph_divs, sent_divs, passks = [], [], []

        for idx, recs in items_data.items():
            recs = sorted(recs, key=lambda r: r["rollout_idx"])
            pk = float(any(r["correct"] for r in recs))

            graph_vecs, sent_vecs = [], []
            for i, r in enumerate(recs):
                print(f"  {method} item={idx} rollout={i}...", end="\r")
                nodes, _ = parse_graph(r["text"], args.parser)
                graph_vecs.append(graph_embedding(nodes))
                sent_vecs.append(sentence_embedding(r["text"]))

            graph_divs.append(pairwise_div(graph_vecs))
            sent_divs.append(pairwise_div(sent_vecs))
            passks.append(pk)

        n = len(items_data)
        g_r = pearson_r(graph_divs, passks) if n >= 3 else float("nan")
        s_r = pearson_r(sent_divs, passks) if n >= 3 else float("nan")

        def fmt_r(r):
            return f"{r:>7.3f}" if not math.isnan(r) else "      —"

        print(
            f"{method:<16}  "
            f"{sum(graph_divs)/n:>10.4f}  "
            f"{sum(sent_divs)/n:>9.4f}  "
            f"{sum(passks)/n:>7.3f}  "
            f"{fmt_r(g_r)}  {fmt_r(s_r)}"
        )

    print("\nColumns: graph_div=graph-embedding pairwise cosine dist, sent_div=sentence-level baseline")
    print("g_r / s_r = Pearson r of each diversity metric vs pass@k")
    print("If g_r > s_r: graph structure captures reasoning diversity better than surface text.")


if __name__ == "__main__":
    main()
