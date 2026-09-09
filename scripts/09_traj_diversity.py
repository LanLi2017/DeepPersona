"""
Trajectory diversity analysis on Tinker eval samples.

Reads runs/.../raw/samples.jsonl (from 08_f0a_tinker.py eval) and computes
per-item diversity for each condition using THREE methods, then correlates
each with pass@k.

─────────────────────────────────────────────────────────────────────────────
METHOD 1 — Graph-embedding (mean-pool)                            [existing]
  Assumption: a trajectory is a BAG OF CONCEPTS, order doesn't matter.
  Pipeline: LLM parses reasoning into {nodes} → embed each node → mean-pool
            → one vector per trajectory → pairwise cosine distance.
  Weakness: loses step order; collapses all reasoning into a single point.

METHOD 2 — DTW over step embeddings                                   [new]
  Assumption: ORDER MATTERS — two trajectories are similar if they visit
              similar reasoning states in a similar sequence.
  Pipeline: split trajectory into steps (sentences) → embed each step →
            T×d matrix → DTW(A, B) with cosine as local cost.
  Handles variable length via non-linear time warping (no padding needed).
  Use when: you care whether step 3 of trace A aligns with step 3 of trace B.

METHOD 3 — Wasserstein over step embeddings                           [new]
  Assumption: ORDER DOESN'T MATTER — two trajectories are similar if they
              visit the same REGIONS of reasoning space, regardless of order.
  Pipeline: split trajectory into steps → embed each step →
            treat T embeddings as a point cloud / empirical distribution →
            Wasserstein distance (optimal transport) between two clouds.
  Handles variable length naturally (OT matches unequal-size point sets).
  Use when: you care about which reasoning concepts were explored, not when.
─────────────────────────────────────────────────────────────────────────────

Output: per-condition mean diversity for each method + per-item Pearson r
        between each diversity metric and pass@k.

Usage:
  python scripts/09_traj_diversity.py runs/<run-dir>/raw/samples.jsonl
  python scripts/09_traj_diversity.py runs/<run-dir>/raw/samples.jsonl --parser qwen2.5:1.5b
  python scripts/09_traj_diversity.py runs/<run-dir>/raw/samples.jsonl --methods graph dtw wass
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict

import ollama
import requests

# ── shared constants ──────────────────────────────────────────────────────────

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

GRAPH_PARSE_PROMPT = """Extract the key reasoning steps from the math solution below.
Return ONLY valid JSON (no markdown, no explanation):
{{"nodes": ["step or concept"], "edges": [["from", "to"]]}}

Rules:
- 3 to 6 nodes max
- Nodes are key reasoning steps, quantities, or conclusions (short phrases)
- Edges show logical flow or dependency
- node_from and node_to must both appear in nodes

Solution:
{text}"""


# ── shared utilities ──────────────────────────────────────────────────────────

def embed_texts(texts):
    texts = [t[:500] if t.strip() else "empty" for t in texts]
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
    data = r.json()
    if "embeddings" not in data:
        raise RuntimeError(f"embed API error: {data.get('error', data)}")
    return data["embeddings"]  # list of list[float]


def split_steps(text):
    """Split a trajectory into sentence-level steps."""
    steps = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    return steps if steps else ([text[:200]] if text.strip() else ["empty"])


def cosine_dist(a, b):
    """Cosine distance = 1 - cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 1.0 - dot / (na * nb + 1e-9)


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs) + 1e-9)
    dy = math.sqrt(sum((y - my) ** 2 for y in ys) + 1e-9)
    return num / (dx * dy)


def mean_pairwise(values):
    """Mean over all unique pairs."""
    n = len(values)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += values[i][j]
            count += 1
    return total / count


# ── METHOD 1: Graph-embedding (mean-pool) ────────────────────────────────────
#
# Each trajectory → LLM-parsed node list → embed nodes → mean-pool → 1 vector.
# Diversity = mean pairwise cosine distance between trajectory vectors.
#
# Strength:  fast, interpretable nodes, cheap inference.
# Weakness:  collapses structure to a single point; loses step order.

def _parse_graph_nodes(text, model):
    prompt = GRAPH_PARSE_PROMPT.format(text=text[:1500])
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 256},
        )
        raw = resp["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("no JSON found")
        nodes = json.loads(m.group()).get("nodes", [])
        if not nodes:
            raise ValueError("empty nodes")
        return nodes
    except Exception:
        return [text[:150]]  # fallback: treat truncated text as single node


def graph_trajectory_vector(text, parser_model):
    """LLM parse → embed nodes → mean-pool → single vector."""
    nodes = _parse_graph_nodes(text, parser_model)
    vecs = embed_texts(nodes)
    d = len(vecs[0])
    pooled = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
    return pooled


def graph_div(traj_vectors):
    """Mean pairwise cosine distance between mean-pooled trajectory vectors."""
    n = len(traj_vectors)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_dist(traj_vectors[i], traj_vectors[j])
            count += 1
    return total / count


# ── METHOD 2: DTW over step embeddings ───────────────────────────────────────
#
# Each trajectory → sentence steps → embed each step → T×d matrix.
# DTW(A, B) finds the optimal non-linear alignment between step sequences,
# accumulating cosine distance at each matched pair.
# Normalized by alignment path length so short vs long trajectories are comparable.
#
# Strength:  captures step-order similarity; handles variable length without padding.
# Weakness:  O(T1×T2) per pair; sensitive to step-splitting granularity.

def _dtw(seq_a, seq_b):
    """
    Dynamic Time Warping between two sequences of embeddings.
    Local cost = cosine distance between embeddings.
    Returns normalized DTW distance (total cost / path length).
    """
    T1, T2 = len(seq_a), len(seq_b)
    # dtw[i][j] = min accumulated cost to align seq_a[:i+1] with seq_b[:j+1]
    INF = float("inf")
    dtw = [[INF] * T2 for _ in range(T1)]
    dtw[0][0] = cosine_dist(seq_a[0], seq_b[0])
    for i in range(1, T1):
        dtw[i][0] = dtw[i - 1][0] + cosine_dist(seq_a[i], seq_b[0])
    for j in range(1, T2):
        dtw[0][j] = dtw[0][j - 1] + cosine_dist(seq_a[0], seq_b[j])
    for i in range(1, T1):
        for j in range(1, T2):
            cost = cosine_dist(seq_a[i], seq_b[j])
            dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
    # normalize by path length (T1 + T2 - 1 is the max possible path length)
    return dtw[T1 - 1][T2 - 1] / (T1 + T2 - 1)


def dtw_trajectory_steps(text):
    """Split into steps and embed each → list of vectors (T×d)."""
    steps = split_steps(text)
    return embed_texts(steps)  # list of T vectors


def dtw_div(traj_step_seqs):
    """Mean pairwise normalized DTW distance across trajectory step sequences."""
    n = len(traj_step_seqs)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _dtw(traj_step_seqs[i], traj_step_seqs[j])
            count += 1
    return total / count


# ── METHOD 3: Wasserstein over step embeddings ───────────────────────────────
#
# Each trajectory → sentence steps → embed each step → point cloud in R^d.
# Wasserstein distance (Earth Mover's Distance) between two point clouds =
# minimum total "work" to transport one distribution onto the other.
# Approximated here with the Sinkhorn algorithm (entropy-regularized OT).
#
# Strength:  order-invariant; naturally handles variable-length trajectories;
#            sensitive to WHICH regions of reasoning space were visited.
# Weakness:  order-blindness means two traces that visit the same concepts
#            in opposite order score as similar; O(T^2) cost matrix + Sinkhorn.

def _sinkhorn(cost_matrix, reg=0.05, n_iter=50):
    """
    Entropy-regularized optimal transport (Sinkhorn-Knopp).
    cost_matrix: T1 × T2 array (list of lists).
    Uniform marginals (1/T1, 1/T2).
    Returns scalar OT distance.
    """
    T1 = len(cost_matrix)
    T2 = len(cost_matrix[0])
    # K = exp(-C / reg)
    K = [[math.exp(-cost_matrix[i][j] / reg) for j in range(T2)] for i in range(T1)]
    u = [1.0 / T1] * T1
    v = [1.0 / T2] * T2
    for _ in range(n_iter):
        # v = (1/T2) / (K^T u)
        Ktu = [sum(K[i][j] * u[i] for i in range(T1)) for j in range(T2)]
        v = [(1.0 / T2) / (Ktu[j] + 1e-9) for j in range(T2)]
        # u = (1/T1) / (K v)
        Kv = [sum(K[i][j] * v[j] for j in range(T2)) for i in range(T1)]
        u = [(1.0 / T1) / (Kv[i] + 1e-9) for i in range(T1)]
    # transport plan P[i][j] = u[i] * K[i][j] * v[j]
    # OT cost = sum_{i,j} P[i][j] * C[i][j]
    ot_cost = sum(
        u[i] * K[i][j] * v[j] * cost_matrix[i][j]
        for i in range(T1) for j in range(T2)
    )
    return ot_cost


def wasserstein_div(traj_step_seqs):
    """
    Mean pairwise Wasserstein distance between trajectory step-embedding clouds.
    Each trajectory is treated as a uniform empirical distribution over its steps.
    """
    n = len(traj_step_seqs)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            A, B = traj_step_seqs[i], traj_step_seqs[j]
            # build pairwise cosine-distance cost matrix
            cost = [[cosine_dist(A[r], B[c]) for c in range(len(B))] for r in range(len(A))]
            total += _sinkhorn(cost)
            count += 1
    return total / count


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="Path to runs/.../raw/samples.jsonl")
    p.add_argument("--parser", default="qwen2.5:1.5b",
                   help="Ollama model for graph parsing (Method 1)")
    p.add_argument("--methods", nargs="+", default=["graph", "dtw", "wass"],
                   choices=["graph", "dtw", "wass"],
                   help="Which diversity methods to run")
    args = p.parse_args()

    # ── load ──────────────────────────────────────────────────────────────────
    raw = defaultdict(lambda: defaultdict(list))
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            raw[r["condition"]][r["item_idx"]].append(r)

    n_groups = sum(len(v) for v in raw.values())
    print(f"Loaded {n_groups} (condition, item) groups from {args.jsonl}")
    print(f"Methods: {args.methods}  parser: {args.parser}\n")

    order = ["vanilla_T", "vanilla_Thi", "axis_T"]
    conditions = [c for c in order if c in raw] + [c for c in raw if c not in order]

    run_graph = "graph" in args.methods
    run_dtw   = "dtw"   in args.methods
    run_wass  = "wass"  in args.methods

    # ── header ────────────────────────────────────────────────────────────────
    cols = []
    if run_graph: cols.append(("graph_div", 10))
    if run_dtw:   cols.append(("dtw_div",   9))
    if run_wass:  cols.append(("wass_div",  9))
    cols += [("pass@k", 7), ("pass@1", 7), ("mean_acc", 9)]

    hdr = f"{'condition':<16}  " + "  ".join(f"{name:>{w}}" for name, w in cols)
    print(hdr)
    print("-" * len(hdr))

    # ── per-condition loop ────────────────────────────────────────────────────
    all_stats = {}  # condition → list of dicts per item

    for cond in conditions:
        items_data = raw[cond]
        per_item = []

        for item_idx, recs in items_data.items():
            recs = sorted(recs, key=lambda r: r["sample_slot"])
            print(f"  [{cond}] item={item_idx} ({len(recs)} rollouts)...", end="\r")

            # embed trajectories once, reuse for DTW + Wasserstein
            step_seqs = None
            if run_dtw or run_wass:
                step_seqs = [dtw_trajectory_steps(r["generation"]) for r in recs]

            stat = {
                "pass_k": float(any(r["correct"] for r in recs)),
                "pass_1": float(recs[0]["correct"]),
            }

            if run_graph:
                gvecs = [graph_trajectory_vector(r["generation"], args.parser) for r in recs]
                stat["graph_div"] = graph_div(gvecs)

            if run_dtw:
                stat["dtw_div"] = dtw_div(step_seqs)

            if run_wass:
                stat["wass_div"] = wasserstein_div(step_seqs)

            per_item.append(stat)

        all_stats[cond] = per_item
        n = len(per_item)
        mean_acc = sum(r["correct"] for items in items_data.values() for r in items) / \
                   sum(len(v) for v in items_data.values())

        row = f"{cond:<16}  "
        if run_graph: row += f"{sum(s['graph_div'] for s in per_item)/n:>10.4f}  "
        if run_dtw:   row += f"{sum(s['dtw_div']   for s in per_item)/n:>9.4f}  "
        if run_wass:  row += f"{sum(s['wass_div']  for s in per_item)/n:>9.4f}  "
        row += f"{sum(s['pass_k'] for s in per_item)/n:>7.3f}  "
        row += f"{sum(s['pass_1'] for s in per_item)/n:>7.3f}  "
        row += f"{mean_acc:>9.3f}"
        print(row)

    # ── Pearson r table ───────────────────────────────────────────────────────
    print(f"\n## Per-item Pearson r: each diversity metric ↔ pass@k\n")
    r_cols = []
    if run_graph: r_cols.append(("graph_div", "graph↔passk", 14))
    if run_dtw:   r_cols.append(("dtw_div",   "dtw↔passk",   12))
    if run_wass:  r_cols.append(("wass_div",  "wass↔passk",  12))

    hdr2 = f"{'condition':<16}  " + "  ".join(f"{label:>{w}}" for _, label, w in r_cols) + f"  {'n_items':>8}"
    print(hdr2)
    print("-" * len(hdr2))

    for cond in conditions:
        stats = all_stats[cond]
        if len(stats) < 3:
            print(f"{cond:<16}  (too few items for correlation: n={len(stats)})")
            continue
        passks = [s["pass_k"] for s in stats]
        row = f"{cond:<16}  "
        for key, label, w in r_cols:
            vals = [s[key] for s in stats]
            r = pearson_r(vals, passks)
            row += f"{r:>{w}.3f}  "
        row += f"{len(stats):>8}"
        print(row)

    print()
    print("Interpretation:")
    print("  Positive r  → items where rollouts are more diverse have higher pass@k.")
    print("  graph_div   → mean-pooled concept bag; fast but order-blind and structure-collapsed.")
    print("  dtw_div     → step-aligned; captures ORDER similarity; sensitive to reasoning sequence.")
    print("  wass_div    → point-cloud OT; captures COVERAGE of reasoning space; order-invariant.")
    print("  dtw > wass  → step order predicts coverage (sequential reasoning matters here).")
    print("  wass > dtw  → concept coverage predicts pass@k better than sequential alignment.")


if __name__ == "__main__":
    main()
