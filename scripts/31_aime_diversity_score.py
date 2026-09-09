"""
Step 4a of docs/aime_passk_diversity_design.md: score each group (scripts/30 output) with the
SIMPLE diversity metrics (DTW + Wasserstein over nomic-embed-text step embeddings) from
scripts/09_traj_diversity.py.

Metric (b), the validated logdist ensemble (scripts/13-15/17/23/25/26), is BLOCKED in this
environment: no CUDA GPU (this machine is Apple Silicon Mac) and the trained projection heads
(head_l15_l4_L18*.pt) do not exist in this checkout's runs/logdist-testbed/ -- they were produced
in a prior GPU session and never synced here. This script covers metric (a) only.

Step embeddings are computed ONCE per unique pool trajectory (not per group) and pairwise
distances are memoized by trajectory-id pair, since the same trajectory/pair appears in many
groups -- avoids redundant embedding calls and O(T1*T2) DTW recomputation.

Usage:
  python scripts/31_aime_diversity_score.py <run_dir>
"""
import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

s09 = None  # loaded in main() after argparse so --help doesn't require ollama running


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    global s09
    s09 = _load(Path(__file__).parent / "09_traj_diversity.py", "s09")

    pool = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.run_dir) / "raw" / "pool.jsonl")}
    groups = [json.loads(l) for l in open(Path(args.run_dir) / "raw" / "groups.jsonl")]

    print(f"pool trajectories: {len(pool)}  groups: {len(groups)}")
    print("computing step embeddings for each unique trajectory (one-time cost)...")
    step_seqs = {}
    for i, (tid, rec) in enumerate(pool.items()):
        step_seqs[tid] = s09.dtw_trajectory_steps(rec["generation"])
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(pool)}", flush=True)

    pair_dtw = {}
    pair_wass = {}

    def dtw_pair(a, b):
        key = tuple(sorted((a, b)))
        if key not in pair_dtw:
            pair_dtw[key] = s09._dtw(step_seqs[a], step_seqs[b])
        return pair_dtw[key]

    def wass_pair(a, b):
        key = tuple(sorted((a, b)))
        if key not in pair_wass:
            A, B = step_seqs[a], step_seqs[b]
            cost = [[s09.cosine_dist(A[r], B[c]) for c in range(len(B))] for r in range(len(A))]
            pair_wass[key] = s09._sinkhorn(cost)
        return pair_wass[key]

    print("scoring groups...")
    out_path = Path(args.run_dir) / "raw" / "groups_scored.jsonl"
    with open(out_path, "x") as f:
        for i, g in enumerate(groups):
            ids = g["member_ids"]
            pairs = list(combinations(ids, 2))
            dtw_scores = [dtw_pair(a, b) for a, b in pairs]
            wass_scores = [wass_pair(a, b) for a, b in pairs]
            g_out = dict(g)
            g_out["dtw_div"] = sum(dtw_scores) / len(dtw_scores)
            g_out["wass_div"] = sum(wass_scores) / len(wass_scores)
            f.write(json.dumps(g_out) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  scored {i + 1}/{len(groups)}", flush=True)

    print(f"\nSaved: {out_path}")
    print(f"unique trajectory pairs scored: dtw={len(pair_dtw)} wass={len(pair_wass)}")


if __name__ == "__main__":
    main()
