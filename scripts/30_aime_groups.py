"""
Step 3 of docs/aime_passk_diversity_design.md: construct groups from the trusted trajectory pool
(raw mixed-group trajectories + complete/non-flipped LLM revisions from scripts/29) and read off
pass@k directly -- no API calls, pure resampling of already-collected labels.

Two group-construction strategies per (model, item, k):
  random     -- uniform random k-subsets of the pool.
  controlled -- stratified by ANSWER ENTROPY (number of distinct final numeric answers among the
                k members), not by number-correct. Stratifying by number-correct is a dead end:
                pass@k = 1{>=1 correct} is a deterministic function of that count, so every group
                in a fixed-number-correct stratum has identical pass@k and zero within-stratum
                variance. Answer-entropy avoids this -- see docs/aime_passk_diversity_design.md
                §4.3 for the full argument.

Outputs two files under <run_dir>/raw/:
  pool.jsonl   -- one row per unique trajectory used in any group (id, generation, pred, correct)
  groups.jsonl -- one row per constructed group (member ids + pass_k), no generation text (join
                  against pool.jsonl by id in Step 4 to avoid duplicating text across groups)

Usage:
  python scripts/30_aime_groups.py <run_dir> --models qwen32b llama70b mistrallarge3
"""
import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

K_VALUES = [4, 8]
N_RANDOM_DRAWS = 10       # per (model, item, k)
N_DRAWS_PER_STRATUM = 5   # per (model, item, k, entropy stratum)
MAX_REJECTION_ATTEMPTS = 200
SEED = 0


def load_pool(run_dir, models):
    raw = [json.loads(l) for l in open(Path(run_dir) / "raw" / "samples.jsonl")]
    aug_path = Path(run_dir) / "raw" / "augmented.jsonl"
    aug = [json.loads(l) for l in open(aug_path)] if aug_path.exists() else []
    trusted_aug = [r for r in aug if not r["incomplete"] and not r["flipped"]]

    by_group = defaultdict(list)
    for r in raw:
        if models and r["condition"] not in models:
            continue
        tid = f"raw:{r['condition']}:{r['item_idx']}:{r['sample_slot']}"
        by_group[(r["condition"], r["item_idx"])].append(
            {"id": tid, "pred": r["pred"], "correct": r["correct"], "generation": r["generation"]})
    for r in trusted_aug:
        if models and r["condition"] not in models:
            continue
        tid = f"aug:{r['condition']}:{r['item_idx']}:{r['parent_slot']}:{r['style']}"
        by_group[(r["condition"], r["item_idx"])].append(
            {"id": tid, "pred": r["pred"], "correct": r["correct"], "generation": r["generation"]})

    mixed = {}
    for key, members in by_group.items():
        # only keep groups that were genuinely mixed in the RAW step-1 labels (design §4.1) --
        # augmentation must not turn a degenerate raw group into a usable one via re-grading noise
        raw_members = [m for m in members if m["id"].startswith("raw:")]
        corrects = [m["correct"] for m in raw_members]
        if any(corrects) and not all(corrects):
            mixed[key] = members
    return mixed


def answer_entropy(members):
    return len({m["pred"] for m in members if m["pred"] is not None})


def random_subset(pool, k, rng):
    return tuple(sorted(rng.sample(range(len(pool)), k)))


def build_groups(pool, k, rng):
    groups = []
    seen = set()
    attempts = 0
    while len(groups) < N_RANDOM_DRAWS and attempts < MAX_REJECTION_ATTEMPTS:
        attempts += 1
        idxs = random_subset(pool, k, rng)
        if idxs in seen:
            continue
        seen.add(idxs)
        groups.append({"strategy": "random", "stratum_entropy": None, "idxs": idxs})
        if len(seen) >= _max_distinct_subsets(len(pool), k):
            break

    # controlled: stratify by answer entropy of the CHOSEN subset via rejection sampling
    by_stratum = defaultdict(set)
    attempts = 0
    while attempts < MAX_REJECTION_ATTEMPTS:
        attempts += 1
        idxs = random_subset(pool, k, rng)
        e = answer_entropy([pool[i] for i in idxs])
        if idxs not in by_stratum[e]:
            by_stratum[e].add(idxs)
        full_strata = [v for v in by_stratum.values() if len(v) >= N_DRAWS_PER_STRATUM]
        if len(full_strata) >= 2 and all(len(v) >= N_DRAWS_PER_STRATUM for v in by_stratum.values()):
            break
    by_stratum = {e: set(list(v)[:N_DRAWS_PER_STRATUM]) for e, v in by_stratum.items()}
    for e, idx_sets in by_stratum.items():
        for idxs in idx_sets:
            groups.append({"strategy": "controlled", "stratum_entropy": e, "idxs": idxs})
    return groups


def _max_distinct_subsets(n, k):
    from math import comb
    return comb(n, k) if n >= k else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    mixed = load_pool(args.run_dir, args.models)
    rng = random.Random(SEED)

    pool_out = {}
    group_rows = []
    skipped = []

    for (cond, item_idx), members in sorted(mixed.items()):
        for m in members:
            pool_out[m["id"]] = m
        for k in K_VALUES:
            if len(members) < k:
                skipped.append((cond, item_idx, k, len(members)))
                continue
            for g_idx, g in enumerate(build_groups(members, k, rng)):
                sub = [members[i] for i in g["idxs"]]
                n_correct = sum(s["correct"] for s in sub)
                group_rows.append({
                    "condition": cond, "item_idx": item_idx, "k": k,
                    "strategy": g["strategy"], "stratum_entropy": g["stratum_entropy"],
                    "group_idx": g_idx, "member_ids": [s["id"] for s in sub],
                    "n_correct": n_correct, "n_distinct_answers": answer_entropy(sub),
                    "pass_k": int(n_correct >= 1),
                })

    pool_path = Path(args.run_dir) / "raw" / "pool.jsonl"
    groups_path = Path(args.run_dir) / "raw" / "groups.jsonl"
    with open(pool_path, "x") as f:
        for tid, m in pool_out.items():
            f.write(json.dumps({"id": tid, **m}) + "\n")
    with open(groups_path, "x") as f:
        for r in group_rows:
            f.write(json.dumps(r) + "\n")

    print(f"mixed (model,item) groups: {len(mixed)}")
    print(f"pool trajectories referenced: {len(pool_out)}")
    print(f"groups constructed: {len(group_rows)}")
    for k in K_VALUES:
        n_k = sum(1 for r in group_rows if r["k"] == k)
        n_rand = sum(1 for r in group_rows if r["k"] == k and r["strategy"] == "random")
        n_ctrl = sum(1 for r in group_rows if r["k"] == k and r["strategy"] == "controlled")
        print(f"  k={k}: {n_k} groups ({n_rand} random, {n_ctrl} controlled)")
    if skipped:
        print(f"\nskipped (pool smaller than k): {len(skipped)}")
        for cond, item_idx, k, n in skipped:
            print(f"  {cond} item={item_idx} k={k} (pool size {n})")

    pass_k_rate = sum(r["pass_k"] for r in group_rows) / len(group_rows)
    print(f"\noverall pass@k rate across constructed groups: {pass_k_rate:.3f}")
    print(f"Saved: {pool_path}, {groups_path}")


if __name__ == "__main__":
    main()
