"""
Analyze rollout diversity JSONL from 01_rollout_diversity.py.

Metrics per method:
  distinct_3    — unique word-trigrams / total trigrams across 8 rollouts per item, mean over items
  answer_ent    — entropy (bits) of A-E answer distribution over 8 rollouts, mean over items
  pairwise_div  — mean pairwise Jaccard distance on unigrams over 8 rollouts, mean over items
  pass_at_1     — accuracy of rollout 0
  pass_at_k     — fraction of items with ≥1 correct rollout
  mean_acc      — mean accuracy across all rollouts

Usage:
  python 02_analyze.py runs/diversity-<ts>.jsonl
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def trigrams(text):
    toks = text.lower().split()
    return [tuple(toks[i:i+3]) for i in range(len(toks) - 2)]


def distinct_3(texts):
    all_tg = [tg for t in texts for tg in trigrams(t)]
    if not all_tg:
        return 0.0
    return len(set(all_tg)) / len(all_tg)


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
            total += 1 - len(inter) / len(union) if union else 0.0
            count += 1
    return total / count


def main():
    path = Path(sys.argv[1])
    records = defaultdict(lambda: defaultdict(list))  # method -> item_idx -> list of records

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("__manifest__"):
                manifest = row
                n_rollouts = manifest.get("n_rollouts", 8)
                print(f"Model: {manifest.get('model', manifest.get('model_id'))}  n_items={manifest['n_items']}  n_rollouts={n_rollouts}\n")
                continue
            records[row["method"]][row["item_idx"]].append(row)

    # Sort rollouts within each item
    for method in records:
        for idx in records[method]:
            records[method][idx].sort(key=lambda r: r["rollout_idx"])

    header = f"{'method':<16}  {'distinct_3':>10}  {'ans_ent':>8}  {'pair_div':>9}  {'pass@1':>7}  {'pass@k':>7}  {'mean_acc':>9}"
    print(header)
    print("-" * len(header))

    for method in sorted(records):
        items_data = records[method]
        d3_scores, ent_scores, pdiv_scores = [], [], []
        pass1_scores, passk_scores, mean_acc_scores = [], [], []

        for idx, recs in items_data.items():
            texts = [r["text"] for r in recs]
            preds = [r["pred"] for r in recs]
            corrects = [r["correct"] for r in recs]

            d3_scores.append(distinct_3(texts))
            ent_scores.append(answer_entropy(preds))
            pdiv_scores.append(pairwise_jaccard_div(texts))
            pass1_scores.append(float(corrects[0]))
            passk_scores.append(float(any(corrects)))
            mean_acc_scores.append(sum(corrects) / len(corrects))

        n = len(items_data)
        print(
            f"{method:<16}  "
            f"{sum(d3_scores)/n:>10.4f}  "
            f"{sum(ent_scores)/n:>8.4f}  "
            f"{sum(pdiv_scores)/n:>9.4f}  "
            f"{sum(pass1_scores)/n:>7.3f}  "
            f"{sum(passk_scores)/n:>7.3f}  "
            f"{sum(mean_acc_scores)/n:>9.3f}"
        )


if __name__ == "__main__":
    main()
