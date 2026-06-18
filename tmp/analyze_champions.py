"""Diagnostic: why do the evolved champion prompts not improve held-out performance?
Examines (1) the champion principle content, (2) per-item solved-set overlap root vs
champion, (3) gen_len / truncation shift, (4) a few root-only / champ-only examples.
Uses the saved scaled-run BeyondAIME eval rollouts. Local only. Delete after use."""
import json
from collections import defaultdict

import numpy as np

RUN = "runs/espl-axis-20260617-212056Z-aime"
TRUNC = 28000
fe = json.load(open(f"{RUN}/final_eval.json"))

print("############## CHAMPION PROMPTS ##############")
for c in fe["champions"]:
    print(f"\n=== id{c['program_id']} ({c['origin']}, {c['n_principles']}p, BeyondAIME pass@1={c['pass1']:.3f}) ===")
    by = defaultdict(list)
    for pid, pr in c["principles"].items():
        by[pr["axis"]].append(pr["text"])
    for ax in ["role", "method", "verification", "pitfalls", "other"]:
        for txt in by.get(ax, []):
            print(f"  [{ax:<12}] {txt}")


def load(tag):
    d = defaultdict(lambda: {"c": 0, "n": 0, "lens": [], "preds": []})
    for line in open(f"{RUN}/raw/eval_{tag}.jsonl"):
        r = json.loads(line)
        s = d[r["item_idx"]]
        s["c"] += r["correct"]; s["n"] += 1; s["lens"].append(r["gen_len"]); s["preds"].append(r["pred"])
        s["gold"] = r["gold"]
    return d


root = load("root")
items = sorted(root)
print("\n############## ROLLOUT COMPARISON (100 BeyondAIME items x 10 samples) ##############")
for tag, cid in [("final-id15", 15), ("final-id14", 14)]:
    ch = load(tag)
    both = ronly = conly = neither = 0
    for i in items:
        cr, cc = root[i]["c"] > 0, ch[i]["c"] > 0
        both += cr and cc; ronly += cr and not cc; conly += cc and not cr; neither += not cr and not cc
    rl = [l for i in items for l in root[i]["lens"]]
    cl = [l for i in items for l in ch[i]["lens"]]
    cr_arr = np.array([root[i]["c"] for i in items]); cc_arr = np.array([ch[i]["c"] for i in items])
    print(f"\n--- root vs id{cid} ---")
    print(f"  solved-set (>=1 of 10): both={both}  root-only={ronly}  champ-only={conly}  neither={neither}")
    print(f"  per-item correct-count corr(root,champ) = {np.corrcoef(cr_arr, cc_arr)[0, 1]:.3f}"
          f"   (mean c/10: root={cr_arr.mean():.2f} champ={cc_arr.mean():.2f})")
    print(f"  mean gen_len: root={np.mean(rl):.0f}  id{cid}={np.mean(cl):.0f}"
          f"   |  truncated@{TRUNC}: root={np.mean([l >= TRUNC for l in rl]):.1%}  id{cid}={np.mean([l >= TRUNC for l in cl]):.1%}")
    if cid == 15:
        ro = [i for i in items if root[i]["c"] > 0 and ch[i]["c"] == 0]
        co = [i for i in items if ch[i]["c"] > 0 and root[i]["c"] == 0]
        print(f"  root-only items (champ regressed): {ro[:8]}")
        print(f"  champ-only items (champ gained):   {co[:8]}")
        for i in (ro[:2] + co[:1]):
            tr = np.mean([l >= TRUNC for l in root[i]["lens"]]); tc = np.mean([l >= TRUNC for l in ch[i]["lens"]])
            print(f"    item {i} gold={root[i]['gold']}: root c={root[i]['c']}/10 (trunc {tr:.0%}) | "
                  f"champ c={ch[i]['c']}/10 (trunc {tc:.0%}) | champ preds={[p for p in ch[i]['preds'] if p][:4]}")
