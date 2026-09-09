"""
Step 2 of docs/aime_passk_diversity_design.md: augment raw AIME rollouts via LLM revision
(N -> N*r), scoped to only the (model, item) groups that came back MIXED in Step 1 -- degenerate
groups get dropped before analysis anyway (design §4.1), so augmenting them would waste money.

Revision model: openai.gpt-oss-120b on Bedrock (no OPENAI_API_KEY configured in this repo, so we
use a cheap Bedrock-hosted model instead of the gpt-4.1-mini the design doc assumed). Reuses the
PARA_SYS prompt + STYLES from scripts/15_logdist_testbed.py, which already encodes the
manipulation-check requirement ("preserving its logic EXACTLY").

Each revision is re-graded independently against gold (never inherits the parent's label) and
compared to the parent's own correctness to compute the flip rate -- the §4.2 manipulation check.
Expected: flip rate ~= 0. A high flip rate means the rewrite prompt is leaking logic changes and
these revisions should NOT be trusted as "style-only" in the diversity-metric validation (§4.4).

Usage:
  python scripts/29_aime_augment.py <run_dir> --r 3 --models qwen32b llama70b mistrallarge3 --yes
"""
import argparse
import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

REGION = "us-east-2"
REVISION_MODEL = "openai.gpt-oss-120b-1:0"
REVISION_PRICE = (0.15, 0.60)  # $/M in, out -- confirmed via AWS Pricing API 2026-09-02


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s15 = _load(Path(__file__).parent / "15_logdist_testbed.py", "s15")
s28 = _load(Path(__file__).parent / "28_aime_multimodel_rollout.py", "s28")

STYLE_KEYS = ["A_concise", "B_pedagogical", "C_casual", "D_restructured"]


def load_mixed_trajectories(run_dir, models):
    recs = [json.loads(l) for l in open(Path(run_dir) / "raw" / "samples.jsonl")]
    if models:
        recs = [r for r in recs if r["condition"] in models]
    by_group = {}
    for r in recs:
        by_group.setdefault((r["condition"], r["item_idx"]), []).append(r)
    mixed = []
    for (cond, item_idx), group in by_group.items():
        corrects = [g["correct"] for g in group]
        if any(corrects) and not all(corrects):
            mixed.extend(group)
    return mixed


def strict_extract_answer(text):
    """Unlike scripts/28's extract_answer, no last-number-in-text fallback: a revision that never
    reaches a real boxed/stated answer (truncated, or hedged with a symbolic placeholder like
    \\boxed{p+q}) must come back as None, not silently graded via a garbage number -- that
    inflates the §4.2 flip rate with pipeline artifacts instead of genuine logic drift."""
    import re
    m = re.search(r"\\boxed\{(\d+)\}", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:answer is|answer:|final answer[:\s]+)\s*(\d+)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def revise_one(client, parent, style_key, gold):
    style_instr = s15.STYLES[style_key]
    user = f"{style_instr}\n\nSOLUTION TO REWRITE:\n{parent['generation'][:6000]}"
    for attempt in range(4):
        try:
            resp = client.converse(
                modelId=REVISION_MODEL,
                system=[{"text": s15.PARA_SYS}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 12000, "temperature": 0.7},
            )
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 3)
    content = resp["output"]["message"]["content"]
    text = "".join(c.get("text", "") for c in content)
    u = resp.get("usage", {})
    pred = strict_extract_answer(text)
    incomplete = pred is None
    correct = pred is not None and pred == gold
    return {
        "item_idx": parent["item_idx"], "condition": parent["condition"],
        "parent_slot": parent["sample_slot"], "style": style_key,
        "gold": gold, "pred": pred, "correct": correct, "incomplete": incomplete,
        "parent_correct": parent["correct"],
        "flipped": (not incomplete) and correct != parent["correct"],
        "in_tok": u.get("inputTokens", 0), "out_tok": u.get("outputTokens", 0),
        "generation": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--r", type=int, default=3, help="N -> N*r; r-1 revisions per trajectory")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    parents = load_mixed_trajectories(args.run_dir, args.models)
    n_rev = args.r - 1
    jobs = [(p, STYLE_KEYS[i % len(STYLE_KEYS)]) for p in parents for i in range(n_rev)]
    est_cost = sum(len(p["generation"]) // 3 for p, _ in jobs) * REVISION_PRICE[0] / 1e6 \
        + len(jobs) * 800 * REVISION_PRICE[1] / 1e6
    print(f"mixed-group trajectories to augment: {len(parents)}  revisions/traj: {n_rev}  "
          f"total jobs: {len(jobs)}")
    print(f"est cost (rough, gpt-oss-120b): ${est_cost:.2f}")

    if not args.yes:
        ans = input("Proceed? [y/N] ")
        if ans.lower() != "y":
            return

    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(read_timeout=120, connect_timeout=10))
    out_path = Path(args.run_dir) / "raw" / "augmented.jsonl"
    write_lock = threading.Lock()
    results = []
    total_in = total_out = 0

    with open(out_path, "x") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(revise_one, client, p, style, p["gold"]): (p, style)
                for p, style in jobs}
        for i, fut in enumerate(as_completed(futs)):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            results.append(rec)
            total_in += rec["in_tok"]
            total_out += rec["out_tok"]
            with write_lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} done", flush=True)

    n_flipped = sum(r["flipped"] for r in results)
    n_incomplete = sum(r["incomplete"] for r in results)
    n_scored = len(results) - n_incomplete
    cost = total_in * REVISION_PRICE[0] / 1e6 + total_out * REVISION_PRICE[1] / 1e6
    print(f"\nSaved: {out_path}")
    print(f"Revisions: {len(results)}/{len(jobs)}")
    print(f"Incomplete (no clean boxed/stated answer -- excluded from flip rate): "
          f"{n_incomplete}/{len(results)} ({100 * n_incomplete / max(len(results), 1):.1f}%)")
    print(f"Flip rate (among {n_scored} scored revisions): {n_flipped}/{n_scored} "
          f"({100 * n_flipped / max(n_scored, 1):.1f}%)")
    print(f"Tokens: in={total_in:,} out={total_out:,}  actual cost ${cost:.3f}")
    print("\nManipulation check (design §4.2): expected flip rate ~0.")
    if n_scored and n_flipped / n_scored > 0.05:
        print("WARNING: flip rate is non-trivial -- these revisions should NOT be treated as "
              "style-only until the rewrite prompt is investigated (see §4.2).")

    by_style = {}
    for r in results:
        if r["incomplete"]:
            continue
        d = by_style.setdefault(r["style"], [0, 0])
        d[0] += 1
        d[1] += r["flipped"]
    print("\nFlip rate by style (scored revisions only):")
    for style, (n, nf) in by_style.items():
        print(f"  {style:15s} {nf}/{n} ({100 * nf / n:.1f}%)")


if __name__ == "__main__":
    main()
