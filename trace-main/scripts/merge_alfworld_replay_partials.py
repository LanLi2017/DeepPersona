#!/usr/bin/env python3
"""Merge rollout-level replay reports into one task-shard report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("partial_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-shard", type=int, required=True)
    args = parser.parse_args()

    paths = sorted(args.partial_root.glob("rollout_*.json"))
    if len(paths) != 6:
        raise AssertionError(f"expected six rollout reports, found {len(paths)}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rollout_indices = sorted(report["rollout_index"] for report in reports)
    if rollout_indices != list(range(6)):
        raise AssertionError(f"unexpected rollout indices: {rollout_indices}")
    merged = {
        "schema": "alfworld-replay-validation/1",
        "shard_index": args.task_shard,
        "num_shards": 16,
        "trajectories": sum(report["trajectories"] for report in reports),
        "reward_mismatches": sum(
            (report["reward_mismatches"] for report in reports), []
        ),
        "observation_mismatches": sum(
            report["observation_mismatches"] for report in reports
        ),
        "passed": all(report["passed"] for report in reports),
        "merged_from_rollout_reports": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    if not merged["passed"]:
        raise AssertionError("one or more rollout-level reports failed")
    print(f"merged six passing rollout reports into {args.output}")


if __name__ == "__main__":
    main()
