#!/usr/bin/env python3
"""Aggregate and validate reports from the ALFWorld replay SLURM array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--expected-shards", type=int, default=16)
    args = parser.parse_args()

    paths = sorted(args.report_root.glob("shard_*.json"))
    if len(paths) != args.expected_shards:
        raise AssertionError(
            f"expected {args.expected_shards} shard reports, found {len(paths)}"
        )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    indices = sorted(report["shard_index"] for report in reports)
    if indices != list(range(args.expected_shards)):
        raise AssertionError(f"unexpected shard indices: {indices}")
    failed = [report["shard_index"] for report in reports if not report["passed"]]
    trajectories = sum(report["trajectories"] for report in reports)
    if failed:
        raise AssertionError(f"replay validation failed in shards: {failed}")
    if trajectories != 96:
        raise AssertionError(f"expected 96 replayed trajectories, found {trajectories}")
    print(
        f"validated {trajectories} ALFWorld trajectories across {len(reports)} "
        "shards: all rewards and observations match"
    )


if __name__ == "__main__":
    main()
