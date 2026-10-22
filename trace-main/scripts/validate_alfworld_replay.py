#!/usr/bin/env python3
"""Replay every selected ALFWorld trajectory against the copied task assets.

Run with the existing ALFWorld environment and sibling icrl adapter available:

    PYTHONPATH=../icrl ../alfworld-venv/bin/python \
        scripts/validate_alfworld_replay.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_alfworld.alfworld_env import AlfredTextEnv


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--rollout-index", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must be in [0, num-shards)")

    repo_root = Path(__file__).resolve().parents[1]
    normalized_path = repo_root / "data/g1/normalized/trajectories.jsonl"
    all_trajectories = [
        row for row in read_jsonl(normalized_path) if row["benchmark"] == "alfworld"
    ]
    task_ids = list(dict.fromkeys(row["task"]["id"] for row in all_trajectories))
    assigned_task_ids = set(task_ids[args.shard_index :: args.num_shards])
    trajectories = [
        row for row in all_trajectories if row["task"]["id"] in assigned_task_ids
    ]
    if args.rollout_index is not None:
        trajectories = [
            row for row in trajectories if row["rollout"]["index"] == args.rollout_index
        ]
        if not trajectories:
            raise ValueError("rollout index is absent from the selected replay shard")
    mismatches: list[str] = []
    observation_mismatches = 0

    progress = 0
    for task_id in task_ids:
        task_trajectories = [row for row in trajectories if row["task"]["id"] == task_id]
        if not task_trajectories:
            continue
        asset_path = (
            normalized_path.parent
            / task_trajectories[0]["task"]["environment_asset"]
        ).resolve()
        environment = AlfredTextEnv(str(asset_path), max_steps=40)
        try:
            for row in task_trajectories:
                _, info = environment.reset()
                for event in row["events"]:
                    observation, _, done, info = environment.step(event["action"])
                    if observation.strip() != event["observation"].strip():
                        observation_mismatches += 1
                    if done:
                        break
                replayed_success = bool(info["won"])
                if replayed_success != row["rollout"]["success"]:
                    mismatches.append(row["trajectory_id"])
                progress += 1
                print(
                    f"[{progress}/{len(trajectories)}] {row['trajectory_id']} "
                    f"reward_match={replayed_success == row['rollout']['success']}",
                    flush=True,
                )
        finally:
            environment.close()

    report = {
        "schema": "alfworld-replay-validation/1",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rollout_index": args.rollout_index,
        "trajectories": len(trajectories),
        "reward_mismatches": mismatches,
        "observation_mismatches": observation_mismatches,
        "passed": not mismatches and observation_mismatches == 0,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} reward mismatches during replay: {mismatches[:8]}"
        )
    if observation_mismatches:
        raise AssertionError(
            f"{observation_mismatches} step observations differed during replay"
        )
    print(
        f"replayed {len(trajectories)} ALFWorld trajectories: "
        "all rewards and observations match"
    )


if __name__ == "__main__":
    main()
