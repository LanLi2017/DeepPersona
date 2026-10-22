#!/usr/bin/env python3
"""Validate the committed G1 seed snapshots and normalized trajectories."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data/g1"
    alf_tasks = read_jsonl(root / "source/alfworld_tasks.jsonl")
    tau_tasks = read_jsonl(root / "source/taubench_retail_tasks.jsonl")
    trajectories = read_jsonl(root / "normalized/trajectories.jsonl")
    manifest = json.loads((root / "manifests/manifest.json").read_text(encoding="utf-8"))

    require(len(alf_tasks) == 16, "expected 16 selected ALFWorld tasks")
    require(len(tau_tasks) == 16, "expected 16 selected tau-bench tasks")
    require(sum(len(row["trajectories"]) for row in alf_tasks) == 96, "ALFWorld snapshot count changed")
    require(sum(len(row["episodes"]) for row in tau_tasks) == 64, "tau-bench snapshot count changed")
    require(len(trajectories) == 160, "expected 160 normalized trajectories")
    require(
        Counter(row["task"] for row in alf_tasks)
        == Counter({"clean": 3, "cool": 3, "examine": 3, "heat": 3, "pick_two": 2, "put": 2}),
        "ALFWorld family quotas changed",
    )
    require(
        all(not str(row["gamefile"]).startswith("/") for row in alf_tasks),
        "ALFWorld snapshot leaked an absolute game path",
    )
    require(
        all(all(episode.get("messages") for episode in row["episodes"]) for row in tau_tasks),
        "tau-bench snapshot contains an empty episode",
    )
    ids = [row["trajectory_id"] for row in trajectories]
    require(len(ids) == len(set(ids)), "normalized trajectory IDs are not unique")
    require(
        Counter(row["benchmark"] for row in trajectories)
        == Counter({"alfworld": 96, "taubench_retail": 64}),
        "unexpected normalized benchmark counts",
    )
    alf_asset_files = list((root / "assets/alfworld").glob("**/*"))
    alf_asset_files = [path for path in alf_asset_files if path.is_file()]
    require(len(alf_asset_files) == 48, "expected 48 ALFWorld replay asset files")
    tau_asset_root = root / "assets/taubench_retail"
    tau_asset_files = [path for path in tau_asset_root.glob("*") if path.is_file()]
    require(len(tau_asset_files) == 4, "expected four tau-bench retail asset files")
    tau_asset_tasks = json.loads((tau_asset_root / "tasks.json").read_text(encoding="utf-8"))
    require(
        {str(task["id"]) for task in tau_asset_tasks}
        == {str(task["task_id"]) for task in tau_tasks},
        "tau-bench task assets do not match the selected trajectory tasks",
    )
    for row in trajectories:
        require(row["schema"] == "g1-trajectory/1", "unexpected trajectory schema")
        require(row["rollout"]["length"] == len(row["events"]), "event length mismatch")
        require(bool(row["events"]), "empty normalized trajectory")
        snapshot_line = row["source_ref"]["snapshot_line"]
        rollout_index = row["rollout"]["index"]
        if row["benchmark"] == "alfworld":
            source_task = alf_tasks[snapshot_line - 1]
            source_rollout = source_task["trajectories"][rollout_index]
            require(
                row["rollout"]["success"] == bool(source_rollout["won"]),
                "ALFWorld normalized reward differs from snapshot",
            )
            require(
                [event["action"] for event in row["events"]]
                == [step["action"] for step in source_rollout["steps_log"]],
                "ALFWorld normalized actions differ from snapshot",
            )
            asset = (root / "normalized" / row["task"]["environment_asset"]).resolve()
            require(asset.is_file(), f"missing normalized ALFWorld asset: {asset}")
        else:
            source_task = tau_tasks[snapshot_line - 1]
            source_rollout = source_task["episodes"][rollout_index]
            require(
                row["rollout"]["success"] == bool(source_rollout["won"]),
                "tau-bench normalized reward differs from snapshot",
            )
            require(
                [(event["actor"], event["text"]) for event in row["events"]]
                == [(message["role"], message["text"]) for message in source_rollout["messages"]],
                "tau-bench normalized messages differ from snapshot",
            )

    parse_stats = manifest["normalization_diagnostics"]["taubench_tool_call_parsing"]
    require(parse_stats.get("tool_call_name_mismatches", 0) == 0, "tau tool-call parse mismatch")
    require(parse_stats.get("messages_with_tool_calls", 0) > 0, "no tau tool calls found")

    checksum_rows = (root / "manifests/checksums.sha256").read_text(encoding="utf-8").splitlines()
    require(len(checksum_rows) == 56, "expected 56 generated-file checksums")
    for checksum_row in checksum_rows:
        expected, relative_path = checksum_row.split("  ", 1)
        target = root / relative_path
        require(target.is_file(), f"missing checksummed file: {relative_path}")
        require(sha256_file(target) == expected, f"checksum mismatch: {relative_path}")
    print(
        "validated: 16 ALFWorld tasks / 96 trajectories; "
        "16 tau-bench tasks / 64 trajectories; 160 total"
    )


if __name__ == "__main__":
    main()
