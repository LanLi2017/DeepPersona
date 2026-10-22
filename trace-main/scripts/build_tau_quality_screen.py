#!/usr/bin/env python3
"""Build blind packets for screening replacement tau success candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_tau_success_calibration import (
    collect_sources,
    compact_transcript,
    tool_summary,
)


CANDIDATE_IDS = (
    "taubench-retail:3:memoryT_tau27:r0",
    "taubench-retail:3:memoryT_tau27:r1",
    "taubench-retail:3:memoryT_tau27:r2",
    "taubench-retail:16:memoryT_tau27:r0",
    "taubench-retail:29:memoryT_tau27:r0",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/g1/tau_success_v2/quality_screen",
    )
    args = parser.parse_args()
    candidate_ids = tuple(args.candidate_ids or CANDIDATE_IDS)
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output = output.resolve()
    all_tasks = {
        str(task["id"]): task
        for task in json.loads(
            (root / "../tau2-bench/data/tau2/domains/retail/tasks.json")
            .resolve()
            .read_text()
        )
    }
    task_ids = tuple(sorted({value.split(":")[1] for value in candidate_ids}, key=int))
    pools, _ = collect_sources(root, all_tasks, task_ids)
    by_id = {
        row["trajectory_id"]: row for rows in pools.values() for row in rows
    }
    assert set(candidate_ids) <= set(by_id)
    policy = (root / "data/g1/assets/taubench_retail/policy.md").read_text()
    packets = []
    mapping = []
    for index, trajectory_id in enumerate(candidate_ids, start=1):
        row = by_id[trajectory_id]
        candidate_id = f"quality-{index:02d}"
        packets.append({
            "schema": "g1-tau-quality-screen-packet/1",
            "candidate_id": candidate_id,
            "task_instruction": row["task"]["instruction"],
            "retail_policy": policy,
            "recorded_reward": row["rollout"]["reward"],
            "tool_steps": tool_summary(row["events"]),
            "conversation_evidence": compact_transcript({"events": row["events"]}),
        })
        mapping.append({
            "candidate_id": candidate_id,
            "trajectory_id": trajectory_id,
            "task_id": row["task"]["id"],
        })
    output.mkdir(parents=True, exist_ok=True)
    packet_path = output / "packets.json"
    provenance_path = output / "provenance.json"
    packet_path.write_text(json.dumps(packets, ensure_ascii=False, indent=2) + "\n")
    provenance_path.write_text(
        json.dumps({"schema": "g1-tau-quality-screen-provenance/1", "items": mapping}, indent=2)
        + "\n"
    )
    (output / "checksums.sha256").write_text(
        f"{sha256(packet_path)}  {packet_path.relative_to(root)}\n"
        f"{sha256(provenance_path)}  {provenance_path.relative_to(root)}\n"
    )
    print(f"wrote {len(packets)} blind quality-screen packets")


if __name__ == "__main__":
    main()
