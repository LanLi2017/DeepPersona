#!/usr/bin/env python3
"""Validate and summarize the blind tau candidate-quality screens."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_tau_quality_judge import validate


JUDGE_IDS = ("gpt56", "sonnet", "opus")
ROUND_NAMES = ("quality_screen", "quality_screen_round2", "quality_screen_round3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "data/g1/tau_success_v2"
    aggregate: list[dict[str, Any]] = []
    for round_name in ROUND_NAMES:
        directory = base / round_name
        packets = json.loads((directory / "packets.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        trajectory_by_candidate = {
            item["candidate_id"]: item["trajectory_id"]
            for item in provenance["items"]
        }
        expected_ids = {packet["candidate_id"] for packet in packets}
        assert set(trajectory_by_candidate) == expected_ids
        panel: dict[str, dict[str, dict[str, Any]]] = {}
        judge_paths = []
        returned_models = {}
        for judge_id in JUDGE_IDS:
            path = directory / f"{judge_id}.json"
            result = json.loads(path.read_text())
            assert result["judge_id"] == judge_id
            validate(result, packets)
            panel[judge_id] = {
                row["candidate_id"]: row for row in result["assessments"]
            }
            returned_models[judge_id] = result.get("returned_model")
            judge_paths.append(path)

        consensus = []
        for candidate_id in sorted(expected_ids):
            votes = {
                judge_id: panel[judge_id][candidate_id]["high_quality"]
                for judge_id in JUDGE_IDS
            }
            counts = Counter(votes.values())
            majority, majority_count = counts.most_common(1)[0]
            consensus.append({
                "candidate_id": candidate_id,
                "trajectory_id": trajectory_by_candidate[candidate_id],
                "votes": votes,
                "majority_high_quality": majority,
                "unanimous": majority_count == len(JUDGE_IDS),
                "requires_human_review": (
                    majority != "yes" or majority_count != len(JUDGE_IDS)
                ),
            })
        summary = {
            "schema": "g1-tau-quality-screen-summary/1",
            "round": round_name,
            "judges": list(JUDGE_IDS),
            "returned_models": returned_models,
            "candidates": len(consensus),
            "majority_high_quality": dict(
                Counter(row["majority_high_quality"] for row in consensus)
            ),
            "unanimous_candidates": sum(row["unanimous"] for row in consensus),
            "human_review": [
                row["trajectory_id"]
                for row in consensus
                if row["requires_human_review"]
            ],
        }
        consensus_path = directory / "consensus.json"
        summary_path = directory / "summary.json"
        consensus_path.write_text(
            json.dumps(consensus, ensure_ascii=False, indent=2) + "\n"
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        checksum_paths = [*judge_paths, consensus_path, summary_path]
        (directory / "judgment_checksums.sha256").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(root)}"
                for path in sorted(checksum_paths)
            )
            + "\n"
        )
        aggregate.append(summary)
        print(
            f"{round_name}: majority={summary['majority_high_quality']}; "
            f"unanimous={summary['unanimous_candidates']}/{len(consensus)}"
        )
    (base / "quality_screens_summary.json").write_text(
        json.dumps({"rounds": aggregate}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
