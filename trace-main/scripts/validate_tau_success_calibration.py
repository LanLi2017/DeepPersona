#!/usr/bin/env python3
"""Validate the tau-bench success-first calibration artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_tau_success_calibration import (
    APPROVED_TRAJECTORY_IDS,
    ANNOTATION_SCHEMA,
    RUBRIC_VERSION,
    SCHEMA,
    TASK_IDS,
    WRITE_TOOLS,
    annotation_rows,
    build,
    judge_packets,
    surface_trajectory,
    tool_summary,
    visible_trajectory,
)


FORBIDDEN_BLIND_KEYS = {
    "role",
    "expected_label",
    "source_trajectory_id",
    "anchor_trajectory_id",
    "natural_near",
    "natural_far",
    "surface_equivalent",
    "source_ref",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(walk_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(walk_keys(item) for item in value))
    return set()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data/g1/tau_success_v2"
    triplets = read_jsonl(output / "calibration_triplets.jsonl")
    sources = read_jsonl(output / "source_trajectories.jsonl")
    annotations = read_jsonl(output / "annotations.template.jsonl")
    packets = read_jsonl(output / "judge_packets.jsonl")
    provenance = json.loads((output / "provenance.json").read_text())
    task_specs = json.loads((output / "task_specs.json").read_text())
    source_by_id = {row["trajectory_id"]: row for row in sources}
    triplet_by_id = {row["triplet_id"]: row for row in triplets}
    spec_by_id = {str(row["id"]): row for row in task_specs}
    split = json.loads(
        (root / "data/g1/assets/taubench_retail/split_tasks.json").read_text()
    )

    assert len(triplets) == 4 and len(sources) == 12 and len(annotations) == 12
    retail_policy = (
        root / "data/g1/assets/taubench_retail/policy.md"
    ).read_text(encoding="utf-8")
    assert packets == judge_packets(triplets, retail_policy)
    assert not (walk_keys(packets) & FORBIDDEN_BLIND_KEYS)
    assert [row["task_id"] for row in triplets] == list(TASK_IDS)
    assert len(source_by_id) == len(sources)
    assert len(triplet_by_id) == len(triplets)
    assert set(spec_by_id) == set(TASK_IDS)
    assert set(TASK_IDS) <= set(map(str, split["train"]))
    assert not (set(TASK_IDS) & set(map(str, split["test"])))
    assert annotations == annotation_rows(triplets)
    assert all(row["schema"] == ANNOTATION_SCHEMA for row in annotations)
    assert all(row["rubric_version"] == RUBRIC_VERSION for row in annotations)
    assert not (walk_keys(triplets) & FORBIDDEN_BLIND_KEYS)
    assert provenance["selection"]["approved_trajectory_ids"] == {
        task_id: list(APPROVED_TRAJECTORY_IDS[task_id]) for task_id in TASK_IDS
    }
    assert {row["trajectory_id"] for row in sources} == {
        trajectory_id
        for task_id in TASK_IDS
        for trajectory_id in APPROVED_TRAJECTORY_IDS[task_id]
    }

    for task_id, spec in spec_by_id.items():
        actions = (spec.get("evaluation_criteria") or {}).get("actions") or []
        assert any(action["name"] in WRITE_TOOLS for action in actions), task_id

    for source in sources:
        assert source["rollout"]["success"] is True
        assert source["rollout"]["reward"] >= 1.0
        assert source["quality"]["messages_nonempty"] is True
        assert source["quality"]["tool_errors"] == 0
        assert source["quality"]["human_transfer"] is False
        assert source["quality"]["declared_calls_parsed"] > 0
        assert len(source["events"]) == source["rollout"]["length"]

    natural_ids = set()
    for item in provenance["items"]:
        triplet = triplet_by_id[item["triplet_id"]]
        assert triplet["schema"] == SCHEMA
        assert triplet["track"] == "success_strategy"
        assert triplet["task_id"] == item["task_id"]
        assert item["split"] == "train"
        anchor = source_by_id[item["anchor_trajectory_id"]]
        assert triplet["anchor"] == visible_trajectory(anchor)
        assert triplet["anchor"]["tool_steps"] == tool_summary(triplet["anchor"]["events"])
        roles = [value["role"] for value in item["candidate_slots"].values()]
        assert sorted(roles) == ["natural_far", "natural_near", "surface_equivalent"]
        assert item["natural_candidate_normalized_action_edit"]["near"] \
            <= item["natural_candidate_normalized_action_edit"]["far"]
        for candidate in triplet["candidates"]:
            assert candidate["outcome"] == "success"
            assert candidate["tool_steps"] == tool_summary(candidate["events"])
            role_info = item["candidate_slots"][candidate["slot"]]
            if role_info["role"] == "surface_equivalent":
                expected, transform = surface_trajectory(anchor)
                assert {key: value for key, value in candidate.items() if key != "slot"} \
                    == expected
                assert item["surface_transform"] == transform
            else:
                source_id = role_info["source_trajectory_id"]
                source = source_by_id[source_id]
                natural_ids.add(source_id)
                assert source["task"]["id"] == anchor["task"]["id"]
                assert {key: value for key, value in candidate.items() if key != "slot"} \
                    == visible_trajectory(source)

    assert len(natural_ids) == 8
    surface_slots = Counter(
        slot
        for item in provenance["items"]
        for slot, value in item["candidate_slots"].items()
        if value["role"] == "surface_equivalent"
    )
    assert set(surface_slots) == {"A", "B", "C"}
    assert max(surface_slots.values()) - min(surface_slots.values()) <= 1

    browser = (root / "annotation/tau_success_data.js").read_text()
    prefix = "window.TAU_SUCCESS_DATA = "
    assert browser.startswith(prefix) and browser.endswith(";\n")
    assert json.loads(browser[len(prefix) : -2]) == triplets

    checksum_lines = (output / "checksums.sha256").read_text().splitlines()
    assert len(checksum_lines) == 7
    for line in checksum_lines:
        expected, relative_path = line.split("  ", 1)
        artifact = root / relative_path
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected

    expected_triplets, expected_sources, hidden = build(root)
    assert expected_triplets == triplets
    assert expected_sources == sources
    assert hidden["provenance"] == provenance
    assert hidden["task_specs"] == task_specs
    print("validated: 4 tau success-candidate triplets / 12 comparisons / execution gate")


if __name__ == "__main__":
    main()
