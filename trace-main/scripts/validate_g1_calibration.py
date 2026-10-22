#!/usr/bin/env python3
"""Validate the blind G1-P1 calibration package."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_g1_calibration import (
    ALF_FAMILIES,
    ALF_PREFERRED_OUTCOMES,
    ANNOTATION_SCHEMA,
    RUBRIC_VERSION,
    TAU_TASK_IDS,
    annotation_rows,
    transform_alf_surface,
    transform_tau_surface,
    visible_trajectory,
)


FORBIDDEN_BLIND_KEYS = {
    "role",
    "expected_label",
    "source_trajectory_id",
    "grounded_trajectory_id",
    "natural_near",
    "natural_far",
    "surface_equivalent",
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
    pilot = root / "data/g1/pilot"
    triplets = read_jsonl(pilot / "calibration_triplets.jsonl")
    annotations = read_jsonl(pilot / "annotations.template.jsonl")
    provenance = json.loads((pilot / "provenance.json").read_text(encoding="utf-8"))
    tau_splits = json.loads(
        (root / "data/g1/assets/taubench_retail/split_tasks.json").read_text(encoding="utf-8")
    )
    normalized = {
        row["trajectory_id"]: row
        for row in read_jsonl(root / "data/g1/normalized/trajectories.jsonl")
    }

    assert len(triplets) == 8
    assert Counter(item["benchmark"] for item in triplets) == {
        "alfworld": 4,
        "taubench_retail": 4,
    }
    assert {
        item["anchor"]["outcome"]
        for item in triplets
        if item["benchmark"] == "alfworld"
    } == {"success", "failure"}
    assert {
        item["anchor"]["outcome"]
        for item in triplets
        if item["benchmark"] == "taubench_retail"
    } == {"success", "failure"}
    assert len(annotations) == 24
    assert annotations == annotation_rows(triplets)
    assert all(row["schema"] == ANNOTATION_SCHEMA for row in annotations)
    assert all(row["rubric_version"] == RUBRIC_VERSION for row in annotations)
    assert all(row["annotator_id"] is None for row in annotations)
    assert len(provenance["items"]) == 8
    assert provenance["selection"]["alfworld_families"] == list(ALF_FAMILIES)
    assert provenance["selection"]["alfworld_preferred_outcomes"] \
        == ALF_PREFERRED_OUTCOMES
    assert provenance["selection"]["taubench_retail_task_ids"] == list(TAU_TASK_IDS)
    assert not (walk_keys(triplets) & FORBIDDEN_BLIND_KEYS)
    blind_payload = json.dumps(triplets, ensure_ascii=False)
    assert not any(value in blind_payload for value in FORBIDDEN_BLIND_KEYS)

    annotation_keys = {
        (row["triplet_id"], row["candidate_slot"]) for row in annotations
    }
    visible_keys = set()
    for triplet in triplets:
        assert triplet["schema"] == "g1-calibration-triplet/1"
        assert {candidate["slot"] for candidate in triplet["candidates"]} == {"A", "B", "C"}
        anchor_outcome = triplet["anchor"]["outcome"]
        anchor_length = len(triplet["anchor"]["events"])
        assert anchor_length == triplet["anchor"]["length"]
        for candidate in triplet["candidates"]:
            assert candidate["outcome"] == anchor_outcome
            assert len(candidate["events"]) == candidate["length"]
            visible_keys.add((triplet["triplet_id"], candidate["slot"]))
    assert annotation_keys == visible_keys

    for item in provenance["items"]:
        assert item["split"] == "train"
        if item["benchmark"] == "taubench_retail":
            assert item["task_id"] in tau_splits["train"]
            assert item["task_id"] not in tau_splits["test"]
        anchor_source = normalized[item["anchor_trajectory_id"]]
        visible = next(row for row in triplets if row["triplet_id"] == item["triplet_id"])
        assert visible["anchor"] == visible_trajectory(anchor_source)
        expected_surface, expected_transform = (
            transform_alf_surface(anchor_source)
            if item["benchmark"] == "alfworld"
            else transform_tau_surface(anchor_source)
        )
        roles = [value["role"] for value in item["candidate_slots"].values()]
        assert sorted(roles) == ["natural_far", "natural_near", "surface_equivalent"]
        assert item["natural_candidate_normalized_action_edit"]["near"] \
            <= item["natural_candidate_normalized_action_edit"]["far"]
        surface_slot = next(
            slot
            for slot, value in item["candidate_slots"].items()
            if value["role"] == "surface_equivalent"
        )
        surface = next(row for row in visible["candidates"] if row["slot"] == surface_slot)
        assert {key: value for key, value in surface.items() if key != "slot"} \
            == expected_surface
        assert item["surface_transform"] == expected_transform
        assert surface["events"] != visible["anchor"]["events"]
        assert item["surface_transform"]["grounded_trajectory_id"] \
            == item["anchor_trajectory_id"]
        mapping = item["surface_transform"].get(
            "entity_mapping",
            item["surface_transform"].get("identifier_mapping", {}),
        )
        assert mapping and len(mapping.values()) == len(set(mapping.values()))
        natural_ids = [
            value["source_trajectory_id"]
            for value in item["candidate_slots"].values()
            if value["source_trajectory_id"] is not None
        ]
        assert len(natural_ids) == 2 and len(set(natural_ids)) == 2
        for trajectory_id in natural_ids:
            candidate_source = normalized[trajectory_id]
            assert candidate_source["task"]["id"] == anchor_source["task"]["id"]
            assert candidate_source["rollout"]["success"] \
                == anchor_source["rollout"]["success"]
            source_slot = next(
                slot
                for slot, value in item["candidate_slots"].items()
                if value["source_trajectory_id"] == trajectory_id
            )
            candidate = next(
                row for row in visible["candidates"] if row["slot"] == source_slot
            )
            assert {key: value for key, value in candidate.items() if key != "slot"} \
                == visible_trajectory(candidate_source)
        if item["benchmark"] == "alfworld":
            assert [event["kind"] for event in surface["events"]] \
                == [event["kind"] for event in visible["anchor"]["events"]]
            assert [event["matched"] for event in surface["events"]] \
                == [event["matched"] for event in visible["anchor"]["events"]]
        else:
            assert [event["actor"] for event in surface["events"]] \
                == [event["actor"] for event in visible["anchor"]["events"]]
            surface_tools = [
                call["name"] for event in surface["events"]
                for call in event.get("tool_calls", [])
            ]
            anchor_tools = [
                call["name"] for event in visible["anchor"]["events"]
                for call in event.get("tool_calls", [])
            ]
            assert surface_tools == anchor_tools

    surface_slots = Counter(
        slot
        for item in provenance["items"]
        for slot, value in item["candidate_slots"].items()
        if value["role"] == "surface_equivalent"
    )
    assert set(surface_slots) == {"A", "B", "C"}
    assert max(surface_slots.values()) - min(surface_slots.values()) <= 1

    browser_data = (root / "annotation/calibration_data.js").read_text(encoding="utf-8")
    prefix = "window.G1_CALIBRATION_DATA = "
    assert browser_data.startswith(prefix) and browser_data.endswith(";\n")
    embedded = json.loads(browser_data[len(prefix) : -2])
    assert embedded == triplets

    checksum_lines = (pilot / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 4
    for line in checksum_lines:
        expected, relative_path = line.split("  ", 1)
        artifact = root / relative_path
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
    print("validated: 8 blind triplets / 24 comparisons; no provenance keys leaked")


if __name__ == "__main__":
    main()
