#!/usr/bin/env python3
"""Build the eight-anchor blind calibration batch for G1-P1."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "g1-calibration-triplet/1"
ANNOTATION_SCHEMA = "g1-calibration-annotation/1"
RUBRIC_VERSION = "g1-e0-e2-v1"
ALF_FAMILIES = ("clean", "cool", "examine", "pick_two")
# Include one successful ALFWorld anchor without relaxing same-outcome matching.
# The other families retain the deterministic maximum-diversity selection rule.
ALF_PREFERRED_OUTCOMES = {"cool": True}
TAU_TASK_IDS = ("14", "30", "47", "57")
BLIND_ROLE_ORDERS = (
    ("surface_equivalent", "natural_near", "natural_far"),
    ("natural_near", "surface_equivalent", "natural_far"),
    ("natural_far", "natural_near", "surface_equivalent"),
    ("surface_equivalent", "natural_far", "natural_near"),
    ("natural_far", "surface_equivalent", "natural_near"),
    ("natural_near", "natural_far", "surface_equivalent"),
)
ALF_ENTITY_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)\s+(\d+)\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
ORDER_RE = re.compile(r"#?W\d+\b")
USER_RE = re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def alpha_label(index: int) -> str:
    value = index
    output = ""
    while True:
        output = chr(ord("A") + value % 26) + output
        value = value // 26 - 1
        if value < 0:
            return output


def action_sequence(row: dict[str, Any]) -> tuple[str, ...]:
    if row["benchmark"] == "alfworld":
        return tuple(event["action"] for event in row["events"])
    return tuple(
        f"{call['name']}({json.dumps(call.get('arguments') or {}, sort_keys=True)})"
        for event in row["events"]
        for call in event.get("tool_calls", [])
    )


def edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    return edit_distance(left, right) / max(len(left), len(right), 1)


def viable_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_outcome: dict[bool, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_outcome[bool(row["rollout"]["success"])].append(row)
    return [
        group
        for group in by_outcome.values()
        if len(group) >= 3 and len({action_sequence(row) for row in group}) >= 3
    ]


def group_spread(group: list[dict[str, Any]]) -> float:
    return max(
        normalized_edit(action_sequence(left), action_sequence(right))
        for index, left in enumerate(group)
        for right in group[index + 1 :]
    )


def select_group(
    rows: list[dict[str, Any]], preferred_outcome: bool | None = None
) -> list[dict[str, Any]]:
    groups = viable_groups(rows)
    if not groups:
        raise ValueError(f"task {rows[0]['task']['id']} has no viable outcome group")
    if preferred_outcome is not None:
        groups = [
            group
            for group in groups
            if bool(group[0]["rollout"]["success"]) == preferred_outcome
        ]
        if not groups:
            raise ValueError(
                f"task {rows[0]['task']['id']} has no viable "
                f"outcome={preferred_outcome} group"
            )
    return sorted(
        groups,
        key=lambda group: (
            -len({action_sequence(row) for row in group}),
            -group_spread(group),
            -len(group),
            bool(group[0]["rollout"]["success"]),
        ),
    )[0]


def select_anchor_and_naturals(
    group: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
    distances: dict[tuple[str, str], float] = {}
    for left in group:
        for right in group:
            if left is right:
                continue
            distances[(left["trajectory_id"], right["trajectory_id"])] = normalized_edit(
                action_sequence(left), action_sequence(right)
            )

    anchor = min(
        group,
        key=lambda row: (
            sum(
                distances[(row["trajectory_id"], other["trajectory_id"])]
                for other in group
                if other is not row
            )
            / (len(group) - 1),
            row["rollout"]["index"],
        ),
    )
    alternatives = [
        row
        for row in group
        if row is not anchor and action_sequence(row) != action_sequence(anchor)
    ]
    if len(alternatives) < 2:
        raise ValueError(f"anchor {anchor['trajectory_id']} lacks two distinct candidates")
    alternatives.sort(
        key=lambda row: (
            distances[(anchor["trajectory_id"], row["trajectory_id"])],
            row["rollout"]["index"],
        )
    )
    near = alternatives[0]
    far = alternatives[-1]
    return anchor, near, far, {
        "near": distances[(anchor["trajectory_id"], near["trajectory_id"])],
        "far": distances[(anchor["trajectory_id"], far["trajectory_id"])],
    }


def visible_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_instruction": row["task"]["instruction"],
        "outcome": "success" if row["rollout"]["success"] else "failure",
        "length": row["rollout"]["length"],
        "events": copy.deepcopy(row["events"]),
    }


def replace_alf_entities(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    mapping: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    for text in texts:
        for match in ALF_ENTITY_RE.finditer(text):
            original = match.group(0)
            if original not in mapping:
                entity_type = match.group(1)
                counts[entity_type] += 1
                mapping[original] = f"{entity_type} instance_{alpha_label(counts[entity_type] - 1)}"

    pattern = re.compile(
        "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
    ) if mapping else None
    transformed = [pattern.sub(lambda match: mapping[match.group(0)], text) for text in texts] \
        if pattern else list(texts)
    return transformed, mapping


def transform_alf_surface(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    visible = visible_trajectory(row)
    strings = [row["task"]["instruction"]]
    for event in visible["events"]:
        strings.extend([event["action"], event["observation"]])
    transformed, mapping = replace_alf_entities(strings)
    cursor = iter(transformed)
    instruction = next(cursor)
    aliases = {
        "go to ": "navigate to ",
        "take ": "pick up ",
        "move ": "place ",
        "examine ": "inspect ",
        "clean ": "wash ",
        "heat ": "warm ",
        "cool ": "chill ",
    }
    observation_aliases = {
        "You arrive at ": "You reach ",
        "You pick up ": "You take ",
        "You move ": "You place ",
        "You clean ": "You wash ",
        "You heat ": "You warm ",
        "You cool ": "You chill ",
    }
    for event in visible["events"]:
        action = next(cursor)
        observation = next(cursor)
        for source, target in aliases.items():
            if action.startswith(source):
                action = target + action[len(source) :]
                break
        for source, target in observation_aliases.items():
            if observation.startswith(source):
                observation = target + observation[len(source) :]
                break
        event["action"] = action
        event["observation"] = observation
    visible["task_instruction"] = instruction
    return visible, {
        "method": "consistent_entity_renaming_and_fixed_action_aliases",
        "entity_mapping": mapping,
        "grounded_trajectory_id": row["trajectory_id"],
    }


def identifier_values(row: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    counters: dict[str, int] = defaultdict(int)

    def add(value: str, category: str) -> None:
        if not value or value in values:
            return
        counters[category] += 1
        values[value] = f"<{category}_{alpha_label(counters[category] - 1)}>"

    def visit_argument(key: str, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit_argument(key, item)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                visit_argument(str(nested_key), nested_value)
        elif isinstance(value, str):
            lowered = key.lower()
            if "email" in lowered:
                add(value, "EMAIL")
            elif "order" in lowered and "id" in lowered:
                add(value, "ORDER")
            elif "id" in lowered:
                add(value, "ID")

    all_text = [row["task"]["instruction"]]
    for event in row["events"]:
        all_text.append(event.get("text") or "")
        for call in event.get("tool_calls", []):
            for key, value in (call.get("arguments") or {}).items():
                visit_argument(str(key), value)
    joined = "\n".join(all_text)
    for value in EMAIL_RE.findall(joined):
        add(value, "EMAIL")
    for value in ORDER_RE.findall(joined):
        add(value, "ORDER")
    for value in USER_RE.findall(joined):
        add(value, "USER")
    return values


def replace_recursive(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for original in sorted(mapping, key=len, reverse=True):
            value = value.replace(original, mapping[original])
        return value
    if isinstance(value, list):
        return [replace_recursive(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: replace_recursive(item, mapping) for key, item in value.items()}
    return value


def transform_tau_surface(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    visible = visible_trajectory(row)
    mapping = identifier_values(row)
    visible = replace_recursive(visible, mapping)
    for event in visible["events"]:
        if event.get("tool_calls"):
            event["text"] = "\n".join(
                f"CALL {call['name']} WITH "
                f"{json.dumps(call.get('arguments') or {}, ensure_ascii=False, sort_keys=True)}"
                for call in event["tool_calls"]
            )
        elif event["actor"] == "tool":
            event["text"] = f"TOOL RESULT: {event['text']}"
    return visible, {
        "method": "consistent_identifier_anonymization_and_tool_serialization",
        "identifier_mapping": mapping,
        "grounded_trajectory_id": row["trajectory_id"],
    }


def choose_tasks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]["id"]].append(row)

    selected_groups: list[list[dict[str, Any]]] = []
    for family in ALF_FAMILIES:
        preferred_outcome = ALF_PREFERRED_OUTCOMES.get(family)
        candidates = [
            task_rows
            for task_rows in by_task.values()
            if task_rows[0]["benchmark"] == "alfworld"
            and task_rows[0]["task"]["family"] == family
            and any(
                preferred_outcome is None
                or bool(group[0]["rollout"]["success"]) == preferred_outcome
                for group in viable_groups(task_rows)
            )
        ]
        candidates.sort(
            key=lambda task_rows: (
                -max(
                    group_spread(group)
                    for group in viable_groups(task_rows)
                    if preferred_outcome is None
                    or bool(group[0]["rollout"]["success"]) == preferred_outcome
                ),
                task_rows[0]["task"]["id"],
            )
        )
        if not candidates:
            raise ValueError(f"no viable ALFWorld calibration task for {family}")
        selected_groups.append(select_group(candidates[0], preferred_outcome))

    for task_id in TAU_TASK_IDS:
        task_rows = [
            row
            for row in by_task.get(task_id, [])
            if row["benchmark"] == "taubench_retail"
        ]
        if not task_rows:
            raise ValueError(f"missing tau-bench calibration task {task_id}")
        selected_groups.append(select_group(task_rows))
    return selected_groups


def build_triplets(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    triplets: list[dict[str, Any]] = []
    provenance_items: list[dict[str, Any]] = []
    for item_index, group in enumerate(choose_tasks(rows), start=1):
        anchor, near, far, distances = select_anchor_and_naturals(group)
        triplet_id = f"cal-{item_index:02d}"
        if anchor["benchmark"] == "alfworld":
            surface, transform = transform_alf_surface(anchor)
        else:
            surface, transform = transform_tau_surface(anchor)
        candidate_by_role = {
            "surface_equivalent": (surface, None),
            "natural_near": (visible_trajectory(near), near["trajectory_id"]),
            "natural_far": (visible_trajectory(far), far["trajectory_id"]),
        }
        role_order = BLIND_ROLE_ORDERS[(item_index - 1) % len(BLIND_ROLE_ORDERS)]
        internal_candidates = [
            (role, *candidate_by_role[role]) for role in role_order
        ]
        visible_candidates = []
        slot_provenance = {}
        for slot_index, (role, candidate, source_id) in enumerate(internal_candidates):
            slot = alpha_label(slot_index)
            visible_candidates.append({"slot": slot, **candidate})
            slot_provenance[slot] = {
                "role": role,
                "source_trajectory_id": source_id,
            }
        triplets.append({
            "schema": SCHEMA,
            "triplet_id": triplet_id,
            "benchmark": anchor["benchmark"],
            "task_instruction": anchor["task"]["instruction"],
            "anchor": visible_trajectory(anchor),
            "candidates": visible_candidates,
        })
        provenance_items.append({
            "triplet_id": triplet_id,
            "benchmark": anchor["benchmark"],
            "task_id": anchor["task"]["id"],
            "split": "train",
            "anchor_trajectory_id": anchor["trajectory_id"],
            "outcome": anchor["rollout"]["success"],
            "candidate_slots": slot_provenance,
            "natural_candidate_normalized_action_edit": distances,
            "surface_transform": transform,
        })
    return triplets, {
        "schema": "g1-calibration-provenance/1",
        "selection": {
            "alfworld_families": list(ALF_FAMILIES),
            "alfworld_preferred_outcomes": ALF_PREFERRED_OUTCOMES,
            "taubench_retail_task_ids": list(TAU_TASK_IDS),
            "natural_candidate_rule": (
                "same-task and same-outcome; nearest and farthest distinct "
                "action sequences from a deterministic medoid anchor"
            ),
        },
        "items": provenance_items,
    }


def annotation_rows(triplets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema": ANNOTATION_SCHEMA,
            "rubric_version": RUBRIC_VERSION,
            "annotator_id": None,
            "triplet_id": triplet["triplet_id"],
            "candidate_slot": candidate["slot"],
            "same_information": None,
            "same_state_effect": None,
            "same_subgoal_order": None,
            "same_recovery_behavior": None,
            "training_interchangeable": None,
            "label": None,
            "confidence": None,
            "notes": "",
        }
        for triplet in triplets
        for candidate in triplet["candidates"]
    ]


def write_browser_data(path: Path, triplets: list[dict[str, Any]]) -> None:
    payload = json.dumps(triplets, ensure_ascii=False).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"window.G1_CALIBRATION_DATA = {payload};\n", encoding="utf-8")


def write_checksums(root: Path, paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    (root / "data/g1/pilot/checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = read_jsonl(root / "data/g1/normalized/trajectories.jsonl")
    triplets, provenance = build_triplets(rows)
    pilot_root = root / "data/g1/pilot"
    write_jsonl(pilot_root / "calibration_triplets.jsonl", triplets)
    write_jsonl(pilot_root / "annotations.template.jsonl", annotation_rows(triplets))
    (pilot_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    browser_data = root / "annotation/calibration_data.js"
    write_browser_data(browser_data, triplets)
    write_checksums(
        root,
        [
            pilot_root / "calibration_triplets.jsonl",
            pilot_root / "annotations.template.jsonl",
            pilot_root / "provenance.json",
            browser_data,
        ],
    )
    print(
        f"wrote {len(triplets)} triplets / "
        f"{sum(len(item['candidates']) for item in triplets)} comparisons"
    )


if __name__ == "__main__":
    main()
