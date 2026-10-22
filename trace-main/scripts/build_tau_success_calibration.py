#!/usr/bin/env python3
"""Build the tau-bench success-strategy candidate calibration batch.

The builder merges existing train-split rollout pools, applies a mechanical
execution gate, and deterministically selects three reward-positive
trajectories per task.
One is the anchor, while the other two are blind natural candidates. A third
candidate is a deterministic E0 rendering of the anchor.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_g1_calibration import (
    alpha_label,
    normalized_edit,
    transform_tau_surface,
)


SCHEMA = "g1-tau-success-triplet/1"
ANNOTATION_SCHEMA = "g1-tau-success-annotation/1"
RUBRIC_VERSION = "g1-success-strategy-v2"
TASK_IDS = ("16", "28", "69", "75")
APPROVED_TRAJECTORY_IDS = {
    "16": (
        "taubench-retail:16:memory4b_tau:r2",
        "taubench-retail:16:memoryT_tau27:r0",
        "taubench-retail:16:memoryT_tau27:r1",
    ),
    "28": (
        "taubench-retail:28:memoryT_tau27:r0",
        "taubench-retail:28:memoryT_tau27:r1",
        "taubench-retail:28:memoryT_tau27:r2",
    ),
    "69": (
        "taubench-retail:69:memoryT_tau27:r0",
        "taubench-retail:69:memoryT_tau27:r1",
        "taubench-retail:69:memoryT_tau27:r2",
    ),
    "75": (
        "taubench-retail:75:memoryT_tau27:r0",
        "taubench-retail:75:memoryT_tau27:r1",
        "taubench-retail:75:memoryT_tau27:r2",
    ),
}
SOURCE_LOCATORS = (
    "../npm/repro_tau/results/memory4b_tau.jsonl",
    "../npm/repro_tau/results/memoryT_tau27.jsonl",
    "../npm/repro_tau/results/memory_tau27_retail.jsonl",
    "../npm/repro_tau/results/memory_tau_retail.jsonl",
)
BLIND_ROLE_ORDERS = (
    ("surface_equivalent", "natural_near", "natural_far"),
    ("natural_far", "surface_equivalent", "natural_near"),
    ("natural_near", "natural_far", "surface_equivalent"),
    ("surface_equivalent", "natural_far", "natural_near"),
)
JSON_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
XML_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(.*?)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
XML_PARAM_RE = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)
WRITE_TOOLS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coerce_xml_value(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false", "null"} or value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def parse_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(message.get("text") or "")
    calls: list[dict[str, Any]] = []
    for match in JSON_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("name"):
            arguments = payload.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append({"name": str(payload["name"]), "arguments": arguments})
    if not calls:
        for match in XML_CALL_RE.finditer(text):
            arguments = {
                key.strip(): coerce_xml_value(value)
                for key, value in XML_PARAM_RE.findall(match.group(2))
            }
            calls.append({"name": match.group(1).strip(), "arguments": arguments})
    declared = [str(name) for name in (message.get("tool_calls") or [])]
    if [call["name"] for call in calls] != declared:
        raise ValueError(
            f"tool parse mismatch: declared={declared}, "
            f"parsed={[call['name'] for call in calls]}"
        )
    return calls


def normalize_episode(
    task_id: str,
    query: str,
    source_path: Path,
    source_sha256: str,
    rollout_index: int,
    episode: dict[str, Any],
) -> dict[str, Any]:
    events = []
    for event_index, message in enumerate(episode["messages"]):
        event = {
            "index": event_index,
            "kind": "message",
            "actor": str(message.get("role") or "unknown"),
            "text": str(message.get("text") or ""),
        }
        calls = parse_calls(message)
        if calls:
            event["tool_calls"] = calls
        if "is_error" in message:
            event["is_error"] = bool(message["is_error"])
        events.append(event)
    source_name = source_path.name
    trajectory_id = f"taubench-retail:{task_id}:{source_path.stem}:r{rollout_index}"
    tool_names = [
        call["name"] for event in events for call in event.get("tool_calls", [])
    ]
    return {
        "schema": "g1-tau-success-source/1",
        "trajectory_id": trajectory_id,
        "benchmark": "taubench_retail",
        "task": {"id": task_id, "family": "retail", "instruction": query},
        "rollout": {
            "index": rollout_index,
            "success": bool(episode["won"]),
            "reward": float(episode["reward"]),
            "length": len(events),
        },
        "events": events,
        "quality": {
            "messages_nonempty": bool(events),
            "tool_errors": sum(bool(event.get("is_error")) for event in events),
            "human_transfer": "transfer_to_human_agents" in tool_names,
            "declared_calls_parsed": len(tool_names),
        },
        "source_ref": {
            "file": source_name,
            "sha256": source_sha256,
            "rollout_index": rollout_index,
        },
    }


def action_sequence(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        f"{call['name']}({json.dumps(call.get('arguments') or {}, sort_keys=True)})"
        for event in row["events"]
        for call in event.get("tool_calls", [])
    )


def tool_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for event in events:
        for call in event.get("tool_calls", []):
            summary.append({
                "event_index": event["index"],
                "name": call["name"],
                "arguments": copy.deepcopy(call.get("arguments") or {}),
            })
    return summary


def visible_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    events = copy.deepcopy(row["events"])
    return {
        "task_instruction": row["task"]["instruction"],
        "outcome": "success",
        "length": len(events),
        "tool_steps": tool_summary(events),
        "events": events,
    }


def surface_trajectory(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    transformed, provenance = transform_tau_surface(row)
    transformed["tool_steps"] = tool_summary(transformed["events"])
    return transformed, provenance


def execution_gate(row: dict[str, Any]) -> bool:
    return (
        row["rollout"]["success"]
        and row["rollout"]["reward"] >= 1.0
        and row["quality"]["messages_nonempty"]
        and row["quality"]["tool_errors"] == 0
        and not row["quality"]["human_transfer"]
        and bool(action_sequence(row))
    )


def choose_three(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["trajectory_id"]):
        unique.setdefault(action_sequence(row), row)
    candidates = list(unique.values())
    if len(candidates) < 3:
        raise ValueError(f"task {rows[0]['task']['id']} lacks 3 strict unique successes")

    def score(group: tuple[dict[str, Any], ...]) -> tuple[Any, ...]:
        distances = [
            normalized_edit(action_sequence(left), action_sequence(right))
            for left, right in itertools.combinations(group, 2)
        ]
        sources = {row["source_ref"]["file"] for row in group}
        return (
            -len(sources),
            -min(distances),
            -sum(distances),
            tuple(row["trajectory_id"] for row in group),
        )

    return list(min(itertools.combinations(candidates, 3), key=score))


def anchor_near_far(
    group: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
    def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        return normalized_edit(action_sequence(left), action_sequence(right))

    anchor = min(
        group,
        key=lambda row: (
            sum(distance(row, other) for other in group if other is not row),
            row["trajectory_id"],
        ),
    )
    alternatives = sorted(
        (row for row in group if row is not anchor),
        key=lambda row: (distance(anchor, row), row["trajectory_id"]),
    )
    near, far = alternatives
    return anchor, near, far, {
        "near": distance(anchor, near),
        "far": distance(anchor, far),
    }


def collect_sources(
    root: Path,
    tasks: dict[str, dict[str, Any]],
    task_ids: tuple[str, ...] = TASK_IDS,
) -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, str]
]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_hashes = {}
    for locator in SOURCE_LOCATORS:
        path = (root / locator).resolve()
        source_hashes[path.name] = sha256_file(path)
        for source_row in read_jsonl(path):
            task_id = str(source_row["task_id"])
            if task_id not in task_ids:
                continue
            query = str(source_row.get("query") or "")
            for rollout_index, episode in enumerate(source_row["episodes"]):
                if not episode.get("messages"):
                    continue
                row = normalize_episode(
                    task_id,
                    query,
                    path,
                    source_hashes[path.name],
                    rollout_index,
                    episode,
                )
                if execution_gate(row):
                    selected[task_id].append(row)
    for task_id in task_ids:
        criterion = tasks[task_id]["evaluation_criteria"]
        expected_writes = [
            action for action in (criterion.get("actions") or [])
            if action["name"] in WRITE_TOOLS
        ]
        if not expected_writes:
            raise ValueError(f"task {task_id} lacks a non-vacuous DB write target")
        if len(selected[task_id]) < 3:
            raise ValueError(f"task {task_id} lacks strict successful episodes")
    return selected, source_hashes


def build(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_path = (root / "../tau2-bench/data/tau2/domains/retail/tasks.json").resolve()
    split_path = (
        root / "../tau2-bench/data/tau2/domains/retail/split_tasks.json"
    ).resolve()
    all_tasks = {str(task["id"]): task for task in json.loads(task_path.read_text())}
    split = json.loads(split_path.read_text())
    train, test = set(map(str, split["train"])), set(map(str, split["test"]))
    assert set(TASK_IDS) <= train and not (set(TASK_IDS) & test)
    selected, source_hashes = collect_sources(root, all_tasks)

    source_rows: list[dict[str, Any]] = []
    triplets: list[dict[str, Any]] = []
    provenance_items = []
    for item_index, task_id in enumerate(TASK_IDS, start=1):
        by_id = {row["trajectory_id"]: row for row in selected[task_id]}
        approved_ids = APPROVED_TRAJECTORY_IDS[task_id]
        missing = set(approved_ids) - set(by_id)
        if missing:
            raise ValueError(f"approved trajectories missing from gate: {sorted(missing)}")
        group = [by_id[trajectory_id] for trajectory_id in approved_ids]
        source_rows.extend(group)
        anchor, near, far, distances = anchor_near_far(group)
        surface, transform = surface_trajectory(anchor)
        candidate_by_role = {
            "surface_equivalent": (surface, None),
            "natural_near": (visible_trajectory(near), near["trajectory_id"]),
            "natural_far": (visible_trajectory(far), far["trajectory_id"]),
        }
        role_order = BLIND_ROLE_ORDERS[item_index - 1]
        visible_candidates = []
        slot_provenance = {}
        for slot_index, role in enumerate(role_order):
            slot = alpha_label(slot_index)
            candidate, source_id = candidate_by_role[role]
            visible_candidates.append({"slot": slot, **candidate})
            slot_provenance[slot] = {"role": role, "source_trajectory_id": source_id}
        triplet_id = f"tau-success-{item_index:02d}"
        triplets.append({
            "schema": SCHEMA,
            "track": "success_strategy",
            "triplet_id": triplet_id,
            "benchmark": "taubench_retail",
            "task_id": task_id,
            "task_instruction": anchor["task"]["instruction"],
            "anchor": visible_trajectory(anchor),
            "candidates": visible_candidates,
        })
        provenance_items.append({
            "triplet_id": triplet_id,
            "task_id": task_id,
            "split": "train",
            "anchor_trajectory_id": anchor["trajectory_id"],
            "candidate_slots": slot_provenance,
            "natural_candidate_normalized_action_edit": distances,
            "surface_transform": transform,
            "execution_gate": {
                "success": True,
                "reward_at_least_one": True,
                "non_vacuous_db_write_target": True,
                "tool_errors": 0,
                "human_transfer": False,
            },
        })
    provenance = {
        "schema": "g1-tau-success-provenance/1",
        "selection": {
            "task_ids": list(TASK_IDS),
            "approved_trajectory_ids": {
                task_id: list(APPROVED_TRAJECTORY_IDS[task_id])
                for task_id in TASK_IDS
            },
            "split": "train",
            "source_sha256": dict(sorted(source_hashes.items())),
            "triplet_rule": (
                "freeze three independently generated trajectories that passed "
                "the execution gate and received majority high-quality approval "
                "in the three-model candidate screen"
            ),
            "anchor_rule": "medoid of the selected three trajectories",
        },
        "items": provenance_items,
    }
    task_specs = [all_tasks[task_id] for task_id in TASK_IDS]
    return triplets, sorted(source_rows, key=lambda row: row["trajectory_id"]), {
        "provenance": provenance,
        "task_specs": task_specs,
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
            "evidence": "",
        }
        for triplet in triplets
        for candidate in triplet["candidates"]
    ]


def compact_transcript(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for event in trajectory["events"]:
        actor = event["actor"]
        if event.get("tool_calls"):
            continue
        if actor not in {"user", "assistant", "tool"}:
            continue
        text = THINK_RE.sub("", str(event.get("text") or "")).strip()
        if not text:
            continue
        output.append({
            "event_index": event["index"],
            "actor": actor,
            "text": text,
            **({"is_error": bool(event.get("is_error"))} if actor == "tool" else {}),
        })
    return output


def judge_packets(triplets: list[dict[str, Any]], retail_policy: str) -> list[dict[str, Any]]:
    def compact(trajectory: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_instruction_as_seen": trajectory["task_instruction"],
            "outcome": trajectory["outcome"],
            "tool_steps": trajectory["tool_steps"],
            "conversation_evidence": compact_transcript(trajectory),
        }

    return [
        {
            "schema": "g1-tau-success-judge-packet/1",
            "triplet_id": triplet["triplet_id"],
            "track": triplet["track"],
            "retail_policy": retail_policy,
            "task_instruction": triplet["task_instruction"],
            "anchor": compact(triplet["anchor"]),
            "candidates": [
                {"slot": candidate["slot"], **compact(candidate)}
                for candidate in triplet["candidates"]
            ],
        }
        for triplet in triplets
    ]


def write_checksums(root: Path, paths: list[Path]) -> None:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root)}"
        for path in sorted(paths)
    ]
    (root / "data/g1/tau_success_v2/checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data/g1/tau_success_v2"
    triplets, source_rows, hidden = build(root)
    write_jsonl(output / "source_trajectories.jsonl", source_rows)
    write_jsonl(output / "calibration_triplets.jsonl", triplets)
    write_jsonl(output / "annotations.template.jsonl", annotation_rows(triplets))
    retail_policy = (
        root / "data/g1/assets/taubench_retail/policy.md"
    ).read_text(encoding="utf-8")
    write_jsonl(output / "judge_packets.jsonl", judge_packets(triplets, retail_policy))
    (output / "provenance.json").write_text(
        json.dumps(hidden["provenance"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "task_specs.json").write_text(
        json.dumps(hidden["task_specs"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    browser_data = root / "annotation/tau_success_data.js"
    browser_data.write_text(
        "window.TAU_SUCCESS_DATA = "
        + json.dumps(triplets, ensure_ascii=False).replace("</", "<\\/")
        + ";\n",
        encoding="utf-8",
    )
    write_checksums(
        root,
        [
            output / "source_trajectories.jsonl",
            output / "calibration_triplets.jsonl",
            output / "annotations.template.jsonl",
            output / "judge_packets.jsonl",
            output / "provenance.json",
            output / "task_specs.json",
            browser_data,
        ],
    )
    print(
        f"wrote {len(triplets)} tau success-candidate triplets / "
        f"{len(source_rows)} sources"
    )


if __name__ == "__main__":
    main()
