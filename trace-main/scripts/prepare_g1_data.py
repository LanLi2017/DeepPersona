#!/usr/bin/env python3
"""Select and normalize existing ALFWorld and tau-bench trajectories for G1.

The script intentionally reads from sibling experiment directories and writes
only a compact, deterministic snapshot into this repository. It uses Python's
standard library so preparing the data does not require either benchmark's
runtime environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "g1-trajectory/1"
ALFWORLD_QUOTAS = {
    "clean": 3,
    "cool": 3,
    "examine": 3,
    "heat": 3,
    "pick_two": 2,
    "put": 2,
}
TAUBENCH_TASK_COUNT = 16
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            row["__source_line__"] = line_number
            rows.append(row)
    return rows


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


def stable_record_hash(row: dict[str, Any]) -> str:
    clean = {key: value for key, value in row.items() if key != "__source_line__"}
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def outcome_class(successes: int, total: int) -> str:
    if successes == 0:
        return "all_failure"
    if successes == total:
        return "all_success"
    return "mixed"


def alf_unique_sequences(row: dict[str, Any]) -> int:
    sequences = {
        tuple(str(step.get("action", "")) for step in trajectory["steps_log"])
        for trajectory in row["trajectories"]
    }
    return len(sequences)


def tau_tool_sequence(episode: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(name)
        for message in episode.get("messages", [])
        for name in (message.get("tool_calls") or [])
    )


def tau_unique_sequences(row: dict[str, Any]) -> int:
    return len({tau_tool_sequence(episode) for episode in row["episodes"]})


def validate_alf_source(rows: list[dict[str, Any]]) -> None:
    if len({row.get("gamefile") for row in rows}) != len(rows):
        raise ValueError("ALFWorld source contains duplicate gamefile records")
    for row in rows:
        trajectories = row.get("trajectories")
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError(f"ALFWorld line {row['__source_line__']} has no trajectories")
        if row.get("n_traj") != len(trajectories):
            raise ValueError(f"ALFWorld line {row['__source_line__']} has inconsistent n_traj")
        successes = sum(bool(trajectory.get("won")) for trajectory in trajectories)
        if row.get("n_succ") != successes:
            raise ValueError(f"ALFWorld line {row['__source_line__']} has inconsistent n_succ")
        for trajectory in trajectories:
            steps = trajectory.get("steps_log")
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"ALFWorld line {row['__source_line__']} has an empty trajectory")


def tau_row_is_complete(row: dict[str, Any]) -> bool:
    episodes = row.get("episodes")
    return bool(episodes) and all(
        isinstance(episode.get("messages"), list) and episode["messages"]
        for episode in episodes
    )


def validate_tau_source(rows: list[dict[str, Any]]) -> None:
    if len({row.get("task_id") for row in rows}) != len(rows):
        raise ValueError("tau-bench source contains duplicate task IDs")
    for row in rows:
        episodes = row.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"tau-bench line {row['__source_line__']} has no episodes")
        if row.get("n_traj") != len(episodes):
            raise ValueError(f"tau-bench line {row['__source_line__']} has inconsistent n_traj")
        successes = sum(bool(episode.get("won")) for episode in episodes)
        if row.get("n_succ") != successes:
            raise ValueError(f"tau-bench line {row['__source_line__']} has inconsistent n_succ")


def select_alf(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["task"])].append(row)

    selected: list[dict[str, Any]] = []
    for family, quota in ALFWORLD_QUOTAS.items():
        candidates = by_family.get(family, [])
        if len(candidates) < quota:
            raise ValueError(f"ALFWorld family {family!r} has fewer than {quota} tasks")
        ranked = sorted(
            candidates,
            key=lambda row: (
                0 if outcome_class(row["n_succ"], row["n_traj"]) == "mixed" else 1,
                -alf_unique_sequences(row),
                abs((row["n_succ"] / row["n_traj"]) - 0.5),
                str(row["gamefile"]),
            ),
        )
        selected.extend(ranked[:quota])
    return sorted(selected, key=lambda row: (str(row["task"]), str(row["gamefile"])))


def numeric_task_id(row: dict[str, Any]) -> tuple[int, str]:
    task_id = str(row["task_id"])
    return (int(task_id), task_id) if task_id.isdigit() else (10**12, task_id)


def select_tau(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in rows if tau_row_is_complete(row)]
    ranked = sorted(
        complete,
        key=lambda row: (
            0 if outcome_class(row["n_succ"], row["n_traj"]) == "mixed" else 1,
            -tau_unique_sequences(row),
            abs((row["n_succ"] / row["n_traj"]) - 0.5),
            numeric_task_id(row),
        ),
    )
    if len(ranked) < TAUBENCH_TASK_COUNT:
        raise ValueError("not enough complete tau-bench tasks")
    return sorted(ranked[:TAUBENCH_TASK_COUNT], key=numeric_task_id)


def relative_alf_gamefile(gamefile: str) -> str:
    marker = "json_2.1.1/"
    if marker not in gamefile:
        return Path(gamefile).name
    return marker + gamefile.split(marker, 1)[1]


def public_source_row(row: dict[str, Any], benchmark: str) -> dict[str, Any]:
    clean = copy.deepcopy(row)
    clean.pop("__source_line__", None)
    if benchmark == "alfworld":
        clean["gamefile"] = relative_alf_gamefile(str(clean["gamefile"]))
    return clean


def parse_tool_calls(message: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    names = [str(name) for name in (message.get("tool_calls") or [])]
    parsed: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(str(message.get("text") or "")):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            parsed.append({
                "name": payload["name"],
                "arguments": payload.get("arguments") or {},
            })
    parsed_names = [call["name"] for call in parsed]
    exact = parsed_names == names
    if not parsed and names:
        parsed = [{"name": name, "arguments": {}} for name in names]
    return parsed, exact


def normalize_alf(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_index, row in enumerate(selected, start=1):
        gamefile = relative_alf_gamefile(str(row["gamefile"]))
        task_key = gamefile.removesuffix("/game.tw-pddl")
        short_id = hashlib.sha256(task_key.encode()).hexdigest()[:12]
        for rollout_index, trajectory in enumerate(row["trajectories"]):
            events = []
            for event_index, step in enumerate(trajectory["steps_log"]):
                events.append({
                    "index": event_index,
                    "kind": "environment_transition",
                    "actor": "agent",
                    "action": str(step.get("action") or ""),
                    "observation": str(step.get("obs") or ""),
                    "matched": bool(step.get("matched", True)),
                    "invalid": bool(step.get("invalid", False)),
                    "redundant": bool(step.get("redundant", False)),
                    "thinking_present": bool(step.get("think", False)),
                })
            output.append({
                "schema": SCHEMA_VERSION,
                "trajectory_id": f"alfworld:{short_id}:r{rollout_index}",
                "benchmark": "alfworld",
                "task": {
                    "id": task_key,
                    "family": row["task"],
                    "instruction": trajectory.get("goal") or row.get("goal") or "",
                    "environment_asset": f"../assets/alfworld/{gamefile}",
                },
                "rollout": {
                    "index": rollout_index,
                    "success": bool(trajectory["won"]),
                    "reward": int(bool(trajectory["won"])),
                    "length": len(events),
                },
                "events": events,
                "source_ref": {
                    "snapshot": "../source/alfworld_tasks.jsonl",
                    "snapshot_line": source_index,
                    "upstream_line": row["__source_line__"],
                    "upstream_record_sha256": stable_record_hash(row),
                },
            })
    return output


def copy_alf_assets(selected: list[dict[str, Any]], asset_root: Path) -> list[Path]:
    copied: list[Path] = []
    required_names = ("game.tw-pddl", "initial_state.pddl", "traj_data.json")
    for row in selected:
        source_gamefile = Path(str(row["gamefile"]))
        relative_gamefile = Path(relative_alf_gamefile(str(source_gamefile)))
        source_dir = source_gamefile.parent
        destination_dir = asset_root / relative_gamefile.parent
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name in required_names:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"missing ALFWorld replay asset: {source}")
            destination = destination_dir / name
            shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def copy_tau_assets(
    selected: list[dict[str, Any]], source_root: Path, asset_root: Path
) -> list[Path]:
    task_path = source_root / "tasks.json"
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    selected_ids = {str(row["task_id"]) for row in selected}
    selected_tasks = [task for task in tasks if str(task.get("id")) in selected_ids]
    if {str(task["id"]) for task in selected_tasks} != selected_ids:
        raise ValueError("tau-bench asset tasks do not cover all selected task IDs")

    asset_root.mkdir(parents=True, exist_ok=True)
    destination_tasks = asset_root / "tasks.json"
    destination_tasks.write_text(
        json.dumps(selected_tasks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    copied = [destination_tasks]
    for name in ("db.json", "policy.md", "split_tasks.json"):
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(f"missing tau-bench retail asset: {source}")
        destination = asset_root / name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def normalize_tau(selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    parse_stats: Counter[str] = Counter()
    for source_index, row in enumerate(selected, start=1):
        for rollout_index, episode in enumerate(row["episodes"]):
            events = []
            for event_index, message in enumerate(episode["messages"]):
                calls, exact = parse_tool_calls(message)
                if message.get("tool_calls"):
                    parse_stats["messages_with_tool_calls"] += 1
                    parse_stats["tool_call_name_matches"] += int(exact)
                    parse_stats["tool_call_name_mismatches"] += int(not exact)
                event = {
                    "index": event_index,
                    "kind": "message",
                    "actor": str(message.get("role") or "unknown"),
                    "text": str(message.get("text") or ""),
                }
                if calls:
                    event["tool_calls"] = calls
                if "is_error" in message:
                    event["is_error"] = bool(message["is_error"])
                events.append(event)
            output.append({
                "schema": SCHEMA_VERSION,
                "trajectory_id": f"taubench-retail:{row['task_id']}:r{rollout_index}",
                "benchmark": "taubench_retail",
                "task": {
                    "id": str(row["task_id"]),
                    "family": "retail",
                    "instruction": row.get("query") or "",
                },
                "rollout": {
                    "index": rollout_index,
                    "success": bool(episode["won"]),
                    "reward": float(episode["reward"]),
                    "length": len(events),
                },
                "events": events,
                "source_ref": {
                    "snapshot": "../source/taubench_retail_tasks.jsonl",
                    "snapshot_line": source_index,
                    "upstream_line": row["__source_line__"],
                    "upstream_record_sha256": stable_record_hash(row),
                },
            })
    return output, dict(sorted(parse_stats.items()))


def selection_summary(rows: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    if benchmark == "alfworld":
        sequence_counts = [alf_unique_sequences(row) for row in rows]
        families = Counter(str(row["task"]) for row in rows)
        trajectory_count = sum(len(row["trajectories"]) for row in rows)
        success_count = sum(row["n_succ"] for row in rows)
        ids = [relative_alf_gamefile(str(row["gamefile"])) for row in rows]
    else:
        sequence_counts = [tau_unique_sequences(row) for row in rows]
        families = Counter(str(row["domain"]) for row in rows)
        trajectory_count = sum(len(row["episodes"]) for row in rows)
        success_count = sum(row["n_succ"] for row in rows)
        ids = [str(row["task_id"]) for row in rows]
    outcomes = Counter(outcome_class(row["n_succ"], row["n_traj"]) for row in rows)
    return {
        "tasks": len(rows),
        "trajectories": trajectory_count,
        "successful_trajectories": success_count,
        "task_groups": dict(sorted(families.items())),
        "task_outcomes": dict(sorted(outcomes.items())),
        "mean_unique_exact_sequences_per_task": sum(sequence_counts) / len(sequence_counts),
        "selected_task_ids": ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent
    parser.add_argument(
        "--alfworld-source",
        type=Path,
        default=project_root / "npm/repro_alf_icrl/results/memory_trajectories.jsonl",
    )
    parser.add_argument(
        "--taubench-source",
        type=Path,
        default=project_root / "npm/repro_tau/results/memory_tau_retail.jsonl",
    )
    parser.add_argument(
        "--taubench-assets-root",
        type=Path,
        default=project_root / "tau2-bench/data/tau2/domains/retail",
    )
    parser.add_argument("--output-root", type=Path, default=repo_root / "data/g1")
    args = parser.parse_args()

    alf_rows = read_jsonl(args.alfworld_source)
    tau_rows = read_jsonl(args.taubench_source)
    validate_alf_source(alf_rows)
    validate_tau_source(tau_rows)

    selected_alf = select_alf(alf_rows)
    selected_tau = select_tau(tau_rows)
    normalized_alf = normalize_alf(selected_alf)
    normalized_tau, tau_parse_stats = normalize_tau(selected_tau)

    source_dir = args.output_root / "source"
    normalized_dir = args.output_root / "normalized"
    manifest_dir = args.output_root / "manifests"
    copied_alf_assets = copy_alf_assets(
        selected_alf, args.output_root / "assets/alfworld"
    )
    copied_tau_assets = copy_tau_assets(
        selected_tau,
        args.taubench_assets_root,
        args.output_root / "assets/taubench_retail",
    )
    write_jsonl(
        source_dir / "alfworld_tasks.jsonl",
        (public_source_row(row, "alfworld") for row in selected_alf),
    )
    write_jsonl(
        source_dir / "taubench_retail_tasks.jsonl",
        (public_source_row(row, "taubench") for row in selected_tau),
    )
    write_jsonl(normalized_dir / "trajectories.jsonl", normalized_alf + normalized_tau)

    malformed_tau_ids = [
        str(row["task_id"]) for row in tau_rows if not tau_row_is_complete(row)
    ]
    manifest = {
        "schema": "g1-seed-manifest/1",
        "sources": {
            "alfworld": {
                "locator": "../npm/repro_alf_icrl/results/memory_trajectories.jsonl",
                "sha256": sha256_file(args.alfworld_source),
                "records": len(alf_rows),
            },
            "taubench_retail": {
                "locator": "../npm/repro_tau/results/memory_tau_retail.jsonl",
                "sha256": sha256_file(args.taubench_source),
                "records": len(tau_rows),
                "excluded_incomplete_task_ids": malformed_tau_ids,
                "asset_fingerprints": {
                    name: sha256_file(args.taubench_assets_root / name)
                    for name in ("db.json", "policy.md", "split_tasks.json", "tasks.json")
                },
            },
        },
        "selection_policy": {
            "alfworld": {
                "quotas": ALFWORLD_QUOTAS,
                "ranking": [
                    "mixed outcome first",
                    "more unique exact action sequences first",
                    "success fraction closest to 0.5",
                    "gamefile lexical order",
                ],
            },
            "taubench_retail": {
                "count": TAUBENCH_TASK_COUNT,
                "filters": ["all episodes contain at least one message"],
                "ranking": [
                    "mixed outcome first",
                    "more unique exact tool-name sequences first",
                    "success fraction closest to 0.5",
                    "numeric task ID",
                ],
            },
        },
        "selected": {
            "alfworld": selection_summary(selected_alf, "alfworld"),
            "taubench_retail": selection_summary(selected_tau, "taubench"),
            "normalized_trajectories": len(normalized_alf) + len(normalized_tau),
        },
        "normalization_diagnostics": {
            "taubench_tool_call_parsing": tau_parse_stats,
            "alfworld_replay_asset_files": len(copied_alf_assets),
            "taubench_retail_asset_files": len(copied_tau_assets),
        },
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    generated_files = sorted(copied_alf_assets + copied_tau_assets) + [
        source_dir / "alfworld_tasks.jsonl",
        source_dir / "taubench_retail_tasks.jsonl",
        normalized_dir / "trajectories.jsonl",
        manifest_path,
    ]
    checksum_path = manifest_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(args.output_root)}\n"
            for path in generated_files
        ),
        encoding="utf-8",
    )

    print(json.dumps(manifest["selected"], ensure_ascii=False, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
