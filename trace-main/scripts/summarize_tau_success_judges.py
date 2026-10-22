#!/usr/bin/env python3
"""Validate and summarize the three-model tau success judge panel."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("E0", "E1", "E2", "ambiguous")
JUDGE_IDS = ("gpt56", "sonnet", "opus")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fleiss_kappa(votes: list[list[str]]) -> tuple[float, float, float]:
    raters = len(votes[0])
    observed = sum(
        sum(count * (count - 1) for count in Counter(row).values())
        / (raters * (raters - 1))
        for row in votes
    ) / len(votes)
    totals = Counter(label for row in votes for label in row)
    expected = sum((totals[label] / (len(votes) * raters)) ** 2 for label in LABELS)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return observed, expected, kappa


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    judgment_dir = root / "data/g1/tau_success_v2/judgments"
    paths = {judge: judgment_dir / f"{judge}.jsonl" for judge in JUDGE_IDS}
    provenance = json.loads(
        (root / "data/g1/tau_success_v2/provenance.json").read_text()
    )
    hidden = {
        (item["triplet_id"], slot): value["role"]
        for item in provenance["items"]
        for slot, value in item["candidate_slots"].items()
    }

    panel: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for judge, path in paths.items():
        rows = read_jsonl(path)
        assert len(rows) == 4
        flattened = {}
        for row in rows:
            assert row["judge_id"] == judge
            assert len(row["judgments"]) == 3
            for judgment in row["judgments"]:
                key = (row["triplet_id"], judgment["candidate_slot"])
                assert judgment["label"] in LABELS and key not in flattened
                flattened[key] = judgment
        assert set(flattened) == set(hidden)
        panel[judge] = flattened

    consensus_rows = []
    vote_rows = []
    anchor_quality_votes = []
    for key in sorted(hidden):
        votes = [panel[judge][key]["label"] for judge in JUDGE_IDS]
        quality_votes = [
            panel[judge][key]["candidate_high_quality"] for judge in JUDGE_IDS
        ]
        vote_rows.append(votes)
        counts = Counter(votes)
        majority, majority_count = counts.most_common(1)[0]
        quality_counts = Counter(quality_votes)
        majority_quality, majority_quality_count = quality_counts.most_common(1)[0]
        consensus_rows.append({
            "schema": "g1-tau-success-consensus/1",
            "triplet_id": key[0],
            "candidate_slot": key[1],
            "hidden_role": hidden[key],
            "votes": dict(zip(JUDGE_IDS, votes)),
            "quality_votes": dict(zip(JUDGE_IDS, quality_votes)),
            "majority_label": majority,
            "majority_high_quality": majority_quality,
            "unanimous": majority_count == len(JUDGE_IDS),
            "quality_unanimous": majority_quality_count == len(JUDGE_IDS),
            "requires_human_review": (
                majority_count != len(JUDGE_IDS)
                or majority_quality_count != len(JUDGE_IDS)
                or majority_quality != "yes"
            ),
        })
    for judge in JUDGE_IDS:
        for row in read_jsonl(paths[judge]):
            assert row.get("anchor_high_quality") in {"yes", "no", "unclear"}
            anchor_quality_votes.append(row["anchor_high_quality"])

    observed, expected, kappa = fleiss_kappa(vote_rows)
    pairwise = {}
    for left_index, left in enumerate(JUDGE_IDS):
        for right in JUDGE_IDS[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = sum(
                panel[left][key]["label"] == panel[right][key]["label"]
                for key in hidden
            ) / len(hidden)
    surface = [row for row in consensus_rows if row["hidden_role"] == "surface_equivalent"]
    natural = [row for row in consensus_rows if row["hidden_role"] != "surface_equivalent"]
    summary = {
        "schema": "g1-tau-success-panel-summary/1",
        "judges": list(JUDGE_IDS),
        "comparisons": len(consensus_rows),
        "unanimous_comparisons": sum(row["unanimous"] for row in consensus_rows),
        "pairwise_label_agreement": pairwise,
        "fleiss_kappa": kappa,
        "observed_pair_agreement": observed,
        "chance_agreement": expected,
        "surface_e0_votes_correct": sum(
            label == "E0" for row in surface for label in row["votes"].values()
        ),
        "surface_e0_votes_total": len(surface) * len(JUDGE_IDS),
        "anchor_high_quality_yes_votes": sum(
            value == "yes" for value in anchor_quality_votes
        ),
        "anchor_high_quality_votes_total": len(anchor_quality_votes),
        "natural_high_quality_yes_votes": sum(
            value == "yes" for row in natural
            for value in row["quality_votes"].values()
        ),
        "natural_high_quality_votes_total": len(natural) * len(JUDGE_IDS),
        "natural_majority_high_quality": dict(
            Counter(row["majority_high_quality"] for row in natural)
        ),
        "natural_majority_labels": dict(Counter(row["majority_label"] for row in natural)),
        "human_review": [
            {"triplet_id": row["triplet_id"], "candidate_slot": row["candidate_slot"]}
            for row in consensus_rows if row["requires_human_review"]
        ],
        "limitations": [
            "LLM consensus is pre-annotation, not human ground truth.",
            "The panel must remain hidden until independent human annotation is complete.",
            "Natural near/far sampling roles are not expected E1/E2 labels.",
        ],
    }
    consensus_path = judgment_dir / "consensus.jsonl"
    with consensus_path.open("w", encoding="utf-8") as handle:
        for row in consensus_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary_path = judgment_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checksum_paths = [*paths.values(), consensus_path, summary_path]
    (judgment_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(root)}"
            for path in sorted(checksum_paths)
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"panel: {summary['unanimous_comparisons']}/{summary['comparisons']} "
        f"unanimous; Fleiss kappa={kappa:.3f}; "
        f"E0={summary['surface_e0_votes_correct']}/{summary['surface_e0_votes_total']}; "
        f"natural HQ-majority={summary['natural_majority_high_quality']}"
    )


if __name__ == "__main__":
    main()
