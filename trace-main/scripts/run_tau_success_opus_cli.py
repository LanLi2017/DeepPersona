#!/usr/bin/env python3
"""Run a blind Opus judge through the authenticated Claude CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from run_tau_success_judge import SYSTEM_PROMPT, read_jsonl, validate_response


def parse_cli_wrapper(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "result" in value:
            return value
    raise ValueError("Claude CLI did not return its JSON wrapper")


def parse_result(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/g1/tau_success_v2/judge_packets.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="opus")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    packets = read_jsonl(args.input)
    required = {
        "results": [
            {
                "triplet_id": "tau-success-01",
                "anchor_high_quality": "yes|no|unclear",
                "judgments": [
                    {
                        "candidate_slot": "A",
                        "same_information": "yes|mostly|no|unclear",
                        "same_state_effect": "yes|mostly|no|unclear",
                        "same_subgoal_order": "yes|mostly|no|unclear",
                        "same_recovery_behavior": (
                            "yes|mostly|no|not_applicable|unclear"
                        ),
                        "training_interchangeable": "yes|mostly|no|unclear",
                        "label": "E0|E1|E2|ambiguous",
                        "confidence": "high|medium|low",
                        "candidate_high_quality": "yes|no|unclear",
                        "rationale": "concise explanation",
                        "evidence": ["specific tool-step comparison"],
                    }
                ],
            }
        ]
    }
    prompt = (
        SYSTEM_PROMPT
        + "\n\nJudge all four packets. Return exactly one JSON object with this shape:\n"
        + json.dumps(required, ensure_ascii=False)
        + "\nThere must be four results and three judgments per result.\n\n"
        + json.dumps(packets, ensure_ascii=False)
    )
    completed = subprocess.run(
        [
            "claude",
            "--",
            "-p",
            "--model",
            args.model,
            "--tools",
            "",
            "--output-format",
            "json",
        ],
        cwd=root,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=True,
    )
    wrapper = parse_cli_wrapper(completed.stdout)
    parsed = parse_result(wrapper["result"])
    results = parsed.get("results")
    assert isinstance(results, list) and len(results) == len(packets)
    by_id = {packet["triplet_id"]: packet for packet in packets}
    assert {result.get("triplet_id") for result in results} == set(by_id)
    model_usage = wrapper.get("modelUsage") or {}
    canonical_models = sorted(
        {
            str(value.get("canonicalModel") or key)
            for key, value in model_usage.items()
            if isinstance(value, dict)
        }
    )
    rows = []
    for result in results:
        packet = by_id[result["triplet_id"]]
        validate_response(result, packet)
        packet_line = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        rows.append({
            "schema": "g1-tau-success-judge-output/1",
            "judge_id": "opus",
            "requested_model": args.model,
            "returned_model": ",".join(canonical_models) or None,
            "usage": wrapper.get("usage"),
            "total_cost_usd": wrapper.get("total_cost_usd"),
            "prompt_sha256": hashlib.sha256(packet_line.encode()).hexdigest(),
            **result,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        f"opus: wrote {len(rows)} outputs; returned_model="
        f"{','.join(canonical_models) or 'unknown'}"
    )


if __name__ == "__main__":
    main()
