#!/usr/bin/env python3
"""Run one OpenAI-compatible LLM judge over blind tau success packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


SYSTEM_PROMPT = """You are an independent trajectory-equivalence annotator.
All examples are from the same task and passed a strict success gate. Success
alone does not imply the same strategy. Judge each candidate against the anchor.

Labels:
- E0: only identifiers, wording, or serialization differ; grounded calls and
  their effects are the same.
- E1: low-level calls, harmless ordering, or redundant checks differ, but the
  task-relevant information, meaningful state effects, ordered subgoals, and
  recovery policy are materially the same.
- E2: information coverage, meaningful state effects, dependency order, or
  recovery strategy differs materially, even though both trajectories succeed.
- ambiguous: the supplied evidence cannot support a reliable distinction.

For each slot, independently answer same_information, same_state_effect,
same_subgoal_order, same_recovery_behavior, and training_interchangeable with
one of yes/mostly/no/unclear (same_recovery_behavior may also be
not_applicable). Then give label E0/E1/E2/ambiguous, confidence high/medium/low,
a concise rationale, and an evidence list citing tool-step positions. Do not
infer roles from slot order or raw edit distance. Also judge whether the anchor
and each candidate are high-quality successes: they must complete the requested
task and respect the supplied retail policy, not merely receive reward 1. E0
identifier placeholders are intentional and should be interpreted consistently.
Return JSON only."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


def validate_response(value: dict[str, Any], packet: dict[str, Any]) -> None:
    expected_slots = {candidate["slot"] for candidate in packet["candidates"]}
    assert value.get("triplet_id") == packet["triplet_id"]
    judgments = value.get("judgments")
    assert isinstance(judgments, list) and len(judgments) == len(expected_slots)
    assert value.get("anchor_high_quality") in {"yes", "no", "unclear"}
    assert {row.get("candidate_slot") for row in judgments} == expected_slots
    allowed_component = {"yes", "mostly", "no", "unclear"}
    for row in judgments:
        assert row.get("same_information") in allowed_component
        assert row.get("same_state_effect") in allowed_component
        assert row.get("same_subgoal_order") in allowed_component
        assert row.get("same_recovery_behavior") in allowed_component | {"not_applicable"}
        assert row.get("training_interchangeable") in allowed_component
        assert row.get("label") in {"E0", "E1", "E2", "ambiguous"}
        assert row.get("confidence") in {"high", "medium", "low"}
        assert row.get("candidate_high_quality") in {"yes", "no", "unclear"}
        assert isinstance(row.get("rationale"), str) and row["rationale"].strip()
        assert isinstance(row.get("evidence"), list)


def call_judge(
    packet: dict[str, Any], base_url: str, api_key: str, model: str, timeout: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema_hint = {
        "triplet_id": packet["triplet_id"],
        "anchor_high_quality": "yes|no|unclear",
        "judgments": [{
            "candidate_slot": "A",
            "same_information": "yes|mostly|no|unclear",
            "same_state_effect": "yes|mostly|no|unclear",
            "same_subgoal_order": "yes|mostly|no|unclear",
            "same_recovery_behavior": "yes|mostly|no|not_applicable|unclear",
            "training_interchangeable": "yes|mostly|no|unclear",
            "label": "E0|E1|E2|ambiguous",
            "confidence": "high|medium|low",
            "candidate_high_quality": "yes|no|unclear",
            "rationale": "concise explanation",
            "evidence": ["anchor step ... vs candidate step ..."],
        }],
    }
    user_prompt = (
        "Return one judgment for every candidate slot using this shape:\n"
        + json.dumps(schema_hint, ensure_ascii=False)
        + "\n\nBLIND PACKET:\n"
        + json.dumps(packet, ensure_ascii=False)
    )
    payload = {
        "model": model.removeprefix("openai/"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 3000,
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    raw = response.json()
    content = raw["choices"][0]["message"]["content"]
    parsed = parse_json(content)
    validate_response(parsed, packet)
    metadata = {
        "returned_model": raw.get("model"),
        "usage": raw.get("usage"),
        "prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
    }
    return parsed, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/g1/tau_success_v2/judge_packets.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url-env", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    base_url = os.environ.get(args.base_url_env, "").strip()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not base_url or not api_key:
        raise SystemExit("configured base URL and API key environment variables are required")

    rows = []
    for packet in read_jsonl(args.input):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                judgment, metadata = call_judge(
                    packet, base_url, api_key, args.model, args.timeout
                )
                rows.append({
                    "schema": "g1-tau-success-judge-output/1",
                    "judge_id": args.judge_id,
                    "requested_model": args.model,
                    **metadata,
                    **judgment,
                })
                print(f"{args.judge_id}: {packet['triplet_id']} complete", flush=True)
                break
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(f"judge failed for {packet['triplet_id']}") from last_error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} judge outputs to {args.output}")


if __name__ == "__main__":
    main()
