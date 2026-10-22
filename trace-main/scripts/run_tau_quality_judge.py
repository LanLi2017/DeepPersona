#!/usr/bin/env python3
"""Run a quality-only judge using HTTP or the authenticated Claude CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import requests


PROMPT = """You are independently screening reward-positive tau-bench retail
trajectories for use as high-quality successful demonstrations. Reward 1 is not
sufficient. A candidate is high quality only if it actually resolves the full
user request, obeys the supplied retail policy (including authentication and
confirmation), makes no unsupported claims, and has no material omission.
Harmless extra read calls or wording do not make it low quality. Judge each
candidate independently. Return JSON only, with one assessment per candidate:
candidate_id, task_complete (yes/no/unclear), policy_compliant
(yes/no/unclear), claims_grounded (yes/no/unclear), high_quality
(yes/no/unclear), confidence (high/medium/low), rationale, and evidence list."""


def parse_result(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        values = []
        cursor = 0
        while cursor < len(text):
            starts = [index for index in (text.find("{", cursor), text.find("[", cursor)) if index >= 0]
            if not starts:
                break
            start = min(starts)
            try:
                parsed, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            values.append(parsed)
            cursor = end
        flattened = []
        for parsed in values:
            if isinstance(parsed, list):
                flattened.extend(item for item in parsed if isinstance(item, dict))
            elif isinstance(parsed, dict) and "candidate_id" in parsed:
                flattened.append(parsed)
            elif isinstance(parsed, dict) and isinstance(parsed.get("assessments"), list):
                flattened.extend(parsed["assessments"])
        if not flattened:
            raise
        value = {"assessments": flattened}
    if isinstance(value, list):
        value = {"assessments": value}
    if not isinstance(value, dict):
        raise ValueError("quality judge response must be an object or assessment list")
    return value


def validate(result: dict[str, Any], packets: list[dict[str, Any]]) -> None:
    rows = result.get("assessments")
    expected = {packet["candidate_id"] for packet in packets}
    assert isinstance(rows, list) and {row.get("candidate_id") for row in rows} == expected
    for row in rows:
        for field in ("task_complete", "policy_compliant", "claims_grounded", "high_quality"):
            assert row.get(field) in {"yes", "no", "unclear"}
        assert row.get("confidence") in {"high", "medium", "low"}
        assert isinstance(row.get("rationale"), str) and row["rationale"].strip()
        assert isinstance(row.get("evidence"), list)


def cli_wrapper(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "result" in value:
            return value
    raise ValueError("Claude CLI did not return a JSON wrapper")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/g1/tau_success_v2/quality_screen/packets.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--transport", choices=("http", "claude-cli"), default="http")
    parser.add_argument("--base-url-env")
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    packets = json.loads(args.input.read_text())
    body = PROMPT + "\n\nCANDIDATES:\n" + json.dumps(packets, ensure_ascii=False)
    usage = None
    cost = None
    returned_model = None
    if args.transport == "http":
        base_url = os.environ.get(args.base_url_env or "", "").strip()
        api_key = os.environ.get(args.api_key_env or "", "").strip()
        if not base_url or not api_key:
            raise SystemExit("HTTP transport requires configured env variable names")
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": args.model.removeprefix("openai/"),
                "messages": [{"role": "user", "content": body}],
                "max_completion_tokens": 6000,
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        wrapper = response.json()
        result = parse_result(wrapper["choices"][0]["message"]["content"])
        usage = wrapper.get("usage")
        returned_model = wrapper.get("model")
    else:
        completed = subprocess.run(
            ["claude", "--", "-p", "--model", args.model, "--tools", "", "--output-format", "json"],
            input=body,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=True,
        )
        wrapper = cli_wrapper(completed.stdout)
        result = parse_result(wrapper["result"])
        usage = wrapper.get("usage")
        cost = wrapper.get("total_cost_usd")
        models = {
            str(value.get("canonicalModel") or key)
            for key, value in (wrapper.get("modelUsage") or {}).items()
            if isinstance(value, dict)
        }
        returned_model = ",".join(sorted(models)) or None
    validate(result, packets)
    output = {
        "schema": "g1-tau-quality-screen-output/1",
        "judge_id": args.judge_id,
        "requested_model": args.model,
        "returned_model": returned_model,
        "usage": usage,
        "total_cost_usd": cost,
        "prompt_sha256": hashlib.sha256(body.encode()).hexdigest(),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.judge_id}: wrote {len(result['assessments'])} quality assessments")


if __name__ == "__main__":
    main()
