#!/usr/bin/env python3
"""Small declared structural probes, separate from the frozen 26-pair development run."""

import argparse
import importlib.util
import json
from pathlib import Path

sp = importlib.util.spec_from_file_location("grounded", Path(__file__).with_name("43_swe_grounded_representation.py"))
g = importlib.util.module_from_spec(sp)
sp.loader.exec_module(g)
OUT = g.OUT / "stress"


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "inputs.json").exists():
        raise FileExistsError("Stress inputs already frozen")
    straight = (
        "def f(value):\n    out = []\n    out.append(value)\n    value += 1\n    out.append(value)\n    return out\n"
    )
    branch = "def f(value, flag):\n    out = []\n    if flag:\n        out.append(value)\n    else:\n        out.append(value)\n    return out\n"
    rows = []
    for name, before in [("same_function_stage", straight), ("if_else_branch", branch)]:
        old, new = "out.append(value)", "out.append(value * 10)"
        first, second = before.index(old), before.rindex(old)
        rows.append(
            {
                "id": name,
                "before": before,
                "a": before[:first] + before[first:].replace(old, new, 1),
                "b": before[:second] + before[second:].replace(old, new, 1),
                "expected_combined_relation": "distinct",
                "rationale": "Identical replacement at different execution positions in one qualified function. For straight-line f(1), A returns [10, 2] and B [1, 20] by arithmetic; no benchmark code executed.",
            }
        )
    before = "def f(value):\n    out = []\n    return out\n"
    rows.append(
        {
            "id": "duplicate_count",
            "before": before,
            "a": before.replace("    return", "    out.append(value)\n    return"),
            "b": before.replace("    return", "    out.append(value)\n    out.append(value)\n    return"),
            "expected_combined_relation": "distinct",
            "rationale": "One versus two appends; multiset must preserve multiplicity.",
        }
    )
    a = straight.replace("value += 1", "value += 2")
    rows.append(
        {
            "id": "format_and_comment",
            "before": straight,
            "a": a,
            "b": a.replace("    value += 2", "    # formatting-only variation\n    value   +=   2"),
            "expected_combined_relation": "equal",
            "rationale": "Same statement ASTs; comments and whitespace excluded by protocol.",
        }
    )
    g.s.write_json(OUT / "inputs.json", rows)
    g.s.write_json(
        OUT / "protocol.json",
        {
            "inputs_sha256": g.digest(OUT / "inputs.json"),
            "representation_runner_sha256": g.digest(Path(g.__file__)),
            "probe_runner_sha256": g.digest(Path(__file__)),
            "scope": "Four handcrafted checks declared after the 26-pair run, before their scores; not independent validation or sampled evidence.",
            "api_cost_usd": 0,
            "candidate_code_executed": False,
        },
    )
    print("Four stress inputs frozen.")


def run():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert g.digest(OUT / "inputs.json") == protocol["inputs_sha256"]
    assert g.digest(Path(g.__file__)) == protocol["representation_runner_sha256"]
    assert g.digest(Path(__file__)) == protocol["probe_runner_sha256"]
    result = []
    for row in json.loads((OUT / "inputs.json").read_text()):
        a, b = (g.represent(row["before"], row[side], "probe.py") for side in ["a", "b"])
        scores = {
            mode: g.distance(g.signature(a, mode), g.signature(b, mode)) for mode in ["operation", "scope", "combined"]
        }
        observed = "equal" if scores["combined"] == 0 else "distinct"
        result.append(
            {
                **row,
                "scores": scores,
                "passed": observed == row["expected_combined_relation"],
                "records_a": a,
                "records_b": b,
            }
        )
    g.s.write_json(OUT / "results.json", result)
    summary = {
        "passed": sum(r["passed"] for r in result),
        "total": len(result),
        "probes": {r["id"]: {"passed": r["passed"], "scores": r["scores"]} for r in result},
    }
    g.s.write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
