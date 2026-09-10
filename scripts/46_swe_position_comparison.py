#!/usr/bin/env python3
"""Frozen local comparison of edit-position signatures on development and fresh probes."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


g = module("grounded", "43_swe_grounded_representation.py")
OUT = g.s.ROOT / "position-comparison"
FILES = {
    "baseline": "43_swe_grounded_representation.py",
    "anchors": "swe_position_anchors.py",
    "sequence": "swe_position_sequence.py",
}


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "comparison_protocol.json").exists():
        raise FileExistsError("Comparison protocol already frozen")
    inputs = [g.OUT / "records.json", g.OUT / "stress/inputs.json", OUT / "fresh_probes.json"]
    protocol = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "seed": g.s.SEED,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": None,
        "api_cost_usd": 0,
        "method_hashes": {name: g.digest(Path(__file__).with_name(file)) for name, file in FILES.items()},
        "runner_sha256": g.digest(Path(__file__)),
        "input_hashes": {str(p): g.digest(p) for p in inputs},
        "scope": "26 reused development pairs, 4 known toy probes, and a separately authored fresh toy suite frozen without inspecting candidate methods. No human validation or benchmark outcomes.",
        "comparison": "Unweighted multiset Jaccard of each method signature. Equality tolerance 1e-12; this tests feature retention/invariance, not calibrated semantic similarity.",
        "gate": "Report all methods, no fitting. Preserve known auxiliary invariance, cross-function distinctions and P18/P19 mechanisms; pass four old probes and all fresh hard probes. Exploratory contextual probes reported separately. Any revised method after fresh scores requires new validation.",
        "smoke": ["P03", "P18", "L01", "old:same_function_stage", "old:if_else_branch", "old:format_and_comment"],
        "smoke_pass": "Technical validity, symmetry, identity and [0,1] range only. Scientific failures are results, not runner failures.",
    }
    g.s.write_json(OUT / "comparison_protocol.json", protocol)
    print("Methods, inputs, and comparison rules frozen without reading fresh probe content.")


def cases(fresh):
    old_pairs = {r["pair_id"]: r for r in json.loads((g.loc.OUT / "pairs.json").read_text())}
    result = []
    for r in json.loads((g.OUT / "records.json").read_text()):
        expected = None
        if old_pairs.get(r["pair_id"], {}).get("sampling_stratum") == "same_runtime_auxiliary_candidate":
            expected = "equal"
        elif r["pair_id"].startswith("L") or r["pair_id"] in ["P18", "P19"]:
            expected = "distinct"
        result.append(
            {
                "id": r["pair_id"],
                "suite": "development",
                "gate": "known",
                "expected_relation": expected,
                "a": r["sides"]["a"]["records"],
                "b": r["sides"]["b"]["records"],
            }
        )
    for r in json.loads((g.OUT / "stress/inputs.json").read_text()):
        result.append(
            {
                "id": "old:" + r["id"],
                "suite": "old_stress",
                "gate": "known",
                "expected_relation": r["expected_combined_relation"],
                **{side: g.represent(r["before"], r[side], "probe.py") for side in ["a", "b"]},
            }
        )
    if fresh:
        for r in json.loads((OUT / "fresh_probes.json").read_text()):
            result.append(
                {
                    "id": r["id"],
                    "suite": "fresh_stress",
                    "gate": r["gate"],
                    "category": r["category"],
                    "expected_relation": r["expected_relation"],
                    **{side: g.represent(r["before_" + side], r["after_" + side], "probe.py") for side in ["a", "b"]},
                }
            )
    return result


def run(smoke):
    protocol = json.loads((OUT / "comparison_protocol.json").read_text())
    assert g.digest(Path(__file__)) == protocol["runner_sha256"]
    for path, expected in protocol["input_hashes"].items():
        assert g.digest(Path(path)) == expected
    for name, expected in protocol["method_hashes"].items():
        assert g.digest(Path(__file__).with_name(FILES[name])) == expected
    if not smoke:
        assert json.loads((OUT / "smoke_summary.json").read_text())["technical_checks_passed"]
    functions = {"baseline": lambda records: g.signature(records, "combined")}
    functions.update({name: module(name, file).signature for name, file in FILES.items() if name != "baseline"})
    rows = cases(fresh=not smoke)
    if smoke:
        rows = [r for r in rows if r["id"] in protocol["smoke"]]
    result = []
    for row in rows:
        record = {k: v for k, v in row.items() if k not in ["a", "b"]}
        record["methods"] = {}
        for name, function in functions.items():
            a, b = function(row["a"]), function(row["b"])
            value = g.distance(a, b)
            assert 0 <= value <= 1 and value == g.distance(b, a) and g.distance(a, a) == 0
            l1 = sum(abs(a[k] - b[k]) for k in a.keys() | b.keys())
            denom = sum(a.values()) + sum(b.values()) + l1
            assert abs(value - (2 * l1 / denom if denom else 0.0)) < 1e-12
            relation = "equal" if value <= 1e-12 else "distinct"
            record["methods"][name] = {
                "distance": value,
                "observed_relation": relation,
                "passed": relation == row["expected_relation"] if row["expected_relation"] else None,
                "signature_a": dict(a),
                "signature_b": dict(b),
            }
        result.append(record)
    summary = {"technical_checks_passed": True, "cases": len(result), "methods": {}}
    for name in functions:
        groups = {}
        for suite, gate in [
            ("development", "known"),
            ("old_stress", "known"),
            ("fresh_stress", "hard"),
            ("fresh_stress", "exploratory"),
        ]:
            subset = [r for r in result if r["suite"] == suite and r["gate"] == gate and r["expected_relation"]]
            groups[suite + ":" + gate] = {
                "passed": sum(r["methods"][name]["passed"] for r in subset),
                "total": len(subset),
                "failed_ids": [r["id"] for r in subset if not r["methods"][name]["passed"]],
            }
        summary["methods"][name] = groups
    prefix = "smoke_" if smoke else ""
    g.s.write_json(OUT / f"{prefix}comparison.json", result)
    g.s.write_json(OUT / f"{prefix}summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run(args.smoke)
