#!/usr/bin/env python3
"""Final frozen ordering comparison; both earlier fresh suites are now development."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import importlib.util

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).with_name("46_swe_position_comparison.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
g = c.g
OUT = c.OUT / "ordered-round3"
METHODS = {**c.FILES, "normalized": "swe_position_normalized.py", "ordered": "swe_position_ordered.py"}


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Round 3 already frozen")
    files = [
        g.OUT / "records.json",
        g.OUT / "stress/inputs.json",
        c.OUT / "fresh_probes.json",
        c.OUT / "round2_probes.json",
        c.OUT / "round3_probes.json",
    ]
    g.s.write_json(
        OUT / "protocol.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": g.s.SEED,
            "api_cost_usd": 0,
            "method_hashes": {name: g.digest(Path(__file__).with_name(file)) for name, file in METHODS.items()},
            "input_hashes": {str(p): g.digest(p) for p in files},
            "runner_sha256": g.digest(Path(__file__)),
            "scope": "Original 26 pairs, four old probes and both earlier 18-probe suites are development for the ordered method. A final independently authored suite is opened after candidate freeze. No benchmark outcomes or human gold.",
            "comparison": "All four frozen methods, unweighted multiset Jaccard; equality tolerance 1e-12, no fitted thresholds. Hard and exploratory assumptions stay separate.",
            "smoke_ids": ["P03", "P19", "L01", "old:same_function_stage", "F13", "F14"],
            "limitations": "Normalization has an explicitly narrow support envelope and falls back to unnormalized anchors outside it. Report support counts; passing toy checks does not authorize a utility claim.",
        },
    )
    print("Final candidate, input suite hashes, and rules frozen.")


def run(smoke):
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert g.digest(Path(__file__)) == protocol["runner_sha256"]
    for path, expected in protocol["input_hashes"].items():
        assert g.digest(Path(path)) == expected
    for name, expected in protocol["method_hashes"].items():
        assert g.digest(Path(__file__).with_name(METHODS[name])) == expected
    if not smoke:
        assert json.loads((OUT / "smoke_summary.json").read_text())["technical_checks_passed"]
    functions = {"baseline": lambda records: g.signature(records, "combined")}
    modules = {name: c.module(name, file) for name, file in METHODS.items() if name != "baseline"}
    functions.update({name: m.signature for name, m in modules.items()})
    rows = c.cases(fresh=True)
    for r in rows:
        if r["suite"] == "fresh_stress":
            r["suite"] = "round1_now_development"
    for r in json.loads((c.OUT / "round2_probes.json").read_text()):
        rows.append(
            {
                "id": r["id"],
                "suite": "round2_now_development",
                "gate": r["gate"],
                "category": r["category"],
                "expected_relation": r["expected_relation"],
                **{side: g.represent(r["before_" + side], r["after_" + side], "probe.py") for side in ["a", "b"]},
            }
        )
    if smoke:
        rows = [r for r in rows if r["id"] in protocol["smoke_ids"]]
    else:
        for r in json.loads((c.OUT / "round3_probes.json").read_text()):
            rows.append(
                {
                    "id": r["id"],
                    "suite": "round3_fresh",
                    "gate": r["gate"],
                    "category": r["category"],
                    "expected_relation": r["expected_relation"],
                    **{side: g.represent(r["before_" + side], r["after_" + side], "probe.py") for side in ["a", "b"]},
                }
            )
    result = []
    for row in rows:
        record = {k: v for k, v in row.items() if k not in ["a", "b"]}
        record["methods"] = {}
        for name, function in functions.items():
            a, b = function(row["a"]), function(row["b"])
            value = g.distance(a, b)
            assert 0 <= value <= 1 and g.distance(a, a) == 0 and value == g.distance(b, a)
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
    summary = {"technical_checks_passed": True, "cases": len(rows), "methods": {}}
    for name in functions:
        groups = {}
        for suite, gate in [
            ("development", "known"),
            ("old_stress", "known"),
            ("round1_now_development", "hard"),
            ("round1_now_development", "exploratory"),
            ("round2_now_development", "hard"),
            ("round2_now_development", "exploratory"),
            ("round3_fresh", "hard"),
            ("round3_fresh", "exploratory"),
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
