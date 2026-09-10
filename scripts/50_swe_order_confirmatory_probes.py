#!/usr/bin/env python3
"""Freeze final independent order probes without candidate imports or execution."""

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/swe-diversity-selection/position-comparison"


def cases():
    rows = []

    def add(name, base, a, b, category, rationale, relation="distinct", gate="hard", base_b=None):
        rows.append(
            dict(
                id=name,
                before_a=base,
                after_a=a,
                before_b=base if base_b is None else base_b,
                after_b=b,
                expected_relation=relation,
                category=category,
                gate=gate,
                rationale=rationale,
            )
        )

    base = "def f(value):\n    return value\n"
    a = "def f(value):\n    first = value + 1\n    second = first * 2\n    return second\n"
    b = "def f(value):\n    second = first * 2\n    first = value + 1\n    return second\n"
    add(
        "T01",
        base,
        a,
        b,
        "inserted_block_dependency_order",
        "Same added/replaced statement multiset in one formerly single-return block. A returns 2*(value+1); B raises UnboundLocalError by reading first before binding it. Relative event order has an observable dependency witness.",
    )

    base = "def f(value):\n    out = []\n    out.append(1)\n    out.append(2)\n    out.append(3)\n    out.append(2)\n    out.append(1)\n    out.append(3)\n    return out\n"
    a = base.replace("    out.append(1)\n    out.append(2)\n", "", 1)
    b = base.replace("    out.append(2)\n    out.append(1)\n", "", 1)
    add(
        "T02",
        base,
        a,
        b,
        "deleted_block_order_and_location",
        "Delete the same multiset of appends in opposite source order at different locations. A returns [3, 2, 1, 3]; B returns [1, 2, 3, 3]. This tests both deleted event order and occurrence placement, not a pure isolated order contrast.",
    )

    base = "def f(value):\n    out = []\n    return out\n"
    a = base.replace("    return", "    out.append(value)\n    value += 1\n    out.append(value)\n    return")
    b = base.replace("    return", "    out.append(value)\n    out.append(value)\n    value += 1\n    return")
    add(
        "T03",
        base,
        a,
        b,
        "duplicate_operations_around_new_update",
        "Same added block multiset with two identical appends and one state update. At value=1, A returns [1, 2], B [1, 1]. Duplicates and their relative order around a newly added update both matter.",
    )

    base = "def f(value):\n    out = []\n    result = value\n    return out, result\n"
    a = base.replace("    return", "    discarded = out.append(result)\n    result += 1\n    return")
    b = base.replace("    return", "    result += 1\n    discarded = out.append(result)\n    return")
    add(
        "T04",
        base,
        a,
        b,
        "unread_call_assignment_order",
        "An unread assignment target still invokes append. At value=1, A returns ([1], 2), B ([2], 2). Filtering unused targets must preserve side-effecting RHS expressions and their placement around updates.",
    )

    base = "def f(value):\n    def inner():\n        return value\n    return inner()\n"
    a = base.replace("        return value", "        result = value\n        ignored = 0\n        return result")
    b = base.replace("        return value", "        result = value\n        value = 0\n        return result")
    add(
        "T05",
        base,
        a,
        b,
        "late_inner_binding_shadowing",
        "Adding a late value assignment makes the earlier inner read local rather than captured. A returns the input; B raises UnboundLocalError. This is a binding/context sensitivity trap, not an isolated event-order comparison.",
    )

    base = "def f(value):\n    out = []\n    return out\n"
    a = base.replace("    return", "    out.append(1)\n    out.append(1)\n    out.append(2)\n    return")
    b = base.replace("    return", "    out.append(1)\n    out.append(2)\n    out.append(1)\n    return")
    add(
        "T06",
        base,
        a,
        b,
        "identical_duplicate_occurrence_order",
        "The same block has two identical appends plus one differing append. Returns [1, 1, 2] versus [1, 2, 1]. Matching duplicate operations as an unordered multiset erases an observable output position.",
    )

    base = "def f(value):\n    initial = value + 1\n    return initial\n"
    a = "def f(value):\n    initial = value + 1\n    doubled = initial * 2\n    answer = doubled + 3\n    return answer\n"
    base_b = base.replace("initial", "start")
    b = a.replace("initial", "start").replace("doubled", "middle").replace("answer", "finish")
    add(
        "T07",
        base,
        a,
        b,
        "renamed_added_dependency_block",
        "Different bases and consistent fresh internal renaming across retained and newly added bindings. Ordered dependencies, public parameter name, and returned values coincide. Desired invariance assumes ordinary integer inputs without tracing or local introspection.",
        "equal",
        "exploratory",
        base_b,
    )

    base = "def f(value):\n    result = value + 1\n    return result\n"
    a = "def f(value):\n    result = value + 1\n    result *= 2\n    result += 3\n    return result\n"
    base_b = "def f(value):\n    unused_head = 0\n    result = value + 1\n    unused_tail = None\n    return result\n"
    b = "def f(value):\n    unused_tail = None\n    result = value + 1\n    result *= 2\n    unused_head = 0\n    result += 3\n    return result\n"
    add(
        "T08",
        base,
        a,
        b,
        "unread_literals_moved_across_added_block",
        "Different bases: two unread literal assignments occur at different positions, and B also moves them while applying the same ordered arithmetic repair. Under the observation contract these local literal writes are harmless. Tests composition of context filtering and order preservation rather than order alone.",
        "equal",
        "exploratory",
        base_b,
    )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "round3_probes.json"
    protocol_path = OUT / "round3_probes_protocol.json"
    if destination.exists() or protocol_path.exists():
        raise FileExistsError("Final order probes already frozen")
    rows = cases()
    for row in rows:
        for field in ["before_a", "after_a", "before_b", "after_b"]:
            ast.parse(row[field])
            compile(row[field], f"{row['id']}:{field}", "exec")
    payload = json.dumps(rows, indent=2) + "\n"
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": None,
        "deterministic": True,
        "source": "Third independently authored pure Python toy suite; final bounded toy iteration for this phase.",
        "blindness": "Author knew candidate intent (relative order within unmatched blocks plus narrow normalization), but did not inspect ordered-candidate implementation or candidate scores. Sources and expected relations frozen before evaluation.",
        "probe_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "count": len(rows),
        "gates": dict(Counter(row["gate"] for row in rows)),
        "relations": dict(Counter(row["expected_relation"] for row in rows)),
        "hard_relation_rule": "All six hard distinctions have explicit return-value or exception witnesses; positive distance is required. These are selected engineering checks, not population accuracy.",
        "exploratory_rule": "Exact zero desired for two normalization combinations under stated observations, reported separately from hard gates.",
        "observation_contract": "Ordinary integer input; observe returned values and raised exception types. No tracing, frame/locals introspection, resource exhaustion, or overloaded arithmetic. Cross-base cases explicitly identify their altered base. Witnesses are reasoned directly from source, not executed.",
        "syntax_validation": "All 32 source versions parse and compile; snippets were not executed.",
        "api_cost_usd": 0,
        "candidate_code_executed": False,
    }
    destination.write_text(payload)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({key: protocol[key] for key in ["count", "gates", "relations", "probe_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
