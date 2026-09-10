#!/usr/bin/env python3
"""Freeze a second independent suite without importing candidate methods."""

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

    base = "def f(value):\n    first = value + 1\n    second = first * 2\n    result = second - 3\n    return result\n"
    b = base.replace("first", "start").replace("second", "middle").replace("result", "answer")
    add(
        "R01",
        base,
        base.replace("second - 3", "second - 4"),
        b.replace("middle - 3", "middle - 4"),
        "straightline_local_rename",
        "Different bases: consistently rename three internal locals while retaining the public argument and data dependencies. Return behavior and the repair coincide under ordinary integer inputs without introspection.",
        "equal",
        "exploratory",
        b,
    )

    base = "def f(value):\n    left = value + 2\n    right = left * 3\n    return right - left\n"
    b = "def f(value):\n    right = value + 2\n    left = right * 3\n    return left - right\n"
    add(
        "R02",
        base,
        base.replace("* 3", "* 4"),
        b.replace("* 3", "* 4"),
        "local_name_permutation",
        "Different bases: left/right spellings are permuted, with every binding and use preserved. This is alpha-equivalent internal data flow, not swapping the two values.",
        "equal",
        "exploratory",
        b,
    )

    base = "def f(value):\n    result = value * 3\n    return result\n"
    b = "def f(value):\n    unused_before = 0\n    result = value * 3\n    unused_after = None\n    return result\n"
    add(
        "R03",
        base,
        base.replace("* 3", "* 4"),
        b.replace("* 3", "* 4"),
        "unused_literal_insertions",
        "Different bases: only two unread local literal assignments are inserted around the same repaired calculation. No external writes or calls occur; return-value invariance excludes tracing and local-variable introspection.",
        "equal",
        "exploratory",
        b,
    )

    base = "def f(value):\n    ignored_a = 0\n    ignored_b = False\n    result = value * 3\n    return result\n"
    b = base.replace("    ignored_a = 0\n    ignored_b = False", "    ignored_b = False\n    ignored_a = 0")
    add(
        "R04",
        base,
        base.replace("* 3", "* 4"),
        b.replace("* 3", "* 4"),
        "unused_literals_reordered",
        "Different bases: reorder independent unread local constants without moving the repair relative to any dependency or observable side effect.",
        "equal",
        "exploratory",
        b,
    )

    base = "def f(value):\n    result = value + 1\n    result = result * 2\n    return result\n"
    b = base.replace("result", "running")
    add(
        "R05",
        base,
        base.replace("* 2", "* 3"),
        b.replace("* 2", "* 3"),
        "renamed_reassigned_local",
        "Different bases: one internal binding is renamed consistently across its initialization, reassignment and use. Same mathematical repair and return values under the stated observations.",
        "equal",
        "exploratory",
        b,
    )

    base = "def f(value):\n    left = value + 1\n    right = value + 3\n    return left + right\n"
    a = base.replace("left", "result")
    b = base.replace("left", "right")
    add(
        "R06",
        base,
        a,
        b,
        "binding_capture",
        "Same base: a fresh local rename preserves behavior, while capturing the existing right binding changes data flow. At value=1, A returns 6 and B returns 8. Name normalization must not treat capture as alpha-renaming.",
    )

    base = "def f(value):\n    left = value + 1\n    right = value + 3\n    return left + right\n"
    add(
        "R07",
        base,
        base.replace("right = value + 3", "right = left + 3"),
        base.replace("right = value + 3", "right = value + 4"),
        "changed_dependency",
        "Same base: one repair depends on the computed local, the other on the input parameter. They happen to have identical integer outputs here but use different dependency mechanisms; this hard check preserves source dependency evidence rather than asserting semantic nonequivalence.",
    )

    base = "def f(value):\n    def inner():\n        return value\n    value += 1\n    return inner()\n"
    a = base.replace("        return value", "        ignored = 3\n        return value")
    b = base.replace("        return value", "        value = 3\n        return value")
    add(
        "R08",
        base,
        a,
        b,
        "closure_shadowing",
        "Same base: adding an unread inner local preserves closure lookup; adding a value binding shadows the captured outer variable. At value=5, A returns 6 and B returns 3. Nested scopes require binding-aware treatment or conservative fallback.",
    )

    base = "counter = 0\n\ndef f(value):\n    global counter\n    counter = 1\n    return value\n"
    add(
        "R09",
        base,
        base.replace("counter = 1", "counter = 2"),
        base.replace("counter = 1", "counter = 1\n    ignored = 2"),
        "global_write",
        "Same base: A changes an externally observable global write to 2; B adds only an unread local constant and leaves counter at 1. Observation includes module state, so a literal RHS does not imply harmlessness.",
    )

    base = "def f(value):\n    out = []\n    return out\n"
    add(
        "R10",
        base,
        base.replace("    return", "    ignored = out.append(value)\n    return"),
        base.replace("    return", "    ignored = 0\n    return"),
        "unread_assignment_with_side_effect",
        "Same base: the unread target does not make its RHS harmless. Assigning the result of append mutates the returned list; assigning literal 0 does not.",
    )

    base = "def f(value):\n    return value\n"
    a = "def f(value):\n    temporary = value\n    result = temporary\n    return result\n"
    b = "def f(value):\n    result = temporary\n    temporary = value\n    return result\n"
    add(
        "R11",
        base,
        a,
        b,
        "read_before_write_order",
        "Same base and same added statements in different order: A returns its input, while B raises UnboundLocalError. Runtime exceptions are observable and compile success alone is insufficient.",
    )

    base = "def f(value):\n    return value\n"
    a = "def f(value):\n    result = value + 1\n    return result\n"
    b = "def f(item):\n    result = item + 1\n    return result\n"
    add(
        "R12",
        base,
        a,
        b,
        "public_keyword_parameter_name",
        "Same base: repaired bodies agree for positional calls, but f(value=1) succeeds only for A. Public parameter names are part of the callable interface and must not be normalized as internal locals.",
    )

    base = (
        "def f(value):\n    def combine(*, left=0, right=0):\n        return left + 2 * right\n    return combine()\n"
    )
    add(
        "R13",
        base,
        base.replace("return combine()", "return combine(left=value)"),
        base.replace("return combine()", "return combine(right=value)"),
        "call_keyword_name",
        "Same base: changed keyword spellings select different callee parameters. At value=3, A returns 3 and B returns 6; keyword names must survive local renaming.",
    )

    base = "def f(value, flag):\n    result = value\n    if flag:\n        result += 1\n    else:\n        result += 1\n    return result\n"
    a = base.replace("result += 1", "result += 2", 1)
    pos = base.rindex("result += 1")
    b = base[:pos] + base[pos:].replace("result += 1", "result += 2", 1)
    add(
        "R14",
        base,
        a,
        b,
        "conditional_boundary",
        "Same replacement appears on opposite branches. At value=1 and flag=True, A returns 3 and B returns 2. Conservative fallback must retain the branch distinction.",
    )

    base = "def f(value):\n    result = value\n    for item in range(2):\n        result += 1\n    result += 1\n    return result\n"
    a = base.replace("result += 1", "result += 2", 1)
    pos = base.rindex("result += 1")
    b = base[:pos] + base[pos:].replace("result += 1", "result += 2", 1)
    add(
        "R15",
        base,
        a,
        b,
        "loop_boundary",
        "Same replacement appears inside versus after a loop. At value=1, A returns 6 and B returns 5; ignoring loops during normalization would erase execution multiplicity.",
    )

    base = "def f(value):\n    out = []\n    return out\n"
    add(
        "R16",
        base,
        base.replace("    return", "    out.append(1)\n    out.append(2)\n    return"),
        base.replace("    return", "    out.append(2)\n    out.append(1)\n    return"),
        "added_block_order",
        "Same base with the same multiset of two additions at the same gap, but opposite order. Returns [1, 2] versus [2, 1]. Event ordering inside an added block must remain observable.",
    )

    base = "offset = 5\n\ndef f(value):\n    result = offset\n    return result + value\n"
    add(
        "R17",
        base,
        base.replace("    return", "    offset = 0\n    return"),
        base.replace("    return", "    ignored = 0\n    return"),
        "late_assignment_changes_earlier_lookup",
        "Same base: inserting offset=0 after an earlier read makes offset local throughout f, causing UnboundLocalError. Inserting ignored=0 does not. Local binding effects are not limited to following statements.",
    )

    base = "def f(value):\n    first, second = value, value + 1\n    return first + second\n"
    b = base.replace("first", "left").replace("second", "right")
    add(
        "R18",
        base,
        base.replace("first + second", "first - second"),
        b.replace("left + right", "left - right"),
        "destructuring_local_rename",
        "Different bases: consistently rename tuple-unpacked locals with identical binding order and uses. This is an exploratory invariance; conservative unsupported-syntax fallback may retain names and fail it without violating a hard safety gate.",
        "equal",
        "exploratory",
        b,
    )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "round2_probes.json"
    protocol_path = OUT / "round2_probes_protocol.json"
    if destination.exists() or protocol_path.exists():
        raise FileExistsError("Round-two probes already frozen")
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
        "source": "Second independently authored pure Python toy suite; no benchmark samples or outcome labels.",
        "blindness": "Author knew normalized-candidate intent and first-suite categories, but did not inspect normalized candidate code or any candidate scores. Expectations frozen before evaluation of this suite.",
        "probe_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "count": len(rows),
        "gates": dict(Counter(row["gate"] for row in rows)),
        "relations": dict(Counter(row["expected_relation"] for row in rows)),
        "hard_relation_rule": "Equal is exactly zero and distinct is positive. Preserve observable behavior/interface distinctions and explicit dependency mechanisms; R07 intentionally distinguishes mechanisms with coincident integer behavior.",
        "exploratory_rule": "Alpha-renaming and harmless-runtime-context invariances are desirable under the observation contract, but remain outside the hard gate because the intended metric may retain lexical evidence or conservatively reject unsupported syntax.",
        "observation_contract": "Ordinary integer/bool arguments, returned values, raised exception types, caller-visible keyword interface and module state. Excludes tracing, frame/locals introspection, resource exhaustion and overloaded arithmetic. Each cross-base case explicitly declares its changed base.",
        "syntax_validation": "All 72 source versions parse and compile; no snippets executed.",
        "api_cost_usd": 0,
        "candidate_code_executed": False,
    }
    destination.write_text(payload)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({key: protocol[key] for key in ["count", "gates", "relations", "probe_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
