#!/usr/bin/env python3
"""Freeze independently authored placement probes; never evaluate candidate methods."""

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/swe-diversity-selection/position-comparison"


def replacement(source, old, new, occurrence=0):
    locations = []
    offset = 0
    while (position := source.find(old, offset)) >= 0:
        locations.append(position)
        offset = position + len(old)
    position = locations[occurrence]
    return source[:position] + new + source[position + len(old) :]


def cases():
    rows = []

    def add(name, before, a, b, relation, category, rationale, gate="hard", before_b=None):
        rows.append(
            dict(
                id=name,
                before_a=before,
                after_a=a,
                before_b=before if before_b is None else before_b,
                after_b=b,
                expected_relation=relation,
                category=category,
                gate=gate,
                rationale=rationale,
            )
        )

    base = "def f(value):\n    out = []\n    out.append(value)\n    value *= 3\n    out.append(value)\n    return out\n"
    add(
        "F01",
        base,
        replacement(base, "out.append(value)", "out.append(value + 7)"),
        replacement(base, "out.append(value)", "out.append(value + 7)", 1),
        "distinct",
        "state_update",
        "Same base: an identical replacement occurs before versus after multiplication. At value=2, returns [9, 6] versus [2, 13].",
    )

    base = "def f(value):\n    out = []\n    if value > 0:\n        out.append(value)\n    value -= 2\n    if value > 0:\n        out.append(value)\n    return out\n"
    add(
        "F02",
        base,
        replacement(base, "out.append(value)", "out.append(value + 7)"),
        replacement(base, "out.append(value)", "out.append(value + 7)", 1),
        "distinct",
        "repeated_identical_conditions",
        "Same base and identical lexical conditions: the first versus second if occurrence is changed. At value=1, returns [8] versus [1].",
    )

    base = "def f(value):\n    out = []\n    out.append(value)\n    out.append(value)\n    return out\n"
    add(
        "F03",
        base,
        replacement(base, "out.append(value)", "out.append(value + 1)"),
        replacement(base, "out.append(value)", "out.append(value + 1)", 1),
        "distinct",
        "adjacent_repeated_statements",
        "The ordered list exposes which identical adjacent statement changed. At value=1, returns [2, 1] versus [1, 2].",
    )

    base = "def f(value):\n    out = []\n" + "    out.append(value)\n" * 7 + "    return out\n"
    add(
        "F04",
        base,
        replacement(base, "out.append(value)", "out.append(value + 1)", 2),
        replacement(base, "out.append(value)", "out.append(value + 1)", 4),
        "distinct",
        "repeated_neighbors",
        "Changes the third versus fifth of seven identical appends, whose immediate unchanged neighbors coincide. The returned list positions differ.",
    )

    base = "def f(value, flag):\n    out = []\n    if flag:\n        out.append(value)\n    else:\n        out.append(value)\n    return out\n"
    add(
        "F05",
        base,
        replacement(base, "out.append(value)", "out.append(value + 1)"),
        replacement(base, "out.append(value)", "out.append(value + 1)", 1),
        "distinct",
        "branch",
        "Same replacement in if versus else. At value=1 and flag=True, returns [2] versus [1].",
    )

    base = "def f(value):\n    out = []\n    return out\n"
    add(
        "F06",
        base,
        base.replace("    return", "    out.append(value)\n    return"),
        base.replace("    return", "    out.append(value)\n    out.append(value)\n    return"),
        "distinct",
        "multiplicity",
        "One versus two new appends must retain multiplicity: list lengths differ.",
    )

    base = "def f(value):\n    return value\n"
    add(
        "F07",
        base,
        base.replace("return value", "return value + 1"),
        base.replace("return value", "return value * 2"),
        "distinct",
        "same_scope_different_mechanisms",
        "Different arithmetic repair operations in the same scope. At value=3, returns 4 versus 6.",
    )

    after = base.replace("return value", "return value + 1")
    add(
        "F08",
        base,
        after,
        "# formatting only\ndef f(value):\n\n    return (value   +   1)  # same expression\n",
        "equal",
        "comments_formatting",
        "Same base and same changed AST; comments, blank lines and redundant parentheses are outside the declared representation.",
    )

    before_b = "# different rendering of base\ndef f(value):\n\n    return (value)\n"
    add(
        "F09",
        base,
        after,
        before_b.replace("return (value)", "return (value + 1)"),
        "equal",
        "base_formatting",
        "Different source renderings of an AST-identical base and repair. Added base comments and blank lines cannot determine execution position.",
        before_b=before_b,
    )

    base = "def f(value):\n    out = []\n    for item in range(2):\n        out.append(value)\n    out.append(value)\n    return out\n"
    add(
        "F10",
        base,
        replacement(base, "out.append(value)", "out.append(value + 1)"),
        replacement(base, "out.append(value)", "out.append(value + 1)", 1),
        "distinct",
        "loop_boundary",
        "An identical replacement occurs inside versus after the loop. At value=1, returns [2, 2, 1] versus [1, 1, 2].",
    )

    base = "def f(value):\n    out = []\n    out.append(value)\n    marker = 0\n    marker += 1\n    value *= 2\n    marker = 0\n    marker += 1\n    out.append(value)\n    return out\n"
    add(
        "F11",
        base,
        replacement(base, "out.append(value)", "out.append(value + 1)"),
        replacement(base, "out.append(value)", "out.append(value + 1)", 1),
        "distinct",
        "distant_dependency_update",
        "The relevant state update is separated from edits by unrelated assignments. At value=2, returns [3, 4] versus [2, 5].",
    )

    base = "def f(value):\n    out = []\n    out.append(value)\n    value += 1\n    out.append(value)\n    return out\n"
    add(
        "F12",
        base,
        replacement(base, "    out.append(value)\n", ""),
        replacement(base, "    out.append(value)\n", "", 1),
        "distinct",
        "deletion_position",
        "Deletion must retain occurrence identity just as replacement does. At value=1, returns [2] versus [1].",
    )

    base = "def f(value):\n    result = value + 1\n    return result\n"
    after = base.replace("value + 1", "value + 2")
    before_b = base.replace("    result", "    unused = 0\n    result", 1)
    add(
        "F13",
        base,
        after,
        before_b.replace("value + 1", "value + 2"),
        "equal",
        "unused_literal_context",
        "Different bases: an unused local literal assignment precedes the same repair. Under ordinary integer inputs, return-value observation, and no tracing/introspection, it adds no relevant behavior. Invariance to this runtime statement is explicitly exploratory.",
        "exploratory",
        before_b,
    )

    before_b = base.replace("result", "answer")
    add(
        "F14",
        base,
        after,
        before_b.replace("value + 1", "value + 2"),
        "equal",
        "consistent_local_renaming",
        "Different bases: consistently rename only the internal result variable, preserving the function signature. Return values and repair operations coincide absent local-variable introspection; name-insensitive repair similarity is exploratory.",
        "exploratory",
        before_b,
    )

    base = "def f(value):\n    left = value + 1\n    right = left * 2\n    return right\n"
    after = base.replace("left * 2", "left * 3")
    before_b = "def f(value):\n    right = value + 1\n    left = right * 2\n    return left\n"
    add(
        "F15",
        base,
        after,
        before_b.replace("right * 2", "right * 3"),
        "equal",
        "binding_preserving_name_permutation",
        "Different bases: swap two internal variable spellings consistently while preserving bindings and data flow. Pure integer return behavior is identical before and after; a lexical-name comparison need not satisfy this exploratory invariance.",
        "exploratory",
        before_b,
    )

    base = "def f(value):\n    unused_a = 0\n    unused_b = 1\n    return value + 1\n"
    before_b = base.replace("    unused_a = 0\n    unused_b = 1", "    unused_b = 1\n    unused_a = 0")
    add(
        "F16",
        base,
        base.replace("value + 1", "value + 2"),
        before_b.replace("value + 1", "value + 2"),
        "equal",
        "independent_literal_context_reordering",
        "Different bases: reorder two unused local literal assignments, preserving the same return repair. No calls, external writes, dependency, tracing or introspection are assumed. Ignoring this runtime context is exploratory.",
        "exploratory",
        before_b,
    )

    base = "def f(value):\n    result = value + 1\n    return result\n"
    before_b = base.replace("value + 1", "value * 2")
    add(
        "F17",
        base,
        base.replace("return result", "return result * 10"),
        before_b.replace("return result", "return result * 10"),
        "distinct",
        "different_incoming_dependency_context",
        "Different bases and identical edit: the retained incoming definition differs. At value=3, base/after outputs are 4/40 versus 6/60. Distinctness tests contextual repair effects, not operation identity, and is explicitly exploratory because the repair operation itself is the same.",
        "exploratory",
        before_b,
    )

    base = "def f(value):\n    result = value + 1\n    return result\n"
    before_b = base.replace("return result", "return result * 2")
    add(
        "F18",
        base,
        base.replace("value + 1", "value + 2"),
        before_b.replace("value + 1", "value + 2"),
        "distinct",
        "different_outgoing_dependency_context",
        "Different bases and identical edit: retained downstream consumers differ. At value=3, base/after outputs are 4/5 versus 8/10. This tests contextual effects rather than operation identity and is exploratory.",
        "exploratory",
        before_b,
    )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "fresh_probes.json"
    protocol_path = OUT / "fresh_probes_protocol.json"
    if destination.exists() or protocol_path.exists():
        raise FileExistsError("Independent probes already frozen")
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
        "source": "Independently authored pure Python toy snippets; no natural benchmark samples or outcomes.",
        "blindness": "Author read prior grounded report and script44 but neither new candidate implementation nor any candidate scores. Sources and expectations frozen before candidate evaluation.",
        "probe_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "count": len(rows),
        "gates": dict(Counter(row["gate"] for row in rows)),
        "relations": dict(Counter(row["expected_relation"] for row in rows)),
        "hard_relation_rule": "Equal requires distance exactly zero; distinct requires positive distance. These selected engineering checks are not sampled population accuracy.",
        "exploratory_rule": "Renaming/irrelevant-runtime-context invariance and cross-base contextual-effect sensitivity depend on the declared notion of repair similarity; do not count them in the hard promotion gate.",
        "syntax_validation": "All 72 source versions parse and compile; snippets were not executed.",
        "interpretation": "All witnesses assume ordinary integer/bool inputs and observation of returned values, without tracing, reflection, resource exhaustion, or overloaded arithmetic. Different bases are declared in each rationale.",
        "api_cost_usd": 0,
        "candidate_code_executed": False,
    }
    destination.write_text(payload)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({key: protocol[key] for key in ["count", "gates", "relations", "probe_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
