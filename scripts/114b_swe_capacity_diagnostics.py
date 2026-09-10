#!/usr/bin/env python3
"""Posthoc source-only changed-region coverage in frozen24k contexts."""

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "runs/swe-diversity-selection/swe-sympy-patch-sft"
OUT = ORIGINAL.parent / "swe-sympy-context-capacity"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def changed_locations(patch):
    result = {}
    name = None
    old_cursor = None
    for line in patch.split(b"\n"):
        if line.startswith(b"diff --git "):
            name = None
            old_cursor = None
        elif line.startswith(b"+++ b/"):
            name = line[6:].decode()
            result[name] = {"deleted_lines": set(), "insertion_boundaries": set()}
        elif line.startswith(b"@@ "):
            match = re.match(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            assert match
            start = int(match[1])
            count = int(match[2]) if match[2] is not None else 1
            old_cursor = start - 1 if count else start
        elif old_cursor is not None and line[:1] == b" ":
            old_cursor += 1
        elif old_cursor is not None and line[:1] == b"-":
            result[name]["deleted_lines"].add(old_cursor + 1)
            old_cursor += 1
        elif old_cursor is not None and line[:1] == b"+":
            result[name]["insertion_boundaries"].add(old_cursor)
    return result


fixtures = [
    (
        b"diff --git a/sympy/a.py b/sympy/a.py\n--- a/sympy/a.py\n+++ b/sympy/a.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n",
        {2},
        {2},
    ),
    (
        b"diff --git a/sympy/a.py b/sympy/a.py\n--- a/sympy/a.py\n+++ b/sympy/a.py\n@@ -0,0 +1,1 @@\n+first\n",
        set(),
        {0},
    ),
    (
        b"diff --git a/sympy/a.py b/sympy/a.py\n--- a/sympy/a.py\n+++ b/sympy/a.py\n@@ -3,0 +4,2 @@\n+last\n+last2\n",
        set(),
        {3},
    ),
    (
        b"diff --git a/sympy/a.py b/sympy/a.py\n--- a/sympy/a.py\n+++ b/sympy/a.py\n@@ -2,1 +1,0 @@\n-gone\n\\ No newline at end of file\n",
        {2},
        set(),
    ),
]
for patch, deletions, insertions in fixtures:
    parsed = changed_locations(patch)["sympy/a.py"]
    assert parsed["deleted_lines"] == deletions and parsed["insertion_boundaries"] == insertions
spec = importlib.util.spec_from_file_location("converter110", ROOT / "scripts/110_swe_patch_sft_targets.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
protocol = read(OUT / "protocol.json")
build = read(OUT / "build_manifest.json")
scores = read(OUT / "score_summary.json")
assert read(OUT / "independent_public_build_audit.json")["all_checks_passed"]
assert read(OUT / "independent_capacity_score_audit.json")["all_checks_passed"]
for path, value in protocol["source_sha256"].items():
    assert sha(path) == value, path
completed = read(ORIGINAL / "source_targets/completed_manifest.json")
rows = []
inputs = {}
for task in protocol["source_tasks"]:
    iid = task["instance_id"]
    assert task["role"] == "source"
    context_path = OUT / "24576" / (iid + ".json")
    context = read(context_path)
    assert context["status"] == "ready" and sha(context_path) == build["artifact_sha256"][str(context_path)]
    record_path = ORIGINAL / "source_targets" / iid / "record.json"
    patch_path = record_path.parent / "gold.patch"
    for p in [record_path, patch_path]:
        assert sha(p) == completed[str(p)]
    for p in [record_path, patch_path, context_path]:
        inputs[str(p)] = sha(p)
    record = read(record_path)
    patch = patch_path.read_bytes()
    locations = changed_locations(patch)
    base = {name: (ORIGINAL / "public" / iid / "source" / name).read_bytes() for name in locations}
    metadata = read(ORIGINAL / "public" / iid / "task.json")
    for name, data in base.items():
        assert hashlib.sha256(data).hexdigest() == metadata["source_sha256"][name]
    assert set(converter._parse(base, patch)) == set(locations)
    visible = defaultdict(list)
    for s in context["retrieval"]["source_spans"]:
        visible[s["path"]].append((s["start_line"], s["end_line"]))

    def line_visible(name, line):
        return any(a <= line <= b for a, b in visible[name])

    files = []
    for name, changes in locations.items():
        n = len(converter.byte_lines(base[name]))
        deletions = sorted(changes["deleted_lines"])
        insertions = []
        assert all(1 <= line <= n for line in deletions)
        for boundary in sorted(changes["insertion_boundaries"]):
            assert 0 <= boundary <= n
            adjacent = [line for line in [boundary, boundary + 1] if 1 <= line <= n]
            flags = [line_visible(name, line) for line in adjacent]
            insertions.append(
                dict(
                    boundary_after_old_line=boundary,
                    adjacent_old_lines=adjacent,
                    any_adjacent_visible=any(flags),
                    all_existing_adjacent_visible=bool(flags) and all(flags),
                )
            )
        files.append(
            dict(
                path=name,
                file_selected=bool(visible[name]),
                deleted_lines=deletions,
                missing_deleted_lines=[line for line in deletions if not line_visible(name, line)],
                insertion_boundaries=insertions,
            )
        )
    all_files = all(f["file_selected"] for f in files)
    all_deletions = all(not f["missing_deleted_lines"] for f in files)
    all_insertions = all(b["any_adjacent_visible"] for f in files for b in f["insertion_boundaries"])
    both_insertions = all(b["all_existing_adjacent_visible"] for f in files for b in f["insertion_boundaries"])
    local = all_files and all_deletions and all_insertions
    strict_local = all_files and all_deletions and both_insertions
    anchors = []
    full = None
    if record["conversion_supported"]:
        for edit in record["edits"]:
            name = edit["path"]
            old = edit["old"].encode()
            assert base[name].count(old) == 1
            start = base[name].index(old)
            end = start + len(old)
            a = base[name][:start].count(b"\n") + 1
            b = base[name][: end - 1].count(b"\n") + 1
            anchors.append(
                dict(path=name, start_line=a, end_line=b, fully_visible=any(x <= a <= b <= y for x, y in visible[name]))
            )
        full = all(a["fully_visible"] for a in anchors)
        expected = next(
            r for cap in scores["caps"] if cap["cap"] == 24576 for r in cap["tasks"] if r["instance_id"] == iid
        )
        assert full == expected["anchor_visible"]
    category = (
        "conversion_unsupported"
        if full is None
        else (
            "missing_changed_file"
            if not all_files
            else ("missing_anchor_region_in_retrieved_file" if not full else "all_anchors_visible")
        )
    )
    rows.append(
        dict(
            instance_id=iid,
            category=category,
            conversion_supported=record["conversion_supported"],
            all_changed_files_selected=all_files,
            all_deletion_lines_visible=all_deletions,
            all_insertion_boundaries_any_adjacent_visible=all_insertions,
            all_insertion_boundaries_both_existing_adjacent_visible=both_insertions,
            changed_region_locality_proxy=local,
            strict_changed_region_locality_proxy=strict_local,
            full_anchors_visible=full,
            locality_pass_full_anchor_fail=local and full is False,
            strict_locality_pass_full_anchor_fail=strict_local and full is False,
            completion_fits=record["completion_fits"],
            files=files,
            anchors=anchors,
        )
    )
summary = dict(
    categories=dict(Counter(r["category"] for r in rows)),
    **{
        key: sum(r[key] is True for r in rows)
        for key in [
            "all_changed_files_selected",
            "all_deletion_lines_visible",
            "all_insertion_boundaries_any_adjacent_visible",
            "all_insertion_boundaries_both_existing_adjacent_visible",
            "changed_region_locality_proxy",
            "strict_changed_region_locality_proxy",
            "full_anchors_visible",
            "locality_pass_full_anchor_fail",
            "strict_locality_pass_full_anchor_fail",
        ]
    },
    locality_and_completion_fit=sum(r["changed_region_locality_proxy"] and r["completion_fits"] is True for r in rows),
    full_anchor_unavailable_conversion_unsupported=sum(r["full_anchors_visible"] is None for r in rows),
)
for p in [
    Path(__file__).resolve(),
    OUT / "protocol.json",
    OUT / "build_manifest.json",
    OUT / "score_summary.json",
    OUT / "independent_public_build_audit.json",
    OUT / "independent_capacity_score_audit.json",
    ROOT / "scripts/110_swe_patch_sft_targets.py",
]:
    inputs[str(p)] = sha(p)
result = dict(
    posthoc=True,
    source_slots=32,
    cap=24576,
    synthetic_boundary_checks=4,
    definitions={
        "deleted_lines": "Original1-based lines marked minus in each exact unified-diff hunk; unchanged hunk context omitted.",
        "insertion_boundaries": "For each plus run, b is the number of original lines consumed at that point; replacement plus lines follow consumed deletions. Distinct b counted once per file.",
        "insertion_locality": "At least one existing adjacent original line b or b+1 must be visible; at BOF/EOF only the existing neighbor applies. This is a locality proxy, not an unambiguous edit anchor or semantic sufficiency claim.",
        "strict_insertion_locality": "Supplementary sensitivity requires BOTH adjacent original lines when both exist; one at BOF/EOF.",
        "vacuity": "No deletions makes deletion criterion true; no insertions makes insertion criterion true. Combined locality also requires every changed file selected.",
        "full_anchor": "Unchanged111canonicalold strings wholly inside one frozen retrieved interval; undefined for the one>20edit target.",
        "interpretation": "Locality-pass/full-anchor-fail identifies additional coverage demanded by the exactold-anchor contract, including unchanged hunk context and unique-anchor expansion; it does not prove a correct repair can be inferred or uniquely applied.",
    },
    summary=summary,
    tasks=rows,
    inputs_sha256=inputs,
    evaluation_private_values_read=False,
    contexts_modified=False,
    reranking=False,
    api_spend_usd=0,
    gpu_used=False,
)
with (OUT / "changed_region_diagnostics.json").open("x") as f:
    f.write(json.dumps(result, indent=2) + "\n")
print(json.dumps(summary))
