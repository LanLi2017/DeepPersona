#!/usr/bin/env python3
"""Source-grounded structural controls on frozen development repair pairs."""

import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

sp = importlib.util.spec_from_file_location("location", Path(__file__).with_name("41_swe_location_audit.py"))
loc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(loc)
s = loc.s
OUT = s.ROOT / "grounded-representation"
FOLLOW = s.ROOT / "location-followup"
NAMED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
BLOCKS = {"body", "orelse", "finalbody", "handlers", "cases"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statements(text):
    result = []

    def walk(node, scope, guards):
        named = isinstance(node, NAMED)
        if named:
            scope = scope + [node.name]
        if isinstance(node, (ast.stmt, ast.ExceptHandler, ast.match_case)):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return
            header = copy.copy(node)
            for field in BLOCKS:
                if hasattr(header, field):
                    setattr(header, field, [])
            canonical = ast.dump(header, include_attributes=False)
            record = {
                "scope": ".".join(scope) or "<module>",
                "operation": canonical,
                "code": ast.unparse(header),
                "guards": list(guards),
                "line": getattr(node, "lineno", None),
                "end_line": getattr(node, "end_lineno", None),
            }
            result.append(record)
        else:
            canonical = None
        for field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            branch = guards
            if named and field == "body":
                branch = []
            elif canonical and field in BLOCKS:
                branch = guards + [
                    {"kind": type(node).__name__, "branch": field, "condition": canonical, "code": ast.unparse(header)}
                ]
            for child in children:
                if isinstance(child, ast.AST):
                    walk(child, scope, branch)

    walk(ast.parse(text), [], [])
    return result


def key(record, guarded):
    return json.dumps([record["operation"], record["guards"]] if guarded else record["operation"], sort_keys=True)


def changes(before, after, guarded):
    matcher = SequenceMatcher(a=[key(r, guarded) for r in before], b=[key(r, guarded) for r in after], autojunk=False)
    result = []
    for tag, i, j, k, end in matcher.get_opcodes():
        if tag != "equal":
            result.extend({"action": "remove", **r} for r in before[i:j])
            result.extend({"action": "add", **r} for r in after[k:end])
    return result


def represent(before, after, file):
    sides = {"before": statements(before), "after": statements(after)}
    scopes = sorted({r["scope"] for values in sides.values() for r in values})
    result = []
    for scope in scopes:
        a, b = ([r for r in sides[side] if r["scope"] == scope] for side in ["before", "after"])
        events = changes(a, b, True)
        if events:
            result.append(
                {
                    "file": file,
                    "scope": scope,
                    "before": a,
                    "after": b,
                    "events": events,
                    "operation_events": changes(a, b, False),
                }
            )
    return result


def signature(records, mode):
    counts = Counter()
    for scope in records:
        if mode == "scope":
            counts[json.dumps([scope["file"], scope["scope"]])] = 1
            continue
        events = scope["operation_events"] if mode == "operation" else scope["events"]
        for event in events:
            item = [event["action"], event["operation"]]
            if mode == "combined":
                item += [scope["file"], scope["scope"], event["guards"]]
            counts[json.dumps(item, sort_keys=True)] += 1
    return counts


def distance(a, b):
    union = sum((a | b).values())
    return 1 - sum((a & b).values()) / union if union else 0.0


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen")
    plan = json.loads((s.OUT / "split.json").read_text())
    pairs = json.loads((loc.OUT / "pairs.json").read_text())
    wanted = {r[side] for r in pairs for side in ["a", "b"]}
    raw = {
        r["id"]: r["generated_patch"]
        for p in sorted((s.OUT / "patches").glob("*.jsonl"))
        for r in s.read_jsonl(p)
        if r["id"] in wanted
    }
    sources = [json.loads(p.read_text()) for p in sorted((loc.OUT / "sources").glob("*.json"))]
    source_map = {(r["repo"], r["base_commit"], r["path"]): r for r in sources}
    jobs = []
    for pair in pairs:
        issue = plan["issues"][pair["instance_id"]]
        assert issue["split"] == "dev"
        row = {
            "pair_id": pair["pair_id"],
            "instance_id": pair["instance_id"],
            "origin": "historical_development",
            "sides": {},
        }
        for side in ["a", "b"]:
            files, skipped = [], []
            for block in loc.PatchSet(raw[pair[side]]):
                if loc.auxiliary(block.path):
                    skipped.append(block.path)
                    continue
                source = (
                    None
                    if block.is_added_file
                    else source_map.get((issue["repo"], issue["base_commit"], block.source_file.removeprefix("a/")))
                )
                files.append(
                    {
                        "file": block.path,
                        "diff": str(block),
                        "before": "" if block.is_added_file else source["text"] if source else None,
                        "source_url": source["url"] if source else None,
                        "source_sha256": source["sha256"] if source else None,
                    }
                )
            row["sides"][side] = {"files": files, "excluded_auxiliary": skipped}
        jobs.append(row)
    for pair in json.loads((FOLLOW / "controlled_pairs.json").read_text()):
        source = next(r for r in sources if r["sha256"] == pair["source_sha256"])
        jobs.append(
            {
                "pair_id": pair["pair_id"],
                "instance_id": pair["instance_id"],
                "origin": pair["origin"],
                "sides": {
                    side: {
                        "files": [
                            {
                                "file": v["file"],
                                "diff": v["patch"],
                                "before": source["text"],
                                "source_url": source["url"],
                                "source_sha256": source["sha256"],
                            }
                        ],
                        "excluded_auxiliary": [],
                    }
                    for side, v in pair["sides"].items()
                },
            }
        )
    s.write_json(OUT / "inputs.json", jobs)
    s.write_json(
        OUT / "protocol.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": s.SEED,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(),
            "model": None,
            "api_cost_usd": 0,
            "inputs_sha256": digest(OUT / "inputs.json"),
            "runner_sha256": digest(Path(__file__)),
            "scope": "26 previously inspected development pairs; no held-out inference or benchmark outcomes.",
            "representation": "Exact reconstruction; AST statement/header records with qualified scope, source spans, lexical guard/branch ancestry, and ordered before/after statements in affected scopes. Anonymous expression control remains inside AST operation. No reachability or interprocedural inference.",
            "normalization": "Ignore comments, whitespace and standalone string expressions including docstrings. Preserve identifiers, literal values, call arguments, multiplicity and statement ordering in evidence. Exclude auxiliary files using frozen script41 heuristic; report exclusions.",
            "comparisons": "Unweighted multiset Jaccard of operation edit events; scope set Jaccard; unweighted multiset Jaccard of (edit, operation, file, scope, lexical guards). SequenceMatcher within each scope, autojunk=False; no fitted weights or thresholds.",
            "empty_and_failure": "Two empty verified signatures distance 0; one empty distance 1. Missing source, invalid Python, or hunk mismatch abstains for the whole pair, never silently maps to zero.",
            "development_acceptance": [
                "All 6 L pairs have equal operation-only signatures and nonidentical combined signatures.",
                "All 6 frozen identical-runtime/auxiliary candidates have equal combined signatures.",
                "P18 and P19 differ under combined signatures despite equal scope sets.",
                "Evidence preserves P14 actual factory, P19 forced text=True, P20 both guarded and retained unguarded append.",
            ],
            "smoke": ["L01", "L06", "P03", "P14", "P19", "P20"],
            "limitations": "Structural instrument only. Known examples informed construction. Docstring behavior, tests as requested repairs, identifier renaming, data-flow equivalence and same-scope ordering remain validity risks. Passing is not metric validity or downstream utility.",
        },
    )
    print(json.dumps({"frozen_pairs": len(jobs), "new_api_cost_usd": 0}))


def run(smoke):
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(OUT / "inputs.json") == protocol["inputs_sha256"]
    assert digest(Path(__file__)) == protocol["runner_sha256"]
    jobs = json.loads((OUT / "inputs.json").read_text())
    if smoke:
        jobs = [r for r in jobs if r["pair_id"] in protocol["smoke"]]
    else:
        assert json.loads((OUT / "smoke_findings.json").read_text())["status"] == "pass"
    result = []
    for job in jobs:
        row = {k: job[k] for k in ["pair_id", "instance_id", "origin"]}
        row["sides"] = {}
        for side, value in job["sides"].items():
            records, failures, sources = [], [], []
            for f in value["files"]:
                try:
                    if f["before"] is None or not f["file"].endswith(".py"):
                        raise ValueError("Missing source or unsupported language")
                    if f["source_sha256"]:
                        assert hashlib.sha256(f["before"].encode()).hexdigest() == f["source_sha256"]
                    after, _, _ = loc.patched(f["before"], loc.PatchSet(f["diff"])[0])
                    records.extend(represent(f["before"], after, f["file"]))
                    sources.append(
                        {
                            "file": f["file"],
                            "before_sha256": hashlib.sha256(f["before"].encode()).hexdigest(),
                            "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
                            "source_url": f["source_url"],
                        }
                    )
                except (ValueError, SyntaxError) as e:
                    failures.append({"file": f["file"], "reason": str(e)})
            row["sides"][side] = {
                "records": records,
                "failures": failures,
                "sources": sources,
                "excluded_auxiliary": value["excluded_auxiliary"],
            }
        row["distances"] = {
            mode: None
            if any(v["failures"] for v in row["sides"].values())
            else distance(*(signature(row["sides"][side]["records"], mode) for side in ["a", "b"]))
            for mode in ["operation", "scope", "combined"]
        }
        result.append(row)
    prefix = "smoke_" if smoke else ""
    s.write_json(OUT / f"{prefix}records.json", result)
    old = {r["pair_id"]: r for r in json.loads((loc.OUT / "pairs.json").read_text())}
    checks = {
        "no_unresolved_pairs": all(r["distances"]["combined"] is not None for r in result),
        "controlled_operation_match_location_distinct": all(
            r["distances"]["operation"] == 0 and r["distances"]["combined"] > 0
            for r in result
            if r["pair_id"].startswith("L")
        ),
        "same_runtime_invariant": all(
            r["distances"]["combined"] == 0
            for r in result
            if old.get(r["pair_id"], {}).get("sampling_stratum") == "same_runtime_auxiliary_candidate"
        ),
        "same_scope_mechanisms_distinct": all(
            r["distances"]["scope"] == 0 and r["distances"]["combined"] > 0
            for r in result
            if r["pair_id"] in ["P18", "P19"]
        ),
    }
    findings = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "pairs": len(result),
        "distances": {r["pair_id"]: r["distances"] for r in result},
        "fidelity_evidence_review": "Required separately; scores alone do not satisfy evidence gate.",
    }
    s.write_json(OUT / f"{prefix}findings.json", findings)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run(args.smoke)
