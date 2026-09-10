#!/usr/bin/env python3
"""Freeze an unscored, fresh-to-pair-audit development annotation packet."""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request

sp = importlib.util.spec_from_file_location("location", Path(__file__).with_name("41_swe_location_audit.py"))
loc = importlib.util.module_from_spec(sp)
sp.loader.exec_module(loc)
s = loc.s
OUT = s.ROOT / "natural-validation"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_job(issue, block):
    path = block.source_file.removeprefix("a/")
    url = f"https://raw.githubusercontent.com/{issue['repo']}/{issue['base_commit']}/{urllib.parse.quote(path)}"
    old = re.search(r"^index ([0-9a-f]+)\.\.", str(block), re.M)
    return {
        "url": url,
        "repo": issue["repo"],
        "base_commit": issue["base_commit"],
        "path": path,
        "declared_old_git_blob": old[1] if old else None,
    }


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Sample already frozen")
    plan = json.loads((s.OUT / "split.json").read_text())
    prior = json.loads((loc.OUT / "pairs.json").read_text())
    excluded = {r["instance_id"] for r in prior} | {"AnalogJ__lexicon-336"}
    issues = {r["instance_id"]: r for r in s.read_jsonl(s.OUT / "issue_metadata.jsonl")}
    wanted = {
        r["id"]
        for r in plan["candidates"]
        if plan["issues"][r["instance_id"]]["split"] == "dev" and r["instance_id"] not in excluded
    }
    candidates, rejection = defaultdict(list), Counter()
    for path in sorted((s.OUT / "patches").glob("*.jsonl")):
        for raw in s.read_jsonl(path):
            if raw["id"] not in wanted:
                continue
            patch = raw["generated_patch"] or ""
            if not patch.strip() or len(patch) > 16000:
                rejection["empty_or_over_16000_chars"] += 1
                continue
            check = subprocess.run(["git", "apply", "--numstat", "-z"], input=patch, text=True, capture_output=True)
            if check.returncode or not check.stdout.strip():
                rejection["invalid_diff"] += 1
                continue
            blocks = [b for b in loc.PatchSet(patch) if not loc.auxiliary(b.path)]
            if not blocks or any(not b.path.endswith(".py") for b in blocks):
                rejection["empty_or_non_python_runtime"] += 1
                continue
            changed = sum(1 for b in blocks for h in b for line in h if line.is_added or line.is_removed)
            if changed > 80:
                rejection["over_80_runtime_changed_lines"] += 1
                continue
            candidates[raw["instance_id"]].append(
                {
                    "id": raw["id"],
                    "patch": patch,
                    "patch_sha256": s.digest(patch),
                    "runtime_diff_sha256": s.digest("".join(str(b) for b in blocks)),
                    "files": sorted({b.path for b in blocks}),
                    "runtime_changed_lines": changed,
                }
            )
    available, support = {}, []
    for iid, rows in candidates.items():
        pairs = [
            (a, b)
            for a, b in combinations(sorted(rows, key=lambda r: r["id"]), 2)
            if a["runtime_diff_sha256"] != b["runtime_diff_sha256"]
        ]
        same = [(a, b) for a, b in pairs if a["files"] == b["files"]]
        support.append(
            {
                "instance_id": iid,
                "eligible_candidates": len(rows),
                "nonidentical_runtime_pairs": len(pairs),
                "same_runtime_file_set_pairs": len(same),
            }
        )
        if pairs:
            available[iid] = min(
                same or pairs, key=lambda pair: s.digest(f"{s.SEED}:natural-pair:{pair[0]['id']}:{pair[1]['id']}")
            )
    chosen = sorted(available, key=lambda iid: s.digest(f"{s.SEED}:natural-issue:{iid}"))[:12]
    assert len(chosen) == 12
    selected, sources = [], {}
    for index, iid in enumerate(chosen, 1):
        a, b = available[iid]
        if int(s.digest(f"{s.SEED}:natural-orientation:{iid}"), 16) % 2:
            a, b = b, a
        issue = {key: issues[iid][key] for key in ["repo", "base_commit", "problem_statement"]}
        assert (
            issue["repo"] == plan["issues"][iid]["repo"] and issue["base_commit"] == plan["issues"][iid]["base_commit"]
        )
        row = {"pair_id": f"N{index:02d}", "instance_id": iid, **issue, "sides": {"a": a, "b": b}}
        for value in [a, b]:
            for block in loc.PatchSet(value["patch"]):
                if not loc.auxiliary(block.path) and not block.is_added_file:
                    job = source_job(issue, block)
                    sources[job["url"]] = job
        selected.append(row)
    s.write_json(OUT / "pairs.json", selected)
    s.write_json(OUT / "source_requests.json", list(sources.values()))
    s.write_json(
        OUT / "support.json",
        {
            "excluded_issues": sorted(excluded),
            "remaining_dev_issues": len({r["instance_id"] for r in plan["candidates"] if r["id"] in wanted}),
            "candidate_rejections": dict(rejection),
            "issue_support": sorted(support, key=lambda r: r["instance_id"]),
            "supported_issues": len(available),
            "selected_pairs": len(selected),
            "selected_same_file_set": sum(r["sides"]["a"]["files"] == r["sides"]["b"]["files"] for r in selected),
        },
    )
    s.write_json(
        OUT / "protocol.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": s.SEED,
            "runner_sha256": digest(Path(__file__)),
            "split_sha256": digest(s.OUT / "split.json"),
            "pairs_sha256": digest(OUT / "pairs.json"),
            "requests_sha256": digest(OUT / "source_requests.json"),
            "dataset": "nebius/SWE-agent-trajectories",
            "dataset_revision": s.REV,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "scope": "12 distinct development issues fresh to pair audit; previously model-reviewed in semantic development. Not unseen data or an independent utility test.",
            "selection": "Exclude 20 historical audit issues and AnalogJ__lexicon-336. Choose SHA-ordered issue IDs among issues with eligible pairs; per issue prefer same runtime file sets then SHA-ordered distinct runtime-diff pair. Frozen before source retrieval; no replacement after source/reconstruction/AST failures.",
            "eligibility": "Valid diffs, Python runtime files under frozen auxiliary heuristic, <=80 runtime changed lines and <=16000 raw patch characters per side. Nonidentical runtime diff is syntactic eligibility, not proof of semantic distinction.",
            "sources": "Pinned public GitHub base requests and patch-declared old Git blob hashes frozen now. Downloaded SHA256 values locked upon retrieval; unknown hashes cannot be known before fetching.",
            "human_annotation": "Two separate humans annotate location, operation, conditions, issue relevance, uncertainty and evidence. Mixed axes allowed. Adjudicate disagreements before any scores are computed or exposed.",
            "outcomes_and_scores_loaded": False,
            "api_spend_usd": 0,
            "candidate_code_executed": False,
            "smoke": [r["pair_id"] for r in selected[:2]],
        },
    )
    print(
        json.dumps(
            {"selected_pairs": len(selected), "supported_issues": len(available), "source_requests": len(sources)}
        )
    )


def inputs(smoke):
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(Path(__file__)) == protocol["runner_sha256"]
    assert digest(OUT / "pairs.json") == protocol["pairs_sha256"]
    assert digest(OUT / "source_requests.json") == protocol["requests_sha256"]
    rows = json.loads((OUT / "pairs.json").read_text())
    return rows[:2] if smoke else rows


def fetch(smoke):
    rows = inputs(smoke)
    if not smoke:
        assert len(json.loads((OUT / "smoke_verification.json").read_text())) == 2
    urls = {
        source_job(row, b)["url"]
        for row in rows
        for v in row["sides"].values()
        for b in loc.PatchSet(v["patch"])
        if not loc.auxiliary(b.path) and not b.is_added_file
    }
    jobs = [j for j in json.loads((OUT / "source_requests.json").read_text()) if j["url"] in urls]
    dest = OUT / "sources"
    dest.mkdir(exist_ok=True)

    def download(job):
        path = dest / (s.digest(job["url"]) + ".json")
        if path.exists():
            return json.loads(path.read_text())
        cached = loc.OUT / "sources" / path.name
        if cached.exists() and json.loads(cached.read_text())["status"] == "ok":
            record = {**json.loads(cached.read_text()), **job, "retrieval": "existing_public_source_cache"}
        else:
            try:
                with urllib.request.urlopen(job["url"], timeout=30) as response:
                    data = response.read()
                record = {
                    **job,
                    "status": "ok",
                    "retrieval": "public_https_get",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "text": data.decode("utf-8"),
                }
            except (urllib.error.URLError, UnicodeDecodeError) as error:
                record = {**job, "status": "unavailable", "error_type": type(error).__name__}
        s.write_json(path, record)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, jobs))
    all_sources = [json.loads(p.read_text()) for p in sorted(dest.glob("*.json"))]
    s.write_json(OUT / "source_lock.json", [{k: v for k, v in r.items() if k != "text"} for r in all_sources])
    print(json.dumps({"requested": len(results), "statuses": dict(Counter(r["status"] for r in results))}))


def packet(smoke):
    rows = inputs(smoke)
    dest = OUT / "annotation"
    dest.mkdir(exist_ok=True)
    (dest / "base").mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    verified = []
    for row in rows:
        evidence, failures, fingerprints = [], [], {}
        for side, value in row["sides"].items():
            evidence.append(f"## Patch {side.upper()}\n\n```diff\n{value['patch']}\n```\n")
            (OUT / "raw" / f"{row['pair_id']}_{side}.patch").write_text(value["patch"])
            fingerprints[side] = []
            for block in loc.PatchSet(value["patch"]):
                if loc.auxiliary(block.path):
                    continue
                try:
                    if block.is_added_file:
                        before, url = "", None
                    else:
                        job = source_job(row, block)
                        source = json.loads((OUT / "sources" / (s.digest(job["url"]) + ".json")).read_text())
                        if source["status"] != "ok":
                            raise ValueError("Pinned public source unavailable")
                        before, url = source["text"], source["url"]
                        assert s.digest(before) == source["sha256"]
                        data = before.encode()
                        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                        if job["declared_old_git_blob"]:
                            assert blob.startswith(job["declared_old_git_blob"]), "Git blob hash mismatch"
                    after, oldlines, _ = loc.patched(before, block)
                    ast.parse(before)
                    tree = ast.parse(after)
                    fingerprints[side].append([block.path, ast.dump(tree, include_attributes=False)])
                    name = s.digest(before) + ".py"
                    (dest / "base" / name).write_text(before)
                    spans = {
                        i
                        for line in oldlines
                        for i in range(max(0, line - 13), min(len(before.splitlines()), line + 12))
                    }
                    if not spans:
                        spans = {
                            i
                            for h in block
                            for i in range(
                                max(0, h.source_start - 7),
                                min(len(before.splitlines()), h.source_start + h.source_length + 6),
                            )
                        }
                    numbered = "\n".join(f"{i + 1}: {before.splitlines()[i]}" for i in sorted(spans))
                    evidence.append(
                        f"### Base evidence for {side.upper()}: `{block.path}`\n\n[Full base file](base/{name})"
                        + (f" · [Pinned public source]({url})" if url else " (new file)")
                        + f"\n\n```python\n{numbered}\n```\n"
                    )
                except (ValueError, SyntaxError, AssertionError, KeyError, FileNotFoundError) as error:
                    failures.append(
                        {"side": side, "file": block.path, "reason": str(error), "error_type": type(error).__name__}
                    )
                    evidence.append(
                        f"### Unresolved source evidence: {side.upper()} `{block.path}`\n\n{type(error).__name__}: {error}\n"
                    )
        identical = sorted(fingerprints["a"]) == sorted(fingerprints["b"]) if not failures else None
        if identical:
            failures.append(
                {
                    "reason": "Different raw runtime diffs reconstruct to identical runtime ASTs; retained as eligibility failure, not replaced."
                }
            )
        verified.append(
            {
                "pair_id": row["pair_id"],
                "instance_id": row["instance_id"],
                "failures": failures,
                "runtime_file_instances": sum(len(v) for v in fingerprints.values()),
                "identical_reconstructed_runtime_asts": identical,
            }
        )
        (dest / f"{row['pair_id']}.md").write_text(
            f"# {row['pair_id']}\n\n## Issue\n\n{row['problem_statement']}\n\n" + "\n".join(evidence)
        )
    prefix = "smoke_" if smoke else ""
    s.write_json(OUT / f"{prefix}verification.json", verified)
    if not smoke:
        template = [
            {
                "pair_id": r["pair_id"],
                "reviewer_id": None,
                "location_relation": None,
                "location_evidence": None,
                "operation_relation": None,
                "operation_evidence": None,
                "condition_relation": None,
                "condition_evidence": None,
                "relevance_a": None,
                "relevance_b": None,
                "relevance_evidence": None,
                "uncertainty": None,
                "uncertainty_reason": None,
                "mixed_case_notes": None,
            }
            for r in rows
        ]
        for reviewer in [1, 2]:
            (dest / f"reviewer_{reviewer}_blank.jsonl").write_text("".join(json.dumps(r) + "\n" for r in template))
        (dest / "README.md").write_text("""# Natural repair-pair annotation packet

Twelve distinct development issues, fresh to the pair audit. These issues were previously model-reviewed in semantic development: this is not unseen data or an independent utility test. No outcome labels, metric scores, extracted claims, or provisional annotations are included. Syntactically different runtime diffs need not differ semantically. Eligibility/source failures remain in the packet and must be marked uncertain; none were replaced.

Two humans independently complete their separate blank JSONL files before discussing disagreements. Do not inspect previous review results or compute distances until annotation and adjudication are complete. Review the full issue, both complete patches, and pinned base evidence; linked full base files supplement excerpts. Code is evidence only; no candidate execution is required.

Use the [existing location rubric](../../../../docs/swe_repair_location_audit.md) for evidence standards, with the following separate axes instead of a forced single relation label. Mixed cases are explicitly allowed.

- `location_relation`: same / branch / within_function_position / function / file / mixed / unclear. Describe actual A and B locations and cite code.
- `operation_relation`: same / different / mixed / unclear. Compare calls, arguments, assignments, returns and exception behavior.
- `condition_relation`: same / different / mixed / unclear. Compare guards, branch membership and preconditions; do not equate lexical context with proven reachability.
- `relevance_a`, `relevance_b`: relevant / partly_relevant / irrelevant / unclear relative to the issue. This is not a test-passing or successful-repair judgment.
- `uncertainty`: low / medium / high, with reasons. Fill `mixed_case_notes` when more than one axis changes or the label hides a distinction.

Record evidence on every axis. Preserve both original annotations during adjudication, recording the agreed judgment and reason separately. These twelve selected development pairs support a small construct audit, not population accuracy or downstream utility claims.
""")
        summary = {
            "pairs": len(verified),
            "verified_pairs": sum(not r["failures"] for r in verified),
            "unresolved_or_ineligible_pairs": [r["pair_id"] for r in verified if r["failures"]],
            "runtime_file_instances": sum(r["runtime_file_instances"] for r in verified),
            "human_annotations_completed": 0,
            "scores_computed": False,
            "api_spend_usd": 0,
        }
        s.write_json(OUT / "summary.json", summary)
        (OUT / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
        s.write_json(
            OUT / "completed_manifest.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "status": "unscored_packet_ready",
                "files": {
                    str(p.relative_to(OUT)): digest(p)
                    for p in sorted(OUT.rglob("*"))
                    if p.is_file() and p.name != "completed_manifest.json"
                },
            },
        )
        print(json.dumps(summary))
    else:
        print(json.dumps({"smoke_pairs": len(verified), "verified": sum(not r["failures"] for r in verified)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "fetch", "packet"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "fetch":
        fetch(args.smoke)
    else:
        packet(args.smoke)
