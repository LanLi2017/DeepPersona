#!/usr/bin/env python3
"""Two frozen public-only production retrieval policies at24k native tokens."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "runs/swe-diversity-selection/swe-sympy-patch-sft"
CAPACITY = ORIGINAL.parent / "swe-sympy-context-capacity"
OUT = ORIGINAL.parent / "swe-sympy-retrieval-policies"
POLICIES = ["production_chunks", "production_files"]
BUDGET = 24576
WORKERS = 8


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(p, value):
    with p.open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize():
    global HELPER, ENCODING, RANKING, TARGETS
    HELPER = load("context109", ROOT / "scripts/109_swe_patch_sft_context.py")
    ENCODING = HELPER.NativeEncoding(HELPER.tokenizer())
    RANKING = HELPER.load_ranking()
    TARGETS = load("targets110", ROOT / "scripts/110_swe_patch_sft_targets.py")
    assert RANKING.SOFT_FILE_TOKENS == 6500 and RANKING.MAX_SELECTED_FILES == 16 and RANKING.MAX_CANDIDATES == 600


def build_packet(task_dir, encoding, budget, policy):
    m = RANKING
    EXTENSIONS, MAX_FILE_BYTES = m.EXTENSIONS, m.MAX_FILE_BYTES
    MAX_CANDIDATES, MAX_SELECTED_FILES = m.MAX_CANDIDATES, m.MAX_SELECTED_FILES
    SOFT_FILE_TOKENS, WORD = m.SOFT_FILE_TOKENS, m.WORD
    spans, terms, merge, render, digest = m.spans, m.terms, m.merge, m.render, m.digest
    public = task_dir.resolve()
    for name in ["task.json", "problem_statement.md", "source"]:
        assert (public / name).exists(), (public.name, name)
        assert (public / name).resolve().is_relative_to(public), "No external symlink reads"
    meta_blob = (public / "task.json").read_bytes()
    issue_blob = (public / "problem_statement.md").read_bytes()
    metadata = json.loads(meta_blob)
    issue = issue_blob.decode("utf-8")
    files, skipped, chunks = {}, [], []
    source = public / "source"
    for file in sorted(source.rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(source).as_posix()
        if file.is_symlink() or not file.resolve().is_relative_to(source.resolve()):
            skipped.append({"path": relative, "reason": "symlink"})
            continue
        if file.suffix.lower() not in EXTENSIONS and file.name not in {"Makefile", "Dockerfile", "setup.py"}:
            skipped.append({"path": relative, "reason": "extension"})
            continue
        if file.stat().st_size > MAX_FILE_BYTES:
            skipped.append({"path": relative, "reason": "size"})
            continue
        blob = file.read_bytes()
        if b"\x00" in blob:
            skipped.append({"path": relative, "reason": "binary"})
            continue
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append({"path": relative, "reason": "encoding"})
            continue
        files[relative] = {"sha256": digest(blob), "bytes": len(blob), "lines": text.splitlines(keepends=True)}
        if TARGETS.path_allowed(relative):
            chunks.extend(spans(relative, text))
    assert chunks, "No source chunks available"
    query = Counter(terms(issue))
    df = Counter(term for chunk in chunks for term in chunk.tf if term in query)
    avg = sum(chunk.length for chunk in chunks) / len(chunks)
    issue_lower = issue.lower()
    mentioned = set(WORD.findall(issue))
    calls = set(re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", issue))
    for chunk in chunks:
        total = 0.0
        for term in sorted(query.keys() & chunk.tf.keys()):
            tf = chunk.tf[term]
            idf = math.log(1 + (len(chunks) - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * chunk.length / avg)) * (1 + math.log(query[term]))
        chunk.bm25 = total
        path = chunk.path.lower()
        leaf = Path(path).name
        if path in issue_lower or path.replace("/", ".").removesuffix(".py") in issue_lower:
            chunk.path_boost = 24.0
        elif leaf in issue_lower and len(leaf) > 5:
            chunk.path_boost = 12.0
        if chunk.symbol and any(s in mentioned for s in chunk.symbol.split(".")):
            chunk.symbol_boost = 8.0
        owner, _, member = chunk.symbol.rpartition(".")
        chunk.direct_symbol_anchor = bool(
            chunk.symbol and (chunk.symbol in calls or (owner in calls and member in {"__new__", "__init__"}))
        )
        chunk.score = total + chunk.path_boost + chunk.symbol_boost
    ranked = sorted(chunks, key=lambda c: (not c.direct_symbol_anchor, -c.score, c.path, c.start, c.end, c.kind))
    chosen, selected = {}, []
    full_attempts, attempted_files = [], set()

    def count(text):
        return len(encoding.encode(text, disallowed_special=()))

    assert count(render(public.name, metadata, issue, {}, files)) < budget, "Issue alone exceeds packet budget"
    # First pass limits any single file from monopolizing the packet; the second
    # permits additional relevant material when there is remaining room.
    for limited in [True, False]:
        for chunk in ranked[:MAX_CANDIDATES]:
            if chunk.score <= 0:
                continue
            if chunk.path not in chosen and len(chosen) >= MAX_SELECTED_FILES:
                continue
            if policy == "production_files" and chunk.path not in attempted_files:
                attempted_files.add(chunk.path)
                full = {**chosen, chunk.path: [(1, len(files[chunk.path]["lines"]))]}
                full_count = count(render(public.name, metadata, issue, full, files))
                accepted = full_count <= budget
                full_attempts.append({"path": chunk.path, "native_prompt_tokens": full_count, "accepted": accepted})
                if accepted:
                    chosen = full
                    selected.append(
                        {
                            "path": chunk.path,
                            "lines": [1, len(files[chunk.path]["lines"])],
                            "symbol": "",
                            "kind": "full_file",
                            "score": chunk.score,
                            "bm25": chunk.bm25,
                            "path_boost": chunk.path_boost,
                            "symbol_boost": chunk.symbol_boost,
                            "direct_symbol_anchor": chunk.direct_symbol_anchor,
                            "selection_pass": 1 if limited else 2,
                        }
                    )
                    continue
            old = chosen.get(chunk.path, [])
            proposed = merge(old + [(1, min(60, len(files[chunk.path]["lines"]))), (chunk.start, chunk.end)])
            if merge(old) == proposed:
                continue
            if limited:
                file_text = "".join("".join(files[chunk.path]["lines"][a - 1 : b]) for a, b in proposed)
                if count(file_text) > SOFT_FILE_TOKENS:
                    continue
            candidate = {**chosen, chunk.path: proposed}
            packet = render(public.name, metadata, issue, candidate, files)
            if count(packet) > budget:
                continue
            chosen = candidate
            selected.append(
                {
                    "path": chunk.path,
                    "lines": [chunk.start, chunk.end],
                    "symbol": chunk.symbol,
                    "kind": chunk.kind,
                    "score": chunk.score,
                    "bm25": chunk.bm25,
                    "path_boost": chunk.path_boost,
                    "symbol_boost": chunk.symbol_boost,
                    "direct_symbol_anchor": chunk.direct_symbol_anchor,
                    "selection_pass": 1 if limited else 2,
                }
            )
    assert chosen, "No source span fits budget"
    packet = render(public.name, metadata, issue, chosen, files)
    token_count = count(packet)
    assert token_count <= budget
    source_spans = []
    for path, intervals in chosen.items():
        for start, end in merge(intervals):
            assert 1 <= start <= end <= len(files[path]["lines"])
            exact = "".join(files[path]["lines"][start - 1 : end])
            assert exact.rstrip("\n") in packet
            source_spans.append(
                {
                    "path": path,
                    "start_line": start,
                    "end_line": end,
                    "file_sha256": files[path]["sha256"],
                    "span_utf8_sha256": digest(exact.encode()),
                }
            )
    manifest = {
        "task_id": public.name,
        "packet_sha256": digest(packet.encode()),
        "packet_tokens": token_count,
        "tokenizer": "tiktoken/o200k_base",
        "token_count_is_model_specific_guarantee": False,
        "budget": budget,
        "issue_sha256": digest(issue_blob),
        "public_task_json_sha256": digest(meta_blob),
        "public_source_files_read": {
            path: {"sha256": value["sha256"], "bytes": value["bytes"], "lines": len(value["lines"])}
            for path, value in files.items()
        },
        "skipped_files": skipped,
        "candidate_chunks": len(chunks),
        "source_spans": source_spans,
        "selection_events": selected,
        "top_retrieval_candidates": [
            {
                "path": c.path,
                "lines": [c.start, c.end],
                "symbol": c.symbol,
                "score": c.score,
                "direct_symbol_anchor": c.direct_symbol_anchor,
            }
            for c in ranked[:40]
        ],
        "private_or_evaluator_inputs_read": False,
        "policy": policy,
        "full_file_attempts": full_attempts,
    }
    return packet, manifest


def freeze():
    assert not OUT.exists() or all(p.name == "retrieval_policy_static_review.json" for p in OUT.iterdir())
    review_path = OUT / "retrieval_policy_static_review.json"
    review = read(review_path)
    assert review["all_checks_passed"] and review["source_sha256"] == sha(Path(__file__).resolve())
    helper = load("context109", ROOT / "scripts/109_swe_patch_sft_context.py")
    original = helper.validate()
    tasks = [t for t in read(ORIGINAL / "selection.json")["tasks"] if t["role"] == "source"]
    assert len(tasks) == 32 and len({t["instance_id"] for t in tasks}) == 32
    paths = [
        review_path,
        Path(__file__).resolve(),
        ROOT / "scripts/110_swe_patch_sft_targets.py",
        ORIGINAL / "selection.json",
        ORIGINAL / "export_manifest.json",
        ORIGINAL / "context/protocol.json",
        CAPACITY / "protocol.json",
        CAPACITY / "build_manifest.json",
    ]
    hashes = {str(p): sha(p) for p in paths} | original["source_sha256"]
    for task in tasks:
        iid = task["instance_id"]
        for p in [
            ORIGINAL / "public" / iid / "task.json",
            ORIGINAL / "public" / iid / "problem_statement.md",
            CAPACITY / "24576" / (iid + ".json"),
        ]:
            hashes[str(p)] = sha(p)
    OUT.mkdir(exist_ok=True)
    save(
        OUT / "protocol.json",
        dict(
            utc=utc(),
            source_tasks=tasks,
            policies=POLICIES,
            prompt_cap=BUDGET,
            completion_cap_including_eos=2048,
            minimum_prequalified_sources=16,
            workers=WORKERS,
            system=helper.SYSTEM,
            model=str(helper.MODEL),
            native_chat=original["native_chat"],
            policy_definitions={
                "production_chunks": "Same91 BM25/path/symbol ranking, chunk/header and two passes, but chunk candidates only110.path_allowed existingproductionPython. DF/average lengths recomputed on eligible chunks; no target inputs.",
                "production_files": "Same eligible-chunk ranking. At first positive-ranked encounter of each file within top600/file-count allowance, try its complete original contents. If whole native prompt fits24576, accept; otherwise fall back to original chunk/header logic. Subsequent chunkfills follow91 normally. Whole-file attempts explicitly bypass soft6500 per-file limit; hard24576 and MAX16files remain.",
            },
            hard_limits={"max_selected_files": 16, "max_ranked_candidates": 600, "soft_file_native_tokens": 6500},
            selection_rule="Highest source prequalified count>=16; tie production_chunks. Any technical error blocks choice. Cached11424k baseline is a comparison only.",
            stage_boundary="All64new public contexts frozen before separate source-only visibility/target-token scoring. No source patches or source_targets records opened in builder.",
            failures="Preserve every fixedslot; no replacement. Expected context infeasibility retained; technical failures block choice.",
            interpretation="Posthoc source training-development policy selection after capacity8/32; not independent replication, quality admission, learning or diversity evidence.",
            source_sha256=hashes,
            versions={n: importlib.metadata.version(n) for n in ["transformers", "tokenizers", "tiktoken"]},
            evaluation_new_contexts=0,
            evaluation_private_values_read=False,
            quality_tests_run=0,
            api_spend_usd=0,
            gpu_used=False,
        ),
    )
    for policy in POLICIES:
        (OUT / policy).mkdir()


def validate():
    p = read(OUT / "protocol.json")
    for path, value in p["source_sha256"].items():
        assert sha(path) == value, path
    return p


def build_one(job):
    policy, task = job
    iid = task["instance_id"]
    folder = OUT / policy
    path = folder / (iid + ".json")
    assert policy in POLICIES and not path.exists()
    start = time.monotonic()
    row = dict(instance_id=iid, role="source", policy=policy, cap=BUDGET, protocol_sha256=sha(OUT / "protocol.json"))
    try:
        public = ORIGINAL / "public" / iid
        meta = read(public / "task.json")
        assert meta["base_commit"] == task["base_commit"] and meta["role"] == "source"
        assert sha(public / "problem_statement.md") == meta["problem_statement_sha256"]
        assert {str(p.relative_to(public / "source")) for p in (public / "source").rglob("*") if p.is_file()} == set(
            meta["source_sha256"]
        )
        for name, value in meta["source_sha256"].items():
            file = public / "source" / name
            assert file.resolve().is_relative_to((public / "source").resolve()) and sha(file) == value
        packet, retrieval = build_packet(public, ENCODING, BUDGET, policy)
        ids = ENCODING.encode(packet)
        prompt = ENCODING.render(packet)
        direct = ENCODING.tok.apply_chat_template(
            ENCODING.messages(packet), tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(direct, list):
            direct = direct["input_ids"]
        assert ids == direct and len(ids) == retrieval["packet_tokens"] <= BUDGET
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert (public / "problem_statement.md").read_bytes().decode() in packet
        for name in {s["path"] for s in retrieval["source_spans"]}:
            assert TARGETS.path_allowed(name)
            text = (public / "source" / name).read_bytes().decode()
            parts = text.split("\n")
            assert text.splitlines(keepends=True) == [p + "\n" for p in parts[:-1]] + (
                [parts[-1]] if parts[-1] else []
            ), "visible_source_line_convention_mismatch"
        retrieval.update(
            tokenizer=str(HELPER.MODEL),
            token_count_is_model_specific_guarantee=True,
            token_count_scope="Complete native system/user/generation prompt",
        )
        (folder / (iid + ".packet.txt")).write_bytes(packet.encode())
        (folder / (iid + ".prompt.txt")).write_bytes(prompt.encode())
        row.update(
            status="ready",
            messages=ENCODING.messages(packet),
            prompt_token_ids=ids,
            prompt_tokens=len(ids),
            packet_sha256=sha(folder / (iid + ".packet.txt")),
            prompt_sha256=sha(folder / (iid + ".prompt.txt")),
            retrieval=retrieval,
        )
    except Exception as error:
        reason = str(error)
        row.update(
            status="context_infeasible"
            if reason in ["Issue alone exceeds packet budget", "No source span fits budget"]
            else "technical_failure",
            error_type=type(error).__name__,
            error=reason,
        )
    row["elapsed_seconds"] = time.monotonic() - start
    save(path, row)
    return {k: row.get(k) for k in ["instance_id", "policy", "status", "prompt_tokens", "elapsed_seconds"]}


def smoke():
    p = validate()
    initialize()
    task = p["source_tasks"][0]
    rows = [build_one((policy, task)) for policy in POLICIES]
    parity = False
    if rows[0]["status"] == "ready":
        original_spans = RANKING.spans
        RANKING.spans = lambda path, text: original_spans(path, text) if TARGETS.path_allowed(path) else []
        try:
            packet, reference = RANKING.build_task(ORIGINAL / "public" / task["instance_id"], ENCODING, BUDGET)
        finally:
            RANKING.spans = original_spans
        saved = read(OUT / "production_chunks" / (task["instance_id"] + ".json"))
        assert packet == (OUT / "production_chunks" / (task["instance_id"] + ".packet.txt")).read_bytes().decode()
        for key in [
            "source_spans",
            "selection_events",
            "top_retrieval_candidates",
            "packet_tokens",
            "public_source_files_read",
        ]:
            assert reference[key] == saved["retrieval"][key], key
        assert ENCODING.encode(packet) == saved["prompt_token_ids"]
        parity = True
    save(
        OUT / "smoke.json",
        dict(
            utc=utc(),
            protocol_sha256=sha(OUT / "protocol.json"),
            tasks=rows,
            production_chunks_matches_frozen91_eligible_spans_wrapper=parity,
            technical_pass=parity and all(r["status"] == "ready" for r in rows),
            record_sha256={
                str(OUT / r["policy"] / (r["instance_id"] + ".json")): sha(
                    OUT / r["policy"] / (r["instance_id"] + ".json")
                )
                for r in rows
            },
        ),
    )
    assert parity and all(r["status"] == "ready" for r in rows)
    print(json.dumps(rows), flush=True)


def build():
    p = validate()
    smoke = read(OUT / "smoke.json")
    assert smoke["technical_pass"]
    for path, value in smoke["record_sha256"].items():
        assert sha(path) == value, path
    assert not (OUT / "build_started.json").exists() and not (OUT / "build_manifest.json").exists()
    jobs = [(policy, task) for policy in POLICIES for task in p["source_tasks"][1:]]
    assert all(not (OUT / policy / (task["instance_id"] + ".json")).exists() for policy, task in jobs)
    save(
        OUT / "build_started.json",
        dict(utc=utc(), workers=WORKERS, jobs=62, protocol_sha256=sha(OUT / "protocol.json")),
    )
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
    ) as pool:
        futures = {pool.submit(build_one, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                policy, task = futures[future]
                path = OUT / policy / (task["instance_id"] + ".json")
                if path.exists():
                    result = read(path)
                else:
                    result = dict(
                        instance_id=task["instance_id"],
                        role="source",
                        policy=policy,
                        cap=BUDGET,
                        protocol_sha256=sha(OUT / "protocol.json"),
                        status="technical_failure",
                        error_type=type(error).__name__,
                        error=str(error),
                        replacement=False,
                    )
                    save(path, result)
            print(
                json.dumps(
                    {k: result.get(k) for k in ["instance_id", "policy", "status", "prompt_tokens", "elapsed_seconds"]}
                ),
                flush=True,
            )
    rows = []
    hashes = {}
    for policy in POLICIES:
        for task in p["source_tasks"]:
            iid = task["instance_id"]
            path = OUT / policy / (iid + ".json")
            row = read(path)
            hashes[str(path)] = sha(path)
            rows.append({k: row.get(k) for k in ["instance_id", "policy", "status", "prompt_tokens"]})
            if row["status"] == "ready":
                for suffix, key in [(".packet.txt", "packet_sha256"), (".prompt.txt", "prompt_sha256")]:
                    path = OUT / policy / (iid + suffix)
                    assert sha(path) == row[key]
                    hashes[str(path)] = sha(path)
    for path, value in smoke["record_sha256"].items():
        assert sha(path) == value, path
    save(
        OUT / "build_manifest.json",
        dict(
            utc=utc(),
            elapsed_seconds=time.monotonic() - start,
            assigned_slots=64,
            tasks=rows,
            counts=dict(Counter(r["status"] for r in rows)),
            artifact_sha256=hashes,
            protocol_sha256=sha(OUT / "protocol.json"),
            private_inputs_read=False,
        ),
    )


def score():
    p = validate()
    manifest = read(OUT / "build_manifest.json")
    ids = [t["instance_id"] for t in p["source_tasks"]]
    assert manifest["assigned_slots"] == len(manifest["tasks"]) == 64
    assert {(r["instance_id"], r["policy"]) for r in manifest["tasks"]} == {
        (iid, policy) for iid in ids for policy in POLICIES
    }
    for path, value in manifest["artifact_sha256"].items():
        assert sha(path) == value, path
    completed = read(ORIGINAL / "source_targets/completed_manifest.json")
    paths = [
        OUT / "protocol.json",
        OUT / "build_manifest.json",
        ORIGINAL / "source_targets/protocol.json",
        ORIGINAL / "source_targets/completed_manifest.json",
        CAPACITY / "score_summary.json",
        CAPACITY / "completed_manifest.json",
    ]
    for iid in ids:
        path = ORIGINAL / "source_targets" / iid / "record.json"
        assert sha(path) == completed[str(path)]
        paths.append(path)
    save(
        OUT / "score_protocol.json",
        dict(
            utc=utc(),
            build_manifest_sha256=sha(OUT / "build_manifest.json"),
            source_ids=ids,
            inputs_sha256={str(x): sha(x) for x in paths},
            target_policy="Only alreadyexposed111source target records. Unchanged110visibility and2048EOS-inclusive targettokens. No newpatch/test/evaluation-private values; no context modifications.",
        ),
    )
    converter = load("targets110", ROOT / "scripts/110_swe_patch_sft_targets.py")
    helper = load("context109", ROOT / "scripts/109_swe_patch_sft_context.py")
    tok = helper.tokenizer()
    results = []
    for policy in POLICIES:
        rows = []
        for iid in ids:
            target = read(ORIGINAL / "source_targets" / iid / "record.json")
            context = read(OUT / policy / (iid + ".json"))
            assert target["role"] == "source" and context["role"] == "source" and context["policy"] == policy
            fit = visible = False
            reason = None
            if target["conversion_supported"]:
                serialized = converter.serialize_target(target["edits"], tok)
                assert (
                    serialized["target"] == target["target"]
                    and serialized["target_token_ids"] == target["target_token_ids"]
                )
                fit = serialized["target_tokens_including_eos"] <= 2048
                if context["status"] == "ready":
                    public = ORIGINAL / "public" / iid
                    meta = read(public / "task.json")
                    base = {
                        name: (public / "source" / name).read_bytes() for name in {e["path"] for e in target["edits"]}
                    }
                    assert all(
                        hashlib.sha256(blob).hexdigest() == meta["source_sha256"][name] for name, blob in base.items()
                    )
                    intervals = defaultdict(list)
                    for span in context["retrieval"]["source_spans"]:
                        intervals[span["path"]].append((span["start_line"], span["end_line"]))
                    try:
                        converter.validate_visibility(base, target["edits"], intervals)
                        visible = True
                    except converter.TargetError as error:
                        assert str(error) == "anchor_not_wholly_visible"
                        reason = str(error)
                else:
                    reason = context["status"]
            else:
                reason = "conversion_unsupported"
            rows.append(
                dict(
                    instance_id=iid,
                    policy=policy,
                    conversion_supported=target["conversion_supported"],
                    completion_fits=fit,
                    anchor_visible=visible,
                    prequalified=bool(fit and visible),
                    context_status=context["status"],
                    reason=reason,
                )
            )
        results.append(
            dict(
                policy=policy,
                assigned_sources=32,
                conversion_supported=sum(r["conversion_supported"] for r in rows),
                completion_fits=sum(r["completion_fits"] for r in rows),
                anchor_visible=sum(r["anchor_visible"] for r in rows),
                prequalified=sum(r["prequalified"] for r in rows),
                tasks=rows,
            )
        )
    technical = manifest["counts"].get("technical_failure", 0)
    eligible = [r for r in results if r["prequalified"] >= 16]
    chosen = (
        min(eligible, key=lambda r: (-r["prequalified"], POLICIES.index(r["policy"])))["policy"]
        if eligible and not technical
        else None
    )
    baseline = next(r for r in read(CAPACITY / "score_summary.json")["caps"] if r["cap"] == 24576)
    assert [r["instance_id"] for r in baseline["tasks"]] == ids and baseline["prequalified"] == 8
    save(
        OUT / "score_summary.json",
        dict(
            utc=utc(),
            policies=results,
            cached_baseline={
                k: baseline[k]
                for k in [
                    "cap",
                    "assigned_sources",
                    "conversion_supported",
                    "completion_fits",
                    "anchor_visible",
                    "prequalified",
                ]
            },
            chosen_policy=chosen,
            minimum_required=16,
            prompt_cap=BUDGET,
            technical_failures=technical,
            interpretation="Training-development policy selection only. Prequalified source targets still require quality verification. No independent replication, learning or diversity result; old8/32 unchanged.",
            quality_tests_run=0,
            evaluation_new_contexts=0,
            evaluation_private_values_read=False,
            api_spend_usd=0,
            gpu_used=False,
        ),
    )
    save(
        OUT / "completed_manifest.json",
        {
            str(x): sha(x)
            for x in [
                OUT / "protocol.json",
                OUT / "smoke.json",
                OUT / "build_manifest.json",
                OUT / "score_protocol.json",
                OUT / "score_summary.json",
            ]
        },
    )
    print(
        json.dumps(
            {"policies": [{k: v for k, v in r.items() if k != "tasks"} for r in results], "chosen_policy": chosen}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "smoke", "build", "score"])
    args = parser.parse_args()
    {"freeze": freeze, "smoke": smoke, "build": build, "score": score}[args.stage]()
