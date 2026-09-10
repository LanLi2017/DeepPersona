#!/usr/bin/env python3
"""Frozen source-only context-capacity grid; public build precedes target scoring."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "runs/swe-diversity-selection/swe-sympy-patch-sft"
OUT = ORIGINAL.parent / "swe-sympy-context-capacity"
CAPS = [12288, 24576]
WORKERS = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze():
    assert not OUT.exists() or all(p.name == "learner_static_review.json" for p in OUT.iterdir())
    helper = load("context109", ROOT / "scripts/109_swe_patch_sft_context.py")
    original = helper.validate()
    selection = read(ORIGINAL / "selection.json")
    tasks = [t for t in selection["tasks"] if t["role"] == "source"]
    assert len(tasks) == 32 and len({t["instance_id"] for t in tasks}) == 32
    exports = read(ORIGINAL / "export_manifest.json")
    assert exports["selection_sha256"] == sha(ORIGINAL / "selection.json")
    assert all(next(r for r in exports["tasks"] if r["instance_id"] == t["instance_id"])["ready"] for t in tasks)
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/110_swe_patch_sft_targets.py",
        ORIGINAL / "selection.json",
        ORIGINAL / "export_manifest.json",
        ORIGINAL / "context/protocol.json",
        ORIGINAL / "context/manifest.json",
    ]
    hashes = {str(p): sha(p) for p in paths} | original["source_sha256"]
    for task in tasks:
        iid = task["instance_id"]
        for p in [
            ORIGINAL / "public" / iid / "task.json",
            ORIGINAL / "public" / iid / "problem_statement.md",
            ORIGINAL / "context" / (iid + ".json"),
        ]:
            hashes[str(p)] = sha(p)
    OUT.mkdir(exist_ok=True)
    save(
        OUT / "protocol.json",
        dict(
            utc=utc(),
            source_tasks=tasks,
            baseline_cap=6144,
            prompt_caps=CAPS,
            completion_cap_including_eos=2048,
            minimum_prequalified_sources=16,
            workers=WORKERS,
            system=helper.SYSTEM,
            model=str(helper.MODEL),
            model_revision=helper.MODEL.name,
            native_chat=original["native_chat"],
            soft_file_native_tokens=6500,
            retrieval="Import frozen109 helpers and unchanged91 build_task/ranking/chunks/order; only its supplied native prompt cap changes. Original109/91 files and protocol remain unchanged.",
            slots="Every32 fixed source tasks at each declared cap. Smoke firstsource at each cap; then allremaining62. No refill, target-selected files or per-task cap changes.",
            stage_boundary="Build and hash all64 new public contexts before separate source-only scoring. Builder never opens source_targets records or any private patch/test/evaluation values.",
            scoring="Unchanged110 validate_visibility plus canonical serialize_target<=2048 includingEOS on already exposed111 source records; unsupported edits retained. Cached6144 recomputed as baseline without new baseline contexts.",
            decision="Smallest declared cap including6144 with>=16 prequalified sources, if any. Training-development capacity selection after observed5/32; not independent replication, quality admission, learning, or diversity evidence.",
            failures="Each assigned context slot gets a ready/infeasible/technical disposition, no replacement. Any technical failure blocks capacity recommendation.",
            source_sha256=hashes,
            versions={n: importlib.metadata.version(n) for n in ["transformers", "tokenizers", "tiktoken"]},
            api_spend_usd=0,
            gpu_used=False,
        ),
    )
    for cap in CAPS:
        (OUT / str(cap)).mkdir()


def validate():
    protocol = read(OUT / "protocol.json")
    for path, value in protocol["source_sha256"].items():
        assert sha(path) == value, path
    return protocol


def initialize():
    global HELPER, ENCODING, RANKING
    HELPER = load("context109", ROOT / "scripts/109_swe_patch_sft_context.py")
    ENCODING = HELPER.NativeEncoding(HELPER.tokenizer())
    RANKING = HELPER.load_ranking()
    assert RANKING.SOFT_FILE_TOKENS == 6500


def build_one(job):
    cap, task = job
    iid = task["instance_id"]
    folder = OUT / str(cap)
    record = folder / (iid + ".json")
    assert not record.exists()
    start = time.monotonic()
    row = dict(instance_id=iid, role="source", cap=cap, protocol_sha256=sha(OUT / "protocol.json"))
    try:
        public = ORIGINAL / "public" / iid
        meta = read(public / "task.json")
        assert meta["base_commit"] == task["base_commit"] and meta["role"] == "source"
        assert sha(public / "problem_statement.md") == meta["problem_statement_sha256"]
        assert {str(p.relative_to(public / "source")) for p in (public / "source").rglob("*") if p.is_file()} == set(
            meta["source_sha256"]
        )
        for path, value in meta["source_sha256"].items():
            p = public / "source" / path
            assert p.resolve().is_relative_to((public / "source").resolve()) and sha(p) == value
        packet, retrieval = RANKING.build_task(public, ENCODING, cap)
        ids = ENCODING.encode(packet)
        prompt = ENCODING.render(packet)
        direct = ENCODING.tok.apply_chat_template(
            ENCODING.messages(packet), tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(direct, list):
            direct = direct["input_ids"]
        assert ids == direct and len(ids) == retrieval["packet_tokens"] <= cap
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert (public / "problem_statement.md").read_bytes().decode("utf-8") in packet
        for name in {span["path"] for span in retrieval["source_spans"]}:
            text = (public / "source" / name).read_bytes().decode("utf-8")
            parts = text.split("\n")
            assert text.splitlines(keepends=True) == [p + "\n" for p in parts[:-1]] + (
                [parts[-1]] if parts[-1] else []
            ), "visible_source_line_convention_mismatch"
        retrieval.update(
            tokenizer=str(HELPER.MODEL),
            token_count_is_model_specific_guarantee=True,
            token_count_scope="Complete native system/user/generation prompt",
        )
        (folder / (iid + ".packet.txt")).write_bytes(packet.encode("utf-8"))
        (folder / (iid + ".prompt.txt")).write_bytes(prompt.encode("utf-8"))
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
    save(record, row)
    return {k: row.get(k) for k in ["instance_id", "cap", "status", "prompt_tokens", "elapsed_seconds"]}


def smoke():
    p = validate()
    initialize()
    rows = [build_one((cap, p["source_tasks"][0])) for cap in CAPS]
    save(
        OUT / "smoke.json",
        dict(
            utc=utc(),
            protocol_sha256=sha(OUT / "protocol.json"),
            tasks=rows,
            technical_pass=all(r["status"] == "ready" for r in rows),
            record_sha256={
                str(OUT / str(r["cap"]) / (r["instance_id"] + ".json")): sha(
                    OUT / str(r["cap"]) / (r["instance_id"] + ".json")
                )
                for r in rows
            },
        ),
    )
    assert all(r["status"] == "ready" for r in rows)
    print(json.dumps(rows), flush=True)


def build():
    p = validate()
    smoke = read(OUT / "smoke.json")
    assert smoke["technical_pass"]
    for path, value in smoke["record_sha256"].items():
        assert sha(path) == value, path
    assert not (OUT / "build_started.json").exists() and not (OUT / "build_manifest.json").exists()
    jobs = [(cap, t) for cap in CAPS for t in p["source_tasks"][1:]]
    assert all(not (OUT / str(cap) / (t["instance_id"] + ".json")).exists() for cap, t in jobs)
    save(
        OUT / "build_started.json",
        dict(utc=utc(), jobs=len(jobs), workers=WORKERS, protocol_sha256=sha(OUT / "protocol.json")),
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
                cap, task = futures[future]
                path = OUT / str(cap) / (task["instance_id"] + ".json")
                if path.exists():
                    result = read(path)
                else:
                    result = dict(
                        instance_id=task["instance_id"],
                        role="source",
                        cap=cap,
                        protocol_sha256=sha(OUT / "protocol.json"),
                        status="technical_failure",
                        error_type=type(error).__name__,
                        error=str(error),
                        replacement=False,
                    )
                    save(path, result)
            print(
                json.dumps(
                    {k: result.get(k) for k in ["instance_id", "cap", "status", "prompt_tokens", "elapsed_seconds"]}
                ),
                flush=True,
            )
    rows = []
    hashes = {}
    for cap in CAPS:
        for task in p["source_tasks"]:
            path = OUT / str(cap) / (task["instance_id"] + ".json")
            row = read(path)
            hashes[str(path)] = sha(path)
            rows.append({k: row.get(k) for k in ["instance_id", "cap", "status", "prompt_tokens"]})
            if row["status"] == "ready":
                for suffix, key in [(".packet.txt", "packet_sha256"), (".prompt.txt", "prompt_sha256")]:
                    path = OUT / str(cap) / (task["instance_id"] + suffix)
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
    assert len(manifest["tasks"]) == manifest["assigned_slots"] == 64
    assert {(r["instance_id"], r["cap"]) for r in manifest["tasks"]} == {
        (t["instance_id"], cap) for t in p["source_tasks"] for cap in CAPS
    }
    for path, value in manifest["artifact_sha256"].items():
        assert sha(path) == value, path
    completed = read(ORIGINAL / "source_targets/completed_manifest.json")
    paths = [
        OUT / "protocol.json",
        OUT / "build_manifest.json",
        ORIGINAL / "source_targets/protocol.json",
        ORIGINAL / "source_targets/completed_manifest.json",
    ]
    for task in p["source_tasks"]:
        path = ORIGINAL / "source_targets" / task["instance_id"] / "record.json"
        assert sha(path) == completed[str(path)]
        paths.append(path)
    save(
        OUT / "score_protocol.json",
        dict(
            utc=utc(),
            build_manifest_sha256=sha(OUT / "build_manifest.json"),
            source_ids=[t["instance_id"] for t in p["source_tasks"]],
            inputs_sha256={str(x): sha(x) for x in paths},
            target_policy="Already exposed source records only; no patches/tests or evaluation-private values opened. Same110 functions and2048EOS-inclusive cap. No context modification.",
        ),
    )
    converter = load("targets110", ROOT / "scripts/110_swe_patch_sft_targets.py")
    helper = load("context109", ROOT / "scripts/109_swe_patch_sft_context.py")
    tok = helper.tokenizer()
    results = []
    for cap in [6144] + CAPS:
        rows = []
        for task in p["source_tasks"]:
            iid = task["instance_id"]
            target = read(ORIGINAL / "source_targets" / iid / "record.json")
            assert target["instance_id"] == iid and target["role"] == "source"
            context = read((ORIGINAL / "context" if cap == 6144 else OUT / str(cap)) / (iid + ".json"))
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
                    names = {e["path"] for e in target["edits"]}
                    public = ORIGINAL / "public" / iid
                    meta = read(public / "task.json")
                    base = {name: (public / "source" / name).read_bytes() for name in names}
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
            qualified = bool(fit and visible)
            if cap == 6144:
                assert (
                    qualified == target["prequalified"]
                    and fit == bool(target["completion_fits"])
                    and visible == bool(target["anchor_visible"])
                )
            rows.append(
                dict(
                    instance_id=iid,
                    cap=cap,
                    conversion_supported=target["conversion_supported"],
                    completion_fits=fit,
                    anchor_visible=visible,
                    prequalified=qualified,
                    context_status=context["status"],
                    reason=reason,
                )
            )
        results.append(
            dict(
                cap=cap,
                assigned_sources=32,
                conversion_supported=sum(r["conversion_supported"] for r in rows),
                completion_fits=sum(r["completion_fits"] for r in rows),
                anchor_visible=sum(r["anchor_visible"] for r in rows),
                prequalified=sum(r["prequalified"] for r in rows),
                tasks=rows,
            )
        )
    technical = manifest["counts"].get("technical_failure", 0)
    eligible = [r["cap"] for r in results if r["prequalified"] >= 16]
    save(
        OUT / "score_summary.json",
        dict(
            utc=utc(),
            caps=results,
            minimum_required=16,
            technical_failures=technical,
            smallest_cap_meeting_source_gate=min(eligible) if eligible and not technical else None,
            interpretation="Training-development capacity selection, not independent replication or downstream learning. Prequalified targets still need quality verification. Original5/32 control unchanged.",
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
    print(json.dumps([{k: v for k, v in r.items() if k != "tasks"} for r in results]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "smoke", "build", "score"])
    args = parser.parse_args()
    {"freeze": freeze, "smoke": smoke, "build": build, "score": score}[args.stage]()
