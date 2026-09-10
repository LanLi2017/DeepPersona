#!/usr/bin/env python3
"""Freeze/build/score one public symbol-context policy; no private evaluation inputs."""
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

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'runs/swe-diversity-selection/swe-sympy-patch-sft'
CAPACITY=ORIGINAL.parent/'swe-sympy-context-capacity'
PRIOR=ORIGINAL.parent/'swe-sympy-retrieval-policies'
OUT=ORIGINAL.parent/'swe-sympy-symbol-context'
KERNEL_PATH=ROOT/'scripts/122_swe_symbol_dependency_context.py'
POLICIES=['symbol_dependency']
BUDGET=24576
WORKERS=8


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
    global HELPER, ENCODING, RANKING, TARGETS, KERNEL
    HELPER=load('context109',ROOT/'scripts/109_swe_patch_sft_context.py')
    ENCODING=HELPER.NativeEncoding(HELPER.tokenizer())
    RANKING=HELPER.load_ranking()
    TARGETS=load('targets110',ROOT/'scripts/110_swe_patch_sft_targets.py')
    KERNEL=load('symbol122',KERNEL_PATH)
    KERNEL.RANKING=RANKING
    KERNEL.TARGETS=TARGETS
    assert RANKING.SOFT_FILE_TOKENS==6500 and RANKING.MAX_SELECTED_FILES==16 and RANKING.MAX_CANDIDATES==600
    assert KERNEL.BUDGET==BUDGET and KERNEL.MAX_ATTEMPTS==600 and KERNEL.MAX_WINDOWS==12


def build_packet(public,encoding,budget,policy):
    assert policy==POLICIES[0]
    return KERNEL.build_packet(public,encoding,budget,policy)


def freeze():
    allowed={'symbol_static_review.json','kernel_synthetic_checks.json','independent_graph_checks.json','independent_graph_check.py'}
    assert OUT.exists() and {p.name for p in OUT.iterdir()}<=allowed
    review=read(OUT/'symbol_static_review.json')
    assert review['all_checks_passed']
    for p in [Path(__file__).resolve(),KERNEL_PATH]:
        assert review['source_sha256'][str(p.relative_to(ROOT))]==sha(p)
    for name in ['kernel_synthetic_checks.json','independent_graph_checks.json']:
        check=read(OUT/name)
        assert check['all_checks_passed'] and check['kernel_sha256']==sha(KERNEL_PATH)
    helper=load('context109',ROOT/'scripts/109_swe_patch_sft_context.py')
    original=helper.validate()
    tasks=[t for t in read(ORIGINAL/'selection.json')['tasks'] if t['role']=='source']
    assert len(tasks)==len({t['instance_id'] for t in tasks})==32
    previous=read(PRIOR/'score_summary.json')
    assert previous['chosen_policy'] is None and previous['technical_failures']==0
    assert [r['prequalified'] for r in previous['policies']]==[9,12]
    for name,value in read(PRIOR/'completed_manifest.json').items():assert sha(name)==value,name
    paths=[Path(__file__).resolve(),KERNEL_PATH,ROOT/'scripts/117_swe_sympy_retrieval_policies.py',
           ROOT/'scripts/110_swe_patch_sft_targets.py',ORIGINAL/'selection.json',ORIGINAL/'export_manifest.json',
           ORIGINAL/'context/protocol.json',PRIOR/'protocol.json',PRIOR/'score_summary.json',PRIOR/'completed_manifest.json']
    paths += [OUT/name for name in sorted(allowed)]
    hashes={str(p):sha(p) for p in paths}|original['source_sha256']
    for task in tasks:
        iid=task['instance_id']
        for name in ['task.json','problem_statement.md']:
            path=ORIGINAL/'public'/iid/name;hashes[str(path)]=sha(path)
    save(OUT/'protocol.json',dict(utc=utc(),source_tasks=tasks,policies=POLICIES,prompt_cap=BUDGET,
        completion_cap_including_eos=2048,minimum_prequalified_sources=16,workers=WORKERS,
        system=helper.SYSTEM,model=str(helper.MODEL),native_chat=original['native_chat'],
        policy_definition='122 issue-seeded conservative production AST definitions, one incoming/outgoing resolved syntactic call hop, seed-first wholeblocks then ranked boundedwindows, fixed production BM25 fallback. Exact code/hash and synthetic reports are authoritative.',
        limits=dict(prompt=24576,completion_including_eos=2048,files=16,total_proposal_attempts=600,windows_per_node=12,window_lines=140,overlap_lines=12,first_pass_soft_file_native_tokens=6500,second_pass_lifts_soft_limit=True,file_header_lines=60,node_window_header_lines=26,method_owner_header_lines=26),
        packing_pass_order='Each pass: seed blocks/windows, one-hop blocks/windows, then BM25 fallback. First-pass fallback may fill prompt before second-pass soft-limit lifting; this is intentional.',
        selection_rule='Single policy qualifies iff >=16 prequalified sources and zero technical failures; no alternate policy, task replacement or threshold relaxation.',
        stage_boundary='All32public contexts complete and immutable before separate source-only already-exposed target scoring; first fixed source smoke then31remaining.',
        interpretation='Training-interface development on exposed sources. Not independent replication, runtime quality, learning or diversity evidence. This is the final planned retrieval intervention in this interface branch.',
        source_sha256=hashes,versions={n:importlib.metadata.version(n) for n in ['transformers','tokenizers','tiktoken']},
        evaluation_new_contexts=0,evaluation_private_values_read=False,quality_tests_run=0,api_spend_usd=0,gpu_used=False))
    (OUT/POLICIES[0]).mkdir()


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
    p=validate();initialize()
    task=p['source_tasks'][0]
    row=build_one((POLICIES[0],task))
    record=OUT/POLICIES[0]/(task['instance_id']+'.json')
    ready=row['status']=='ready'
    detail=read(record).get('retrieval',{}).get('symbol_graph',{})
    save(OUT/'smoke.json',dict(utc=utc(),protocol_sha256=sha(OUT/'protocol.json'),tasks=[row],technical_pass=ready,
        record_sha256={str(record):sha(record)},seeds=len(detail.get('seeds',[])),one_hop_neighbors=len(detail.get('one_hop_neighbors',[])),
        qualification='Public-context/native-token technical smoke only. No source target or model/quality outcome inspected.'))
    print(json.dumps(row),flush=True)
    assert ready


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
        dict(utc=utc(), workers=WORKERS, jobs=31, protocol_sha256=sha(OUT / "protocol.json")),
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
            assigned_slots=32,
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
    assert manifest["assigned_slots"] == len(manifest["tasks"]) == 32
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
        PRIOR / "score_summary.json",
        PRIOR / "completed_manifest.json",
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
            cached_previous_policy_counts={r["policy"]: r["prequalified"] for r in read(PRIOR / "score_summary.json")["policies"]},
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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','smoke','build','score'])
    args=parser.parse_args();{'freeze':freeze,'smoke':smoke,'build':build,'score':score}[args.stage]()
