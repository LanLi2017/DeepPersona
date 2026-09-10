#!/usr/bin/env python3
"""Freeze correctness/token-matched groups before measuring update utility."""

import argparse
import ast
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess

POOL = Path("runs/swe-diversity-selection/code-learning-pilot")
OUT = Path("runs/swe-diversity-selection/code-update-transfer")
SEED = 20260909


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.open()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def features(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = blocks[-1] if blocks else text
    tokens = set(re.findall(r"[A-Za-z_]\w*|\d+|[^\w\s]", code))
    try:
        tree = ast.parse(code)
        nodes = {type(n).__name__ for n in ast.walk(tree)}
        canonical = ast.dump(tree, include_attributes=False)
    except SyntaxError:
        nodes, canonical = None, None
    return tokens, nodes, digest(canonical) if canonical else None


def jaccard(a, b):
    return 1 - len(a & b) / len(a | b) if a or b else 0.0


def prepare(smoke):
    dest = OUT / "group-smoke" if smoke else OUT
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / "groups.json").exists(), "Groups already frozen"
    inputs, raw, grades = [read(POOL / name) for name in ["inputs.jsonl", "raw.jsonl", "grades.jsonl"]]
    source_protocol = json.loads((POOL / "protocol.json").read_text())
    train = {r["task_id"] for r in inputs if r["split"] == "train"}
    calibration = source_protocol["calibration_task_ids"]
    measurement = source_protocol["measurement_task_ids"]
    assert not (train & set(calibration) or train & set(measurement) or set(calibration) & set(measurement))
    assert len(calibration) == len(measurement) == 21
    raw_by_id = {f"{r['task_id']}:{r['sample']}": r for r in raw}
    grade_by_id = {f"{r['task_id']}:{r['sample']}": r for r in grades}
    assert len(raw_by_id) == len(raw) == len(grade_by_id) == len(grades)
    assert raw_by_id.keys() == grade_by_id.keys()
    assert {r["task_id"] for r in raw} == train
    assert all(sum(r["task_id"] == t for r in raw) == source_protocol["k"] for t in train)
    candidates, support = {}, []
    for task in sorted(train, key=lambda t: digest(f"{SEED}:update-task:{t}")):
        ids = sorted((k for k, r in raw_by_id.items() if r["task_id"] == task), key=lambda k: int(k.split(":")[1]))
        passed = [k for k in ids if grade_by_id[k]["correct"]]
        failed = [k for k in ids if not grade_by_id[k]["correct"]]
        options = [tuple(sorted(a + b)) for a in combinations(passed, 2) for b in combinations(failed, 2)]
        lengths = [sum(len(raw_by_id[k]["generated_token_ids"]) for k in group) for group in options]
        edges = [
            (a, b)
            for a, b in combinations(range(len(options)), 2)
            if abs(lengths[a] - lengths[b]) / min(lengths[a], lengths[b]) <= 0.05
        ]
        edges.sort(key=lambda pair: digest(f"{SEED}:update-pair:{task}:{options[pair[0]]}:{options[pair[1]]}"))
        chosen, used = [], set()
        for a, b in edges:
            if a not in used and b not in used:
                chosen.append((a, b))
                used.update((a, b))
            if len(chosen) == 4:
                break
        status = (
            "supported"
            if chosen
            else "fewer_than_two_successes_or_failures"
            if not options
            else "no_5pct_token_matched_pair"
        )
        support.append(
            dict(
                task_id=task,
                correct=len(passed),
                failed=len(failed),
                candidate_groups=len(options),
                token_matched_edges=len(edges),
                greedy_matched_pairs=len(chosen),
                status=status,
            )
        )
        if chosen:
            candidates[task] = (options, lengths, chosen)
    selected_tasks = list(candidates)[: 1 if smoke else 12]
    groups, pairs = [], []
    for task in selected_tasks:
        options, lengths, chosen = candidates[task]
        for number, (a, b) in enumerate(chosen[:1] if smoke else chosen):
            pair_id = f"T{task}_P{number:02d}"
            group_ids = []
            for side, index in zip("ab", (a, b)):
                record_ids = list(options[index])
                group_id = pair_id + side
                group_ids.append(group_id)
                fs = [features(raw_by_id[k]["generation"]) for k in record_ids]
                token_distance = sum(jaccard(fs[a][0], fs[b][0]) for a, b in combinations(range(4), 2)) / 6
                ast_distance = (
                    None
                    if any(v[1] is None for v in fs)
                    else sum(jaccard(fs[a][1], fs[b][1]) for a, b in combinations(range(4), 2)) / 6
                )
                groups.append(
                    dict(
                        group_id=group_id,
                        task_id=task,
                        record_ids=record_ids,
                        rewards=[int(grade_by_id[k]["correct"]) for k in record_ids],
                        completion_tokens=[len(raw_by_id[k]["generated_token_ids"]) for k in record_ids],
                        total_completion_tokens=lengths[index],
                        matched_pair_id=pair_id,
                        positive_completion_tokens=sum(
                            len(raw_by_id[k]["generated_token_ids"]) for k in record_ids if grade_by_id[k]["correct"]
                        ),
                        negative_completion_tokens=sum(
                            len(raw_by_id[k]["generated_token_ids"])
                            for k in record_ids
                            if not grade_by_id[k]["correct"]
                        ),
                        cheap_token_diversity=token_distance,
                        cheap_ast_node_type_diversity=ast_distance,
                        distinct_generated_texts=len({raw_by_id[k]["generation"] for k in record_ids}),
                        distinct_valid_ast=len({v[2] for v in fs if v[2]}),
                    )
                )
            pairs.append(
                dict(
                    pair_id=pair_id,
                    task_id=task,
                    group_ids=group_ids,
                    relative_token_gap=abs(lengths[a] - lengths[b]) / min(lengths[a], lengths[b]),
                )
            )
    assert groups, "No supported groups; stop without relaxing matching"
    selected_ids = sorted({k for group in groups for k in group["record_ids"]})
    records = [
        dict(
            record_id=k,
            task_id=raw_by_id[k]["task_id"],
            sample=raw_by_id[k]["sample"],
            reward=int(grade_by_id[k]["correct"]),
            completion_tokens=len(raw_by_id[k]["generated_token_ids"]),
            raw_source=str(POOL / "raw.jsonl"),
        )
        for k in selected_ids
    ]
    repeats = [
        dict(control_id=f"repeat_{task}", source_group_id=next(g["group_id"] for g in groups if g["task_id"] == task))
        for task in selected_tasks
    ]
    payload = dict(
        schema_version=1,
        groups=groups,
        matched_pairs=pairs,
        controls=dict(no_update=True, identical_group_repeats=repeats, reset_student_and_optimizer=True),
    )
    assert len({g["group_id"] for g in groups}) == len(groups)
    assert all(len(g["record_ids"]) == len(set(g["record_ids"])) == 4 and sum(g["rewards"]) == 2 for g in groups)
    assert all(p["relative_token_gap"] <= 0.05 for p in pairs)
    paths = [
        POOL / name
        for name in ["protocol.json", "inputs.jsonl", "raw.jsonl", "grades.jsonl", "evaluator_targets.jsonl"]
    ]
    protocol = dict(
        schema_version=1,
        seed=SEED,
        scope="Local code one-step mechanism; not deployed GRPO or SWE evidence",
        model=source_protocol["model"],
        model_revision=source_protocol["model_revision"],
        max_tasks=1 if smoke else 12,
        max_groups_per_task=2 if smoke else 8,
        k=4,
        correct_per_group=2,
        selection="Tasks by SHA256(seed:update-task:task), group pairs by SHA256(seed:update-pair:task:tupleA:tupleB); greedy without group reuse, no metric or gradient selection",
        token_matching="abs(sumA-sumB)/min(sumA,sumB)<=.05; no relaxation; overlap of individual records across groups allowed",
        selected_task_ids=selected_tasks,
        calibration_task_ids=calibration,
        measurement_task_ids=measurement,
        gradient="g_i = gradient of mean completion-token NLL over fixed adapter parameters; H=mean_i((reward_i-.5)*g_i)",
        update="Reset theta/optimizer each group; theta_new=theta-.1*H primary, theta-H secondary; norm-matched update=.02*H/||H|| diagnostic; zero H is no update",
        adapter=dict(
            last_decoder_layer=35, target_modules=["q_proj", "v_proj"], rank=8, alpha=16, dropout=0, parameters=106496
        ),
        metrics=[
            "mean pairwise token-set Jaccard",
            "mean pairwise AST-node-type-set Jaccard; null if any invalid AST",
            "raw adapter gradient mean pairwise cosine distance",
            "H norm",
            "weighted-gradient cancellation",
            "mean calibration NLL gradient dot H",
            "actual measurement NLL reduction after reset finite update",
        ],
        calibration="21 task mean reference-completion NLL; no measurement fitting",
        measurement="21 disjoint task mean reference-completion NLL; fixed parameters before outcomes",
        reporting="Task-cluster inference; groups overlap and are not independent; code reference NLL is not execution success",
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        api_spend_usd=0,
        gpu_used=False,
    )
    summary = dict(
        train_tasks=len(train),
        support_status=dict(Counter(r["status"] for r in support)),
        supported_tasks=len(candidates),
        selected_tasks=len(selected_tasks),
        groups=len(groups),
        pairs=len(pairs),
        unique_records=len(selected_ids),
        identical_group_repeats=len(repeats),
        fewer_than_requested_tasks=len(selected_tasks) < (1 if smoke else 12),
        groups_per_task=dict(Counter(g["task_id"] for g in groups)),
        maximum_token_gap=max(p["relative_token_gap"] for p in pairs),
        exclusions="All unsupported and eligible-but-cap-excluded task IDs retained in support.json; no outcome-based replacement or caliper relaxation",
        api_spend_usd=0,
    )
    for row in support:
        row["selected"] = row["task_id"] in selected_tasks
    for name, value in [
        ("group_protocol.json", protocol),
        ("groups.json", payload),
        ("support.json", support),
        ("group_summary.json", summary),
    ]:
        save(dest / name, value)
    (dest / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    save(
        dest / "group_manifest.json",
        dict(
            files_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    dest / n
                    for n in [
                        "group_protocol.json",
                        "groups.json",
                        "support.json",
                        "group_summary.json",
                        "records.jsonl",
                    ]
                ]
            }
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.smoke)
