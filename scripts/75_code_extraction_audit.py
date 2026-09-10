#!/usr/bin/env python3
"""Posthoc first-versus-last code-fence grading sensitivity; originals immutable."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path("runs/swe-diversity-selection")
OUT = ROOT / "code-extraction-audit"
HELPER = Path("scripts/52_code_learning_pool.py")
PATTERN = r"```(?:python)?\s*\n(.*?)```"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return [json.loads(line) for line in p.open()]


def save(p, value):
    p.write_text(json.dumps(value, indent=2) + "\n")


def inputs():
    all_rows = []
    targets = {}
    paths = [HELPER, Path(__file__)]
    for pool in ["code-learning-pilot", "code-learning-expansion"]:
        folder = ROOT / pool
        paths.extend(folder / n for n in ["raw.jsonl", "grades.jsonl", "evaluator_targets.jsonl"])
        grades = {(r["task_id"], r["sample"]): r for r in rows(folder / "grades.jsonl")}
        for target in rows(folder / "evaluator_targets.jsonl"):
            if target["split"] == "train":
                targets[("train", target["task_id"])] = target
        for r in rows(folder / "raw.jsonl"):
            key = (r["task_id"], r["sample"])
            all_rows.append(
                dict(
                    record_id=f"train:{key[0]}:{key[1]}",
                    split="train",
                    source_pool=pool,
                    task_id=r["task_id"],
                    sample=r["sample"],
                    generation=r["generation"],
                    frozen_grade=grades[key],
                )
            )
    base = ROOT / "code-functional-development/grid-evaluation/base"
    paths.extend(base / n for n in ["eval_raw.jsonl", "grades.jsonl"])
    grades = {r["task_id"]: r for r in rows(base / "grades.jsonl")}
    for target in rows(ROOT / "code-learning-pilot/evaluator_targets.jsonl"):
        if target["split"] == "calibration":
            targets[("calibration", target["task_id"])] = target
    for r in rows(base / "eval_raw.jsonl"):
        all_rows.append(
            dict(
                record_id=f"calibration:{r['task_id']}",
                split="calibration",
                source_pool="base_greedy",
                task_id=r["task_id"],
                sample=None,
                generation=r["generation"],
                frozen_grade=grades[r["task_id"]],
            )
        )
    assert Counter(r["split"] for r in all_rows) == {"train": 960, "calibration": 21}
    assert len({r["record_id"] for r in all_rows}) == 981
    for r in all_rows:
        blocks = re.findall(PATTERN, r["generation"], re.S)
        first, last = (blocks[0], blocks[-1]) if blocks else (r["generation"], r["generation"])
        r.update(block_count=len(blocks), first_code=first, last_code=last, extraction_differs=first != last)
        assert (r["split"], r["task_id"]) in targets
    return all_rows, targets, paths


def prepare():
    assert not OUT.exists()
    all_rows, _, paths = inputs()
    OUT.mkdir()
    save(
        OUT / "protocol.json",
        dict(
            status="Posthoc extraction sensitivity, not replacement of frozen primary reward labels",
            original_rule="Same regex; last capturedPython-or-unlabelled block, otherwise fullgeneration",
            alternative_rule="Same regex; first capturedPython-or-unlabelled block, otherwise fullgeneration; no verifier-based choice",
            regex=PATTERN,
            scope=dict(train_rollouts=960, train_tasks=120, base_calibration_tasks=21),
            execution="Only differing extractedcode: regradeFIRSTandLAST via unchanged52 subprocessgrader; exactsingle-block wrapping preserves code byte-for-byte",
            unchanged_code="Reuse frozen label for identical first/last code; not a full grading-repeatability study",
            grader="Original52 environment, resource limits and timeout; no filesystem/network security isolation",
            analysis="Separate train/calibration transitions, old-label-vsLASTrepeat discrepancies,2<=correct_count<=6 eligibility atN8",
            prohibit=[
                "Mutating original labels",
                "Scoringmeasurement/finaltasks",
                "Newgeneration",
                "Outcome-basedblockselection",
            ],
            input_sha256={str(p): sha(p) for p in paths},
            api_spend_usd=0,
        ),
    )
    (OUT / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    print(
        json.dumps(
            dict(frozen_rows=len(all_rows), differing_extractions=sum(r["extraction_differs"] for r in all_rows))
        )
    )


def run():
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, expected in protocol["input_sha256"].items():
        assert sha(Path(name)) == expected
    assert not (OUT / "regrades.jsonl").exists(), "Prior sensitivity outputs immutable"
    all_rows, targets, _ = inputs()
    assert rows(OUT / "inputs.jsonl") == all_rows
    spec = importlib.util.spec_from_file_location("frozen_grader52", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    def grade(r):
        result = dict(record_id=r["record_id"], split=r["split"], task_id=r["task_id"])
        for mode in ["first", "last"]:
            code = r[mode + "_code"]
            wrapped = "```python\n" + code + "```"
            assert re.findall(PATTERN, wrapped, re.S) == [code]
            result[mode] = helper.grade(wrapped, targets[(r["split"], r["task_id"])])
        return result

    changed = [r for r in all_rows if r["extraction_differs"]]
    with (OUT / "regrades.jsonl").open("w") as output, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(grade, r) for r in changed]
        for number, future in enumerate(as_completed(futures), 1):
            output.write(json.dumps(future.result()) + "\n")
            output.flush()
            if number % 20 == 0:
                print(json.dumps(dict(regraded_records=number, total=len(changed))), flush=True)
    summarize()


def summarize():
    all_rows = rows(OUT / "inputs.jsonl")
    regrades = {r["record_id"]: r for r in rows(OUT / "regrades.jsonl")}
    assert set(regrades) == {r["record_id"] for r in all_rows if r["extraction_differs"]}
    outcomes = []
    for r in all_rows:
        repeat = regrades.get(r["record_id"])
        first = repeat["first"] if repeat else r["frozen_grade"]
        last = repeat["last"] if repeat else r["frozen_grade"]
        outcomes.append(
            dict(
                record_id=r["record_id"],
                split=r["split"],
                source_pool=r["source_pool"],
                task_id=r["task_id"],
                block_count=r["block_count"],
                extraction_differs=r["extraction_differs"],
                frozen_correct=bool(r["frozen_grade"]["correct"]),
                first_correct=bool(first["correct"]),
                last_repeat_correct=bool(last["correct"]),
                first_status=first["status"],
                last_status=last["status"],
            )
        )
    results = {}
    for split in ["train", "calibration"]:
        selected = [r for r in outcomes if r["split"] == split]
        results[split] = dict(
            rows=len(selected),
            multiblock_rows=sum(r["block_count"] > 1 for r in selected),
            block_count_histogram=dict(Counter(r["block_count"] for r in selected)),
            changed_extractions=sum(r["extraction_differs"] for r in selected),
            original_correct=sum(r["frozen_correct"] for r in selected),
            first_correct=sum(r["first_correct"] for r in selected),
            first_vs_frozen_transitions=dict(
                Counter(f"{int(r['frozen_correct'])}->{int(r['first_correct'])}" for r in selected)
            ),
            last_repeat_disagreements=[
                r["record_id"] for r in selected if r["last_repeat_correct"] != r["frozen_correct"]
            ],
            first_vs_last_repeat_transitions=dict(
                Counter(f"{int(r['last_repeat_correct'])}->{int(r['first_correct'])}" for r in selected)
            ),
        )
    tasks = defaultdict(list)
    for r in outcomes:
        if r["split"] == "train":
            tasks[r["task_id"]].append(r)
    eligibility = []
    for task, rs in sorted(tasks.items()):
        assert len(rs) == 8
        old = sum(r["frozen_correct"] for r in rs)
        new = sum(r["first_correct"] for r in rs)
        eligibility.append(
            dict(
                task_id=task,
                source_pool=rs[0]["source_pool"],
                original_correct=old,
                first_correct=new,
                originally_eligible=2 <= old <= 6,
                first_eligible=2 <= new <= 6,
            )
        )
    results["eligibility"] = dict(
        original=sum(r["originally_eligible"] for r in eligibility),
        first=sum(r["first_eligible"] for r in eligibility),
        lost=[r["task_id"] for r in eligibility if r["originally_eligible"] and not r["first_eligible"]],
        gained=[r["task_id"] for r in eligibility if not r["originally_eligible"] and r["first_eligible"]],
    )
    for pool in ["code-learning-pilot", "code-learning-expansion"]:
        rs = [r for r in eligibility if r["source_pool"] == pool]
        results["eligibility"][pool] = dict(
            tasks=len(rs),
            original=sum(r["originally_eligible"] for r in rs),
            first=sum(r["first_eligible"] for r in rs),
        )
    save(OUT / "summary.json", results)
    save(OUT / "task_eligibility.json", eligibility)
    (OUT / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in outcomes))
    save(OUT / "completed_manifest.json", dict(sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
