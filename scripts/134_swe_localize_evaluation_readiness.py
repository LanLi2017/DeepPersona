#!/usr/bin/env python3
"""Fixed20 public initial barrier, then separately isolated host runtime readiness."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
POP = ROOT / "runs/swe-diversity-selection/swe-sympy-patch-sft"
SUPPORT = POP.parent / "swe-localize-repair-support"
OUT = SUPPORT / "evaluation_readiness"
WORKERS = 8


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = load("evaluation112", ROOT / "scripts/112_swe_patch_sft_runtime.py")
p = load("evaluation131", ROOT / "scripts/131_swe_environment_provenance.py")
w.OUT = OUT
w.m.OUT = OUT
w.__file__ = str(Path(__file__).resolve())
sha, read, save, utc = w.sha, w.read, w.save, w.utc
OLD_QUERY = 'import importlib.metadata as m,json,platform,sympy;print(json.dumps(dict(python=platform.python_version(),packages={d.metadata["Name"]:d.version for d in m.distributions()},sympy_file=sympy.__file__)))'


def verify(mapping):
    for path, digest in mapping.items():
        assert sha(ROOT / path) == digest, path


def initialize():
    global kernel, tok
    kernel = load("evaluation125", ROOT / "scripts/125_swe_localize_repair_support.py")
    tok = kernel.tokenizer()


def freeze_initials():
    assert not OUT.exists(), "No retry or replacement namespace"
    population = read(POP / "selection.json")
    tasks = [t for t in population["tasks"] if t["role"] == "evaluation"]
    assert len(tasks) == len({t["instance_id"] for t in tasks}) == 20
    milestone = ROOT / "runs/swe-diversity-selection/paper-program/milestone_thirteen.json"
    verify(read(milestone)["evidence_sha256"])
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/125_swe_localize_repair_support.py",
        ROOT / "scripts/110_swe_patch_sft_targets.py",
        POP / "selection.json",
        POP / "export_manifest.json",
        milestone,
    ]
    initialize()
    files += [
        kernel.MODEL / n
        for n in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
        if (kernel.MODEL / n).exists()
    ]
    for task in tasks:
        files += [POP / "public" / task["instance_id"] / n for n in ["task.json", "problem_statement.md"]]
    OUT.mkdir()
    (OUT / "initials").mkdir()
    save(
        OUT / "initial_protocol.json",
        dict(
            utc=utc(),
            evaluation_tasks=tasks,
            workers=WORKERS,
            source_sha256={str(f): sha(f) for f in files},
            kernel_sha256=sha(ROOT / "scripts/125_swe_localize_repair_support.py"),
            interface="Exact125 initial messages, prompt and token IDs; amend returned metadata role only from source to evaluation.",
            private_barrier="All20 public initial records and manifest before any evaluation private assets or dataset values.",
            initial_limit=12288,
            evaluation_private_values_read=False,
            api_spend_usd=0,
            gpu_used=False,
        ),
    )
    print("Initial protocol frozen", flush=True)


def initial_one(task):
    iid = task["instance_id"]
    public = POP / "public" / iid
    metadata = read(public / "task.json")
    assert metadata["role"] == "evaluation" and metadata["base_commit"] == task["base_commit"]
    assert sha(public / "problem_statement.md") == metadata["problem_statement_sha256"]
    record = kernel.initial(public, tok)
    assert record["instance_id"] == iid and record["initial_public_only"]
    record["role"] = "evaluation"
    path = OUT / "initials" / (iid + ".json")
    save(path, record)
    return dict(
        instance_id=iid,
        status="ready",
        prompt_tokens=record["prompt_tokens"],
        initial_index_fits=record["initial_index_fits"],
        path=str(path),
        sha256=sha(path),
    )


def initials():
    protocol = read(OUT / "initial_protocol.json")
    verify(protocol["source_sha256"])
    save(OUT / "initials_started.json", dict(utc=utc(), protocol_sha256=sha(OUT / "initial_protocol.json")))
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
    ) as pool:
        rows = list(pool.map(initial_one, protocol["evaluation_tasks"]))
    assert len(rows) == 20
    save(
        OUT / "initial_manifest.json",
        dict(
            utc=utc(),
            protocol_sha256=sha(OUT / "initial_protocol.json"),
            tasks=rows,
            source_sha256={r["path"]: r["sha256"] for r in rows},
            all20_public_initials_frozen=True,
            evaluation_private_values_read=False,
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    print(
        json.dumps(
            dict(
                public_initials=20,
                initial_index_fits=sum(r["initial_index_fits"] for r in rows),
                min_tokens=min(r["prompt_tokens"] for r in rows),
                max_tokens=max(r["prompt_tokens"] for r in rows),
            )
        ),
        flush=True,
    )


def barrier():
    protocol = read(OUT / "initial_protocol.json")
    verify(protocol["source_sha256"])
    manifest = read(OUT / "initial_manifest.json")
    assert manifest["all20_public_initials_frozen"] and manifest["evaluation_private_values_read"] is False
    assert manifest["protocol_sha256"] == sha(OUT / "initial_protocol.json")
    assert [t["instance_id"] for t in manifest["tasks"]] == [t["instance_id"] for t in protocol["evaluation_tasks"]]
    verify(manifest["source_sha256"])
    return protocol, manifest


def freeze_runtime():
    initial, manifest = barrier()
    population = read(POP / "selection.json")
    tasks = initial["evaluation_tasks"]
    assert not (OUT / "selection.json").exists()
    (OUT / "private").mkdir()
    (OUT / "private/base").symlink_to(POP / "private/base", target_is_directory=True)
    (OUT / "public").symlink_to(POP / "public", target_is_directory=True)
    shutil.copyfile(POP / "export_manifest.json", OUT / "export_manifest.json")
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/112_swe_patch_sft_runtime.py",
        ROOT / "scripts/90_swe_fresh_development.py",
        ROOT / "scripts/85_swe_execution_bridge.py",
        ROOT / "scripts/131_swe_environment_provenance.py",
        w.m.b.PARSER,
        w.m.b.DATA,
        POP / "selection.json",
        POP / "selection_protocol.json",
        POP / "export_manifest.json",
        OUT / "initial_protocol.json",
        OUT / "initial_manifest.json",
        SUPPORT / "source_runtime_metadata/independent_quality_audit.json",
    ]
    save(
        OUT / "selection.json",
        dict(
            utc=utc(),
            tasks=tasks,
            all_slots=tasks,
            source_prequalification=[],
            source_prequalified_ids=[],
            evaluation_ids=[t["instance_id"] for t in tasks],
            smoke_ids=[tasks[0]["instance_id"]],
            remaining_ids=[t["instance_id"] for t in tasks[1:]],
            source_sha256={str(f): sha(f) for f in files},
            export_manifest_sha256=sha(OUT / "export_manifest.json"),
            export_manifest_original=str(POP / "export_manifest.json"),
            **{k: population[k] for k in ["dataset_revision", "harness_revision", "task_repo_revision"]},
            scope="Fixed20 evaluation host runtime readiness; operator-private gold/test only after all20 public initial barrier; no model generation or private content in model inputs.",
            initial_index_fits=sum(t["initial_index_fits"] for t in manifest["tasks"]),
            limits=dict(
                per_test_wall_seconds=180,
                per_test_cpu_seconds=120,
                per_test_memory_bytes=4 * 1024**3,
                aggregate_outer_test_seconds=7200,
                outer_termination_grace_seconds=5,
            ),
            no_retry=True,
            replacement=False,
            api_spend_usd=0,
            gpu_used=False,
        ),
    )
    print("Runtime cohort frozen:20", flush=True)


def selection():
    barrier()
    chosen = w.selection()
    assert len(chosen["tasks"]) == len(chosen["all_slots"]) == 20 and chosen["source_prequalified_ids"] == []
    assert all(t["role"] == "evaluation" for t in chosen["tasks"])
    assert [t["instance_id"] for t in chosen["tasks"]] == chosen["evaluation_ids"]
    return chosen


def assets():
    selection()
    w.assets()
    # 112's summary wrongly labels embedded test-patch bodies absent. Correct before review/prepare freeze.
    path = OUT / "private/assets_review_summary.json"
    report = read(path)
    report["gold_and_test_body_included"] = True
    report["body_scope_correction"] = (
        "Operator-private evaluation test patches embedded in eval_script. Never execute full eval.sh/Dockerfile or pass this summary to model."
    )
    report["reviewed_source_sha256"].update(
        {str(f): sha(f) for f in [Path(__file__).resolve(), ROOT / "scripts/131_swe_environment_provenance.py"]}
    )
    w.m.save(path, report)


def prepare(review):
    chosen = selection()
    selected_ids = set(chosen["evaluation_ids"])
    assert p.PINS == w.m.b.PACKAGES
    original = w.m.b.call
    count = 0

    def corrected(cmd, *args, **kwargs):
        nonlocal count
        if len(cmd) == 3 and cmd[1] == "-c" and cmd[2] == OLD_QUERY:
            repo = Path(kwargs["cwd"]).resolve()
            assert repo.name == "repo" and repo.is_relative_to(OUT / "private/tasks")
            iid, arm = repo.parent.parent.name, repo.parent.name
            assert arm in {"baseline", "gold"} and iid in selected_ids
            assert Path(cmd[0]) == repo.parent / "venv/bin/python"
            assert kwargs["env"]["VIRTUAL_ENV"] == str(repo.parent / "venv")
            assert kwargs["env"]["PATH"].split(":")[0] == str(repo.parent / "venv/bin")
            count += 1
            cmd = list(cmd)
            cmd[2] = p.PROBE
            raw = original(cmd, *args, **kwargs)
            evidence = json.loads(raw)
            save(OUT / "private/environment_provenance" / iid / (arm + ".json"), evidence)
            assert evidence["provenance_all_checks_passed"] is True, (
                "Strict direct environment provenance failed; raw evidence retained"
            )
            return raw
        return original(cmd, *args, **kwargs)

    w.m.b.call = corrected
    try:
        w.prepare(review)
    finally:
        w.m.b.call = original
        save(
            OUT / "metadata_query_dispatch_receipt.json",
            dict(
                utc=utc(),
                substituted_queries=count,
                exact_old_query_only=True,
                other_commands_unchanged=True,
                package_pins_unchanged=True,
            ),
        )


def aggregate():
    chosen = selection()
    runtime = {}
    preparation = {}
    if (OUT / "protocol.json").exists():
        preparation = {t["instance_id"]: t for t in read(OUT / "protocol.json")["tasks"]}
    elif (OUT / "preparation_progress.json").exists():
        preparation = {t["instance_id"]: t for t in read(OUT / "preparation_progress.json")}
    groups = []
    for group in ["smoke", "remaining"]:
        folder = OUT / "groups" / group
        if (folder / "dispatch.json").exists():
            assert (folder / "terminal.json").exists(), "Do not aggregate an unresolved/running group"
            groups.append(dict(group=group, **read(folder / "terminal.json")))
        result_path = folder / "results.json"
        if not result_path.exists():
            result_path = folder / "results_progress.json"
        if result_path.exists():
            data = read(result_path)
            rows = data["tasks"] if isinstance(data, dict) else data
            for row in rows:
                assert row["instance_id"] not in runtime
                runtime[row["instance_id"]] = row
    screened = {t["instance_id"]: t for t in chosen["source_prequalification"]}
    rows = []
    for task in chosen["all_slots"]:
        iid = task["instance_id"]
        result = runtime.get(iid)
        ready = preparation.get(iid)
        excluded = task["role"] == "source" and not screened[iid]["prequalified"]
        if excluded:
            status = "source_target_not_prequalified"
        elif ready is None:
            status = "preparation_not_completed"
        elif not ready["ready"]:
            status = "preparation_failure"
        elif result is None:
            status = "tests_not_completed"
        else:
            # Reject vacuous or incomplete quality evidence even if an upstream flag were wrong.
            quality = bool(result["host_bridge_passed"] and ready["tests"]["FAIL_TO_PASS"])
            arms = {a["arm"]: a for a in result.get("arms", [])}
            quality = quality and len(result.get("arms", [])) == 2 and set(arms) == {"baseline", "gold"}
            if quality:
                for arm in ["baseline", "gold"]:
                    evidence = arms[arm]
                    expected = {"PASSED"} if arm == "gold" else {"FAILED", "ERROR"}
                    quality = quality and evidence["exit_code"] == (0 if arm == "gold" else 1)
                    for suite, tests in ready["tests"].items():
                        statuses = evidence["statuses"][suite]
                        allowed = expected if suite == "FAIL_TO_PASS" else {"PASSED"}
                        quality = (
                            quality and set(statuses) == set(tests) and all(x in allowed for x in statuses.values())
                        )
            status = "baseline_gold_passed" if quality else "baseline_gold_failed"
        rows.append(
            dict(
                instance_id=iid,
                role=task["role"],
                slot=task["slot"],
                family=task["family"],
                disposition=status,
                prequalified=screened[iid]["prequalified"] if task["role"] == "source" else None,
                preparation_failure=ready.get("preparation_failure") if ready else None,
                runtime_result=result,
                replacement=False,
            )
        )
    passed_source = sum(r["role"] == "source" and r["disposition"] == "baseline_gold_passed" for r in rows)
    passed_eval = sum(r["role"] == "evaluation" and r["disposition"] == "baseline_gold_passed" for r in rows)
    assert len(rows) == 20
    admitted_ids = [
        r["instance_id"] for r in rows if r["role"] == "source" and r["disposition"] == "baseline_gold_passed"
    ]
    runnable_ids = [
        r["instance_id"] for r in rows if r["role"] == "evaluation" and r["disposition"] == "baseline_gold_passed"
    ]
    public_status_rows = []
    for row in rows:
        path = OUT / "private/quality_records" / (row["instance_id"] + ".json")
        save(path, row)
        public_status_rows.append(
            dict(
                **{k: v for k, v in row.items() if k != "runtime_result"},
                quality_record_path=str(path.relative_to(ROOT)),
                quality_record_sha256=sha(path),
            )
        )
    report = dict(
        utc=utc(),
        admitted_source_ids=admitted_ids,
        evaluation_runnable_ids=runnable_ids,
        official_container_evaluation=False,
        assigned_slots=20,
        assigned_source_slots=0,
        assigned_evaluation_slots=20,
        source_prequalified=len(chosen["source_prequalified_ids"]),
        admitted_source_quality=passed_source,
        minimum_source_quality=0,
        evaluation_runtime_valid=passed_eval,
        all20_endpoints_ready=passed_eval == 20,
        readiness_gate_passed=passed_eval == 20,
        initial_index_fits=chosen["initial_index_fits"],
        groups=groups,
        outer_test_seconds=sum(g["seconds"] for g in groups),
        tasks=public_status_rows,
        interpretation="Runtime/target readiness only, no learned repair effect. Partial slots retained; no replacement.",
        api_spend_usd=0,
        gpu_used=False,
    )
    save(OUT / "summary.json", report)
    paths = [
        OUT / "selection.json",
        OUT / "summary.json",
        OUT / "initial_protocol.json",
        OUT / "initial_manifest.json",
        OUT / "metadata_query_dispatch_receipt.json",
    ]
    paths += list((OUT / "initials").glob("*.json"))
    paths += list((OUT / "private/environment_provenance").glob("*/*.json"))
    paths += list(OUT.glob("*review*.json"))
    paths += list((OUT / "private").glob("*review*.json"))
    paths += list((OUT / "groups").glob("*/runner.log"))
    paths += [
        p
        for p in [OUT / "protocol.json", OUT / "assets_manifest.json", OUT / "preparation_terminal.json"]
        if p.exists()
    ]
    paths += list((OUT / "groups").glob("*/*.json"))
    paths += list((OUT / "private/quality_records").glob("*.json"))
    paths += list((OUT / "private/tasks").glob("*/*/result.json"))
    paths += list((OUT / "private/tasks").glob("*/*/environment.json"))
    paths += list((OUT / "private/tasks").glob("*/*/test.log"))
    save(OUT / "completed_manifest.json", {str(p.relative_to(ROOT)): sha(p) for p in set(paths)})
    print(json.dumps({k: v for k, v in report.items() if k not in {"tasks", "groups"}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "freeze-initials",
            "initials",
            "freeze-runtime",
            "assets",
            "prepare",
            "smoke",
            "run",
            "aggregate",
            "_run-group",
        ],
    )
    parser.add_argument("--review")
    parser.add_argument("--group", choices=["smoke", "remaining"])
    args = parser.parse_args()
    if args.stage == "freeze-initials":
        freeze_initials()
        return
    if args.stage == "initials":
        initials()
        return
    if args.stage == "freeze-runtime":
        freeze_runtime()
        return
    selection()
    if args.stage == "assets":
        assets()
    elif args.stage == "prepare":
        prepare(args.review)
    elif args.stage == "aggregate":
        aggregate()
    elif args.stage == "_run-group":
        w.child_group(args.group)
    else:
        w.run_group("smoke" if args.stage == "smoke" else "remaining", args.review)


if __name__ == "__main__":
    main()
