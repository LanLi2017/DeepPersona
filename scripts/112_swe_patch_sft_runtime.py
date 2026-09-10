#!/usr/bin/env python3
"""Gated reuse of90 baseline/gold checks; no assets read at import time."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import io
import json
import os
import re
import shlex
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/swe-diversity-selection"
POP = BASE / "swe-sympy-patch-sft"
OUT = POP / "runtime"
ASSETS = ["task.yaml", "tests.json", "gold.patch", "test.patch", "eval.sh", "Dockerfile"]
TOTAL_TEST_SECONDS = 7200
SPEC = importlib.util.spec_from_file_location("patch_sft_runtime90", ROOT / "scripts/90_swe_fresh_development.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
m.OUT = OUT
m.b.DATA = BASE / "swe-sympy-learning-metadata/full_test.parquet"
sha, read = m.sha, m.read


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_map(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT / name) == digest, name


def freeze():
    """Metadata gate before this wrapper can acquire any evaluator assets."""
    assert not OUT.exists()
    summary = read(POP / "source_targets/summary.json")
    population = read(POP / "selection.json")
    assert summary["assigned_source_slots"] == 32 and summary["minimum_required"] == 16
    assert summary["prequalification_gate_passed"] and summary["prequalified"] >= 16
    assert summary["technical_failures"] == 0
    assert summary["quality_tests_run"] == 0 and summary["evaluation_private_columns_materialized"] is False
    verify_map(read(POP / "source_targets/completed_manifest.json"))
    verify_map(read(POP / "source_targets/protocol.json")["inputs_sha256"])
    all_tasks = population["tasks"]
    sources = [t for t in all_tasks if t["role"] == "source"]
    evaluation = [t for t in all_tasks if t["role"] == "evaluation"]
    assert len(all_tasks) == len({t["instance_id"] for t in all_tasks}) == 52
    assert len(sources) == 32 and len(evaluation) == 20
    screened = {t["instance_id"]: t for t in summary["tasks"]}
    assert set(screened) == {t["instance_id"] for t in sources}
    active_sources = [t for t in sources if screened[t["instance_id"]]["prequalified"]]
    assert len(active_sources) == summary["prequalified"]
    active = active_sources + evaluation
    export = read(POP / "export_manifest.json")
    assert export["selection_sha256"] == sha(POP / "selection.json")
    assert export["ready"] == 52
    context = read(POP / "context/manifest.json")
    assert context["assigned_slots"] == context["completed_slots"] == 52
    assert context["selection_sha256"] == sha(POP / "selection.json")
    OUT.mkdir()
    (OUT / "private").mkdir()
    (OUT / "private/base").symlink_to(POP / "private/base", target_is_directory=True)
    (OUT / "public").symlink_to(POP / "public", target_is_directory=True)
    shutil.copyfile(POP / "export_manifest.json", OUT / "export_manifest.json")
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/90_swe_fresh_development.py",
        ROOT / "scripts/85_swe_execution_bridge.py",
        ROOT / "scripts/107_swe_sympy_compatibility.py",
        ROOT / "scripts/108_swe_patch_sft_population.py",
        ROOT / "scripts/111_swe_patch_sft_admission.py",
        m.b.PARSER,
        m.b.DATA,
        POP / "selection.json",
        POP / "selection_protocol.json",
        POP / "export_manifest.json",
        POP / "context/protocol.json",
        POP / "context/manifest.json",
        POP / "source_targets/protocol.json",
        POP / "source_targets/summary.json",
        POP / "source_targets/completed_manifest.json",
    ]
    files += list((POP / "public").glob("*/task.json"))
    selection = dict(
        utc=utc(),
        tasks=active,
        all_slots=all_tasks,
        source_prequalification=summary["tasks"],
        source_prequalified_ids=[t["instance_id"] for t in active_sources],
        evaluation_ids=[t["instance_id"] for t in evaluation],
        smoke_ids=[active_sources[0]["instance_id"]],
        remaining_ids=[t["instance_id"] for t in active[1:]],
        dataset_revision=population["dataset_revision"],
        harness_revision=population["harness_revision"],
        task_repo_revision=population["task_repo_revision"],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        export_manifest_original=str((POP / "export_manifest.json").relative_to(ROOT)),
        export_manifest_sha256=sha(OUT / "export_manifest.json"),
        scope="Private runtime readiness/quality only. All32source and20evaluation slots retained; no replacement. No model endpoint here.",
        admission="Only prequalified source IDs plus ALL20 assigned evaluation IDs acquire private columns/assets. Source failures never refill slots.",
        private_access="Unchanged90prepare materializes all dataset columns for selected IDs only, after reviewed cached official assets. No private content enters public/context.",
        commands="Same90 exact Python3.9.20, nine package pins, official command and parser; no environment retuning.",
        limits=dict(
            per_test_wall_seconds=180,
            per_test_cpu_seconds=120,
            per_test_memory_bytes=4 * 1024**3,
            aggregate_outer_test_seconds=TOTAL_TEST_SECONDS,
            outer_termination_grace_seconds=5,
        ),
        stages="freeze -> six assets/task and review summary -> root reviewed prepare -> firstsource smoke -> reviewed remaining group -> aggregate",
        smoke="First prequalified source only, never rerun in remaining group. Root reviews runtime/operation evidence; baseline/gold quality failures remain fixed-slot dispositions.",
        no_retry=True,
        api_spend_usd=0,
        gpu_used=False,
    )
    save(OUT / "selection.json", selection)
    print(
        json.dumps(
            dict(
                selection_sha256=sha(OUT / "selection.json"),
                active_sources=len(active_sources),
                evaluation_slots=20,
                retained_slots=52,
            )
        ),
        flush=True,
    )


def selection():
    value = m.frozen_selection()
    assert sha(OUT / "export_manifest.json") == value["export_manifest_sha256"]
    assert (OUT / "private/base").resolve() == (POP / "private/base").resolve()
    assert (OUT / "public").resolve() == (POP / "public").resolve()
    return value


def asset_contract(folder):
    """Fixed command grammar and explicit SymPy Python test paths, no shell syntax."""
    tests = read(folder / "tests.json")
    assert isinstance(tests["FAIL_TO_PASS"], list) and tests["FAIL_TO_PASS"], "Missing repair tests"
    assert isinstance(tests["PASS_TO_PASS"], list)
    assert all(isinstance(t, str) and t for key in ["FAIL_TO_PASS", "PASS_TO_PASS"] for t in tests[key])
    commands = [line for line in (folder / "eval.sh").read_text().splitlines() if line.startswith("PYTHONWARNINGS=")]
    assert len(commands) == 1
    parts = shlex.split(commands[0])
    assert parts[:4] == ["PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning", "bin/test", "-C", "--verbose"]
    paths = parts[4:]
    assert paths and len(paths) == len(set(paths))
    assert all(
        re.fullmatch(r"sympy/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py", name) and ".." not in name.split("/")
        for name in paths
    ), "Command outside explicit Python path whitelist"
    # Tokens must not hide shell expansion inside otherwise parseable quoting.
    assert not any(c in commands[0] for c in ["$", "`", ";", "&", "|", "<", ">", "\r"])
    return dict(
        test_command=commands[0],
        explicit_test_paths=paths,
        fail_to_pass_count=len(tests["FAIL_TO_PASS"]),
        pass_to_pass_count=len(tests["PASS_TO_PASS"]),
    )


def assets():
    chosen = selection()
    assert not (OUT / "assets_started.json").exists()
    save(OUT / "assets_started.json", dict(utc=utc(), selection_sha256=sha(OUT / "selection.json")))
    rows = []
    for task in chosen["tasks"]:
        iid = task["instance_id"]
        folder = OUT / "private/review_assets" / iid
        folder.mkdir(parents=True)
        row = dict(instance_id=iid, role=task["role"], ready=False, files={})
        try:
            for name in ASSETS:
                url = f"https://raw.githubusercontent.com/SWE-bench/swe-bench-tasks/{chosen['task_repo_revision']}/tasks/{iid}/{name}"
                with urllib.request.urlopen(url, timeout=30) as response:
                    body = response.read(10_000_001)
                assert len(body) <= 10_000_000
                (folder / name).write_bytes(body)
                row["files"][name] = dict(url=url, sha256=sha(folder / name))
            if task["role"] == "source":
                assert sha(folder / "gold.patch") == sha(POP / "source_targets" / iid / "gold.patch"), (
                    "Official gold differs from exact prequalified source target"
                )
                row["matches_prequalified_gold_sha256"] = True
            row["command_contract"] = asset_contract(folder)
            row["ready"] = True
        except Exception as error:
            row["failure"] = dict(type=type(error).__name__, message=str(error), replacement=False)
        rows.append(row)
        save(OUT / "private/asset_records" / f"{iid}.json", row)
        print(json.dumps(dict(instance_id=iid, assets_ready=row["ready"])), flush=True)
    save(OUT / "assets_manifest.json", dict(selection_sha256=sha(OUT / "selection.json"), tasks=rows))
    # Summary contains executable setup/command text for the operator, never model inputs.
    review_rows = []
    reviewed = {
        str(p.relative_to(ROOT)): sha(p)
        for p in [
            Path(__file__).resolve(),
            ROOT / "scripts/90_swe_fresh_development.py",
            ROOT / "scripts/85_swe_execution_bridge.py",
            ROOT / "scripts/107_swe_sympy_compatibility.py",
            m.b.PARSER,
        ]
    }
    for row in rows:
        iid = row["instance_id"]
        folder = OUT / "private/review_assets" / iid
        setup = POP / "public" / iid / "source/setup.py"
        setup_sources = []
        for relative in ["setup.py", "sympy/release.py", "bin/test"]:
            source = POP / "public" / iid / "source" / relative
            if source.exists():
                reviewed[str(source.relative_to(ROOT))] = sha(source)
                setup_sources.append(
                    dict(path=str(source.relative_to(ROOT)), sha256=sha(source), text=source.read_text())
                )
            else:
                setup_sources.append(dict(path=str(source.relative_to(ROOT)), missing=True))
        review_rows.append(
            dict(
                instance_id=iid,
                role=row["role"],
                assets_ready=row["ready"],
                command_contract=row.get("command_contract"),
                asset_failure=row.get("failure"),
                setup_and_runner_sources=setup_sources,
                asset_sha256={k: v["sha256"] for k, v in row["files"].items()},
                setup_source_path=str(setup.relative_to(ROOT)),
                setup_sha256=sha(setup) if setup.exists() else None,
                setup_source=setup.read_text() if setup.exists() else None,
                eval_script=(folder / "eval.sh").read_text() if "eval.sh" in row["files"] else None,
                dockerfile=(folder / "Dockerfile").read_text() if "Dockerfile" in row["files"] else None,
            )
        )
    save(
        OUT / "private/assets_review_summary.json",
        dict(
            selection_sha256=sha(OUT / "selection.json"),
            assets_manifest_sha256=sha(OUT / "assets_manifest.json"),
            tasks=review_rows,
            reviewed_source_sha256=reviewed,
            private_operator_only=True,
            gold_and_test_body_included=False,
            source_roots="Original108base/public; private assets never copied there",
        ),
    )


def prepare(review):
    selection()
    auth = read(Path(review))
    manifest = read(OUT / "assets_manifest.json")
    review_summary = read(OUT / "private/assets_review_summary.json")
    assert auth["allow_preparation"] and auth["reason"]
    assert auth["selection_sha256"] == sha(OUT / "selection.json")
    assert auth["assets_manifest_sha256"] == sha(OUT / "assets_manifest.json")
    assert auth["assets_review_summary_sha256"] == sha(OUT / "private/assets_review_summary.json")
    assert auth["reviewed_source_sha256"] == review_summary["reviewed_source_sha256"]
    verify_map(auth["reviewed_source_sha256"])
    assert manifest["selection_sha256"] == sha(OUT / "selection.json")
    cache = {}
    for row in manifest["tasks"]:
        if row["ready"]:
            assert asset_contract(OUT / "private/review_assets" / row["instance_id"]) == row["command_contract"]
        for name, info in row["files"].items():
            path = OUT / "private/review_assets" / row["instance_id"] / name
            assert sha(path) == info["sha256"]
            if row["ready"]:
                cache[info["url"]] = path
    save(OUT / "root_preparation_review.json", auth)
    save(OUT / "preparation_started.json", dict(utc=utc(), retry=False))
    original = m.urllib.request.urlopen

    def cached(url, timeout=30):
        assert url in cache, "Missing frozen official asset; no replacement fetch"
        return io.BytesIO(cache[url].read_bytes())

    m.urllib.request.urlopen = cached
    try:
        m.prepare()
    except Exception as error:
        save(
            OUT / "preparation_terminal.json",
            dict(
                completed=False,
                utc=utc(),
                failure=dict(type=type(error).__name__, message=str(error)),
                replacement=False,
            ),
        )
        raise
    finally:
        m.urllib.request.urlopen = original
    protocol = read(OUT / "protocol.json")
    for path in [
        Path(__file__).resolve(),
        OUT / "assets_manifest.json",
        OUT / "private/assets_review_summary.json",
        OUT / "root_preparation_review.json",
    ]:
        protocol["source_sha256"][str(path)] = sha(path)
    protocol["total_test_stage_wall_seconds"] = TOTAL_TEST_SECONDS
    protocol["scope"] = selection()["scope"]
    # 90 writes this once; amendment completes the wrapper's freeze before any test.
    m.save(OUT / "protocol.json", protocol)
    save(
        OUT / "preparation_terminal.json",
        dict(
            completed=True,
            utc=utc(),
            protocol_sha256=sha(OUT / "protocol.json"),
            ready=sum(t["ready"] for t in protocol["tasks"]),
        ),
    )
    print(
        json.dumps(dict(protocol_sha256=sha(OUT / "protocol.json"), ready=sum(t["ready"] for t in protocol["tasks"]))),
        flush=True,
    )


def run_group(group, review):
    """Own outer process group; enforce combined7200s over smoke and remaining."""
    chosen = selection()
    assert group in {"smoke", "remaining"}
    with (OUT / ".test_stage.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auth = read(Path(review))
        protocol = read(OUT / "protocol.json")
        assert auth["allow_baseline_gold_tests"] and auth["reason"] and auth["group"] == group
        assert auth["protocol_sha256"] == sha(OUT / "protocol.json")
        verify_map(protocol["source_sha256"])
        if group == "remaining":
            terminal = read(OUT / "groups/smoke/terminal.json")
            assert terminal["exit_code"] == 0 and not terminal["timed_out"]
            assert auth["smoke_results_sha256"] == sha(OUT / "groups/smoke/results.json")
        terminals = list((OUT / "groups").glob("*/terminal.json"))
        dispatches = list((OUT / "groups").glob("*/dispatch.json"))
        assert len(dispatches) == len(terminals), "Unresolved prior dispatch; no automatic retry"
        used = sum(read(p)["seconds"] for p in terminals)
        remaining = TOTAL_TEST_SECONDS - used
        assert remaining > 0, "Aggregate outer test budget exhausted"
        folder = OUT / "groups" / group
        assert not folder.exists(), "Group already attempted; never retry"
        folder.mkdir(parents=True)
        (folder / "private").symlink_to(OUT / "private", target_is_directory=True)
        ids = chosen["smoke_ids"] if group == "smoke" else chosen["remaining_ids"]
        child_protocol = dict(
            protocol,
            tasks=[t for t in protocol["tasks"] if t["instance_id"] in ids],
            total_test_stage_wall_seconds=remaining,
        )
        assert [t["instance_id"] for t in child_protocol["tasks"]] == ids
        save(folder / "protocol.json", child_protocol)
        save(folder / "root_test_review.json", auth)
        save(
            folder / "dispatch.json",
            dict(
                utc=utc(),
                group=group,
                ids=ids,
                protocol_sha256=sha(folder / "protocol.json"),
                remaining_outer_seconds=remaining,
                retry=False,
            ),
        )
        started = time.monotonic()
        timed_out = False
        with (folder / "runner.log").open("w") as log:
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "_run-group", "--group", group],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=ROOT,
            )
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
        save(
            folder / "terminal.json",
            dict(
                utc=utc(),
                exit_code=code,
                timed_out=timed_out,
                seconds=time.monotonic() - started,
                termination_grace_seconds=5,
                results_complete=(folder / "results.json").exists(),
                no_retry=True,
            ),
        )
        print(json.dumps(read(folder / "terminal.json")), flush=True)


def child_group(group):
    assert group in {"smoke", "remaining"}
    selection()
    folder = OUT / "groups" / group
    dispatch = read(folder / "dispatch.json")
    assert sha(folder / "protocol.json") == dispatch["protocol_sha256"]
    assert not (folder / "terminal.json").exists()
    # Exclusive claim prevents direct duplicate invocation of the dispatched child.
    save(folder / "child_started.json", dict(utc=utc(), pid=os.getpid()))
    m.OUT = folder
    m.run()


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
            quality = quality and set(arms) == {"baseline", "gold"}
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
    assert len(rows) == 52
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
        assigned_slots=52,
        assigned_source_slots=32,
        assigned_evaluation_slots=20,
        source_prequalified=len(chosen["source_prequalified_ids"]),
        admitted_source_quality=passed_source,
        minimum_source_quality=16,
        evaluation_runtime_valid=passed_eval,
        all20_endpoints_ready=passed_eval == 20,
        readiness_gate_passed=passed_source >= 16 and passed_eval == 20,
        groups=groups,
        outer_test_seconds=sum(g["seconds"] for g in groups),
        tasks=public_status_rows,
        interpretation="Runtime/target readiness only, no learned repair effect. Partial slots retained; no replacement.",
        api_spend_usd=0,
        gpu_used=False,
    )
    save(OUT / "summary.json", report)
    paths = [OUT / "selection.json", OUT / "summary.json"]
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
    save(OUT / "completed_manifest.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})
    print(json.dumps({k: v for k, v in report.items() if k not in {"tasks", "groups"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "assets", "prepare", "smoke", "run", "aggregate", "_run-group"])
    parser.add_argument("--review")
    parser.add_argument("--group", choices=["smoke", "remaining"])
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "assets":
        assets()
    elif args.stage == "prepare":
        prepare(args.review)
    elif args.stage in ["smoke", "run"]:
        run_group("smoke" if args.stage == "smoke" else "remaining", args.review)
    elif args.stage == "_run-group":
        child_group(args.group)
    else:
        aggregate()
