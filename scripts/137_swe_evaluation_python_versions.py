#!/usr/bin/env python3
"""Explicit official per-task Python/dependency readiness, same20 frozen evaluations."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "runs/swe-diversity-selection/swe-localize-repair-support"
PREVIOUS = SUPPORT / "evaluation_readiness"
OUT = SUPPORT / "evaluation_readiness_versions"
REVIEW = SUPPORT / "evaluation_python_versions_independent_review.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load("evaluation134versions", ROOT / "scripts/134_swe_localize_evaluation_readiness.py")
p = load("versioned138", ROOT / "scripts/138_swe_versioned_environment_probe.py")
w = prior.w
prior.OUT = w.OUT = w.m.OUT = OUT
w.__file__ = str(Path(__file__).resolve())
sha, read, save, utc = w.sha, w.read, w.save, w.utc


def official_config(docker):
    """Exact assignment tokens, never substring matches (e.g.3.15 versus3.16)."""
    versions = re.findall(r"(?<![\w-])python=(3\.9\.\d+)=[^\s\\]+", docker)
    assert len(versions) == 1 and versions[0] in {"3.9.20", "3.9.21"}, versions
    packages = {}
    for name in p.NAMES:
        values = re.findall(r"(?<![\w-])" + re.escape(name) + r"(?:==|=)([0-9][A-Za-z0-9.+-]*)(?==|[\s\\]|$)", docker)
        assert len(values) == 1, (name, values)
        packages[name] = values[0]
    p.pins(packages)
    return dict(python=versions[0], packages=packages)


def freeze():
    assert not (OUT / "selection.json").exists() and not (OUT / "protocol.json").exists()
    assert not (PREVIOUS / "preparation_started.json").exists()
    assert not (PREVIOUS / "private/tasks").exists() and not (PREVIOUS / "groups").exists()
    terminal = read(PREVIOUS / "preparation_review_terminal.json")
    assert terminal["environments_created"] == terminal["baseline_gold_tests_run"] == terminal["model_generations"] == 0
    initial_protocol = read(PREVIOUS / "initial_protocol.json")
    initial_manifest = read(PREVIOUS / "initial_manifest.json")
    w.verify_map(initial_protocol["source_sha256"])
    w.verify_map(initial_manifest["source_sha256"])
    assert initial_manifest["protocol_sha256"] == sha(PREVIOUS / "initial_protocol.json")
    assert (
        initial_manifest["all20_public_initials_frozen"] and initial_manifest["evaluation_private_values_read"] is False
    )
    initial_ids = [t["instance_id"] for t in initial_protocol["evaluation_tasks"]]
    assert len(initial_ids) == len(set(initial_ids)) == 20
    assert [t["instance_id"] for t in initial_manifest["tasks"]] == initial_ids
    for row in initial_manifest["tasks"]:
        assert row["status"] == "ready" and sha(Path(row["path"])) == row["sha256"]
        record = read(Path(row["path"]))
        assert record["instance_id"] == row["instance_id"] and record["role"] == "evaluation"
        assert record["initial_public_only"]
    old = read(PREVIOUS / "selection.json")
    assert [t["instance_id"] for t in old["tasks"]] == initial_ids
    assets = read(PREVIOUS / "assets_manifest.json")
    summary = read(PREVIOUS / "private/assets_review_summary.json")
    review = read(REVIEW)
    assert review["all_checks_passed"] and review["evaluation_tasks_audited"] == 20
    for field, name in [
        ("initial_manifest_sha256", "initial_manifest.json"),
        ("selection_sha256", "selection.json"),
        ("assets_manifest_sha256", "assets_manifest.json"),
    ]:
        assert terminal[field] == review[field] == sha(PREVIOUS / name)
    assert terminal["metadata_query_dispatch_receipt_sha256"] == sha(PREVIOUS / "metadata_query_dispatch_receipt.json")
    assert terminal["review_summary_sha256"] == sha(PREVIOUS / "private/assets_review_summary.json")
    w.verify_map(old["source_sha256"])
    w.verify_map(summary["reviewed_source_sha256"])
    ids = [t["instance_id"] for t in old["tasks"]]
    assert len(ids) == len(set(ids)) == 20 and old["source_prequalified_ids"] == []
    assert [t["instance_id"] for t in assets["tasks"]] == ids and all(t["ready"] for t in assets["tasks"])
    config = {}
    for iid in ids:
        path = PREVIOUS / "private/review_assets" / iid / "Dockerfile"
        cfg = official_config(path.read_text())
        assert cfg["python"] == review["required_python_by_task"][iid]
        assert cfg["packages"] == review["required_packages_by_task"][iid]
        cfg["docker_sha256"] = sha(path)
        config[iid] = cfg
    assert [iid for iid in ids if config[iid]["python"] == "3.9.21"] == ["sympy__sympy-19495"]
    acquisition = read(OUT / "python_acquisition_manifest.json")
    assert acquisition["python_version"] == "3.9.21"
    executable = Path(acquisition["python_path"])
    assert executable.resolve().is_relative_to(OUT / "python") and sha(executable) == acquisition["python_sha256"]
    assert sha(Path(acquisition["uv_path"])) == acquisition["uv_sha256"]
    installation = read(OUT / "python_installation_files.json")
    assert Path(installation["python_prefix"]).resolve() == executable.resolve().parents[1]
    w.verify_map(installation["file_sha256"])
    for name, target in installation["symlinks"].items():
        assert str(Path(name).readlink()) == target
    interpreter_paths = {"3.9.20": str(w.m.b.PYTHON), "3.9.21": str(executable)}
    evidence = [
        Path(__file__).resolve(),
        ROOT / "scripts/138_swe_versioned_environment_probe.py",
        ROOT / "scripts/131_swe_environment_provenance.py",
        ROOT / "scripts/134_swe_localize_evaluation_readiness.py",
        REVIEW,
        SUPPORT / "evaluation_official_versions_audit_driver.py",
        PREVIOUS / "preparation_review_terminal.json",
        PREVIOUS / "metadata_query_dispatch_receipt.json",
        PREVIOUS / "selection.json",
        PREVIOUS / "assets_manifest.json",
        PREVIOUS / "private/assets_review_summary.json",
        PREVIOUS / "initial_protocol.json",
        PREVIOUS / "initial_manifest.json",
        OUT / "python_acquisition_manifest.json",
        OUT / "python_installation_files.json",
        Path(acquisition["uv_path"]),
        *map(Path, interpreter_paths.values()),
    ]
    evidence += list(OUT.glob("python_acquisition*"))
    OUT.mkdir(exist_ok=True)
    (OUT / "private").mkdir()
    (OUT / "private/base").symlink_to(prior.POP / "private/base", target_is_directory=True)
    (OUT / "public").symlink_to(prior.POP / "public", target_is_directory=True)
    (OUT / "initials").symlink_to(PREVIOUS / "initials", target_is_directory=True)
    for name in ["initial_protocol.json", "initial_manifest.json"]:
        shutil.copyfile(PREVIOUS / name, OUT / name)
    shutil.copyfile(prior.POP / "export_manifest.json", OUT / "export_manifest.json")
    prior.barrier()
    new = copy.deepcopy(old)
    new["source_sha256"].update({str(f): sha(f) for f in evidence if f.is_file()})
    new.update(
        utc=utc(),
        official_environment_by_task=config,
        interpreter_paths=interpreter_paths,
        installer_path=acquisition["uv_path"],
        prior_preparation=str(PREVIOUS),
        prior_environments=0,
        prior_tests=0,
        correction="Match exact official per-task Python and nine named package versions;19495 requires3.9.21 and flake8-comprehensions3.16.0. No prior runtime outcomes, no replacement tasks, same public initials/official assets/tests.",
        commands="Exact official per-task Python and nine package pins; same90 test commands/parser/resource limits; no task replacement.",
        environment_provenance="138 binds explicit pins into unchanged131 collector; observed versions never replaced.",
    )
    save(OUT / "selection.json", new)
    for task in assets["tasks"]:
        iid = task["instance_id"]
        folder = OUT / "private/review_assets" / iid
        folder.mkdir(parents=True)
        for name, info in task["files"].items():
            source = PREVIOUS / "private/review_assets" / iid / name
            assert sha(source) == info["sha256"]
            shutil.copyfile(source, folder / name)
        assert w.asset_contract(folder) == task["command_contract"]
        save(OUT / "private/asset_records" / (iid + ".json"), task)
    assets["selection_sha256"] = sha(OUT / "selection.json")
    save(OUT / "assets_manifest.json", assets)
    summary["selection_sha256"] = sha(OUT / "selection.json")
    summary["assets_manifest_sha256"] = sha(OUT / "assets_manifest.json")
    summary["reviewed_source_sha256"].update({str(f): sha(f) for f in evidence if f.is_file()})
    summary["official_environment_by_task"] = config
    save(OUT / "private/assets_review_summary.json", summary)
    save(
        OUT / "correction_protocol.json",
        dict(
            utc=utc(),
            selection_sha256=sha(OUT / "selection.json"),
            assigned_evaluation_slots=20,
            official_environment_by_task=config,
            interpreter_paths=interpreter_paths,
            original_initial_manifest_sha256=sha(PREVIOUS / "initial_manifest.json"),
            old_attempt_preserved=True,
            new_asset_downloads=0,
            prior_environments=0,
            prior_tests=0,
            api_spend_usd=0,
            gpu_used=False,
        ),
    )
    print("Frozen20 exact official environments; no preparation/tests", flush=True)


def selection():
    chosen = prior.selection()
    assert set(chosen["official_environment_by_task"]) == set(chosen["evaluation_ids"])
    return chosen


def versioned_prepare():
    """Small explicit90 preparation copy: exact per-task executable/pins/provenance."""
    import pyarrow.parquet as pq
    import yaml

    chosen = selection()
    b = w.m.b
    assert not (OUT / "protocol.json").exists()
    ids = chosen["evaluation_ids"]
    dataset = {r["instance_id"]: r for r in pq.read_table(b.DATA, filters=[("instance_id", "in", ids)]).to_pylist()}
    records = []
    queries = []
    started = time.monotonic()
    uv = chosen["installer_path"]
    asset_rows = {r["instance_id"]: r for r in read(OUT / "assets_manifest.json")["tasks"]}
    for task in chosen["tasks"]:
        iid = task["instance_id"]
        record = dict(task, ready=False)
        cfg = chosen["official_environment_by_task"][iid]
        executable = chosen["interpreter_paths"][cfg["python"]]
        specs = p.pins(cfg["packages"])
        try:
            assert asset_rows[iid]["ready"]
            assert w.asset_contract(OUT / "private/review_assets" / iid) == asset_rows[iid]["command_contract"]
            assets = OUT / "private/official_tasks" / iid
            assets.mkdir(parents=True)
            for name in w.ASSETS:
                shutil.copyfile(OUT / "private/review_assets" / iid / name, assets / name)
            assert official_config((assets / "Dockerfile").read_text()) == {k: cfg[k] for k in ["python", "packages"]}
            row = dataset[iid]
            meta = yaml.safe_load((assets / "task.yaml").read_text())
            assert meta["base_commit"] == row["base_commit"] == task["base_commit"]
            for name, key in [("gold.patch", "patch"), ("test.patch", "test_patch")]:
                assert (assets / name).read_text() == row[key]
            tests = read(assets / "tests.json")
            assert all(tests[k] == json.loads(row[k]) for k in ["FAIL_TO_PASS", "PASS_TO_PASS"])
            commands = [x for x in (assets / "eval.sh").read_text().splitlines() if x.startswith("PYTHONWARNINGS=")]
            assert len(commands) == 1 and "bin/test -C --verbose " in commands[0]
            for arm in ["baseline", "gold"]:
                folder = OUT / "private/tasks" / iid / arm
                folder.mkdir(parents=True)
                repo = folder / "repo"
                venv = folder / "venv"
                log = b.call(["git", "clone", "--shared", "--no-checkout", str(OUT / "private/base" / iid), str(repo)])
                log += b.call(["git", "checkout", "--detach", task["base_commit"]], cwd=repo)
                assert (repo / "setup.py").exists() and b.call(["git", "status", "--porcelain"], cwd=repo) == ""
                actual = json.loads(
                    b.call([executable, "-c", "import json,platform;print(json.dumps(platform.python_version()))"])
                )
                assert actual == cfg["python"]
                log += b.call([uv, "--cache-dir", str(OUT / "uv-cache"), "venv", "--python", executable, str(venv)])
                log += b.call(
                    [
                        uv,
                        "--cache-dir",
                        str(OUT / "uv-cache"),
                        "pip",
                        "install",
                        "--python",
                        str(venv / "bin/python"),
                        *specs,
                    ]
                )
                log += b.call(
                    [str(venv / "bin/python"), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."],
                    cwd=repo,
                    env=b.environment(venv),
                )
                (folder / "setup.log").write_text(log)
                probe = p.probe(cfg["packages"])
                raw = b.call([str(venv / "bin/python"), "-c", probe], cwd=repo, env=b.environment(venv))
                versions = json.loads(raw)
                save(OUT / "private/environment_provenance" / iid / (arm + ".json"), versions)
                queries.append(
                    dict(
                        instance_id=iid,
                        arm=arm,
                        expected_python=cfg["python"],
                        actual_python=versions["python"],
                        executable=executable,
                        packages=cfg["packages"],
                        probe_sha256=w.m.hashlib.sha256(probe.encode()).hexdigest(),
                    )
                )
                assert versions["provenance_all_checks_passed"]
                assert versions["python"] == cfg["python"] and Path(versions["sympy_file"]).resolve().is_relative_to(
                    repo.resolve()
                )
                assert all(versions["packages"][name] == version for name, version in cfg["packages"].items())
                versions["official_environment"] = dict(cfg, selected_interpreter=executable)
                save(folder / "environment.json", versions)
            record.update(
                ready=True,
                test_command=commands[0],
                tests=tests,
                official_environment=cfg,
                selected_interpreter=executable,
            )
        except Exception as error:
            record["preparation_failure"] = dict(type=type(error).__name__, message=str(error), replacement=False)
        records.append(record)
        w.m.save(OUT / "preparation_progress.json", records)
        print(json.dumps(dict(task=iid, ready=record["ready"])), flush=True)
    save(
        OUT / "metadata_query_dispatch_receipt.json",
        dict(
            utc=utc(),
            strict_queries=len(queries),
            queries=queries,
            observed_values_preserved=True,
            per_task_official_pins=True,
        ),
    )
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/138_swe_versioned_environment_probe.py",
        OUT / "selection.json",
        OUT / "export_manifest.json",
        OUT / "correction_protocol.json",
        OUT / "metadata_query_dispatch_receipt.json",
    ]
    paths += [
        ROOT / "scripts/90_swe_fresh_development.py",
        ROOT / "scripts/85_swe_execution_bridge.py",
        ROOT / "scripts/112_swe_patch_sft_runtime.py",
        ROOT / "scripts/134_swe_localize_evaluation_readiness.py",
        ROOT / "scripts/131_swe_environment_provenance.py",
        b.PARSER,
    ]
    paths += list((OUT / "private/official_tasks").glob("*/*")) + list(
        (OUT / "private/tasks").glob("*/*/environment.json")
    )
    paths += list((OUT / "private/environment_provenance").glob("*/*.json"))
    save(
        OUT / "protocol.json",
        dict(
            tasks=records,
            selection_sha256=sha(OUT / "selection.json"),
            source_sha256={str(f): sha(f) for f in paths},
            python="per_task_official",
            official_environment_by_task=chosen["official_environment_by_task"],
            interpreter_paths=chosen["interpreter_paths"],
            test_timeout_seconds=180,
            cpu_limit_seconds=120,
            memory_limit_bytes=4 * 1024**3,
            total_test_stage_wall_seconds=7200,
            preparation_seconds=time.monotonic() - started,
            baseline="Exact base plus official test.patch",
            gold="Exact base plus official gold.patch and same test.patch",
            failures="Retain all20, stop affected task on mismatch; no replacements",
            scope=chosen["scope"],
            deviations=read(b.OUT / "protocol.json")["deviations"][:3],
            api_spend_usd=0,
            gpu_used=False,
        ),
    )


def prepare(review):
    selection()
    original = w.m.prepare
    w.m.prepare = versioned_prepare
    try:
        w.prepare(review)
    finally:
        w.m.prepare = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "prepare", "smoke", "run", "aggregate", "_run-group"])
    parser.add_argument("--review")
    parser.add_argument("--group", choices=["smoke", "remaining"])
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
        return
    selection()
    if args.stage == "prepare":
        prepare(args.review)
    elif args.stage == "aggregate":
        prior.aggregate()
    elif args.stage == "_run-group":
        w.child_group(args.group)
    else:
        w.run_group("smoke" if args.stage == "smoke" else "remaining", args.review)


if __name__ == "__main__":
    main()
