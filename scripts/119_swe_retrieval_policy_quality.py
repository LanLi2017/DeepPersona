#!/usr/bin/env python3
"""Conditional retrieval-policy source-only runtime reuse; no assets/models read at import time."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/swe-diversity-selection"
POP = BASE / "swe-sympy-patch-sft"
STUDY = BASE / "swe-sympy-retrieval-policies"
SMOKE = STUDY / "learner_smoke"
OUT = STUDY / "source_runtime"
HELPER = ROOT / "scripts/112_swe_patch_sft_runtime.py"
SPEC = importlib.util.spec_from_file_location("source_runtime112", HELPER)
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
# The inherited child launcher must dispatch this wrapper, never frozen112's CLI.
w.__file__ = str(Path(__file__).resolve())
w.OUT = OUT
w.m.OUT = OUT
sha, read, save, utc = w.sha, w.read, w.save, w.utc


def verify(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT / name) == digest, name


def freeze():
    assert not OUT.exists()
    scores = read(STUDY / "score_summary.json")
    policy = scores["chosen_policy"]
    assert policy in ["production_chunks", "production_files"] and scores["minimum_required"] == 16
    assert scores["technical_failures"] == 0 and scores["quality_tests_run"] == 0
    assert scores["evaluation_new_contexts"] == 0 and scores["evaluation_private_values_read"] is False
    candidates = scores["policies"]
    assert len(candidates) == 2 and {r["policy"] for r in candidates} == {"production_chunks", "production_files"}
    winner = min(candidates, key=lambda r: (-r["prequalified"], r["policy"] != "production_chunks"))
    assert winner["prequalified"] >= 16 and winner["policy"] == policy
    scored = next(r for r in candidates if r["policy"] == policy)
    assert scored["assigned_sources"] == 32
    cap = 24576
    population = read(POP / "selection.json")
    sources = [t for t in population["tasks"] if t["role"] == "source"]
    assert len(sources) == len({t["instance_id"] for t in sources}) == 32
    for candidate in candidates:
        assert len(candidate["tasks"]) == candidate["assigned_sources"] == 32
        assert {r["instance_id"] for r in candidate["tasks"]} == {t["instance_id"] for t in sources}
        assert candidate["prequalified"] == sum(r["prequalified"] for r in candidate["tasks"])
        assert all(r["policy"] == candidate["policy"] for r in candidate["tasks"])
    screened = {r["instance_id"]: r for r in scored["tasks"]}
    assert len(scored["tasks"]) == 32 and set(screened) == {t["instance_id"] for t in sources}
    active = [t for t in sources if screened[t["instance_id"]]["prequalified"]]
    ids = [t["instance_id"] for t in active]
    assert len(ids) == scored["prequalified"]
    assert all(
        screened[i]["conversion_supported"]
        and screened[i]["completion_fits"]
        and screened[i]["anchor_visible"]
        and screened[i]["context_status"] == "ready"
        for i in ids
    )
    policy_protocol = read(STUDY / "protocol.json")
    assert policy_protocol["source_tasks"] == sources
    assert policy_protocol["prompt_cap"] == cap
    assert all(r["policy"] == policy for r in scored["tasks"])
    technical = read(SMOKE / "run/summary.json")
    technical_protocol = read(SMOKE / "protocol.json")
    outer = read(SMOKE / "outer_terminal.json")
    assert technical["technical_checks_passed"] is True
    assert technical["quality_verified"] is False and technical["full_training_run"] is False
    assert technical["inference_run"] is False and technical["evaluation_private_access"] is False
    assert technical_protocol["chosen_policy"] == policy
    started = read(SMOKE / "run/started.json")
    assert started["protocol_sha256"] == sha(SMOKE / "protocol.json")
    smoke_completed = read(SMOKE / "run/completed_manifest.json")
    smoke_completed_absolute = {str(ROOT / name): digest for name, digest in smoke_completed.items()}
    for path in [SMOKE / "run/started.json", SMOKE / "run/summary.json"]:
        assert smoke_completed_absolute[str(path)] == sha(path)
    assert technical["selected_prompt_cap"] == technical_protocol["chosen_prompt_cap"] == cap
    assert technical_protocol["prequalified_source_ids"] == ids
    assert technical_protocol["max_completion_tokens"] == 2048
    assert technical_protocol["max_sequence_tokens"] == cap + 2048
    assert technical["future_max_dose_fits_internal_budget"] is True
    assert outer["protocol_sha256"] == sha(SMOKE / "protocol.json")
    assert outer["exit_code"] == 0 and outer["timed_out"] is False and outer["error"] is None
    assert outer["hard_limit_seconds"] == 3600 and outer["soft_limit_seconds"] == 3595
    internal, external = technical["total_elapsed_seconds"], outer["elapsed_seconds"]
    estimate = technical["future_max_prequalified_dose_estimate_seconds"]
    assert all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in [internal, external, estimate])
    assert external <= 3600 and technical_protocol["total_shared_gpu_budget_seconds"] == 3600
    assert technical_protocol["prior_gpu_seconds"] == 0
    charged = max(internal, external)
    assert estimate <= 3600 - charged, "Future maximum source dose does not fit externally charged GPU budget"
    # Bind the executed smoke's source/data provenance without opening pretrained model files.
    model_root = Path(technical_protocol["model"])
    smoke_nonmodel_inputs = {
        name: digest
        for name, digest in technical_protocol["inputs_sha256"].items()
        if not (ROOT / name).is_relative_to(model_root)
    }
    maps = [
        smoke_nonmodel_inputs,
        read(STUDY / "completed_manifest.json"),
        read(STUDY / "build_manifest.json")["artifact_sha256"],
        policy_protocol["source_sha256"],
        read(STUDY / "score_protocol.json")["inputs_sha256"],
        smoke_completed,
    ]
    for mapping in maps:
        verify(mapping)
    exports = read(POP / "export_manifest.json")
    assert exports["selection_sha256"] == sha(POP / "selection.json")
    exported = {t["instance_id"]: t for t in exports["tasks"]}
    assert all(exported[t["instance_id"]]["ready"] for t in sources)
    paths = [
        Path(__file__).resolve(),
        HELPER,
        ROOT / "scripts/90_swe_fresh_development.py",
        ROOT / "scripts/85_swe_execution_bridge.py",
        ROOT / "scripts/107_swe_sympy_compatibility.py",
        ROOT / "scripts/108_swe_patch_sft_population.py",
        ROOT / "scripts/117_swe_sympy_retrieval_policies.py",
        ROOT / "scripts/118_swe_retrieval_policy_smoke.py",
        w.m.b.PARSER,
        w.m.b.DATA,
        POP / "selection.json",
        POP / "selection_protocol.json",
        POP / "export_manifest.json",
        STUDY / "protocol.json",
        STUDY / "score_protocol.json",
        STUDY / "score_summary.json",
        STUDY / "build_manifest.json",
        STUDY / "completed_manifest.json",
        SMOKE / "protocol.json",
        SMOKE / "run/summary.json",
        SMOKE / "run/started.json",
        SMOKE / "run/completed_manifest.json",
        SMOKE / "outer_terminal.json",
    ]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    # Freeze selected contexts and target records transitively; no target body is parsed here.
    for mapping in maps:
        for name, digest in mapping.items():
            path = ROOT / name
            normalized = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            assert normalized not in hashes or hashes[normalized] == digest
            hashes[normalized] = digest
    OUT.mkdir()
    (OUT / "private").mkdir()
    (OUT / "private/base").symlink_to(POP / "private/base", target_is_directory=True)
    (OUT / "public").symlink_to(POP / "public", target_is_directory=True)
    shutil.copyfile(POP / "export_manifest.json", OUT / "export_manifest.json")
    save(
        OUT / "selection.json",
        dict(
            utc=utc(),
            tasks=active,
            all_slots=sources,
            source_prequalification=scored["tasks"],
            source_prequalified_ids=ids,
            evaluation_ids=[],
            smoke_ids=ids[:1],
            remaining_ids=ids[1:],
            chosen_policy=policy,
            chosen_prompt_cap=cap,
            dataset_revision=population["dataset_revision"],
            harness_revision=population["harness_revision"],
            task_repo_revision=population["task_repo_revision"],
            source_sha256=hashes,
            export_manifest_sha256=sha(OUT / "export_manifest.json"),
            export_manifest_original=str((POP / "export_manifest.json").relative_to(ROOT)),
            gpu_accounting=dict(
                shared_limit_seconds=3600,
                technical_internal_seconds=internal,
                technical_external_seconds=external,
                charged_prior_seconds=charged,
                remaining_seconds=3600 - charged,
                future_max_source_dose_estimate_seconds=estimate,
            ),
            scope="Source-only host baseline/gold quality; all32 fixed source slots retained. No evaluation assets or model calls.",
            admission="Only selected-policy-prequalified SOURCE IDs acquire assets. No replacement or evaluation admission.",
            private_access="Unchanged90prepare selected-source rows only; original public/base/111 source-gold remain unchanged.",
            asset_guard="Unchanged112prepare cache admits URLs only for ready=true rows after command-contract and hash verification. Failed rows stop before any environment execution.",
            limits=dict(
                per_test_wall_seconds=180,
                per_test_cpu_seconds=120,
                per_test_memory_bytes=4 * 1024**3,
                aggregate_outer_test_seconds=7200,
                outer_termination_grace_seconds=5,
            ),
            stages="freeze -> assets -> exact root-reviewed prepare -> firstsource smoke -> reviewed remaining -> aggregate",
            no_retry=True,
            api_spend_usd=0,
            gpu_used=False,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                selected_policy=policy,
                selected_prompt_cap=cap,
                active_sources=len(ids),
                retained_sources=32,
            )
        ),
        flush=True,
    )


def selection():
    chosen = w.selection()
    assert len(chosen["all_slots"]) == 32 and chosen["evaluation_ids"] == []
    assert all(t["role"] == "source" for t in chosen["all_slots"] + chosen["tasks"])
    return chosen


def prepare(review):
    chosen = selection()
    manifest = read(OUT / "assets_manifest.json")
    assert [r["instance_id"] for r in manifest["tasks"]] == chosen["source_prequalified_ids"]
    assert all(r["role"] == "source" for r in manifest["tasks"])
    # Frozen112 already excludes every ready=false row from its cached URL getter.
    w.prepare(review)
    prepared = {r["instance_id"]: r for r in read(OUT / "protocol.json")["tasks"]}
    for row in manifest["tasks"]:
        if not row["ready"]:
            assert prepared[row["instance_id"]]["ready"] is False
            assert not (OUT / "private/tasks" / row["instance_id"]).exists()


def aggregate():
    chosen = selection()
    preparation, runtime, groups = {}, {}, []
    if (OUT / "protocol.json").exists():
        preparation = {r["instance_id"]: r for r in read(OUT / "protocol.json")["tasks"]}
    elif (OUT / "preparation_progress.json").exists():
        preparation = {r["instance_id"]: r for r in read(OUT / "preparation_progress.json")}
    for group in ["smoke", "remaining"]:
        folder = OUT / "groups" / group
        if (folder / "dispatch.json").exists():
            assert (folder / "terminal.json").exists(), "Cannot aggregate unresolved/running dispatch"
            groups.append(dict(group=group, **read(folder / "terminal.json")))
        result = folder / "results.json"
        if not result.exists():
            result = folder / "results_progress.json"
        if result.exists():
            data = read(result)
            for row in data["tasks"] if isinstance(data, dict) else data:
                assert row["instance_id"] not in runtime
                runtime[row["instance_id"]] = row
    active = set(chosen["source_prequalified_ids"])
    assert set(runtime) <= active and set(preparation) <= active
    rows = []
    for task in chosen["all_slots"]:
        iid = task["instance_id"]
        ready, result = preparation.get(iid), runtime.get(iid)
        if iid not in active:
            status = "source_target_not_prequalified"
        elif ready is None:
            status = "preparation_not_completed"
        elif not ready["ready"]:
            status = "preparation_failure"
        elif result is None:
            status = "tests_not_completed"
        else:
            quality = bool(result["host_bridge_passed"] and ready["tests"]["FAIL_TO_PASS"])
            arms = {a["arm"]: a for a in result.get("arms", [])}
            quality = quality and len(result.get("arms", [])) == 2 and set(arms) == {"baseline", "gold"}
            if quality:
                for arm, evidence in arms.items():
                    quality = quality and evidence["exit_code"] == (0 if arm == "gold" else 1)
                    for suite, tests in ready["tests"].items():
                        allowed = {"FAILED", "ERROR"} if arm == "baseline" and suite == "FAIL_TO_PASS" else {"PASSED"}
                        statuses = evidence["statuses"][suite]
                        quality = (
                            quality and set(statuses) == set(tests) and all(s in allowed for s in statuses.values())
                        )
            status = "baseline_gold_passed" if quality else "baseline_gold_failed"
        row = dict(
            instance_id=iid,
            role="source",
            slot=task["slot"],
            family=task["family"],
            disposition=status,
            prequalified=iid in active,
            replacement=False,
            preparation_failure=ready.get("preparation_failure") if ready else None,
            runtime_result=result,
        )
        path = OUT / "private/quality_records" / (iid + ".json")
        save(path, row)
        rows.append(
            dict(
                **{k: v for k, v in row.items() if k != "runtime_result"},
                quality_record_path=str(path.relative_to(ROOT)),
                quality_record_sha256=sha(path),
            )
        )
    admitted = [r["instance_id"] for r in rows if r["disposition"] == "baseline_gold_passed"]
    report = dict(
        utc=utc(),
        admitted_source_ids=admitted,
        assigned_source_slots=32,
        assigned_evaluation_slots=0,
        source_prequalified=len(active),
        admitted_source_quality=len(admitted),
        minimum_source_quality=16,
        source_quality_gate_passed=len(admitted) >= 16,
        readiness_gate_passed=len(admitted) >= 16,
        evaluation_readiness_assessed=False,
        official_container_evaluation=False,
        selected_policy=chosen["chosen_policy"],
        selected_prompt_cap=chosen["chosen_prompt_cap"],
        groups=groups,
        outer_test_seconds=sum(g["seconds"] for g in groups),
        tasks=rows,
        gpu_accounting=chosen["gpu_accounting"],
        api_spend_usd=0,
        gpu_used=False,
        interpretation="Fixed-slot source quality only, not evaluation readiness or learned/diversity effect; no refill.",
    )
    save(OUT / "summary.json", report)
    paths = list(OUT.glob("*.json")) + list((OUT / "groups").glob("*/*.json"))
    paths += list((OUT / "groups").glob("*/runner.log"))
    paths += list((OUT / "private/quality_records").glob("*.json"))
    paths += list((OUT / "private/asset_records").glob("*.json"))
    paths += list((OUT / "private/review_assets").glob("*/*"))
    paths += list((OUT / "private").glob("*.json"))
    for pattern in ["*/*/result.json", "*/*/environment.json", "*/*/test.log"]:
        paths += list((OUT / "private/tasks").glob(pattern))
    save(OUT / "completed_manifest.json", {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))})
    print(json.dumps({k: v for k, v in report.items() if k not in {"tasks", "groups"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "assets", "prepare", "smoke", "run", "aggregate", "_run-group"])
    parser.add_argument("--review")
    parser.add_argument("--group", choices=["smoke", "remaining"])
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "prepare":
        prepare(args.review)
    elif args.stage == "aggregate":
        aggregate()
    else:
        selection()
        if args.stage == "assets":
            w.assets()
        elif args.stage == "_run-group":
            w.child_group(args.group)
        else:
            w.run_group("smoke" if args.stage == "smoke" else "remaining", args.review)
