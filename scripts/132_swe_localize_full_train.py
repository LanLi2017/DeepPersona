#!/usr/bin/env python3
"""Single gated three-epoch task-balanced control; no evaluation/private target reads."""

import argparse
from collections import Counter
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "runs/swe-diversity-selection/swe-localize-repair-support"
OUT = SUPPORT / "full_training"
SMOKE = SUPPORT / "learner_smoke"
QUALITY = SUPPORT / "source_runtime_metadata"
EVALUATION = SUPPORT / "evaluation_readiness_versions"
INITIALS = SUPPORT / "evaluation_readiness"
INFERENCE = SUPPORT / "autonomous_inference"
OLD = SUPPORT.parent / "swe-sympy-patch-sft"
EPOCHS = 3
TASK_BATCH = 4
TOTAL_SECONDS = 3600.0


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


h = module("full_metadata113", ROOT / "scripts/113_swe_patch_sft_train.py")
k = module("full_support125", ROOT / "scripts/125_swe_localize_repair_support.py")
ops = module("full_learner_ops", ROOT / "scripts/swe_localize_learner_ops.py")
MODEL = h.MODEL


def task_batches(tasks):
    for epoch in range(1, EPOCHS + 1):
        for start in range(0, len(tasks), TASK_BATCH):
            batch = tasks[start : start + TASK_BATCH]
            yield {
                "epoch": epoch,
                "batch_index": start // TASK_BATCH + 1,
                "tasks": batch,
                "actual_task_divisor": len(batch),
            }


def weighted_rows(batch):
    for task in batch["tasks"]:
        for index, row in enumerate(task["rows"], 1):
            yield task["instance_id"], index, row, 1.0 / (batch["actual_task_divisor"] * len(task["rows"]))


def forecast(smoke, rows, tasks):
    # Retain127's deliberately conservative one-optimizer-overhead per task estimate.
    return (
        1.20
        * (smoke["max_observed_row_seconds"] * EPOCHS * rows + smoke["optimizer_overhead_seconds"] * EPOCHS * tasks)
        + 120.0
    )


def verify(mapping):
    for name, digest in mapping.items():
        assert h.sha(h.resolve(name)) == digest, name


def freeze():
    assert not OUT.exists() or all(p.name in {"cpu_checks.json", "static_review.json"} for p in OUT.iterdir()), (
        "One full control only; preserve previous attempts"
    )
    checks = h.read(OUT / "cpu_checks.json")
    assert (
        checks["all_checks_passed"]
        and checks["source_sha256"] == h.sha(Path(__file__))
        and checks["helper_sha256"] == h.sha(ROOT / "scripts/swe_localize_learner_ops.py")
    )
    source = h.read(QUALITY / "summary.json")
    audit = h.read(QUALITY / "independent_quality_audit.json")
    assert source["source_quality_gate_passed"] and source["admitted_source_quality"] >= 16
    assert source["evaluation_readiness_assessed"] is False and source["assigned_evaluation_slots"] == 0
    assert audit["all_checks_passed"] and audit["admitted_source_quality"] == source["admitted_source_quality"]
    assert audit["summary_sha256"] == h.sha(QUALITY / "summary.json") and audit["completed_manifest_sha256"] == h.sha(
        QUALITY / "completed_manifest.json"
    )
    verify(h.read(QUALITY / "completed_manifest.json"))
    smoke = h.read(SMOKE / "run/summary.json")
    outer = h.read(SMOKE / "outer_terminal.json")
    smoke_protocol = h.read(SMOKE / "protocol.json")
    assert smoke["technical_checks_passed"] and smoke["task_balanced_per_turn_loss"] and smoke["optimizer_steps"] == 1
    assert smoke["zero_update_exact"] and smoke["standard_peft_saved_reloaded_exact"] and smoke["initial_reset_exact"]
    assert outer["exit_code"] == 0 and not outer["timed_out"] and outer["error"] is None
    assert outer["protocol_sha256"] == h.sha(SMOKE / "protocol.json")
    verify(h.read(SMOKE / "run/completed_manifest.json"))
    verify(smoke_protocol["inputs_sha256"])
    charged = max(smoke["total_elapsed_seconds"], outer["elapsed_seconds"])
    remaining = TOTAL_SECONDS - charged
    assert math.isclose(remaining, 3386.771692177281, abs_tol=1e-9) and remaining > 5
    # Read only evaluation readiness metadata/attestations; never evaluation tests, patches or logs.
    evaluation = h.read(EVALUATION / "summary.json")
    evaluation_audit = h.read(EVALUATION / "independent_readiness_audit.json")
    assert evaluation["assigned_evaluation_slots"] == 20 and evaluation["evaluation_runtime_valid"] == 20
    assert evaluation["all20_endpoints_ready"] and evaluation["readiness_gate_passed"]
    assert evaluation_audit["all_checks_passed"]
    for field, name in [
        ("summary_sha256", "summary.json"),
        ("completed_manifest_sha256", "completed_manifest.json"),
        ("initial_manifest_sha256", "initial_manifest.json"),
    ]:
        assert evaluation_audit[field] == h.sha((INITIALS if name == "initial_manifest.json" else EVALUATION) / name)
    evaluation_initials = h.read(INITIALS / "initial_manifest.json")
    assert (
        evaluation_initials["all20_public_initials_frozen"]
        and evaluation_initials["evaluation_private_values_read"] is False
    )
    assert evaluation_initials["protocol_sha256"] == h.sha(INITIALS / "initial_protocol.json")
    verify(evaluation_initials["source_sha256"])  # Public initial records only.
    inference = h.read(INFERENCE / "protocol.json")
    assert inference["assigned_evaluation_slots"] == 20 and inference["arms"] == ["base", "trained"]
    assert inference["evaluation_initial_manifest_sha256"] == h.sha(INITIALS / "initial_manifest.json")
    assert Path(inference["prospective_checkpoint_path"]).resolve() == (OUT / "run/final_adapter").resolve()
    assert inference["limits"] == dict(
        global_wall_seconds=7200,
        episode_wall_seconds=180,
        max_assistant_turns=8,
        max_tool_attempts=7,
        max_completion_tokens=2048,
        max_action_tokens=256,
        max_prompt_tokens=24576,
        max_history_tokens=26624,
        max_episode_completion_tokens=16384,
        total_completion_tokens=655360,
    )
    verify(inference["source_sha256"])
    selection = h.read(OLD / "selection.json")
    source_order = [r["instance_id"] for r in selection["tasks"] if r["role"] == "source"]
    eval_ids = [r["instance_id"] for r in selection["tasks"] if r["role"] == "evaluation"]
    admitted = [iid for iid in source_order if iid in source["admitted_source_ids"]]
    assert admitted == source["admitted_source_ids"] and len(admitted) == 23 and not set(admitted) & set(eval_ids)
    assert len(eval_ids) == 20
    assert [row["instance_id"] for row in evaluation_initials["tasks"]] == eval_ids
    assert set(evaluation["evaluation_runnable_ids"]) == set(eval_ids)
    manifest = h.read(SUPPORT / "completed_manifest.json")
    verify(manifest)
    tok = k.tokenizer()
    tasks = []
    paths = [
        Path(__file__),
        ROOT / "scripts/swe_localize_learner_ops.py",
        ROOT / "scripts/133_swe_localize_full_launcher.py",
        ROOT / "scripts/127_swe_localize_repair_smoke.py",
        ROOT / "scripts/129_swe_localize_smoke_launcher.py",
        ROOT / "scripts/125_swe_localize_repair_support.py",
        ROOT / "scripts/113_swe_patch_sft_train.py",
        ROOT / "docs/swe_localize_learner_control_plan.md",
        SUPPORT.parent / "paper-program/milestone_thirteen.json",
        QUALITY / "summary.json",
        QUALITY / "independent_quality_audit.json",
        QUALITY / "completed_manifest.json",
        SMOKE / "protocol.json",
        SMOKE / "run/summary.json",
        SMOKE / "outer_terminal.json",
        SMOKE / "run/completed_manifest.json",
        SMOKE / "run/initial_adapter.pt",
        SMOKE / "smoke_records.json",
        SUPPORT / "completed_manifest.json",
        OLD / "selection.json",
        OUT / "cpu_checks.json",
        INFERENCE / "protocol.json",
        EVALUATION / "summary.json",
        EVALUATION / "independent_readiness_audit.json",
        EVALUATION / "completed_manifest.json",
        INITIALS / "initial_protocol.json",
        INITIALS / "initial_manifest.json",
        EVALUATION / "selection.json",
    ]
    for iid in admitted:
        path = SUPPORT / "demonstrations" / f"{iid}.json"
        record = h.read(path)
        initial = h.read(SUPPORT / "initials" / f"{iid}.json")
        assert record["prequalified"] and record["role"] == "source" and record["instance_id"] == iid
        assert k.replay(OLD / "public" / iid, initial, record, tok)["all_checks_passed"]
        rows = [
            dict(prompt_token_ids=r["prompt_token_ids"], target_token_ids=r["completion_token_ids"], kind=r["kind"])
            for r in record["rows"]
        ]
        tasks.append(dict(instance_id=iid, rows=rows, demonstration_sha256=h.sha(path)))
        paths += [path, SUPPORT / "initials" / f"{iid}.json"]
    row_count = sum(len(t["rows"]) for t in tasks)
    batches = list(task_batches(tasks))
    assert row_count == 85 and len(batches) == 18 and sum(len(b["tasks"]) for b in batches) == 69
    assert sum(len(t["rows"]) for b in batches for t in b["tasks"]) == 255
    estimated = forecast(smoke, row_count, len(tasks))
    assert estimated <= remaining
    # Same pinned model/software files as the successful actual127 smoke.
    paths += [h.resolve(name) for name in smoke_protocol["inputs_sha256"]]
    paths += [h.resolve(name) for name in inference["source_sha256"]]
    if (OUT / "static_review.json").exists():
        paths.append(OUT / "static_review.json")
    assert {name: importlib.metadata.version(name) for name in smoke_protocol["software"]} == smoke_protocol["software"]
    assert h.gpu_identity() == smoke_protocol["hardware"]
    OUT.mkdir(exist_ok=True)
    h.save(OUT / "training_records.json", tasks)
    paths.append(OUT / "training_records.json")
    probe = h.read(SMOKE / "smoke_records.json")[0]
    h.save(OUT / "probe_record.json", probe)
    paths.append(OUT / "probe_record.json")
    protocol = dict(
        utc=h.utc(),
        scope="One fixed ordinary-supervision SWE learner control; no diversity claim",
        model=str(MODEL),
        seed=0,
        inputs_sha256={str(p): h.sha(p) for p in sorted(set(paths))},
        software=smoke_protocol["software"],
        hardware=smoke_protocol["hardware"],
        admitted_source_ids=admitted,
        evaluation_ids=eval_ids,
        evaluation_readiness_attested=True,
        evaluation_private_values_read=False,
        inference_protocol_sha256=h.sha(INFERENCE / "protocol.json"),
        epochs=EPOCHS,
        task_batch_size=TASK_BATCH,
        task_order="Original108sourceorder filtered by qualityadmission, unchanged each epoch",
        expected_optimizer_steps=18,
        expected_task_exposures=69,
        expected_causal_row_exposures=255,
        rows_per_epoch=85,
        expected_target_token_exposures=EPOCHS * sum(len(r["target_token_ids"]) for t in tasks for r in t["rows"]),
        objective="Mean current-completion-token NLL per turn, mean turns per task, mean tasks per actual batch. No clipping/scheduler/dropout/early stopping.",
        optimizer=smoke_protocol["optimizer"],
        adapter=smoke_protocol["adapter"],
        precision=smoke_protocol["precision"],
        initial_adapter_path=str(SMOKE / "run/initial_adapter.pt"),
        initial_adapter_sha256=h.sha(SMOKE / "run/initial_adapter.pt"),
        initialization="Exact savedsmokeINITIAL tensors withBzero; freshAdamW, neverupdatedsmokecheckpoint",
        expected_initial_probe_sha256=smoke["base_probe_sha256"],
        expected_frozen_base_sha256=smoke["frozen_base_parameter_sha256_before"],
        limits=dict(initial_prompt=12288, generation_prefix=24576, action=256, final=2048, causal_row=26624),
        total_shared_gpu_budget_seconds=TOTAL_SECONDS,
        charged_prior_seconds=charged,
        remaining_seconds=remaining,
        forecast_seconds=estimated,
        forecast_formula="1.20*(max127rowtime*3*85 + measured127optimizer_overhead*3*23)+120; conservative planning only",
        external_timeout=dict(
            required=True, term_seconds=remaining - 5, kill_grace_seconds=5, hard_ceiling_seconds=remaining
        ),
        final_checkpoint=str(OUT / "run/final_adapter"),
        checkpoint_selection="Only prescribed final18step checkpoint; no performance selection",
        inference=False,
        quality_tests=False,
        api_spend_usd=0,
        no_retry=True,
        no_backend_fallback=True,
    )
    h.save(OUT / "protocol.json", protocol)
    print(
        json.dumps(
            dict(
                frozen=True,
                source_tasks=len(tasks),
                steps=18,
                rows=255,
                remaining_seconds=remaining,
                forecast_seconds=estimated,
            )
        ),
        flush=True,
    )


def run():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    protocol = h.read(OUT / "protocol.json")
    assert os.environ.get("SWE_FULL_EXTERNAL_TIMEOUT") == json.dumps(protocol["external_timeout"], sort_keys=True)
    work = OUT / "run"
    work.mkdir()
    started = time.monotonic()
    torch = None
    steps = rows_done = tasks_done = targets_done = 0

    def interrupted(signum, frame):
        raise TimeoutError("Remaining sharedtrainingbudget exhausted or external interruption; no retry/fallback")

    old_alarm = signal.signal(signal.SIGALRM, interrupted)
    old_term = signal.signal(signal.SIGTERM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, protocol["external_timeout"]["term_seconds"])
    try:
        h.save(
            work / "started.json",
            dict(
                utc=h.utc(),
                protocol_sha256=h.sha(OUT / "protocol.json"),
                charged_prior_seconds=protocol["charged_prior_seconds"],
                external_timeout=protocol["external_timeout"],
            ),
        )
        verify(protocol["inputs_sha256"])
        assert {name: importlib.metadata.version(name) for name in protocol["software"]} == protocol["software"]
        assert h.gpu_identity() == protocol["hardware"]
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend

        observed = ops.configure_precision(protocol["hardware"]["uuid"])
        torch.cuda.reset_peak_memory_stats()
        h.save(work / "device_identity.json", dict(physical_gpu=2, torch_logical0_uuid=observed, matched=True))
        tasks = h.read(OUT / "training_records.json")
        assert [t["instance_id"] for t in tasks] == protocol["admitted_source_ids"]
        probe_record = h.read(OUT / "probe_record.json")
        initial = torch.load(protocol["initial_adapter_path"], map_location="cpu", weights_only=True)
        model = ops.load_base(MODEL)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            original = ops.probe(model, probe_record)
            assert ops.tensor_hash(original) == protocol["expected_initial_probe_sha256"]
            model, params = ops.attach_initial(model, initial)
            assert torch.equal(original, ops.probe(model, probe_record)), "Initial adapter changes baseprobe"
            before = ops.base_hash(model)
            base_signature = ops.storage_signature(model)
            assert before == protocol["expected_frozen_base_sha256"]
            initial_exact = all(torch.equal(v.detach().cpu(), initial[n]) for n, v in params.items())
            assert initial_exact
            trace, hooks = ops.dtype_hooks(model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            opt = torch.optim.AdamW(params.values(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            assert not opt.state
            with (work / "rows.jsonl").open("x") as row_log, (work / "steps.jsonl").open("x") as step_log:
                for batch in task_batches(tasks):
                    opt.zero_grad(set_to_none=True)
                    step_started = time.monotonic()
                    batch_loss = 0.0
                    for iid, turn, row, weight in weighted_rows(batch):
                        row_started = time.monotonic()
                        loss, logit_dtype = ops.completion(model, row)
                        assert torch.isfinite(loss)
                        (loss * weight).backward()
                        torch.cuda.synchronize()
                        value = float(loss.detach())
                        batch_loss += value * weight
                        rows_done += 1
                        targets_done += len(row["target_token_ids"])
                        row_log.write(
                            json.dumps(
                                dict(
                                    epoch=batch["epoch"],
                                    batch_index=batch["batch_index"],
                                    instance_id=iid,
                                    turn=turn,
                                    kind=row["kind"],
                                    loss=value,
                                    loss_weight=weight,
                                    actual_batch_tasks=batch["actual_task_divisor"],
                                    prompt_tokens=len(row["prompt_token_ids"]),
                                    target_tokens=len(row["target_token_ids"]),
                                    logit_dtype=logit_dtype,
                                    loss_dtype=str(loss.dtype),
                                    seconds=time.monotonic() - row_started,
                                )
                            )
                            + "\n"
                        )
                        row_log.flush()
                        del loss
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
                    norm = float(torch.sqrt(sum(v.grad.square().sum() for v in params.values())))
                    assert math.isfinite(norm) and norm > 0
                    opt.step()
                    assert all(torch.isfinite(v).all() for v in params.values())
                    assert all(
                        s.dtype == torch.float32
                        for state in opt.state.values()
                        for s in state.values()
                        if torch.is_tensor(s)
                    )
                    steps += 1
                    tasks_done += batch["actual_task_divisor"]
                    torch.cuda.synchronize()
                    step_log.write(
                        json.dumps(
                            dict(
                                step=steps,
                                epoch=batch["epoch"],
                                batch_index=batch["batch_index"],
                                task_ids=[t["instance_id"] for t in batch["tasks"]],
                                actual_task_divisor=batch["actual_task_divisor"],
                                task_balanced_loss=batch_loss,
                                gradient_norm=norm,
                                seconds=time.monotonic() - step_started,
                                cumulative_rows=rows_done,
                                cumulative_tasks=tasks_done,
                            )
                        )
                        + "\n"
                    )
                    step_log.flush()
            opt.zero_grad(set_to_none=True)
            assert (steps, tasks_done, rows_done, targets_done) == (
                protocol["expected_optimizer_steps"],
                protocol["expected_task_exposures"],
                protocol["expected_causal_row_exposures"],
                protocol["expected_target_token_exposures"],
            )
            ops.verify_dtypes(trace)
            checkpoint = ops.save_reload_reset(model, params, initial, work / "final_adapter", probe_record, original)
            assert base_signature == ops.storage_signature(model)
            after = ops.base_hash(model)
            assert before == after
            for hook in hooks:
                hook.remove()
            del opt
            for value in params.values():
                value.grad = None
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            assert elapsed <= protocol["remaining_seconds"]
            summary = dict(
                utc=h.utc(),
                training_complete=True,
                technical_checks_passed=True,
                full_training_run=True,
                epochs_completed=EPOCHS,
                optimizer_steps=steps,
                task_exposures=tasks_done,
                causal_row_exposures=rows_done,
                target_token_exposures=targets_done,
                admitted_source_ids=protocol["admitted_source_ids"],
                initial_adapter_sha256=protocol["initial_adapter_sha256"],
                initial_adapter_exact=True,
                fresh_optimizer_state=True,
                initial_probe_sha256=ops.tensor_hash(original),
                frozen_base_parameter_sha256_before=before,
                frozen_base_parameter_sha256_after=after,
                base_storage_version_dtype_unchanged=True,
                actual_dtypes=trace,
                **checkpoint,
                final_adapter_path=str(work / "final_adapter"),
                final_adapter_sha256=h.sha(work / "final_adapter/adapter_model.safetensors"),
                final_adapter_config_sha256=h.sha(work / "final_adapter/adapter_config.json"),
                protocol_sha256=h.sha(OUT / "protocol.json"),
                total_elapsed_seconds=elapsed,
                charged_prior_seconds=protocol["charged_prior_seconds"],
                shared_budget_seconds=TOTAL_SECONDS,
                remaining_seconds_before_external_charge=TOTAL_SECONDS - protocol["charged_prior_seconds"] - elapsed,
                external_terminal_duration_required=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                inference_run=False,
                evaluation_private_values_read=False,
                quality_tests_run=False,
                learner_benefit_established=False,
                diversity_benefit_established=False,
                api_spend_usd=0,
            )
            h.save(work / "summary.json", summary)
            h.save(work / "completed_manifest.json", {str(p): h.sha(p) for p in work.rglob("*") if p.is_file()})
    except BaseException as error:
        h.save(
            work / "failure.json",
            dict(
                utc=h.utc(),
                type=type(error).__name__,
                message=str(error),
                total_elapsed_seconds=time.monotonic() - started,
                optimizer_steps=steps,
                causal_rows=rows_done,
                task_exposures=tasks_done,
                retry=False,
                fallback=False,
                partial_state_not_for_evaluation=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()
                if torch is not None and torch.cuda.is_initialized()
                else None,
            ),
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)


def checks():
    import torch
    from types import SimpleNamespace

    tasks = [
        dict(
            instance_id=f"t{i}",
            rows=[dict(prompt_token_ids=[1, 2, 3], target_token_ids=[4, 5], kind="final_patch") for _ in range(n)],
        )
        for i, n in enumerate([2, 3, 1, 4, 2])
    ]
    batches = list(task_batches(tasks))
    assert len(batches) == 6 and [b["actual_task_divisor"] for b in batches] == [4, 1] * 3
    assert Counter(t["instance_id"] for b in batches for t in b["tasks"]) == Counter({f"t{i}": 3 for i in range(5)})
    assert all(math.isclose(sum(weight for _, _, _, weight in weighted_rows(b)), 1.0) for b in batches)
    # Independent explicit task means vs loop weights including final one-task batch.
    parameters = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    differences = []
    for batch in batches[:2]:
        terms = []
        explicit = []
        for task in batch["tasks"]:
            per_turn = []
            for turn, _ in enumerate(task["rows"], 1):
                per_turn.append(
                    (parameters * torch.tensor([turn, len(task["rows"])], dtype=torch.float64)).square().mean()
                )
            explicit.append(torch.stack(per_turn).mean())
            terms += [value / (len(batch["tasks"]) * len(task["rows"])) for value in per_turn]
        expected = torch.stack(explicit).mean()
        actual = sum(terms)
        ga = torch.autograd.grad(actual, parameters, retain_graph=True)[0]
        gb = torch.autograd.grad(expected, parameters)[0]
        differences.append(float((ga - gb).abs().max()))
        assert torch.allclose(ga, gb, atol=1e-14, rtol=0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(11, 5)
            self.head = torch.nn.Linear(5, 11)

        def forward(self, input_ids, use_cache, logits_to_keep):
            assert not use_cache
            return SimpleNamespace(logits=self.head(self.emb(input_ids))[:, logits_to_keep, :])

    torch.manual_seed(132)
    model = Tiny()
    row = tasks[0]["rows"][0]
    actual, _ = ops.completion(model, row)
    actual_grad = torch.autograd.grad(actual, list(model.parameters()), retain_graph=True)
    ids = torch.tensor([row["prompt_token_ids"] + row["target_token_ids"]])
    all_logits = model.head(model.emb(ids))[:, :-1, :]
    labels = ids[:, 1:].clone()
    labels[:, : len(row["prompt_token_ids"]) - 1] = -100
    expected = torch.nn.functional.cross_entropy(all_logits.reshape(-1, 11), labels.reshape(-1), ignore_index=-100)
    expected_grad = torch.autograd.grad(expected, list(model.parameters()))
    error = max(float((a - b).abs().max()) for a, b in zip(actual_grad, expected_grad))
    assert error == 0
    report = dict(
        all_checks_passed=True,
        source_sha256=h.sha(Path(__file__)),
        helper_sha256=h.sha(ROOT / "scripts/swe_localize_learner_ops.py"),
        actual_tail_task_divisor_verified=True,
        three_epoch_exposures_verified=True,
        task_mean_gradient_max_error=max(differences),
        extracted_completion_gradient_max_error=error,
        scope="Fabricated CPU tasks and tinyCPUmodel only; no real Qwen allocation or training artifacts",
        gpu_used=False,
    )
    OUT.mkdir(exist_ok=True)
    h.save(OUT / "cpu_checks.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["checks", "freeze", "run"])
    globals()[parser.parse_args().stage]()
