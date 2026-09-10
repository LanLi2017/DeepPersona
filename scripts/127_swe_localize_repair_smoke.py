#!/usr/bin/env python3
"""One technical-only multi-turn learner smoke; no full training, quality or inference."""

import argparse
from collections import Counter
import hashlib
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
OUT = SUPPORT / "learner_smoke"
OLD = SUPPORT.parent / "swe-sympy-patch-sft"
LIMIT = 3600


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = module("multi_train113", ROOT / "scripts/113_swe_patch_sft_train.py")
k = module("multi_support125", ROOT / "scripts/125_swe_localize_repair_support.py")
MODEL = h.MODEL


def validate_weights(records, task_ids):
    counts = Counter(r["instance_id"] for r in records)
    assert set(counts) == set(task_ids) and len(counts) == len(task_ids)
    for row in records:
        assert row["task_turns"] == counts[row["instance_id"]]
        assert row["loss_weight"] == 1.0 / (len(task_ids) * row["task_turns"])
    assert math.isclose(sum(r["loss_weight"] for r in records), 1.0, abs_tol=1e-12)
    for iid in task_ids:
        assert math.isclose(
            sum(r["loss_weight"] for r in records if r["instance_id"] == iid), 1.0 / len(task_ids), abs_tol=1e-12
        )


def select_tasks(records):
    def key(row, field):
        return (-row[field], row["instance_id"])

    criteria = ["max_sequence_tokens", "total_sequence_squared", "max_completion_tokens"]
    selected = []
    for field in criteria:
        iid = min(records, key=lambda row: key(row, field))["instance_id"]
        if iid not in selected:
            selected.append(iid)
    return selected


def objective_checks():
    """CPU-only changed-objective regression; no real data or model allocation."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(127)
    tasks = []
    for count in [6, 4]:
        task = []
        for turn in range(count):
            prompt, completion = 3 + turn % 2, 2 + turn % 3
            ids = torch.randint(0, 11, (prompt + completion,))
            logits = torch.randn(prompt + completion - 1, 11, dtype=torch.float64, requires_grad=True)
            task.append((logits, ids, prompt))
        tasks.append(task)
    flattened = []
    masked = []
    explicit_tasks = []
    full_logits = []
    for task in tasks:
        row_losses = []
        for logits, ids, prompt in task:
            value = F.cross_entropy(logits[prompt - 1 :], ids[prompt:])
            flattened.append(value / (len(tasks) * len(task)))
            row_losses.append(value)
            full_logits.append(logits)
            labels = ids[1:].clone()
            labels[: prompt - 1] = -100
            masked.append(F.cross_entropy(logits, labels, ignore_index=-100) / (len(tasks) * len(task)))
        explicit_tasks.append(torch.stack(row_losses).mean())
    flat_loss = sum(flattened)
    nested_loss = torch.stack(explicit_tasks).mean()
    masked_loss = sum(masked)
    flat_grad = torch.autograd.grad(flat_loss, full_logits, retain_graph=True)
    nested_grad = torch.autograd.grad(nested_loss, full_logits, retain_graph=True)
    masked_grad = torch.autograd.grad(masked_loss, full_logits)
    loss_error = float((flat_loss - nested_loss).detach().abs())
    gradient_error = max(float((a - b).abs().max()) for a, b in zip(flat_grad, nested_grad))
    masked_error = max(float((a - b).abs().max()) for a, b in zip(flat_grad, masked_grad))
    perturbed = []
    ignored_zero = True
    index = 0
    for task in tasks:
        for logits, ids, prompt in task:
            ignored_zero &= bool(torch.count_nonzero(flat_grad[index][: prompt - 1]) == 0)
            changed = logits.detach().clone()
            changed[: prompt - 1] += 1000 * torch.randn_like(changed[: prompt - 1])
            labels = ids[1:].clone()
            labels[: prompt - 1] = -100
            perturbed.append(F.cross_entropy(changed, labels, ignore_index=-100) / (len(tasks) * len(task)))
            index += 1
    perturbation_error = float((sum(perturbed) - flat_loss.detach()).abs())
    naive_loss = torch.stack(
        [F.cross_entropy(logits[prompt - 1 :], ids[prompt:]) for task in tasks for logits, ids, prompt in task]
    ).mean()
    naive_difference = float((naive_loss - nested_loss.detach()).detach().abs())
    checks = dict(
        loss_equal=loss_error < 1e-12,
        gradients_equal=gradient_error < 1e-12,
        selected_logits_equal_masked_CE=masked_error < 1e-12,
        masked_history_logits_inert=perturbation_error < 1e-12,
        ignored_history_gradients_zero=ignored_zero,
        unequal_turn_counts_not_uniform_rows=naive_difference > 1e-6,
    )
    assert all(checks.values()), checks
    OUT.mkdir(exist_ok=True)
    report = dict(
        all_checks_passed=True,
        smoke_sha256=h.sha(Path(__file__)),
        checks=checks,
        loss_error=loss_error,
        gradient_error=gradient_error,
        masked_gradient_error=masked_error,
        masked_history_logit_perturbation_error=perturbation_error,
        incorrect_uniform_row_loss_difference=naive_difference,
        task_turn_counts=[6, 4],
        dtype="float64",
        device="CPU",
        real_data_read=False,
        model_allocated=False,
        scope="Only changed task-balanced objective and target-position CE/masks; perturbing actual history input tokens is NOT claimed inert.",
    )
    h.save(OUT / "objective_checks.json", report)
    print(json.dumps(report), flush=True)


def freeze():
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as qwen
    import transformers.integrations.sdpa_attention as sdpa
    import peft.tuners.lora.layer as lora
    import peft.utils.save_and_load as peft_io
    import torch.optim.adamw as adamw

    assert not OUT.exists() or all(p.name in {"static_review.json", "objective_checks.json"} for p in OUT.iterdir()), (
        "One prospective smoke; preserve failures"
    )
    checks = h.read(OUT / "objective_checks.json")
    assert checks["all_checks_passed"] and checks["smoke_sha256"] == h.sha(Path(__file__))
    summary = h.read(SUPPORT / "support_summary.json")
    assert summary["support_gate_passed"] and summary["prequalified"] >= 16 and summary["technical_failures"] == 0
    assert summary["minimum_required"] == 16 and summary["source_tasks"] == 32
    assert summary["quality_tests_run"] == 0 and summary["evaluation_private_values_read"] is False
    audit_path = SUPPORT / "independent_support_audit.json"
    audit = h.read(audit_path)
    assert (
        audit["all_checks_passed"]
        and audit["source_tasks_audited"] == 32
        and audit["prequalified"] == summary["prequalified"]
    )
    for key, path in [
        ("protocol_sha256", SUPPORT / "protocol.json"),
        ("support_summary_sha256", SUPPORT / "support_summary.json"),
        ("completed_manifest_sha256", SUPPORT / "completed_manifest.json"),
        ("initial_manifest_sha256", SUPPORT / "initial_manifest.json"),
        ("kernel_sha256", ROOT / "scripts/125_swe_localize_repair_support.py"),
        ("driver_sha256", ROOT / "scripts/audit_swe_localize_support.py"),
    ]:
        assert audit[key] == h.sha(path), key
    audit_started = h.read(SUPPORT / "independent_support_audit.started.json")
    audit_terminal = h.read(SUPPORT / "independent_support_audit.terminal.json")
    assert audit_started["protocol_sha256"] == audit["protocol_sha256"]
    assert audit_started["completed_manifest_sha256"] == audit["completed_manifest_sha256"]
    assert audit_started["initial_manifest_sha256"] == audit["initial_manifest_sha256"]
    assert audit_started["driver_sha256"] == audit["driver_sha256"]
    assert (
        audit_terminal["exit_code"] == 0
        and audit_terminal["failure"] is None
        and audit_terminal["tasks_returned"] == 32
        and audit_terminal["report_exists"]
    )
    source_ids = [r["instance_id"] for r in h.read(OLD / "selection.json")["tasks"] if r["role"] == "source"]
    assert len(source_ids) == 32 and [r["instance_id"] for r in summary["tasks"]] == source_ids
    manifest = h.read(SUPPORT / "completed_manifest.json")
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/129_swe_localize_smoke_launcher.py",
        ROOT / "scripts/115_swe_sympy_long_context_smoke.py",
        ROOT / "scripts/113_swe_patch_sft_train.py",
        ROOT / "scripts/125_swe_localize_repair_support.py",
        ROOT / "scripts/126_swe_localize_repair_study.py",
        audit_path,
        SUPPORT / "independent_support_audit.started.json",
        SUPPORT / "independent_support_audit.terminal.json",
        ROOT / "scripts/audit_swe_localize_support.py",
        OUT / "objective_checks.json",
        SUPPORT / "support_summary.json",
        SUPPORT / "completed_manifest.json",
        OLD / "selection.json",
    ]
    for name, digest in manifest.items():
        path = h.resolve(name)
        assert h.sha(path) == digest, name
        paths.append(path)
    support_protocol = h.read(SUPPORT / "protocol.json")
    for field in ["source_sha256", "inputs_sha256"]:
        for name, digest in support_protocol.get(field, {}).items():
            path = h.resolve(name)
            assert h.sha(path) == digest, name
            paths.append(path)
    assert h.read(SUPPORT / "initial_manifest.json")["all32_public_initials_frozen"]
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    demographics = []
    all_records = []
    for task in summary["tasks"]:
        if not task["prequalified"]:
            continue
        iid = task["instance_id"]
        path = SUPPORT / "demonstrations" / f"{iid}.json"
        record = h.read(path)
        assert record["prequalified"] and record["quality_verified"] is False and record["role"] == "source"
        assert record["instance_id"] == iid and record["protocol_sha256"] == h.sha(SUPPORT / "protocol.json")
        init = h.read(SUPPORT / "initials" / f"{iid}.json")
        assert k.replay(OLD / "public" / iid, init, record, tok)["all_checks_passed"]
        rows = record["rows"]
        turns = len(rows)
        demographics.append(
            dict(
                instance_id=iid,
                turns=turns,
                max_sequence_tokens=max(r["sequence_tokens"] for r in rows),
                total_sequence_tokens=sum(r["sequence_tokens"] for r in rows),
                total_sequence_squared=sum(r["sequence_tokens"] ** 2 for r in rows),
                max_completion_tokens=max(r["completion_tokens"] for r in rows),
            )
        )
        for index, row in enumerate(rows):
            assert row["labels"] == [-100] * row["prompt_tokens"] + row["completion_token_ids"]
            all_records.append(
                dict(
                    instance_id=iid,
                    row_id=f"{iid}:turn{index + 1:02d}",
                    turn_index=index + 1,
                    kind=row["kind"],
                    task_turns=turns,
                    prompt_token_ids=row["prompt_token_ids"],
                    target_token_ids=row["completion_token_ids"],
                    demonstration_sha256=h.sha(path),
                    quality_verified=False,
                )
            )
        paths.append(path)
    selected = select_tasks(demographics)
    chosen = []
    for iid in selected:
        for row in all_records:
            if row["instance_id"] == iid:
                chosen.append(dict(row, loss_weight=1.0 / (len(selected) * row["task_turns"])))
    validate_weights(chosen, selected)
    assert len(demographics) == summary["prequalified"]
    assert max(len(r["prompt_token_ids"]) + len(r["target_token_ids"]) for r in all_records) <= 26624
    model_files = sorted(
        p for p in MODEL.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja", ".txt"}
    )
    index = h.read(MODEL / "model.safetensors.index.json")
    assert all(MODEL / n in model_files for n in set(index["weight_map"].values()))
    paths += model_files + [Path(m.__file__) for m in [qwen, sdpa, lora, peft_io, adamw]]
    if (OUT / "static_review.json").exists():
        paths.append(OUT / "static_review.json")
    inputs = {str(p): h.sha(p) for p in sorted(set(paths))}
    software = {
        n: importlib.metadata.version(n)
        for n in ["torch", "transformers", "peft", "safetensors", "accelerate", "tokenizers"]
    }
    hardware = h.gpu_identity()
    OUT.mkdir(exist_ok=True)
    h.save(OUT / "smoke_records.json", chosen)
    inputs[str(OUT / "smoke_records.json")] = h.sha(OUT / "smoke_records.json")
    h.save(
        OUT / "protocol.json",
        dict(
            utc=h.utc(),
            scope="Constructed source demonstration technical memory/numerics/checkpoint smoke only; quality unverified",
            interface="public_outline_read_then_canonical_patch",
            inputs_sha256=inputs,
            software=software,
            hardware=hardware,
            model=str(MODEL),
            seed=0,
            cuda_device_order="PCI_BUS_ID",
            torch_device_uuid_must_match_frozen_physical_gpu2=True,
            chosen_prompt_cap=24576,
            initial_prompt_cap=12288,
            max_prompt_tokens=24576,
            max_action_tokens=256,
            max_completion_tokens=2048,
            max_sequence_tokens=26624,
            prequalified_source_ids=[r["instance_id"] for r in demographics],
            smoke_ids=selected,
            smoke_row_ids=[r["row_id"] for r in chosen],
            source_demographics=demographics,
            all_source_assistant_rows=len(all_records),
            all_source_causal_token_exposures=sum(
                len(r["prompt_token_ids"]) + len(r["target_token_ids"]) for r in all_records
            ),
            selection="Union of longest actual sequence, greatest sum(sequence_length squared) task, longest completion task; ties by instance ID. ALL chronological assistant rows for each chosen task.",
            objective="Mean over source tasks of mean over assistant turns of mean current-completion-token NLL including EOS; only current assistant completion supervised. Rowweight1/(smoke_tasks*turns_in_task).",
            steady_state_probe="Same task-balanced rows backward with initialized Adam state resident; no second optimizer step; clear gradients before reset.",
            optimizer=dict(name="AdamW", lr=1e-4, betas=[0.9, 0.999], eps=1e-8, weight_decay=0),
            adapter=dict(layers=36, targets=["q_proj", "v_proj"], r=8, alpha=16, dropout=0, parameters=3833856),
            precision=dict(
                base="BF16",
                adapter_parameters="FP32",
                adapter_compute="FP32 A/B matmuls, result cast to BF16",
                optimizer_state="FP32",
                loss="FP32 crossentropy",
                autocast=False,
                tf32=False,
                attention="SDPA FLASH_ATTENTION only; no fallback",
            ),
            checkpointing="nonreentrant",
            use_cache=False,
            logits="Only current completion prediction positions via tensor logits_to_keep",
            validation=[
                "Every selected causal native row/mask replayed",
                "Equal total task gradient weight despite different turn counts",
                "Zero-adapter probe exact",
                "Finite gradients/nonzero delta",
                "Resident-Adam second backward",
                "Exact standard PEFT save/reload",
                "Exact original adapter reset/probe",
                "Frozen BF16 base parameter hashes/storage/version/dtype",
            ],
            total_shared_gpu_budget_seconds=LIMIT,
            prior_gpu_seconds=0,
            external_timeout=dict(
                required=True,
                term_seconds=3595,
                kill_grace_seconds=5,
                hard_ceiling_seconds=3600,
                launcher="scripts/129_swe_localize_smoke_launcher.py",
            ),
            accounting="Single-use smoke plus any future training share3600seconds; charge max(internal elapsed, external terminal duration), no retry reset. Quality CPU work separately bounded.",
            future_estimate="1.20*(max observed row seconds*3*all_source_rows + measured optimizer overhead*3*source_tasks)+120 load/save reserve; descriptive conservative estimate, not training approval.",
            full_training=False,
            inference=False,
            quality_verification=False,
            evaluation_private_access=False,
            api_spend_usd=0,
        ),
    )
    print(
        json.dumps(
            dict(
                frozen=True,
                smoke_ids=selected,
                smoke_rows=len(chosen),
                task_weights={
                    iid: sum(r["loss_weight"] for r in chosen if r["instance_id"] == iid) for iid in selected
                },
            )
        ),
        flush=True,
    )


def run():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    assert os.environ.get("SWE_SMOKE_EXTERNAL_TIMEOUT") == "3595+5", (
        "Use frozen external process-group timeout and set acknowledgment"
    )
    work = OUT / "run"
    work.mkdir()  # Failed or killed attempts remain permanently consumed.
    started = time.monotonic()
    torch = None

    def interrupted(signum, frame):
        raise TimeoutError("Shared60minute smoke/full budget or external timeout; no fallback/retry")

    old_alarm = signal.signal(signal.SIGALRM, interrupted)
    old_term = signal.signal(signal.SIGTERM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 3595)
    try:
        p = h.read(OUT / "protocol.json")
        h.save(
            work / "started.json",
            dict(utc=h.utc(), protocol_sha256=h.sha(OUT / "protocol.json"), external_timeout=p["external_timeout"]),
        )
        for name, digest in p["inputs_sha256"].items():
            assert h.sha(name) == digest, name
        assert {n: importlib.metadata.version(n) for n in p["software"]} == p["software"]
        assert h.gpu_identity() == p["hardware"]
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from peft.tuners.lora.layer import LoraLayer
        from safetensors.torch import load_file
        from torch.nn.attention import sdpa_kernel, SDPBackend

        assert torch.cuda.device_count() == 1
        torch_uuid = str(torch.cuda.get_device_properties(0).uuid)

        def normalize_uuid(value):
            return str(value).lower().removeprefix("gpu-")

        assert normalize_uuid(torch_uuid) == normalize_uuid(p["hardware"]["uuid"]), (
            "CUDA logical0 is not frozen physical GPU2"
        )
        h.save(
            work / "device_identity.json",
            dict(
                cuda_device_order=os.environ["CUDA_DEVICE_ORDER"],
                visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                torch_logical0_uuid=torch_uuid,
                physical_gpu2_uuid=p["hardware"]["uuid"],
                matched=True,
            ),
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.reset_peak_memory_stats()
        records = h.read(OUT / "smoke_records.json")
        assert [r["row_id"] for r in records] == p["smoke_row_ids"]
        validate_weights(records, p["smoke_ids"])
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL),
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        ).eval()
        model.config.use_cache = False

        def forward(r, probe=False):
            start = len(r["prompt_token_ids"])
            ids = r["prompt_token_ids"] + r["target_token_ids"]
            tokens = torch.tensor([ids], device="cuda", dtype=torch.long)
            keep = torch.arange(start - 1, start if probe else len(ids) - 1, device="cuda")
            logits = model(input_ids=tokens, use_cache=False, logits_to_keep=keep).logits
            assert logits.shape[1] == (1 if probe else len(r["target_token_ids"]))
            if probe:
                return logits.detach().cpu()
            return torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), tokens[:, start:].reshape(-1)
            ), str(logits.dtype)

        def probe():
            model.eval()
            with torch.no_grad():
                return forward(records[0], True)

        def tensor_hash(t):
            return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        def frozen_hash():
            digest = hashlib.sha256()
            for name, v in model.named_parameters():
                if v.requires_grad:
                    continue
                digest.update(json.dumps([name, list(v.shape), str(v.dtype)]).encode())
                for chunk in v.detach().reshape(-1).split(8 * 1024 * 1024):
                    digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            original = probe()
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0,
                    target_modules=["q_proj", "v_proj"],
                    bias="none",
                ),
            )
            params = {n: v for n, v in model.named_parameters() if v.requires_grad}
            assert sum(v.numel() for v in params.values()) == 3833856
            modules = [m for m in model.modules() if isinstance(m, LoraLayer)]
            assert len(modules) == 72
            for v in params.values():
                v.data = v.data.float()
            assert all(v.dtype == torch.float32 for v in params.values())
            base = {
                n: (v.data_ptr(), v._version, str(v.dtype)) for n, v in model.named_parameters() if not v.requires_grad
            }
            assert all(dtype == "torch.bfloat16" for _, _, dtype in base.values())
            initial = {n: v.detach().cpu().clone() for n, v in params.items()}
            assert all(torch.count_nonzero(v) == 0 for n, v in initial.items() if ".lora_B." in n)
            torch.save(initial, work / "initial_adapter.pt")
            assert torch.equal(original, probe()), "Zero adapter changes base probe"
            before_hash = frozen_hash()
            trace = {}

            def layer_hook(module, args, result):
                trace["layer"] = dict(
                    input=str(args[0].dtype), output=str(result.dtype), base_weight=str(module.base_layer.weight.dtype)
                )

            def a_hook(module, args, result):
                trace["A"] = dict(input=str(args[0].dtype), output=str(result.dtype), weight=str(module.weight.dtype))

            def b_hook(module, args, result):
                trace["B"] = dict(input=str(args[0].dtype), output=str(result.dtype), weight=str(module.weight.dtype))

            hooks = [
                modules[0].register_forward_hook(layer_hook),
                modules[0].lora_A["default"].register_forward_hook(a_hook),
                modules[0].lora_B["default"].register_forward_hook(b_hook),
            ]
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            opt = torch.optim.AdamW(params.values(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            assert len(opt.state) == 0
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            update_start = time.monotonic()
            row_times = []
            first_row_seconds = 0.0
            with (work / "smoke_items.jsonl").open("x") as f:
                for r in records:
                    torch.cuda.synchronize()
                    row_start = time.monotonic()
                    loss, logit_dtype = forward(r)
                    assert torch.isfinite(loss)
                    (loss * r["loss_weight"]).backward()
                    torch.cuda.synchronize()
                    row_seconds = time.monotonic() - row_start
                    row_times.append(row_seconds)
                    f.write(
                        json.dumps(
                            dict(
                                instance_id=r["instance_id"],
                                row_id=r["row_id"],
                                turn_index=r["turn_index"],
                                row_kind=r["kind"],
                                row_seconds=row_seconds,
                                nll=float(loss.detach()),
                                prompt_tokens=len(r["prompt_token_ids"]),
                                completion_tokens=len(r["target_token_ids"]),
                                source_tasks_in_accumulation=len(p["smoke_ids"]),
                                turns_in_task=r["task_turns"],
                                loss_weight=r["loss_weight"],
                                logit_dtype=logit_dtype,
                                loss_dtype=str(loss.dtype),
                                quality_verified=False,
                            )
                        )
                        + "\n"
                    )
                    f.flush()
                    del loss
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
            gradnorm = float(torch.sqrt(sum(v.grad.square().sum() for v in params.values())))
            assert gradnorm > 0
            opt.step()
            assert all(torch.isfinite(v).all() for v in params.values())
            assert all(
                s.dtype == torch.float32 for state in opt.state.values() for s in state.values() if torch.is_tensor(s)
            )
            torch.cuda.synchronize()
            update_seconds = time.monotonic() - update_start
            first_row_seconds = sum(row_times)
            # Second backward checks peak memory with Adam state resident, without a second update.
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            steady_start = time.monotonic()
            with (work / "steady_state_items.jsonl").open("x") as f:
                for r in records:
                    torch.cuda.synchronize()
                    row_start = time.monotonic()
                    loss, logit_dtype = forward(r)
                    assert torch.isfinite(loss)
                    (loss * r["loss_weight"]).backward()
                    torch.cuda.synchronize()
                    row_seconds = time.monotonic() - row_start
                    row_times.append(row_seconds)
                    f.write(
                        json.dumps(
                            dict(
                                instance_id=r["instance_id"],
                                row_id=r["row_id"],
                                turn_index=r["turn_index"],
                                row_kind=r["kind"],
                                row_seconds=row_seconds,
                                nll=float(loss.detach()),
                                prompt_tokens=len(r["prompt_token_ids"]),
                                completion_tokens=len(r["target_token_ids"]),
                                source_tasks_in_accumulation=len(p["smoke_ids"]),
                                turns_in_task=r["task_turns"],
                                loss_weight=r["loss_weight"],
                                optimizer_states_resident=True,
                                optimizer_step_performed=False,
                                logit_dtype=logit_dtype,
                                loss_dtype=str(loss.dtype),
                                quality_verified=False,
                            )
                        )
                        + "\n"
                    )
                    f.flush()
                    del loss
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
            torch.cuda.synchronize()
            steady_seconds = time.monotonic() - steady_start
            opt.zero_grad(set_to_none=True)
            changed = {n: v.detach().cpu().clone() for n, v in params.items()}
            delta = float(torch.sqrt(sum((changed[n] - initial[n]).square().sum() for n in params)))
            assert delta > 0
            updated = probe()
            assert (
                trace["A"] == trace["B"] == dict(input="torch.float32", output="torch.float32", weight="torch.float32")
            )
            assert trace["layer"] == dict(input="torch.bfloat16", output="torch.bfloat16", base_weight="torch.bfloat16")
            model.save_pretrained(work / "updated_adapter", safe_serialization=True)
            saved = load_file(str(work / "updated_adapter/adapter_model.safetensors"))
            state = get_peft_model_state_dict(model)
            assert saved.keys() == state.keys() and all(
                torch.equal(saved[n], v.detach().cpu()) for n, v in state.items()
            )
            del state
            with torch.no_grad():
                for v in params.values():
                    v.zero_()
            set_peft_model_state_dict(model, saved, adapter_name="default")
            assert all(torch.equal(v.detach().cpu(), changed[n]) for n, v in params.items())
            assert torch.equal(probe(), updated), "Reload probe differs"
            with torch.no_grad():
                for n, v in params.items():
                    v.copy_(initial[n].to(v.device))
            assert all(torch.equal(v.detach().cpu(), initial[n]) for n, v in params.items())
            assert torch.equal(probe(), original), "Initial reset probe differs"
            assert base == {
                n: (v.data_ptr(), v._version, str(v.dtype)) for n, v in model.named_parameters() if not v.requires_grad
            }
            after_hash = frozen_hash()
            assert before_hash == after_hash
            for hook in hooks:
                hook.remove()
            del opt
            for v in params.values():
                v.grad = None
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            optimizer_overhead = max(0.0, update_seconds - first_row_seconds)
            estimate = (
                1.20
                * (
                    max(row_times) * (3 * p["all_source_assistant_rows"])
                    + optimizer_overhead * (3 * len(p["prequalified_source_ids"]))
                )
                + 120
            )
            h.save(
                work / "summary.json",
                dict(
                    utc=h.utc(),
                    technical_checks_passed=True,
                    quality_verified=False,
                    full_training_run=False,
                    inference_run=False,
                    evaluation_private_access=False,
                    smoke_ids=p["smoke_ids"],
                    selected_prompt_cap=p["chosen_prompt_cap"],
                    accumulation_examples=len(p["smoke_ids"]),
                    accumulation_rows=len(records),
                    accumulation_tasks=len(p["smoke_ids"]),
                    task_balanced_per_turn_loss=True,
                    smoke_row_ids=p["smoke_row_ids"],
                    gradient_norm=gradnorm,
                    adapter_delta_l2=delta,
                    actual_dtypes=trace,
                    base_probe_sha256=tensor_hash(original),
                    updated_probe_sha256=tensor_hash(updated),
                    probe_max_abs_change=float((updated.float() - original.float()).abs().max()),
                    zero_update_exact=True,
                    standard_peft_saved_reloaded_exact=True,
                    initial_reset_exact=True,
                    frozen_base_parameter_sha256_before=before_hash,
                    frozen_base_parameter_sha256_after=after_hash,
                    base_storage_version_dtype_unchanged=True,
                    smoke_update_seconds=update_seconds,
                    steady_state_probe_seconds=steady_seconds,
                    steady_state_optimizer_states_resident=True,
                    optimizer_steps=1,
                    gradient_example_exposures=2 * len(records),
                    steady_state_extra_optimizer_steps=0,
                    gradient_completion_token_exposures=2 * sum(len(r["target_token_ids"]) for r in records),
                    max_observed_row_seconds=max(row_times),
                    optimizer_overhead_seconds=optimizer_overhead,
                    future_source_assistant_rows=p["all_source_assistant_rows"],
                    future_compute_reserve_factor=1.20,
                    future_load_save_reserve_seconds=120,
                    total_elapsed_seconds=elapsed,
                    shared_budget_seconds=LIMIT,
                    remaining_seconds_before_external_charge=LIMIT - elapsed,
                    future_max_prequalified_dose_estimate_seconds=estimate,
                    future_max_dose_fits_internal_budget=estimate <= LIMIT - elapsed,
                    external_terminal_duration_required_before_future_full=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    cuda_runtime=torch.version.cuda,
                    device=torch.cuda.get_device_name(0),
                    api_spend_usd=0,
                    interpretation="Technical task-balanced outline/read/patch memory/numerics/save-reset check only; targets are not execution-quality verified. Future full training requires source-quality dose and all20evaluation runtimes plus remaining shared budget.",
                ),
            )
            h.save(work / "completed_manifest.json", {str(f): h.sha(f) for f in work.rglob("*") if f.is_file()})
    except BaseException as error:
        h.save(
            work / "failure.json",
            dict(
                utc=h.utc(),
                type=type(error).__name__,
                message=str(error),
                total_elapsed_seconds=time.monotonic() - started,
                shared_budget_seconds=LIMIT,
                retry=False,
                fallback=False,
                external_terminal_duration_required=True,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["checks", "freeze", "run"])
    args = parser.parse_args()
    if args.stage == "checks":
        objective_checks()
    else:
        globals()[args.stage]()
