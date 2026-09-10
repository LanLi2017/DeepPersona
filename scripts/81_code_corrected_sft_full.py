#!/usr/bin/env python3
"""One frozen correct-only SFT positive control; held-in/calibration evaluation only."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import signal
import time

HELPER = Path("scripts/78_code_corrected_sft_control.py")
spec = importlib.util.spec_from_file_location("sft_control78", HELPER)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
e = c.e
OUT = c.OUT
DEST = OUT / "full"
BASE_CAL = c.ROOT / "code-functional-development/grid-evaluation/base/eval_raw.jsonl"


def prepare():
    assert not (OUT / "full_protocol.json").exists() and not DEST.exists()
    old = c.read(OUT / "protocol.json")
    for name, digest in old["input_sha256"].items():
        assert e.sha(Path(name)) == digest
    assert c.read(OUT / "headroom_results.json")["headroom_gate_passed"]
    assert c.read(OUT / "smoke_results.json")["technical_checks_passed"]
    assert c.read(OUT / "interface_definition_check.json")["all_required_public_functions_defined"]
    training = c.lines(OUT / "training_inputs.jsonl")
    headroom = c.lines(OUT / "headroom/base_raw.jsonl")
    calibration = c.lines(BASE_CAL)
    rows, _, _ = e.load_inputs([r["task_id"] for r in calibration])
    assert len(training) == 94 and len(rows) == 21 and len(headroom) == 16
    assert [r["task_id"] for r in training] == old["training_tasks"]
    assert [r["task_id"] for r in headroom] == old["headroom_task_ids"]
    ids = c.read(e.POOL / "protocol.json")
    assert set(old["training_tasks"]).isdisjoint(
        ids["calibration_task_ids"] + ids["measurement_task_ids"] + ids["later_test_task_ids"]
    )
    base_grades = {
        r["task_id"]: r["first_correct"] for r in c.lines(c.AUDIT / "outcomes.jsonl") if r["split"] == "calibration"
    }
    assert sum(base_grades.values()) == 19
    paths = [
        Path(__file__),
        HELPER,
        c.EVAL_HELPER,
        e.HELPER,
        BASE_CAL,
        c.AUDIT / "outcomes.jsonl",
        e.POOL / "inputs.jsonl",
        e.POOL / "evaluator_targets.jsonl",
        e.POOL / "protocol.json",
    ]
    paths += [
        OUT / n
        for n in [
            "protocol.json",
            "training_inputs.jsonl",
            "headroom_inputs.jsonl",
            "headroom_targets.json",
            "headroom/base_raw.jsonl",
            "headroom/grades_first_block.jsonl",
            "headroom_results.json",
            "smoke_results.json",
            "smoke/initial_adapter.pt",
            "cleaned_code_checks.json",
            "interface_definition_check.json",
        ]
    ]
    e.save(
        OUT / "full_protocol.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            authorization="Root authorized fixed full run after successful78 headroom/smoke; independent infrastructure code review before launch",
            source_protocol_sha256=e.sha(OUT / "protocol.json"),
            input_sha256={str(p): e.sha(p) for p in paths},
            initialization="Load exact78 smoke/initial_adapter.pt before any update; fresh AdamW, never smoke-trained tensors",
            training_tasks=old["training_tasks"],
            epochs=3,
            epoch_order="Same frozen94-task order each epoch",
            microbatch=1,
            accumulation=4,
            last_batch=2,
            optimizer_steps=72,
            item_exposures=282,
            objective="Mean per-example completion-token NLL including EOS; each accumulation batch normalized by its actual size; correct-only SFT",
            optimizer=old["optimizer"],
            adapter=old["adapter"],
            precision=old["precision"],
            seed=c.SEED,
            training_completion_tokens=3 * sum(len(r["completion_token_ids"]) for r in training),
            training_prompt_tokens=3 * sum(len(r["prompt_token_ids"]) for r in training),
            evaluation=dict(
                headroom_ids=old["headroom_task_ids"],
                calibration_ids=[r["task_id"] for r in calibration],
                batch_size=4,
                boundaries="Exact previous headroom and65 basecal order/batches/padded prompts",
                generation=old["generation"],
                rule="FIRST Python-or-unlabelled fenced block; fallback full text; unchanged52 grader",
                baseline_correct=dict(headroom=9, calibration=19),
            ),
            feasibility_gate="At least1 net correct held-in improvement and no net calibration loss from19/21; descriptive engineering gate, not statistical evidence",
            limits=dict(gpu=2, total_including78_seconds=3600, retry=False, api_dollars=0),
            no_measurement_or_final_scoring=True,
            scope="Learner sensitivity positive control; no diversity comparison or held-out generalization claim",
        ),
    )
    print("Frozen full control:94 tasks x3,72 updates;16 held-in +21 calibration")


def run():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from peft.tuners.lora.layer import LoraLayer
    from safetensors.torch import load_file

    p = c.read(OUT / "full_protocol.json")
    assert not DEST.exists(), "Immutable run, no retry or budget reset"
    for name, digest in p["input_sha256"].items():
        assert e.sha(Path(name)) == digest
    c.check()  # One global alarm subtracts completed78 stage time; no reset during81.
    started = time.monotonic()
    DEST.mkdir()
    e.save(
        DEST / "manifest.json",
        dict(
            protocol_sha256=e.sha(OUT / "full_protocol.json"),
            prior_gpu_seconds=sum(
                c.read(OUT / f"{stage}_results.json")["elapsed_seconds"] for stage in ["headroom", "smoke"]
            ),
            hardware=c.read(OUT / "protocol.json")["hardware"],
            software=c.read(OUT / "protocol.json")["software"],
            utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    model, tok = c.model_and_tokenizer()
    model = get_peft_model(
        model,
        LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0, target_modules=["q_proj", "v_proj"]),
    )
    parameters = {n: v for n, v in model.named_parameters() if v.requires_grad}
    initial = torch.load(OUT / "smoke/initial_adapter.pt", map_location="cpu", weights_only=True)
    assert parameters.keys() == initial.keys() and sum(v.numel() for v in parameters.values()) == 3833856
    assert len([m for m in model.modules() if isinstance(m, LoraLayer)]) == 72
    with torch.no_grad():
        for n, v in parameters.items():
            v.copy_(initial[n].to(v.device))
    assert all(torch.equal(v.detach().cpu(), initial[n]) for n, v in parameters.items())
    base_parameters = {n: (v.data_ptr(), v._version) for n, v in model.named_parameters() if not v.requires_grad}
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    training = c.lines(OUT / "training_inputs.jsonl")
    step = 0
    with (DEST / "training_items.jsonl").open("w") as items, (DEST / "training_steps.jsonl").open("w") as steps:
        for epoch in range(3):
            for offset in range(0, 94, 4):
                batch = training[offset : offset + 4]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for r in batch:
                    tokens = torch.tensor([r["prompt_token_ids"] + r["completion_token_ids"]], device="cuda")
                    start = len(r["prompt_token_ids"])
                    logits = model(input_ids=tokens, use_cache=False).logits[:, start - 1 : -1, :]
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), tokens[:, start:].reshape(-1)
                    )
                    assert torch.isfinite(loss)
                    (loss / len(batch)).backward()
                    value = float(loss.detach())
                    losses.append(value)
                    items.write(
                        json.dumps(
                            dict(
                                epoch=epoch + 1,
                                step=step + 1,
                                task_id=r["task_id"],
                                record_id=r["record_id"],
                                nll=value,
                                prompt_tokens=start,
                                completion_tokens=len(r["completion_token_ids"]),
                                accumulation_divisor=len(batch),
                            )
                        )
                        + "\n"
                    )
                    items.flush()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters.values())
                gradnorm = float(torch.sqrt(sum(v.grad.square().sum() for v in parameters.values())))
                optimizer.step()
                assert all(torch.isfinite(v).all() for v in parameters.values())
                step += 1
                row = dict(
                    step=step,
                    epoch=epoch + 1,
                    task_ids=[r["task_id"] for r in batch],
                    mean_nll=sum(losses) / len(losses),
                    gradient_norm=gradnorm,
                    elapsed_seconds=time.monotonic() - started,
                )
                steps.write(json.dumps(row) + "\n")
                steps.flush()
                if step % 8 == 0:
                    print(json.dumps(row), flush=True)
    assert step == 72
    assert {
        n: (v.data_ptr(), v._version) for n, v in model.named_parameters() if not v.requires_grad
    } == base_parameters
    delta = float(torch.sqrt(sum((v.detach().cpu() - initial[n]).square().sum() for n, v in parameters.items())))
    model.eval()
    model.gradient_checkpointing_disable()
    model.save_pretrained(DEST / "adapter")
    saved = load_file(str(DEST / "adapter/adapter_model.safetensors"))
    state = get_peft_model_state_dict(model)
    assert saved.keys() == state.keys() and all(torch.equal(v.detach().cpu(), saved[n]) for n, v in state.items())
    del optimizer, logits, loss, parameters, state
    base = model.unload()
    del model
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(base, str(DEST / "adapter"), is_trainable=False).eval()
    assert all(v.dtype == torch.float32 for v in model.parameters() if v.is_floating_point())
    assert all(torch.equal(v.detach().cpu(), saved[n]) for n, v in get_peft_model_state_dict(model).items())
    model.config.use_cache = True
    cal_rows, cal_targets, _ = e.load_inputs(p["evaluation"]["calibration_ids"])
    head_rows = c.lines(OUT / "headroom_inputs.jsonl")
    head_targets = {int(k): v for k, v in c.read(OUT / "headroom_targets.json").items()}
    audit = {
        r["task_id"]: r["first_correct"] for r in c.lines(c.AUDIT / "outcomes.jsonl") if r["split"] == "calibration"
    }
    results = {}
    for split, rows, targets, old_raw, old_grades in [
        (
            "headroom",
            head_rows,
            head_targets,
            c.lines(OUT / "headroom/base_raw.jsonl"),
            {r["task_id"]: r["correct"] for r in c.lines(OUT / "headroom/grades_first_block.jsonl")},
        ),
        ("calibration", cal_rows, cal_targets, c.lines(BASE_CAL), audit),
    ]:
        dest = DEST / split
        dest.mkdir()
        records = e.greedy(model, tok, rows, 4, dest, "trained")
        grades = []
        for r, old in zip(records, old_raw):
            for field in [
                "task_id",
                "batch_index",
                "batch_task_ids",
                "prompt_token_ids",
                "padded_prompt_token_ids",
                "attention_mask",
            ]:
                assert r[field] == old[field], field
            blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", r["generation"], re.S)
            wrapped = "```python\n" + blocks[0] + "```" if blocks else r["generation"]
            grade = c.g.grade(wrapped, targets[r["task_id"]])
            grades.append(
                dict(
                    task_id=r["task_id"],
                    **grade,
                    base_correct=old_grades[r["task_id"]],
                    generated_tokens=len(r["generated_token_ids"]),
                    base_generated_tokens=len(old["generated_token_ids"]),
                    exact_output_same=r["generated_token_ids"] == old["generated_token_ids"],
                )
            )
        c.jsonl(dest / "grades_first_block.jsonl", grades)
        results[split] = dict(
            tasks=len(grades),
            correct=sum(r["correct"] for r in grades),
            base_correct=sum(r["base_correct"] for r in grades),
            gained=[r["task_id"] for r in grades if r["correct"] and not r["base_correct"]],
            lost=[r["task_id"] for r in grades if not r["correct"] and r["base_correct"]],
            outputs_changed=sum(not r["exact_output_same"] for r in grades),
            generated_tokens=sum(r["generated_tokens"] for r in grades),
        )
        e.save(dest / "summary.json", results[split])
    elapsed = time.monotonic() - started
    prior = c.read(DEST / "manifest.json")["prior_gpu_seconds"]
    summary = dict(
        technical_checks_passed=True,
        exact_original_initialization=True,
        frozen_base_parameters_unchanged=True,
        exact_saved_and_reloaded_adapter=True,
        epochs=3,
        steps=72,
        item_exposures=282,
        training_completion_tokens=p["training_completion_tokens"],
        parameter_delta_l2=delta,
        evaluation=results,
        feasibility_gate_passed=results["headroom"]["correct"] >= 10 and results["calibration"]["correct"] >= 19,
        elapsed_seconds=elapsed,
        total_gpu_stage_seconds=prior + elapsed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        api_spend_usd=0,
        measurement_or_final_scored=False,
        interpretation="Correct-only SFT learner-sensitivity control; held-in and development calibration outcomes, no diversity or generalization claim",
    )
    e.save(DEST / "summary.json", summary)
    e.save(
        DEST / "completed_manifest.json",
        dict(sha256={str(f.relative_to(DEST)): e.sha(f) for f in DEST.rglob("*") if f.is_file()}),
    )
    signal.alarm(0)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        started = time.monotonic()
        try:
            run()
        except Exception as error:
            if DEST.exists() and not (DEST / "failure.json").exists():
                e.save(
                    DEST / "failure.json",
                    dict(
                        error_type=type(error).__name__,
                        message=str(error),
                        elapsed_seconds=time.monotonic() - started,
                        no_fallback=True,
                        no_retry=True,
                    ),
                )
            raise
