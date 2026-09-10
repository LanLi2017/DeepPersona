#!/usr/bin/env python3
"""Corrected-code positive control: prepare, held-in headroom, and technical smoke only."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
import signal
import time

ROOT = Path("runs/swe-diversity-selection")
OUT = ROOT / "code-corrected-sft-control"
AUDIT = ROOT / "code-extraction-audit"
SEED = 20260909
EVAL_HELPER = Path("scripts/62_code_functional_eval.py")
spec = importlib.util.spec_from_file_location("control_eval62", EVAL_HELPER)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
spec = importlib.util.spec_from_file_location("control_grader52", e.HELPER)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(line) for line in p.open()]


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def jsonl(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def prepare():
    from transformers import AutoTokenizer

    assert not OUT.exists()
    outcomes = {r["record_id"]: r for r in lines(AUDIT / "outcomes.jsonl") if r["split"] == "train"}
    sources = {r["record_id"]: r for r in lines(AUDIT / "inputs.jsonl") if r["split"] == "train"}
    prompts = {}
    targets = {}
    paths = [
        Path(__file__),
        EVAL_HELPER,
        e.HELPER,
        AUDIT / "outcomes.jsonl",
        AUDIT / "inputs.jsonl",
        AUDIT / "task_eligibility.json",
    ]
    for pool in ["code-learning-pilot", "code-learning-expansion"]:
        folder = ROOT / pool
        paths.extend(folder / n for n in ["inputs.jsonl", "evaluator_targets.jsonl"])
        for r in lines(folder / "inputs.jsonl"):
            if r["split"] == "train":
                prompts[r["task_id"]] = r["prompt"]
        for r in lines(folder / "evaluator_targets.jsonl"):
            if r["split"] == "train":
                targets[r["task_id"]] = r
    tasks = sorted(
        {r["task_id"] for r in outcomes.values() if r["first_correct"]},
        key=lambda t: key(f"{SEED}:corrected-sft-order:{t}"),
    )
    assert len(tasks) == 94
    tok = AutoTokenizer.from_pretrained(e.MODEL, local_files_only=True)
    training = []
    for task in tasks:
        candidates = [r for r in outcomes.values() if r["task_id"] == task and r["first_correct"]]
        choice = min(candidates, key=lambda r: key(f"{SEED}:corrected-sft-choice:{r['record_id']}"))
        source = sources[choice["record_id"]]
        code = source["first_code"].strip() + "\n"
        completion = "```python\n" + code + "```"
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompts[task]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = tok.encode(prompt, add_special_tokens=False)
        suffix = tok.encode(completion, add_special_tokens=False) + [tok.eos_token_id]
        assert len(suffix) <= 768 and len(prefix) + len(suffix) <= 2048
        training.append(
            dict(
                task_id=task,
                record_id=choice["record_id"],
                prompt=prompts[task],
                rendered_prompt=prompt,
                completion=completion,
                prompt_token_ids=prefix,
                completion_token_ids=suffix,
                source_first_code_sha256=key(source["first_code"]),
            )
        )
    mixed = [r["task_id"] for r in read(AUDIT / "task_eligibility.json") if 0 < r["first_correct"] < 8]
    assert len(mixed) == 30
    probes = sorted(mixed, key=lambda t: key(f"{SEED}:corrected-sft-headroom:{t}"))[:16]
    smoke = sorted(
        training, key=lambda r: (-(len(r["prompt_token_ids"]) + len(r["completion_token_ids"])), r["task_id"])
    )[:2]
    OUT.mkdir()
    jsonl(OUT / "training_inputs.jsonl", training)
    jsonl(OUT / "headroom_inputs.jsonl", [dict(task_id=t, prompt=prompts[t]) for t in probes])
    e.save(OUT / "headroom_targets.json", {str(t): targets[t] for t in probes})
    old = read(ROOT / "code-learning-pilot/protocol.json")
    assert set(probes).isdisjoint(
        old["calibration_task_ids"] + old["measurement_task_ids"] + old["later_test_task_ids"]
    )
    paths.extend(OUT / n for n in ["training_inputs.jsonl", "headroom_inputs.jsonl", "headroom_targets.json"])
    e.save(
        OUT / "protocol.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            seed=SEED,
            model_snapshot=str(e.MODEL),
            model_revision="b968826d9c46dd6066d109eabc6255188de91218",
            dataset_revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
            software={name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]},
            hardware=subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    "2",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            source_rule="script75FIRST Python-or-unlabelled fence, otherwisefulltext; onepassing candidate/task byfrozenSHA, no reselection",
            training_tasks=tasks,
            headroom_task_ids=probes,
            smoke_task_ids=[r["task_id"] for r in smoke],
            smoke_choice="Two longest selected prompt+completion sequences, tie taskID; oneupdate withmean of2per-example NLLs",
            adapter=dict(
                all_decoder_layers=36,
                modules=["q_proj", "v_proj"],
                rank=8,
                alpha=16,
                dropout=0,
                trainable_parameters=3833856,
            ),
            precision="FullFP32,TF32off,SDPA; gradientcheckpointing use_reentrant=False; noOOMfallback",
            optimizer=dict(name="AdamW", learning_rate=0.0001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0),
            planned_training=dict(
                epochs=3,
                microbatch=1,
                accumulation=4,
                epoch_order=tasks,
                steps_per_epoch=24,
                last_batch_size=2,
                final_batch_normalization="Mean overactual2examples, flush eachepoch",
                status="Notauthorizedtoexecuteuntilrootsmoke/headroomreview",
            ),
            generation=dict(do_sample=False, max_new_tokens=768, batch_size=4, enable_thinking=False),
            gates=dict(
                headroom="Baseheld-in16musthaveatleast1failure; otherwise stopfulltraining, ceilingnotlearnerfailure",
                technical="Finiteupdate,all72LoRAlayersactive,exactno-updatebasegreedytokens,exactsave/reloadlogits; cleanedselectedcodesreproduceFIRSTpass",
                later_feasibility="Atleast1netcorrectheld-inimprovement withno netcalibrationloss from19/21; practicalgate, notsignificance",
            ),
            calibration="No freshcalibrationscoring inthisstage; reusescript75base19/21",
            measurement_and_final="No measurement21 or final107 access",
            limits=dict(
                gpu=2,
                total_gpu_stage_seconds=3600,
                api_dollars=0,
                max_training_completion_tokens=216576,
                max_future_greedy_tokens=56832,
            ),
            scope="Correct-only learner-sensitivity control, notdiversity orheld-outgeneralization",
            input_sha256={str(p): e.sha(p) for p in paths},
            api_spend_usd=0,
        ),
    )
    checks = []
    for r in training:
        grade = g.grade(r["completion"], targets[r["task_id"]])
        assert grade["correct"], "Cleanedcodefailed; stop,noreselection"
        checks.append(dict(task_id=r["task_id"], record_id=r["record_id"], grade=grade))
    e.save(OUT / "cleaned_code_checks.json", checks)
    print(
        json.dumps(
            dict(
                training_tasks=94,
                headroom_ids=probes,
                smoke_ids=[r["task_id"] for r in smoke],
                training_completion_tokens=sum(len(r["completion_token_ids"]) for r in training),
                all_cleaned_targets_pass=True,
            )
        )
    )


def check():
    p = read(OUT / "protocol.json")
    for name, sha in p["input_sha256"].items():
        assert e.sha(Path(name)) == sha
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    elapsed = sum(
        read(OUT / f"{stage}_results.json")["elapsed_seconds"]
        for stage in ["headroom", "smoke"]
        if (OUT / f"{stage}_results.json").exists()
    )
    remaining = int(3600 - elapsed)
    assert remaining > 0

    def timeout(*_):
        raise TimeoutError("Frozen60GPU-minute budget exhausted")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(remaining)
    return p


def model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tok = AutoTokenizer.from_pretrained(e.MODEL, local_files_only=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            e.MODEL, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point())
    return model, tok


def headroom():
    import torch

    check()
    started = time.monotonic()
    dest = OUT / "headroom"
    assert not dest.exists()
    dest.mkdir()
    model, tok = model_and_tokenizer()
    model.config.use_cache = True
    records = e.greedy(model, tok, lines(OUT / "headroom_inputs.jsonl"), 4, dest, "base")
    targets = read(OUT / "headroom_targets.json")
    grades = []
    for r in records:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", r["generation"], re.S)
        code = blocks[0] if blocks else r["generation"]
        wrapped = "```python\n" + code + "```" if blocks else code
        grades.append(dict(task_id=r["task_id"], **g.grade(wrapped, targets[str(r["task_id"])])))
    jsonl(dest / "grades_first_block.jsonl", grades)
    result = dict(
        tasks=16,
        correct=sum(r["correct"] for r in grades),
        headroom_gate_passed=sum(r["correct"] for r in grades) < 16,
        elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        api_spend_usd=0,
        interpretation="Held-inmixed-trainingprobe, notgeneralization; FIRSTblockgrader",
    )
    e.save(OUT / "headroom_results.json", result)
    print(json.dumps(result), flush=True)


def smoke():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from peft.tuners.lora.layer import LoraLayer
    from safetensors.torch import load_file

    p = check()
    started = time.monotonic()
    dest = OUT / "smoke"
    assert not dest.exists()
    dest.mkdir()
    model, tok = model_and_tokenizer()
    model.config.use_cache = True
    probe = lines(OUT / "headroom_inputs.jsonl")[:2]
    baseline = e.greedy(model, tok, probe, 4, dest, "base")
    model = get_peft_model(
        model,
        LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0, target_modules=["q_proj", "v_proj"]),
    ).eval()
    parameters = {n: v for n, v in model.named_parameters() if v.requires_grad}
    assert sum(v.numel() for v in parameters.values()) == 3833856
    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    assert len(layers) == 72
    zero = e.greedy(model, tok, probe, 4, dest, "no_update")
    assert [r["generated_token_ids"] for r in baseline] == [r["generated_token_ids"] for r in zero]
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    initial = {n: v.detach().cpu().clone() for n, v in parameters.items()}
    torch.save(initial, dest / "initial_adapter.pt")
    chosen = {r["task_id"]: r for r in lines(OUT / "training_inputs.jsonl")}
    batch = [chosen[t] for t in p["smoke_task_ids"]]
    optimizer = torch.optim.AdamW(parameters.values(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for r in batch:
        tokens = torch.tensor([r["prompt_token_ids"] + r["completion_token_ids"]], device="cuda")
        start = len(r["prompt_token_ids"])
        logits = model(input_ids=tokens, use_cache=False).logits[:, start - 1 : -1, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, start:].reshape(-1))
        assert torch.isfinite(loss)
        (loss / len(batch)).backward()
        losses.append(float(loss.detach()))
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters.values())
    gradnorm = float(torch.sqrt(sum(v.grad.square().sum() for v in parameters.values())))
    e.save(
        dest / "step.json",
        dict(
            task_ids=p["smoke_task_ids"],
            record_ids=[r["record_id"] for r in batch],
            per_example_nll=losses,
            gradient_norm=gradnorm,
        ),
    )
    optimizer.step()
    assert all(torch.isfinite(v).all() for v in parameters.values())
    delta = float(torch.sqrt(sum((v.detach().cpu() - initial[n]).square().sum() for n, v in parameters.items())))
    assert delta > 0
    model.eval()
    model.gradient_checkpointing_disable()
    model.save_pretrained(dest / "adapter")
    saved = load_file(str(dest / "adapter/adapter_model.safetensors"))
    actual = get_peft_model_state_dict(model)
    assert actual.keys() == saved.keys() and all(torch.equal(v.detach().cpu(), saved[k]) for k, v in actual.items())
    r = batch[0]
    tokens = torch.tensor([r["prompt_token_ids"] + r["completion_token_ids"]], device="cuda")
    with torch.inference_mode():
        before = model(input_ids=tokens, use_cache=False).logits.cpu()
    base = model.unload()
    del model
    reloaded = PeftModel.from_pretrained(base, str(dest / "adapter"), is_trainable=False).eval()
    with torch.inference_mode():
        after = reloaded(input_ids=tokens, use_cache=False).logits.cpu()
    assert torch.equal(before, after)
    result = dict(
        technical_checks_passed=True,
        training_tasks=p["smoke_task_ids"],
        optimizer_steps=1,
        mean_preupdate_nll=sum(losses) / 2,
        gradient_norm=gradnorm,
        parameter_delta_l2=delta,
        trainable_parameters=3833856,
        lora_layers=72,
        no_update_greedy_exact=True,
        save_reload_logits_exact=True,
        elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        api_spend_usd=0,
        full_training_executed=False,
    )
    e.save(OUT / "smoke_results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "headroom", "smoke"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        started = time.monotonic()
        try:
            {"headroom": headroom, "smoke": smoke}[args.stage]()
        except Exception as error:
            e.save(
                OUT / f"{args.stage}_failure.json",
                dict(
                    error_type=type(error).__name__,
                    message=str(error),
                    elapsed_seconds=time.monotonic() - started,
                    no_fallback=True,
                ),
            )
            raise
