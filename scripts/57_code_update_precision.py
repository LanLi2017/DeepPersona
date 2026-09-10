#!/usr/bin/env python3
"""FP32 precision replication of frozen adapter-update groups."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

POOL = Path("runs/swe-diversity-selection/code-learning-pilot")
ORIGINAL = Path("runs/swe-diversity-selection/code-update-transfer")
OUT = Path("runs/swe-diversity-selection/code-update-transfer-fp32")
MODEL = Path(
    "/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
SEED = 20260909


def read(path):
    return [json.loads(line) for line in path.open()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "learner_protocol.json").exists()
    for name in ["groups.json", "group_protocol.json", "records.jsonl"]:
        (OUT / name).write_bytes((ORIGINAL / name).read_bytes())
    assert (
        json.loads(Path("runs/swe-diversity-selection/learning-update-smoke/results.json").read_text())["status"]
        == "pass"
    )
    save(
        OUT / "learner_protocol.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "model_snapshot": str(MODEL),
            "gpu": 3,
            "precision": "Full model FP32, TF32 disabled; replicate unchanged groups and steps after BF16 smoke showed threshold behavior. Same-measurement gradient is numerical sanity only, not a predictive endpoint.",
            "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"]},
            "adapter": {"modules": ["q_proj", "v_proj"], "last_layer_only": True, "rank": 8, "alpha": 16, "dropout": 0},
            "objective": "Completion-token mean NLL gradients g_i; balanced group H=mean((reward-.5)*g_i). Finite SGD theta_new=theta-eta*H, reset each time. A length-normalized centered-reward surrogate, not a full GRPO/PPO training claim.",
            "primary_step": 0.1,
            "sensitivity_step": 1.0,
            "norm_matched_update_l2": 0.02,
            "calibration": "Mean NLL gradient of21 separate calibration references; used only for predictive alignment.",
            "measurement": "Actual finite-step NLL loss change on21 disjoint measurement references; no final test or functional performance claim.",
            "input_hashes": {
                str(p): sha(p)
                for p in [
                    OUT / "groups.json",
                    ORIGINAL / "initial_adapter.pt",
                    POOL / "raw.jsonl",
                    POOL / "inputs.jsonl",
                    POOL / "evaluator_targets.jsonl",
                ]
            },
            "runner_sha256": sha(Path(__file__)),
            "api_cost_usd": 0,
            "smoke": "First fixed matched pair, all its unique rollouts, first2 calibration and first2 measurement tasks; technical checks before full run.",
        },
    )
    print("Learner protocol frozen before gradients or transfer outcomes.")


def run(smoke):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3"
    protocol = json.loads((OUT / "learner_protocol.json").read_text())
    assert sha(Path(__file__)) == protocol["runner_sha256"]
    for name, expected in protocol["input_hashes"].items():
        assert sha(Path(name)) == expected
    if not smoke:
        assert json.loads((OUT / "smoke_learner_summary.json").read_text())["technical_checks_passed"]
    group_input = json.loads((OUT / "groups.json").read_text())
    groups = group_input["groups"]
    if smoke:
        pair = groups[0]["matched_pair_id"]
        groups = [g for g in groups if g["matched_pair_id"] == pair]
    ids = sorted({key for g in groups for key in g["record_ids"]})
    raw = {f"{r['task_id']}:{r['sample']}": r for r in read(POOL / "raw.jsonl")}
    inputs = {r["task_id"]: r for r in read(POOL / "inputs.jsonl")}
    targets = read(POOL / "evaluator_targets.jsonl")
    calibration = sorted([r for r in targets if r["split"] == "calibration"], key=lambda r: r["task_id"])
    measurement = sorted([r for r in targets if r["split"] == "measurement"], key=lambda r: r["task_id"])
    if smoke:
        calibration, measurement = calibration[:2], measurement[:2]
    assert not {r["task_id"] for r in calibration} & {r["task_id"] for r in measurement}
    assert not {raw[i]["task_id"] for i in ids} & {r["task_id"] for r in calibration + measurement}
    prefix = "smoke_" if smoke else ""
    assert not (OUT / f"{prefix}learner_summary.json").exists(), "Run complete; use saved outputs"
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa"
    ).cuda()
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=[model.config.num_hidden_layers - 1],
        ),
    ).eval()
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    fixed_initial = torch.load(ORIGINAL / "initial_adapter.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(fixed_initial[name])
    initial = {n: p.detach().clone() for n, p in parameters}
    torch.save({n: p.cpu() for n, p in initial.items()}, OUT / f"{prefix}initial_adapter.pt")
    arrays = {}
    records = {}

    def batch(prompt, completion):
        assert completion and len(prompt) + len(completion) <= 2048
        return torch.tensor([prompt + completion], device="cuda"), len(prompt)

    for key in ids:
        r = raw[key]
        assert tok.encode(r["rendered_prompt"], add_special_tokens=False) == r["prompt_token_ids"]
        records[key] = batch(r["prompt_token_ids"], r["generated_token_ids"])
    for split, rows in [("calibration", calibration), ("measurement", measurement)]:
        for row in rows:
            text = tok.apply_chat_template(
                [{"role": "user", "content": inputs[row["task_id"]]["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt = tok.encode(text, add_special_tokens=False)
            completion = tok.encode("```python\n" + row["code"].strip() + "\n```", add_special_tokens=False) + [
                tok.eos_token_id
            ]
            records[f"{split}:{row['task_id']}"] = batch(prompt, completion)

    def loss(key):
        tokens, start = records[key]
        logits = model(input_ids=tokens, use_cache=False).logits[:, start - 1 : -1, :].float()
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens[:, start:].reshape(-1), reduction="none"
        )

    baseline = {}
    logprob_checks = {}
    started = time.time()
    gradient_keys = ids + [f"calibration:{r['task_id']}" for r in calibration]
    gradient_keys += [f"measurement:{r['task_id']}" for r in measurement]
    for j, key in enumerate(gradient_keys):
        model.zero_grad(set_to_none=True)
        nll = loss(key)
        baseline[key] = float(nll.mean().detach())
        if key in raw:
            reference = np.array(raw[key]["chosen_token_logprobs"])
            errors = np.abs(-nll.detach().cpu().numpy() - reference)
            logprob_checks[key] = {"mean_abs": float(errors.mean()), "max_abs": float(errors.max())}
        nll.mean().backward()
        gradient = (
            torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
                    for _, p in parameters
                ]
            )
            .cpu()
            .numpy()
        )
        assert np.isfinite(gradient).all()
        arrays[key] = gradient
        if j % 8 == 0:
            print(
                json.dumps(
                    {"gradients": j + 1, "total": len(gradient_keys), "seconds": round(time.time() - started, 2)}
                ),
                flush=True,
            )
    np.savez(OUT / f"{prefix}gradients.npz", ids=np.array(list(arrays)), gradients=np.stack(list(arrays.values())))
    cal = np.mean([arrays[f"calibration:{r['task_id']}"] for r in calibration], axis=0)
    mids = [f"measurement:{r['task_id']}" for r in measurement]
    measurement_gradient = np.mean([arrays[k] for k in mids], axis=0)
    with torch.no_grad():
        base_measurement = [float(loss(k).mean()) for k in mids]
        repeat = [float(loss(k).mean()) for k in mids]
    assert max(abs(a - b) for a, b in zip(base_measurement, repeat)) < 1e-6
    results = []

    def update(vector, scale):
        cursor = 0
        with torch.no_grad():
            for name, p in parameters:
                size = p.numel()
                delta = torch.from_numpy(vector[cursor : cursor + size].copy()).to(p.device).view_as(p)
                p.copy_(initial[name] - scale * delta)
                cursor += size
        assert cursor == len(vector)

    output = OUT / f"{prefix}updates.jsonl"
    assert not output.exists(), "Partial updates exist; inspect before restarting"
    by_id = {g["group_id"]: g for g in groups}
    controls = [c for c in group_input["controls"]["identical_group_repeats"] if c["source_group_id"] in by_id]
    group_list = groups + [dict(by_id[c["source_group_id"]], group_id=c["control_id"]) for c in controls]
    for group in group_list:
        vectors = np.stack([arrays[k] for k in group["record_ids"]])
        rewards = np.array(group["rewards"], dtype=np.float32)
        assert rewards.sum() == 2 and len(rewards) == 4
        h = ((rewards - 0.5)[:, None] * vectors).mean(0)
        norm = float(np.linalg.norm(h))
        norms = np.linalg.norm(vectors, axis=1)
        assert (norms > 0).all()
        unit = vectors / norms[:, None]
        dispersion = float((1 - unit @ unit.T)[np.triu_indices(4, 1)].mean())
        score = {
            "group_id": group["group_id"],
            "task_id": group["task_id"],
            "matched_pair_id": group["matched_pair_id"],
            "gradient_norm": norm,
            "same_measurement_alignment_sanity_only": float(measurement_gradient @ h),
            "gradient_dispersion": dispersion,
            "calibration_alignment": float(cal @ h),
            "calibration_cosine": float(cal @ h / (max(norm, 1e-30) * np.linalg.norm(cal))),
            "cancellation": float(norm / np.mean(np.linalg.norm((rewards - 0.5)[:, None] * vectors, axis=1))),
            "mean_nll": float(np.mean([baseline[k] for k in group["record_ids"]])),
            "total_completion_tokens": group["total_completion_tokens"],
            "updates": {},
        }
        for mode, scale in [("sgd_0.1", 0.1), ("sgd_1.0", 1.0), ("norm_0.02", 0.02 / norm if norm else 0.0)]:
            update(h, scale)
            with torch.no_grad():
                after = [float(loss(k).mean()) for k in mids]
            score["updates"][mode] = {
                "scale": scale,
                "measurement_losses": after,
                "mean_loss_decrease": float(np.mean(np.array(base_measurement) - after)),
                "per_task_loss_decrease": (np.array(base_measurement) - after).tolist(),
            }
        if smoke:
            score["directional_precision_check"] = {}
            for step in [0.01, 0.1, 1.0]:
                directional_losses = []
                for scale in [-step, step]:
                    update(h, scale)
                    with torch.no_grad():
                        directional_losses.append(float(np.mean([float(loss(k).mean()) for k in mids])))
                finite_difference = (directional_losses[0] - directional_losses[1]) / (2 * step)
                score["directional_precision_check"][str(step)] = {
                    "central_derivative": finite_difference,
                    "autograd_derivative": float(measurement_gradient @ h),
                    "absolute_error": abs(finite_difference - float(measurement_gradient @ h)),
                }
        results.append(score)
        with output.open("a") as f:
            f.write(json.dumps(score) + "\n")
        if len(results) % 8 == 0:
            print(
                json.dumps(
                    {"groups": len(results), "total": len(group_list), "seconds": round(time.time() - started, 2)}
                ),
                flush=True,
            )
    by_result = {r["group_id"]: r for r in results}
    for control in controls:
        assert by_result[control["source_group_id"]]["updates"] == by_result[control["control_id"]]["updates"], (
            "Identical group reset/update mismatch"
        )
    with torch.no_grad():
        for name, p in parameters:
            p.copy_(initial[name])
        final_reset = [float(loss(k).mean()) for k in mids]
    assert final_reset == base_measurement
    summary = {
        "technical_checks_passed": True,
        "groups": len(groups),
        "training_tasks": len({g["task_id"] for g in groups}),
        "unique_rollouts": len(ids),
        "calibration_tasks": len(calibration),
        "measurement_tasks": len(measurement),
        "trainable_parameters": sum(p.numel() for _, p in parameters),
        "exact_group_repeat_and_reset": True,
        "identical_group_repeats": len(controls),
        "base_measurement_losses": base_measurement,
        "measurement_task_ids": [r["task_id"] for r in measurement],
        "logprob_recomputation": logprob_checks,
        "elapsed_seconds": time.time() - started,
        "mean_transfer_by_step": {
            mode: float(np.mean([r["updates"][mode]["mean_loss_decrease"] for r in results[: len(groups)]]))
            for mode in ["sgd_0.1", "sgd_1.0", "norm_0.02"]
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "api_cost_usd": 0,
        "interpretation": "Actual finite adapter update and new-task reference-loss change; not functional accuracy or a downstream training gain claim.",
    }
    save(OUT / f"{prefix}baseline_nll.json", baseline)
    save(OUT / f"{prefix}learner_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "logprob_recomputation"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run(args.smoke)
