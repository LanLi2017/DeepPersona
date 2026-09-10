#!/usr/bin/env python3
"""Real-code local LoRA update/reset smoke; no downstream efficacy claim."""

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path("runs/swe-diversity-selection/learning-update-smoke")
MODEL = Path(
    "/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
DATA = Path(
    "/scratch/yirenl2/.cache/huggingface/hub/datasets--google-research-datasets--mbpp/snapshots/4bb6404fdc6cacfda99d4ac4205087b89d32030c"
)
SEED = 20260909


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "Smoke uses reserved GPU2, never GPU0"
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "results.json").exists(), "Completed smoke already recorded"
    torch.manual_seed(SEED)
    rows = pq.read_table(next((DATA / "sanitized").glob("train-*.parquet"))).to_pylist()
    rows.sort(key=lambda r: hashlib.sha256(f"{SEED}:learning-smoke:{r['task_id']}".encode()).hexdigest())
    selected = rows[:2]
    items = []
    for row in selected:
        tree = ast.parse(row["code"])
        functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        interface = "\n".join("def " + n.name + "(" + ast.unparse(n.args) + "): ..." for n in functions)
        prompt = (
            row["prompt"]
            + "\n\nRequired callable interfaces:\n"
            + interface
            + "\n\nReturn a complete Python solution in a ```python code block."
        )
        items.append(
            {"task_id": row["task_id"], "prompt": prompt, "target": "```python\n" + row["code"].strip() + "\n```"}
        )
    save("inputs.json", items)
    save(
        "protocol.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "model_snapshot": str(MODEL),
            "dataset_snapshot": str(DATA),
            "split": "train",
            "purpose": "Local backward/finite-update/reset infrastructure smoke on two real code training examples; no held-out or benchmark performance claim.",
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu": 2,
            "dtype": "bfloat16",
            "adapter": {
                "target_modules": ["q_proj", "v_proj"],
                "last_decoder_layer_only": True,
                "rank": 8,
                "alpha": 16,
                "dropout": 0,
            },
            "loss": "Completion-token mean NLL; prompts masked; deterministic teacher forcing.",
            "learning_rates": [0.001, 0.01, 0.1],
            "api_cost_usd": 0,
            "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"]},
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).cuda()
    model.config.use_cache = False
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=[model.config.num_hidden_layers - 1],
    )
    model = get_peft_model(model, config).eval()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    initial = {n: p.detach().clone() for n, p in params}
    batches = []
    for row in items:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = tokenizer.encode(prefix, add_special_tokens=False)
        completion = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        assert len(prefix) + len(completion) <= 2048
        tokens = torch.tensor([prefix + completion], device="cuda")
        batches.append((tokens, len(prefix), len(completion)))

    def loss(batch):
        tokens, start, _ = batch
        output = model(input_ids=tokens, use_cache=False).logits[:, start - 1 : -1, :].float()
        return torch.nn.functional.cross_entropy(output.reshape(-1, output.shape[-1]), tokens[:, start:].reshape(-1))

    before = []
    with torch.no_grad():
        before = [float(loss(batch)) for batch in batches]
    model.zero_grad(set_to_none=True)
    started = time.time()
    value = loss(batches[0])
    value.backward()
    gradients = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n, p in params}
    assert all(torch.isfinite(v).all() for v in gradients.values())
    norm = float(torch.sqrt(sum(v.float().square().sum() for v in gradients.values())))
    assert norm > 0
    outcomes = []
    for lr in [0.001, 0.01, 0.1]:
        with torch.no_grad():
            for n, p in params:
                p.copy_(initial[n] - lr * gradients[n])
            after = [float(loss(batch)) for batch in batches]
        outcomes.append(
            {
                "learning_rate": lr,
                "losses_after": after,
                "training_loss_decrease": before[0] - after[0],
                "other_training_example_loss_decrease": before[1] - after[1],
            }
        )
    with torch.no_grad():
        for n, p in params:
            p.copy_(initial[n])
        reset = [float(loss(batch)) for batch in batches]
    assert all(torch.equal(p, initial[n]) for n, p in params)
    assert max(abs(a - b) for a, b in zip(before, reset)) < 1e-6
    assert any(r["training_loss_decrease"] > 0 for r in outcomes)
    result = {
        "status": "pass",
        "before": before,
        "reset": reset,
        "update_results": outcomes,
        "trainable_parameters": sum(p.numel() for _, p in params),
        "gradient_norm": norm,
        "nonzero_gradient_blocks": [n for n, v in gradients.items() if bool(v.abs().max() > 0)],
        "token_lengths": [{"prompt": b[1], "completion": b[2]} for b in batches],
        "finite_update_and_exact_reset_verified": True,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "api_cost_usd": 0,
    }
    save("results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
