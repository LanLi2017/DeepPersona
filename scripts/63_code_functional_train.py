#!/usr/bin/env python3
"""Finite offline signed updates for frozen calibration-only functional trials."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path("runs/swe-diversity-selection")
OUT = ROOT / "code-functional-development"
POOL = ROOT / "code-learning-pilot"
INITIAL = ROOT / "code-update-transfer-fp32/initial_adapter.pt"
MODEL = Path(
    "/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    assert not (OUT / "training_protocol.json").exists()
    protocol = json.loads((OUT / "functional_protocol.json").read_text())
    save(
        OUT / "training_protocol.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "model": str(MODEL),
            "gpu": 2,
            "dtype": "float32",
            "tf32": False,
            "versions": {k: importlib.metadata.version(k) for k in ["torch", "transformers", "peft"]},
            "optimizer": protocol["optimizer"],
            "source_protocol_sha256": sha(OUT / "functional_protocol.json"),
            "input_sha256": {str(p): sha(p) for p in [OUT / "selections.json", POOL / "raw.jsonl", INITIAL]},
            "runner_sha256": sha(Path(__file__)),
            "objective": protocol["training_objective"],
            "seed_role": protocol["seed_role"],
            "smoke": protocol["technical_smoke"],
            "api_spend_usd": 0,
        },
    )
    print("Functional learner frozen before training or executable calibration outcomes")


def train(args):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    frozen = json.loads((OUT / "training_protocol.json").read_text())
    assert sha(Path(__file__)) == frozen["runner_sha256"]
    assert sha(OUT / "functional_protocol.json") == frozen["source_protocol_sha256"]
    for name, expected in frozen["input_sha256"].items():
        assert sha(Path(name)) == expected
    protocol = json.loads((OUT / "functional_protocol.json").read_text())
    selections = {r["seed"]: r for r in json.loads((OUT / "selections.json").read_text())["selections"]}
    raw = {f"{r['task_id']}:{r['sample']}": r for line in (POOL / "raw.jsonl").open() if (r := json.loads(line))}
    if args.stage == "grid":
        assert json.loads((OUT / "smoke_gate.json").read_text())["technical_checks_passed"]
        jobs = [
            ("model_typicality", seed, lr, epochs)
            for seed in protocol["seeds"]
            for epochs in protocol["grid"]["epochs"]
            for lr in protocol["grid"]["learning_rates"]
        ]
    else:
        jobs = [(args.arm, args.seed, args.lr, args.epochs)]
    if args.smoke:
        assert jobs == [("model_typicality", 0, 0.1, 1)]
    elif args.stage != "grid":
        assert json.loads((OUT / "smoke_gate.json").read_text())["technical_checks_passed"]
    torch.manual_seed(20260909)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
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
            layers_to_transform=[35],
        ),
    ).eval()
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in parameters.values()) == 106496
    initial = torch.load(INITIAL, map_location="cpu", weights_only=True)
    assert initial.keys() == parameters.keys()
    for arm, seed, lr, epochs in jobs:
        assert arm in protocol["arms"] and seed in protocol["seeds"]
        assert lr in protocol["grid"]["learning_rates"] and epochs in protocol["grid"]["epochs"]
        name = "smoke" if args.smoke else f"{arm}_seed{seed}_lr{lr:g}_epochs{epochs}"
        dest = OUT / "checkpoints" / name
        if (dest / "summary.json").exists():
            print(json.dumps({"already_complete": name}), flush=True)
            continue
        assert not dest.exists(), "Partial checkpoint exists; inspect before restarting"
        dest.mkdir(parents=True)
        with torch.no_grad():
            for n, p in parameters.items():
                p.copy_(initial[n])
        optimizer = torch.optim.SGD(parameters.values(), lr=lr, momentum=0, weight_decay=0)
        chosen = selections[seed]["arms"][arm]
        by_task = {g["task_id"]: g for g in chosen}
        batches = [[by_task[t]] for t in selections[seed]["task_order"]] * epochs
        if args.smoke:
            batches = [[by_task[t] for t in protocol["technical_smoke"]["training_task_ids"]]]
        manifest = {
            "name": name,
            "arm": arm,
            "selection_order_seed": seed,
            "learning_rate": lr,
            "epochs": epochs,
            "smoke": args.smoke,
            "initial_adapter_sha256": sha(INITIAL),
            "training_protocol_sha256": sha(OUT / "training_protocol.json"),
            "groups": [[g["group_id"] for g in batch] for batch in batches],
            "api_spend_usd": 0,
        }
        save(dest / "manifest.json", manifest)
        for step, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            records = []
            for group in batch:
                assert sum(group["rewards"]) == 2 and len(group["record_ids"]) == 4
                for key, reward in zip(group["record_ids"], group["rewards"]):
                    row = raw[key]
                    ids = row["prompt_token_ids"] + row["generated_token_ids"]
                    start = len(row["prompt_token_ids"])
                    assert len(ids) <= 2048
                    tokens = torch.tensor([ids], device="cuda")
                    logits = model(input_ids=tokens, use_cache=False).logits[:, start - 1 : -1, :].float()
                    nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), tokens[:, start:].reshape(-1)
                    )
                    assert torch.isfinite(nll)
                    ((reward - 0.5) * nll / (4 * len(batch))).backward()
                    records.append({"record_id": key, "reward": reward, "nll": float(nll.detach())})
            gradient = torch.cat([p.grad.flatten() for p in parameters.values()])
            assert torch.isfinite(gradient).all()
            norm = float(torch.linalg.vector_norm(gradient))
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in parameters.values())
            with (dest / "steps.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "gradient_norm": norm, "records": records}) + "\n")
        model.save_pretrained(dest / "adapter")
        expected = get_peft_model_state_dict(model)
        loaded = load_file(str(dest / "adapter/adapter_model.safetensors"))
        assert expected.keys() == loaded.keys()
        assert all(torch.equal(v.detach().cpu(), loaded[k]) for k, v in expected.items())
        delta = float(torch.sqrt(sum(((p.detach().cpu() - initial[n]) ** 2).sum() for n, p in parameters.items())))
        assert delta > 0
        save(
            dest / "summary.json",
            {
                "technical_checks_passed": True,
                "steps": len(batches),
                "parameter_delta_l2": delta,
                "exact_saved_adapter_tensors": True,
                "initial_adapter_sha256": sha(INITIAL),
                "adapter_sha256": {p.name: sha(p) for p in (dest / "adapter").iterdir() if p.is_file()},
                "api_spend_usd": 0,
                "interpretation": "Actual offline training only; functional calibration outcomes saved separately, no generalization claim",
            },
        )
        print(json.dumps({"completed": name, "steps": len(batches), "delta_l2": delta}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "grid"])
    parser.add_argument("--arm", default="model_typicality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else train(args)
