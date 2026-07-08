"""
CSQA rollout diversity sweep — compare diversity sources on Qwen2.5-1.5B-Instruct.

Methods (each produces N=8 rollouts per item):
  greedy        : temp=0, same neutral prompt → 8 identical outputs (diversity floor)
  temp{τ}       : temp=τ, same neutral prompt → 8 stochastic samples
  persona       : 8 basic CSQA personas × greedy
  template      : 8 neutral templates × greedy
  persona_temp  : 8 CSQA personas × temp=0.7

Backends:
  --ollama MODEL   use Ollama (local, no CUDA needed) — e.g. --ollama qwen2.5:1.5b
  (default)        HuggingFace transformers + CUDA

Output: experiments/diversity/runs/diversity-<ts>.jsonl
  one record per (method, item_idx, rollout_idx).

Smoke test: --smoke  (8 items, fast)
"""
import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deeppersona.data import load_csqa_items
from deeppersona.personas import ANSWER_INSTRUCTION, CSQA_SPECS
from deeppersona.verifiers import csqa_score, extract_csqa_pred

ANSWER_INSTR = ANSWER_INSTRUCTION["csqa"]

NEUTRAL_TEMPLATES = [
    "Answer the following question.",
    "Solve the following problem.",
    "Read the problem and answer it.",
    "Think step by step and choose the best answer.",
    "You must pick exactly one of the given options.",
    "Consider all options carefully before answering.",
    "What is the correct answer to this question?",
    "Select the most appropriate option.",
]

PERSONA_SYSMSGS = [s["basic"] + ANSWER_INSTR for s in CSQA_SPECS[:8]]
TEMPLATE_SYSMSGS = [t + ANSWER_INSTR for t in NEUTRAL_TEMPLATES]
DEFAULT_SYSMSG = NEUTRAL_TEMPLATES[0] + ANSWER_INSTR


# ── HF backend ───────────────────────────────────────────────────────────────

def load_hf_model(model_id, revision, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for HF backend — use --ollama for local runs.")
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=dt, device_map={"": 0}
    )
    return model.eval(), tok


def fmt_prompts_hf(tok, sys_msgs, user_msgs):
    return [
        tok.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True,
        )
        for s, u in zip(sys_msgs, user_msgs)
    ]


def gen_once_hf(model, tok, prompts, temperature, max_new_tokens, batch_size):
    import torch
    do_sample = temperature > 0
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to("cuda")
            ids = model.generate(
                **enc,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            in_len = enc["input_ids"].shape[1]
            outputs += tok.batch_decode(ids[:, in_len:], skip_special_tokens=True)
    return outputs


# ── Ollama backend ────────────────────────────────────────────────────────────

def gen_once_ollama(ollama_model, sys_msgs, user_msgs, temperature, max_new_tokens):
    import ollama
    outputs = []
    for sys_msg, user_msg in zip(sys_msgs, user_msgs):
        resp = ollama.chat(
            model=ollama_model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens,
                # seed=0 only works for greedy; omit for stochastic
                **({"seed": 0} if temperature == 0 else {}),
            },
        )
        outputs.append(resp["message"]["content"])
    return outputs


# ── unified dispatch ──────────────────────────────────────────────────────────

def gen_once(backend, sys_msgs, user_msgs, temperature, max_new_tokens, batch_size):
    """Single generation pass. backend = (model, tok) for HF, or str for Ollama."""
    if isinstance(backend, str):
        return gen_once_ollama(backend, sys_msgs, user_msgs, temperature, max_new_tokens)
    model, tok = backend
    prompts = fmt_prompts_hf(tok, sys_msgs, user_msgs)
    return gen_once_hf(model, tok, prompts, temperature, max_new_tokens, batch_size)


def run_fixed_prompt(backend, items, sys_msg, temperature, n_rollouts, max_new_tokens, bs):
    """Same system prompt, N rollouts. If greedy: generate once, replicate."""
    user_msgs = [it["question"] for it in items]
    if temperature == 0:
        texts = gen_once(backend, [sys_msg] * len(items), user_msgs, 0, max_new_tokens, bs)
        return [[t] * n_rollouts for t in texts]
    per_pass = [gen_once(backend, [sys_msg] * len(items), user_msgs, temperature, max_new_tokens, bs)
                for _ in range(n_rollouts)]
    return [[per_pass[r][i] for r in range(n_rollouts)] for i in range(len(items))]


def run_prompt_variants(backend, items, sys_msgs, temperature, max_new_tokens, bs):
    """One rollout per system message variant (greedy or temp)."""
    user_msgs = [it["question"] for it in items]
    per_variant = [
        gen_once(backend, [s] * len(items), user_msgs, temperature, max_new_tokens, bs)
        for s in sys_msgs
    ]
    return [[per_variant[v][i] for v in range(len(sys_msgs))] for i in range(len(items))]


def write_records(f, method, items, rollouts_per_item):
    for item, rollouts in zip(items, rollouts_per_item):
        for r_idx, text in enumerate(rollouts):
            pred = extract_csqa_pred(text)
            correct = csqa_score(pred, item["gold_answer"])
            f.write(json.dumps({
                "method": method,
                "item_idx": item["idx"],
                "rollout_idx": r_idx,
                "gold": item["gold_answer"],
                "pred": pred,
                "correct": correct,
                "text": text,
            }) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/diversity/configs/qwen15b_csqa.yaml")
    parser.add_argument("--methods", nargs="+",
                        default=["greedy", "temp0.3", "temp0.7", "temp1.0", "temp1.3",
                                 "persona", "template", "persona_temp"])
    parser.add_argument("--ollama", metavar="MODEL", default=None,
                        help="Use Ollama backend (e.g. qwen2.5:1.5b). No CUDA needed.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(ROOT / args.config).read_text())
    if args.smoke:
        cfg["n_items"] = 8
        cfg["batch_size"] = 4

    random.seed(cfg["seed"])

    items = load_csqa_items("val", n_items=cfg["n_items"], seed=cfg["seed"])
    print(f"Loaded {len(items)} CSQA val items.")

    if args.ollama:
        backend = args.ollama
        model_label = f"ollama:{args.ollama}"
        print(f"Backend: Ollama ({args.ollama})")
    else:
        backend = load_hf_model(cfg["model_id"], cfg.get("model_revision"))
        model_label = cfg["model_id"]
        print(f"Backend: HF ({model_label})")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = ROOT / f"experiments/diversity/runs/diversity-{ts}.jsonl"

    git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    manifest = {
        "git_sha": git_sha, "model": model_label,
        "n_items": len(items), "n_rollouts": cfg["n_rollouts"],
        "max_new_tokens": cfg["max_new_tokens"], "seed": cfg["seed"],
        "methods": args.methods,
    }

    n_rollouts = cfg["n_rollouts"]
    max_tok = cfg["max_new_tokens"]
    bs = cfg["batch_size"]

    with open(out_path, "w") as f:
        f.write(json.dumps({"__manifest__": True, **manifest}) + "\n")

        for method in args.methods:
            print(f"\n=== {method} ===")
            if method == "greedy":
                rollouts = run_fixed_prompt(backend, items, DEFAULT_SYSMSG, 0.0, n_rollouts, max_tok, bs)
            elif method.startswith("temp") and method != "template":
                tau = float(method[4:])
                rollouts = run_fixed_prompt(backend, items, DEFAULT_SYSMSG, tau, n_rollouts, max_tok, bs)
            elif method == "persona":
                rollouts = run_prompt_variants(backend, items, PERSONA_SYSMSGS, 0.0, max_tok, bs)
            elif method == "template":
                rollouts = run_prompt_variants(backend, items, TEMPLATE_SYSMSGS, 0.0, max_tok, bs)
            elif method == "persona_temp":
                rollouts = run_prompt_variants(backend, items, PERSONA_SYSMSGS, 0.7, max_tok, bs)
            else:
                raise ValueError(f"Unknown method: {method}")

            write_records(f, method, items, rollouts)

            accs = [csqa_score(extract_csqa_pred(t), it["gold_answer"])
                    for it, rl in zip(items, rollouts) for t in rl]
            pass1 = sum(csqa_score(extract_csqa_pred(rl[0]), it["gold_answer"]) for it, rl in zip(items, rollouts)) / len(items)
            passK = sum(any(csqa_score(extract_csqa_pred(t), it["gold_answer"]) for t in rl) for it, rl in zip(items, rollouts)) / len(items)
            print(f"  pass@1={pass1:.3f}  pass@{n_rollouts}={passK:.3f}  mean_acc={sum(accs)/len(accs):.3f}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
