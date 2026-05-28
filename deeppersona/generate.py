"""HF model loading + batched greedy generation.

Greedy + matched max_new_tokens across conditions (design doc §3.4). Pinned
revision (cfg.model_revision); fails loudly if CUDA is missing.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RunConfig

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(cfg: RunConfig):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. Refusing CPU fallback.")
    dtype = _DTYPES[cfg.dtype]
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        torch_dtype=dtype,
        device_map={"": 0},
    )
    model.eval()
    return model, tok


def build_chat_prompts(tok, system_msg: str, user_msgs: list[str]) -> list[str]:
    out = []
    for u in user_msgs:
        msgs = [{"role": "system", "content": system_msg}, {"role": "user", "content": u}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


@torch.inference_mode()
def generate_iter(model, tok, prompts: list[str], cfg: RunConfig) -> Iterator[list[str]]:
    """Yields decoded outputs one batch at a time. Lets callers print progress."""
    for i in range(0, len(prompts), cfg.batch_size):
        batch = prompts[i : i + cfg.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to(model.device)
        gen_ids = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        in_len = enc["input_ids"].shape[1]
        new_ids = gen_ids[:, in_len:]
        yield tok.batch_decode(new_ids, skip_special_tokens=True)


def generate(model, tok, prompts: list[str], cfg: RunConfig) -> list[str]:
    return [s for batch in generate_iter(model, tok, prompts, cfg) for s in batch]
