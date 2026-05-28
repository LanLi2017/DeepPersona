from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    model_id: str
    model_revision: str
    dtype: str = "bfloat16"
    device: str = "cuda"

    task: str = "gsm8k"
    condition: str = "none"            # "none" | "prompt" | "vec"
    persona_idx: int = -1              # -1 == no persona (used for "none")
    neutral_template_idx: int = 0      # which neutral template to use as system msg

    max_new_tokens: int = 512
    batch_size: int = 8

    seed: int = 0
    split: str = "val"                 # "calib" | "val" | "test"
    n_items: int | None = None         # None == all items in split

    run_name: str = "baseline"
    smoke: bool = False                # if True: force n_items=4, batch_size=2


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    allowed = {f.name for f in fields(RunConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return RunConfig(**data)


def add_overrides(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags that override config fields."""
    parser.add_argument("--task", default=None)
    parser.add_argument("--condition", default=None, choices=["none", "prompt", "vec"])
    parser.add_argument("--persona-idx", type=int, default=None)
    parser.add_argument("--neutral-template-idx", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default=None, choices=["calib", "val", "test"])
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    updates = {}
    for f in fields(RunConfig):
        val = getattr(args, f.name.replace("-", "_"), None)
        if val is None:
            continue
        if f.name == "smoke" and val is False:
            continue
        updates[f.name] = val
    cfg = replace(cfg, **updates)
    if cfg.smoke:
        cfg = replace(cfg, n_items=4, batch_size=2)
    return cfg
