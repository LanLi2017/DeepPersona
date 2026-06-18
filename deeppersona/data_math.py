"""MATH-500 loader (HuggingFaceH4/MATH-500), mirroring data.load_items shape.

Gold answer is pre-extracted in the `answer` column. No training here, but we
still freeze a calib slice disjoint from the test slice so T_hi calibration never
touches the eval items (same discipline as data.py).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MATH_HF_ID = "HuggingFaceH4/MATH-500"
N_CALIB = 64  # reserved calib pool, disjoint from test


def math500_splits(seed: int = 0) -> dict[str, list[int]]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / "math500_split_indices.json"
    ds = load_dataset(MATH_HF_ID, split="test")
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached["seed"] == seed, f"math500 split seed mismatch: {cached['seed']} vs {seed}"
        return {"calib": cached["calib"], "test": cached["test"]}
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    splits = {"calib": sorted(idx[:N_CALIB]), "test": sorted(idx[N_CALIB:])}
    path.write_text(json.dumps({"seed": seed, **splits}, indent=2))
    return splits


def load_math_items(split: str, n_items: int | None = None, seed: int = 0) -> list[dict]:
    splits = math500_splits(seed=seed)
    if split not in splits:
        raise ValueError(f"Unknown split: {split}. Want 'calib' or 'test'.")
    ds = load_dataset(MATH_HF_ID, split="test")
    indices = splits[split]
    if n_items is not None:
        indices = indices[:n_items]
    return [{"idx": i, "question": ds[i]["problem"],
             "gold_answer": ds[i]["answer"], "gold_solution": ds[i]["solution"]} for i in indices]
