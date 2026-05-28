"""Dataset loading + frozen calib/val/test splits.

Per design doc §3.3: calib + val are disjoint 200-item slices of TRAIN; test is
the full test split, never touched during tuning.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _gsm8k_split_path() -> Path:
    return SPLITS_DIR / "gsm8k_split_indices.json"


def _build_gsm8k_splits(train_len: int, seed: int = 0) -> dict[str, list[int]]:
    rng = random.Random(seed)
    idx = list(range(train_len))
    rng.shuffle(idx)
    return {"calib": sorted(idx[:200]), "val": sorted(idx[200:400])}


def gsm8k_splits(seed: int = 0) -> dict[str, list[int]]:
    """Returns {'calib': [...], 'val': [...], 'test': [...]} with frozen indices.

    Caches to data/splits/gsm8k_split_indices.json on first call.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = _gsm8k_split_path()
    train = load_dataset("gsm8k", "main", split="train")
    test = load_dataset("gsm8k", "main", split="test")
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached.get("seed") == seed, f"split seed mismatch: cached={cached.get('seed')} requested={seed}"
        return {"calib": cached["calib"], "val": cached["val"], "test": list(range(len(test)))}
    splits = _build_gsm8k_splits(len(train), seed=seed)
    path.write_text(json.dumps({"seed": seed, **splits}, indent=2))
    return {"calib": splits["calib"], "val": splits["val"], "test": list(range(len(test)))}


def load_gsm8k_items(split: str, n_items: int | None = None, seed: int = 0) -> list[dict]:
    """Returns list of {idx, question, gold_answer (number string), gold_solution}."""
    splits = gsm8k_splits(seed=seed)
    if split not in splits:
        raise ValueError(f"Unknown split: {split}. Want one of {list(splits)}.")
    src_split = "test" if split == "test" else "train"
    ds = load_dataset("gsm8k", "main", split=src_split)
    indices = splits[split]
    if n_items is not None:
        indices = indices[:n_items]
    items: list[dict] = []
    for i in indices:
        row = ds[i]
        # gold format: "...\n#### 42" -> "42"
        ans = row["answer"]
        assert "####" in ans, f"unexpected gsm8k row {i}: {ans!r}"
        gold_number = ans.split("####", 1)[1].strip().replace(",", "")
        items.append({
            "idx": i,
            "question": row["question"],
            "gold_answer": gold_number,
            "gold_solution": ans,
        })
    return items


# ── CommonsenseQA (CSQA) ──────────────────────────────────────────────────
# 5-way multiple choice (Talmor et al. 2018). Test labels are hidden, so eval
# is on the validation split (1221) — same convention as SRPS (arXiv:2506.07335).
CSQA_HF_ID = "tau/commonsense_qa"


def _csqa_split_path() -> Path:
    return SPLITS_DIR / "csqa_split_indices.json"


def csqa_splits(seed: int = 0) -> dict[str, list[int]]:
    """calib/val = disjoint 200-item slices of TRAIN; test = full validation split."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = _csqa_split_path()
    train = load_dataset(CSQA_HF_ID, split="train")
    val = load_dataset(CSQA_HF_ID, split="validation")
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached.get("seed") == seed, f"split seed mismatch: cached={cached.get('seed')} requested={seed}"
        return {"calib": cached["calib"], "val": cached["val"], "test": list(range(len(val)))}
    rng = random.Random(seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    splits = {"calib": sorted(idx[:200]), "val": sorted(idx[200:400])}
    path.write_text(json.dumps({"seed": seed, **splits}, indent=2))
    return {"calib": splits["calib"], "val": splits["val"], "test": list(range(len(val)))}


def load_csqa_items(split: str, n_items: int | None = None, seed: int = 0) -> list[dict]:
    """Returns {idx, question (with A-E options), gold_answer (letter), question_concept}."""
    splits = csqa_splits(seed=seed)
    if split not in splits:
        raise ValueError(f"Unknown split: {split}. Want one of {list(splits)}.")
    src_split = "validation" if split == "test" else "train"
    ds = load_dataset(CSQA_HF_ID, split=src_split)
    indices = splits[split]
    if n_items is not None:
        indices = indices[:n_items]
    items: list[dict] = []
    for i in indices:
        row = ds[i]
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        options = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        items.append({
            "idx": i,
            "question": f"{row['question']}\n\n{options}",
            "gold_answer": row["answerKey"],
            "question_concept": row.get("question_concept", ""),
        })
    return items


def load_items(task: str, split: str, n_items: int | None = None, seed: int = 0) -> list[dict]:
    if task == "gsm8k":
        return load_gsm8k_items(split, n_items=n_items, seed=seed)
    if task == "csqa":
        return load_csqa_items(split, n_items=n_items, seed=seed)
    raise ValueError(f"unknown task: {task}")
