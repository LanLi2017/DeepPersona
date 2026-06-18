"""E-SPL competition-math benchmark loaders + seeded train/test split.

Ports load_data from refs/E-SPL/tinker_cookbook/recipes/system_prompt_learning_rl.py
for the math sets, normalizing each item to {idx, problem, gold_answer}. Default
focus: aime_and_amc (AIME+AMC; easier than AIME->BeyondAIME, has headroom on an 8B
policy so prompt evolution has fitness signal to move).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# name -> (hf repo, split). `answer` is the gold for all of these.
_HF = {
    "AIME24": ("HuggingFaceH4/aime_2024", "train"),
    "AIME25": ("yentinglin/aime_2025", "train"),
    "amc": ("AI-MO/aimo-validation-amc", "train"),
    "aime": ("AI-MO/aimo-validation-aime", "train"),
    "BeyondAIME": ("ByteDance-Seed/BeyondAIME", "test"),
    "hmmt_nov_2025": ("MathArena/hmmt_nov_2025", "train"),
}


def _load_one(repo: str, split: str) -> list[dict]:
    ds = datasets.load_dataset(repo, split=split)
    return [{"problem": e["problem"], "groundtruth": str(e["answer"])} for e in ds.to_list()]


def load_raw(name: str) -> list[dict]:
    if name == "aime_and_amc":
        data = _load_one(*_HF["aime"]) + _load_one(*_HF["amc"])
        random.Random(42).shuffle(data)  # match E-SPL's seed-42 combine+shuffle
        return data
    if name in _HF:
        return _load_one(*_HF[name])
    raise ValueError(f"unknown dataset {name!r}; have {['aime_and_amc'] + list(_HF)}")


def espl_splits(name: str, n_test: int, seed: int):
    """Disjoint train/test indices over load_raw(name), cached for reproducibility."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cache = SPLITS_DIR / f"espl_{name}_test{n_test}_seed{seed}.json"
    data = load_raw(name)
    if cache.exists():
        idx = json.loads(cache.read_text())
        return data, idx["train"], idx["test"]
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    test_idx = sorted(order[:n_test])
    train_idx = sorted(order[n_test:])
    cache.write_text(json.dumps(
        {"name": name, "n": len(data), "n_test": n_test, "seed": seed,
         "train": train_idx, "test": test_idx}, indent=2))
    return data, train_idx, test_idx


def _items(data: list[dict], idxs) -> list[dict]:
    return [{"idx": i, "problem": data[i]["problem"], "gold_answer": data[i]["groundtruth"]} for i in idxs]


def load_espl_items(train_set: str, test_set: str | None = None, n_test: int = 40,
                    n_train: int | None = None, n_test_items: int | None = None, seed: int = 0):
    """(train_items, test_items), each [{idx, problem, gold_answer}].

    test_set=None -> disjoint split of train_set (in-distribution eval, headroom).
    test_set given -> train_set fully train, test_set fully test (easy->hard).
    """
    if test_set is None:
        data, tr, te = espl_splits(train_set, n_test, seed)
        train, test = _items(data, tr), _items(data, te)
    else:
        trd, ted = load_raw(train_set), load_raw(test_set)
        train, test = _items(trd, range(len(trd))), _items(ted, range(len(ted)))
    if n_train:
        train = train[:n_train]
    if n_test_items:
        test = test[:n_test_items]
    return train, test
