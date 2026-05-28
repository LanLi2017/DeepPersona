"""Paired stats for the persona-scaffolding contrast (structured vs basic).

Shared by the single-process runner (03_scaffolding) and the HPC aggregator
(04_aggregate_scaffolding) so the McNemar / bootstrap logic has one definition.
"""
from __future__ import annotations

from math import comb

import numpy as np


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, j) for j in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_boot_ci(diff, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0):
    diff = np.asarray(diff, dtype=np.float64)
    if diff.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = diff[rng.integers(0, diff.size, size=(n_resamples, diff.size))].mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def contrast_summary(pairs, none_acc, seed: int = 0) -> dict:
    """Structured-vs-basic contrast, per identity and pooled.

    `pairs`: list of (persona_idx, basic_correct, structured_correct), where the
    correctness arrays are 0/1 and ALIGNED item-for-item within each pair.
    """
    per_identity = []
    pooled_basic, pooled_struct = [], []
    for pidx, b_arr, s_arr in pairs:
        b_arr, s_arr = np.asarray(b_arr), np.asarray(s_arr)
        pooled_basic.append(b_arr)
        pooled_struct.append(s_arr)
        b = int(np.sum((b_arr == 1) & (s_arr == 0)))  # basic right, structured wrong
        c = int(np.sum((b_arr == 0) & (s_arr == 1)))  # basic wrong, structured right
        per_identity.append({
            "persona_idx": pidx,
            "basic_acc": float(b_arr.mean()),
            "structured_acc": float(s_arr.mean()),
            "delta": float(s_arr.mean() - b_arr.mean()),
            "discordant_b_basic_only": b,
            "discordant_c_structured_only": c,
            "mcnemar_p": mcnemar_exact_p(b, c),
        })

    bA, sA = np.concatenate(pooled_basic), np.concatenate(pooled_struct)
    pb = int(np.sum((bA == 1) & (sA == 0)))
    pc = int(np.sum((bA == 0) & (sA == 1)))
    lo, hi = paired_boot_ci(sA - bA, seed=seed)
    return {
        "none_acc": none_acc,
        "basic_mean_acc": float(bA.mean()),
        "structured_mean_acc": float(sA.mean()),
        "pooled_delta_structured_minus_basic": float(sA.mean() - bA.mean()),
        "pooled_delta_ci95": [lo, hi],
        "pooled_mcnemar": {"b_basic_only": pb, "c_structured_only": pc, "p": mcnemar_exact_p(pb, pc)},
        "per_identity": per_identity,
    }
