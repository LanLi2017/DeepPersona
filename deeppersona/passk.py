"""Unbiased pass@k (Chen et al. 2021, Codex).

Copied verbatim from E-SPL:
refs/E-SPL/tinker_cookbook/recipes/system_prompt_learning_rl.py:180-210
"""
from __future__ import annotations


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k = 1 - C(n-c, k) / C(n, k); n samples, c correct."""
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def pass_at_k_for_range(n: int, c: int) -> list[float]:
    return [pass_at_k(n, c, k) for k in range(1, n + 1)]
