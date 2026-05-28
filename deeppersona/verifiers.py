"""GSM8K answer extraction + exact-match scoring.

We try a small ordered list of patterns. Keep raw generation in the run log so
extraction is re-runnable offline.
"""
from __future__ import annotations

import re

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
HASH_RE = re.compile(r"####\s*(-?[\d.,]+)")
THE_ANSWER_IS_RE = re.compile(r"(?:answer\s*(?:is|:))\s*\$?(-?[\d.,]+)", re.IGNORECASE)
LAST_NUMBER_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


def _clean_number(s: str) -> str | None:
    s = s.strip().rstrip(".").replace(",", "").replace("$", "")
    # strip a trailing "." or unit-ish garbage but keep a leading minus
    m = re.match(r"^-?\d+(?:\.\d+)?$", s)
    if m:
        # canonicalize: drop trailing zeros after decimal, drop decimal point if integer
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return None


def extract_gsm8k_pred(text: str) -> str | None:
    # 1. \boxed{...}
    m = list(BOXED_RE.finditer(text))
    if m:
        cleaned = _clean_number(m[-1].group(1))
        if cleaned is not None:
            return cleaned
    # 2. "#### X"
    m2 = list(HASH_RE.finditer(text))
    if m2:
        cleaned = _clean_number(m2[-1].group(1))
        if cleaned is not None:
            return cleaned
    # 3. "answer is X" / "answer: X"
    m3 = list(THE_ANSWER_IS_RE.finditer(text))
    if m3:
        cleaned = _clean_number(m3[-1].group(1))
        if cleaned is not None:
            return cleaned
    # 4. last number in the text
    m4 = list(LAST_NUMBER_RE.finditer(text))
    if m4:
        cleaned = _clean_number(m4[-1].group(1))
        if cleaned is not None:
            return cleaned
    return None


def gsm8k_score(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    gp = _clean_number(pred)
    gg = _clean_number(gold)
    if gp is None or gg is None:
        return False
    try:
        return float(gp) == float(gg)
    except ValueError:
        return False
