"""Output-trajectory diversity metrics over a group of rollouts (F1 study).

All operate on one question's group of generations (or a flat batch pool):
- distinct_n      : distinct n-gram ratio (lexical diversity; higher = more diverse).
- self_bleu       : mean BLEU-n of each rollout vs the rest (higher = more SIMILAR / less diverse).
- answer_entropy  : Shannon entropy (bits) over the distribution of extracted final answers.
- token_surprisal : mean per-token negative logprob of sampled tokens (policy-entropy proxy;
                    Tinker returns sampled-token logprobs, not full distributions, so this is the
                    surprisal of the chosen tokens, not exact policy entropy).
"""
from __future__ import annotations

import math
from collections import Counter


def distinct_n(generations: list[str], n: int = 4) -> float:
    grams, total = set(), 0
    for g in generations:
        toks = g.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _sentence_bleu(hyp: str, refs: list[str], max_n: int = 4) -> float:
    """Add-1-smoothed modified-precision BLEU of hyp vs the union of refs, with brevity penalty."""
    h = hyp.split()
    if not h:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hc = _ngrams(h, n)
        if not hc:
            continue
        maxref: Counter = Counter()
        for r in refs:
            for g, c in _ngrams(r.split(), n).items():
                if c > maxref[g]:
                    maxref[g] = c
        clip = sum(min(c, maxref[g]) for g, c in hc.items())
        precisions.append((clip + 1) / (sum(hc.values()) + 1))  # add-1 smoothing
    if not precisions:
        return 0.0
    ref_lens = [len(r.split()) for r in refs] or [len(h)]
    closest = min(ref_lens, key=lambda rl: (abs(rl - len(h)), rl))
    bp = 1.0 if len(h) >= closest else math.exp(1 - closest / max(len(h), 1))
    return bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))


def self_bleu(generations: list[str], max_n: int = 4) -> float:
    gens = [g for g in generations if g.strip()]
    if len(gens) < 2:
        return 0.0
    scores = [_sentence_bleu(g, gens[:i] + gens[i + 1 :], max_n) for i, g in enumerate(gens)]
    return sum(scores) / len(scores)


def answer_entropy(answers: list) -> float:
    """Shannon entropy (bits) over distinct non-null final answers in a group."""
    vals = [str(a) for a in answers if a is not None]
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def token_surprisal(logprobs_per_rollout: list[list[float]]) -> float:
    """Mean over rollouts of (mean per-token -logprob of sampled tokens)."""
    means = [(-sum(lp) / len(lp)) for lp in logprobs_per_rollout if lp]
    return sum(means) / len(means) if means else 0.0
