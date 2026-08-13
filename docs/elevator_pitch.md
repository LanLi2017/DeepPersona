# Elevator pitch — reasoning diversity and task performance

*2026-08-13. Status: everything through N3b is committed and final; N4 (diversity-GRPO) is
in flight — no results yet. ~2.5 minutes spoken.*

---

Everyone believes diversity helps LLMs solve hard problems — sample more, sample
differently, vote. Several recent methods even *train* for diversity, rewarding models for
producing distinct-looking outputs. Our question: **what kind of diversity actually buys
competence, and can you train for it without getting scammed?**

**Finding 1: the diversity everyone can cheaply get is worthless.** We tested surface
diversity — personas, prompt variation, strategy lenses — three ways: on an 8B model, at
the frontier (gpt-5.5, $200 run), and on SWE-bench. Three clean nulls. Style variation
changes how solutions *read*, not which problems get *solved*.

**Finding 2: the signal everyone uses to measure diversity is broken.** The standard
proxy — "did the samples reach different answers?" — turns out to be mostly noise: with an
LLM judge labeling actual solution *methods*, only 32% of distinct-answer pairs are
genuinely different methods; the rest are the same method with arithmetic slips. And 69%
of true method switches land on the *same* answer, invisible to the proxy. Any training
signal built on answer entropy is rewarding sloppiness, not exploration.

**Finding 3: you can build a metric style can't fool — but it has to be hardened, and
that's the paper's teeth.** We built a "logical distance" metric over reasoning traces
(activation-based heads + a claim-level kernel, aggregated by Vendi score). Stress test:
give it 4 paraphrases of one solution vs 4 genuinely different solutions. Every cheap
metric — token overlap, sentence embeddings, raw activations — hands out 88–113% of its
diversity reward for pure style edits; raw activations actually reward paraphrasing *more*
than a method switch. Our hardened ensemble, trained against adversarially mined style
attacks with judge-labeled method pairs, gets that down to **15% in the exact regime we
train in** — passing our pre-registered safety bar.

**Finding 4: where method diversity lives is not where you'd think.** Thinking mode
doesn't create it — long chains of thought converge on one canonical method and polish it.
Problem *hardness* creates it: on AIME/AMC, 72% of problems draw multiple genuine methods
from the same model, vs ~32% on standard MATH.

**The twist that makes this interesting rather than tidy:** everywhere we look, natural
method diversity *anti*-correlates with success — models explore methods when they're
lost. So diversity is a symptom, not (yet) a lever. The live experiment, running right
now, is the causal test: GRPO where correct answers earn a bonus for being
*methodologically* different from other correct answers — "solve it, and solve it a
different way" — with an LLM-judge audit watching for the failure mode we can now measure:
reward going up while genuine method diversity doesn't.

Either the reward moves real method diversity at matched accuracy — first evidence that
trained-for logical diversity is a usable lever — or we show that even a hardened metric
gets gamed under RL pressure, which is a warning the diversity-training literature
currently has no way to even detect.

---

*Receipts, in order: F0a / F2 / F3-E3 (finding 1); N-methods, `scripts/27` (finding 2);
V6 / N2 / N3b, `scripts/22,25,29` (finding 3); N-regime probe, `scripts/28` (finding 4);
N1-dry + per-regime pass@8 splits (the twist); N4, `scripts/30` (in flight). Details:
`docs/experiment_tracker.md`.*
