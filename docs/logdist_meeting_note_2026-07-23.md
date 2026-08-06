# Meeting note — "Logical distance" between reasoning traces (2026-07-20 → 07-23)

**TL;DR.** We built and validated a style-invariant, logic-sensitive distance metric between
LLM reasoning traces. The best construct (a 3-part ensemble) separates paraphrases from
genuinely different solutions at AUC 0.97. Two honest nulls came out along the way: on
final-answer math, logical diversity predicts coverage *no better than the answer multiset*
(pre-registered falsifier confirmed), and derivation-graph structure adds nothing over the bag
of claims. Total spend: ~$4 in API calls + local GPU. Everything below is on branch
`yiren/apa-proposal` (scripts/15–19, `runs/logdist-testbed/`, `docs/logical_distance_*.md`).

## 1. Why

We want a measure of *diversity of reasoning* among a set of traces, to (a) test whether
reasoning diversity causally improves pass@k, and (b) use as a diversity reward in GRPO /
in-context RL. Design constraints from our own earlier results:

- **Beat the answer multiset.** Our consensus-gating work showed answer entropy alone is a
  strong, free diversity statistic on math. A logic metric must add value *conditional* on it.
- **Style-invariance is the whole point.** Our persona work showed style moves surface form
  without moving competence. A metric that scores a paraphrase like a method-switch measures
  the wrong thing — and as an RL reward it would be gamed by register shifts.
- **Aggregate by effective count, not mean distance.** Set-level diversity = Vendi score
  (exp von-Neumann entropy of the pairwise similarity matrix) = "effective number of distinct
  strategies," which is sensitive to cluster structure (7 clones + 1 outlier ≠ 2 balanced
  strategies).

**Positioning** (from a 15-query literature sweep): arXiv 2606.29985 shows surface diversity
metrics are unreliable and LLM-judge rewards get hacked, and poses a reliable approach-level
metric as an open problem — we answer it. Closest existing metric: RPD (2510.26122). All four
of our metric designs and the conditional-validity regression were unclaimed as of the sweep.

## 2. Testbed (Stage 0)

BeyondAIME, 100 problems × 8 neutral gpt-5.5 traces (existing data), plus:

- **Paraphrase positives** (logic identical by construction): 50 problems × 4 rewrite styles —
  concise-formal, verbose-pedagogical, casual-student, and **restructured+renamed** (reorder
  presentation, rename variables). The last style, "D," is held out of ALL metric training.
  200/200 rewrites pass a boxed-answer fidelity check.
- **Distinct-logic negatives** (free): within-problem trace pairs with *different final
  answers* — guaranteed logically divergent. 571 pairs after normalizing answer-equivalence
  (we caught 17 formatting-artifact pairs, e.g. `28357376` vs `28,357,376`).
- **Validation battery** (pre-registered): V1 style-invariance + V2 logic-sensitivity (AUC
  separating paraphrase pairs from distinct-answer pairs; bar = 0.90), V3 parse–reparse
  reliability, V4 known-groups, V5 criterion validity (does trace-set Vendi predict coverage@k
  *after controlling for answer entropy*), V6 gameability (pending).

## 3. The candidate ladder and what happened

| Level | Kernel | V1∧V2 AUC (held-out style D) | Verdict |
|---|---|---|---|
| L1 | Off-shelf text embeddings (Qwen3-Emb, mpnet) | 0.79–0.82 (—) | Fail: style dominates — a pedagogical rewrite looks as distant as a different solution |
| L4 raw | Qwen3-8B hidden states, mean-pooled | 0.19–0.51 (—) | *Inverted*: register moves the representation more than logic |
| L1.5 | Contrastive linear head on hidden states (trained on paraphrase-positives / distinct-answer-negatives, styles A–C only) | held-out D 0.80 | Works — model internals contain a linearly extractable logic representation that text embeddings lack (same recipe on text embeddings: 0.64) |
| L2 | LLM-extracted atomic claims → embedded → Chamfer set-distance | 0.946 (D 0.85) | First kernel past the bar; interpretable ("these traces share 7/9 lemmas") |
| L3 | Derivation graph (edges over the L2 claims) | 0.941 (D 0.86) | **Informative null**: parse–reparse reliability is high (r = 0.99), so the null is real — edges are stable but *redundant* given the claims. Graph structure adds nothing. |
| L4 chunks | 128-token chunk embeddings + Chamfer, no LLM | mpnet raw 0.875; trained Qwen head held-out D 0.73 | The set-of-parts aggregation, not claim extraction, does most of L2's work; the $0 kernel alone falls short but is complementary |

**Adopted construct: 3-way ensemble** (L2 claim-Chamfer + L1.5 trace head + chunk head):
**0.972 overall full-set (worst style 0.93); 0.929 on fully held-out problems (D 0.79)**.
The previous 2-way ensemble is 0.964 / 0.917 — the chunk kernel adds real, non-collinear
signal (unlike L3's edges).

## 4. Methods in detail (paper-style)

**Setup and notation.** Let $T = \{t_1, \dots, t_k\}$ be a set of reasoning traces for one
problem. Each candidate defines a pairwise distance $d(t_i, t_j) \in [0, 1]$; the induced
similarity kernel is $K_{ij} = 1 - d(t_i, t_j)$. Set-level diversity is the Vendi score
$\mathrm{VS}(T) = \exp\!\big(-\sum_i \lambda_i \log \lambda_i\big)$, where $\lambda_i$ are the
eigenvalues of $K/k$ — the effective number of distinct strategies in $T$. All learned
components are trained only on 25 of the 50 paraphrase problems and on rewrite styles A–C;
style D (restructure + rename) and the remaining 25 problems are never seen in training.

**L0 — answer-multiset entropy (deflationary baseline).** Final answers are grouped into
equivalence classes by a normalized comparator (comma/whitespace stripping plus symbolic
equivalence via a sympy-based grader, checked in both directions); $\mathrm{L0}(T)$ is the
Shannon entropy of the class distribution. No trace content is used.

**L1 — off-the-shelf text embeddings.** Two encoders: (i) Qwen3-Embedding-0.6B, last-token
pooling with left padding, 8,192-token context, whole trace; (ii) all-mpnet-base-v2 applied to
384-token chunks, mean-pooled within chunk and averaged across chunks. Embeddings are
$\ell_2$-normalized; $d = 1 - \cos$.

**L1.5 — contrastive linear metric head.** A single linear map $W \in \mathbb{R}^{256 \times
D}$ (no bias) over a frozen base embedding $e(t)$, with $f(t) = \mathrm{norm}(W e(t))$.
Positives: all pairs within a paraphrase clique {original trace, rewrites A, B, C} per
training problem (150 pairs); negatives: distinct-answer pairs from training problems.
Loss $= \mathbb{E}_{\text{pos}}[1 - \cos(f_a, f_b)] + \mathbb{E}_{\text{neg}}[\max(0,
\cos(f_a, f_b) - 0.4)]$; Adam, lr $10^{-3}$, weight decay $10^{-4}$, 400 full-batch steps.
Bases compared: the L1 text embedding and the L4 hidden-state embedding (below). The head
succeeds only on the hidden-state base — evidence the generator family's internal
representation linearly encodes solution logic where text embedders do not.

**L2 — claim-set Chamfer distance.** An extractor LLM (gpt-4.1-mini, temperature 0, JSON
mode) parses each trace into a list of atomic claims under canonicalization instructions:
express claims in the problem's original quantities; describe auxiliary variables by their
defining property rather than their letter name (defeats variable renaming); plain digits;
exclude prose, restatements, and abandoned dead ends. Claims are embedded with
Qwen3-Embedding-0.6B (~14 claims/trace). With $S_{uv}$ the cosine similarity between claim
$u$ of $t_a$ and claim $v$ of $t_b$, the distance is the symmetric Chamfer
$$d(t_a, t_b) = 1 - \tfrac{1}{2}\Big(\mathrm{mean}_u \max_v S_{uv} + \mathrm{mean}_v \max_u
S_{uv}\Big),$$
i.e., how well each trace's claims are covered by the other's. Interpretable by construction:
the argmax matching exhibits which lemmas are shared.

**L3 — derivation-graph distance.** Holding the L2 claims frozen, a second extractor pass
outputs direct derivation dependencies as edges $(u \to v)$ over the numbered claim list
(0-based indices, no self-loops or cycles; invalid indices dropped). Two variants: (a)
*edge-Chamfer* — each edge is represented by the normalized concatenation $[e_u; e_v]$ of its
endpoint claim embeddings and the same symmetric Chamfer is applied to the edge sets; (b)
*fused Gromov–Wasserstein* (POT implementation) with feature cost $M = 1 - \cos$ between
claim embeddings, intra-graph structure given by shortest-path matrices (Floyd–Warshall;
disconnected pairs set to $n$, normalized by $n$), uniform marginals, $\alpha = 0.5$.
Reliability (V3) is estimated by re-extracting all edges at temperature 0.7 and correlating
the two parses' distances over all evaluation pairs.

**L4 — model-internal embeddings.** Traces are encoded by Qwen3-8B (bf16) — the encoder need
not be the generator — taking attention-masked mean-pooled hidden states at layers
{9, 18, 27, 36} (6,144-token context). Two granularities: (i) *trace-level*, one vector per
trace (the L1.5 base above; also probed with a style-subspace ablation: SVD of
paraphrase-minus-original difference vectors on training problems, top-$k$ directions
projected out); (ii) *chunk-level*, layer-18 states mean-pooled over non-overlapping
128-token windows (~5 chunks/trace given ~550 visible tokens), giving a set-of-parts
representation scored with the same Chamfer as L2 — an extraction-free analogue that isolates
the contribution of the aggregation from that of the LLM parse. The chunk head is trained
with the L1.5 recipe but with the (subdifferentiable) Chamfer similarity in the loss,
minibatches of 48 positive + 48 negative pairs, and early stopping by validation AUC on 5
held-out training problems (checked every 25 steps) — without it the max operator memorizes
the training problems.

**Ensembles.** Component distances are $z$-scored using moments computed on the
negative-pair distribution only (no positive-pair statistics leak into the combination) and
averaged. The adopted construct is the 3-way ensemble {L2 claim-Chamfer, L1.5 trace head,
L4 chunk head}.

**Evaluation protocol.** V1∧V2 is the ROC-AUC of $d$ separating paraphrase pairs (label 0)
from distinct-answer pairs (label 1), reported per rewrite style and overall; "held-out" rows
restrict both positives and negatives to the 25 problems disjoint from all training. V5 uses
within-problem subset regression: for each problem with mixed correctness, all
$\binom{8}{4} = 70$ trace subsets are scored (subset Vendi, subset answer entropy,
coverage@4), and the Vendi–coverage association is tested within (problem × answer-entropy)
strata with a bootstrap CI over strata.

## 5. The honest nulls (these matter for the paper)

1. **V5 on math: answer entropy fully mediates.** Within-problem subset test (31 problems ×
   all 70 4-subsets of their 8 traces): subset Vendi correlates with coverage@4 (mean ρ 0.24),
   but subset *answer entropy* does better (0.38), and stratified on answer entropy the Vendi
   effect vanishes (mean −0.04, 95% CI [−0.26, 0.17]). On final-answer math, "logical
   diversity predicts pass@k" is an illusion of the answer multiset. Per our pre-registered
   rule, the construct is *not* adopted for math coverage-prediction; its home is (a) regimes
   with no discrete answer (SWE-bench localization, proofs) and (b) RL-reward use, where the
   value is non-gameability, not prediction.
2. **L3: reasoning-graph structure is redundant.** Contra the implicit assumption in the
   claim-DAG literature, edges carry no discriminative signal beyond the bag of claims — and
   we can say that cleanly because extraction reliability was high.

3. **(added 2026-07-30) Changing regime does not escape the mediator.** We re-ran the V5 design
   on SWE-bench localization (`scripts/20_locdiv_linkb.py`), chosen because pooled recall has no
   discrete answer for answer-entropy to proxy. Two findings. (a) *The traces were never saved* —
   that run persisted only the last 500 chars of each completion, so the logical-distance kernel
   could not be applied; this is an outcome-level (predicted-file-set) analysis only. (b) The
   total effect is positive — Vendi over the Jaccard kernel of prediction sets buys **+0.033
   union recall per SD** [+0.017, +0.048], placebo −0.002 — but it is **fully absorbed by
   dispersion in per-sample recall** (div → +0.005 [−0.002, +0.010] once sd(acc) enters;
   within-stratum corr = 0.55). sd(acc) is equally readable as the mediator through which
   diversity acts on a max-like pooled metric, or as a confounder; the quality-matched cells that
   would separate them contain 1 problem. So the mediator changed identity (answer multiset →
   quality dispersion) rather than disappearing.

4. **(added 2026-07-30) …but the aggregate null hides a narrow regime where pooling does pay.**
   Moderator analysis (`scripts/21_locdiv_moderators.py`) on why only 17/50 problems carry
   information: the other 33 are not the hard ones — they are the ones where **the model does not
   vary**. Of 100 (problem × arm) strata, 38 sit at ceiling and 37 are constant at an intermediate
   value (stuck at recall 0.50, 89% of them with a *permanent blind spot* — gold that no sample
   ever proposes). Uninformative problems have higher per-sample recall (0.71 vs 0.63) but a lower
   pooled ceiling (0.73 vs 0.92). Where the effect lives: an **inverted U in difficulty** (hard
   +0.008, mid +0.049, easy +0.003 per SD — the linear interaction is null *because* it is
   non-monotonic), **absent a shared blind spot** (+0.053 vs +0.004; div×blind −0.021
   [−0.032,−0.012]), and on **small gold sets**. Within the high-headroom subgroup the effect even
   survives the dispersion control (+0.019 [+0.002,+0.033]) — the only place it does — though
   "headroom" is defined from the pooled outcome, so that selection is partly definitional and is
   not a clean discovery. Net: diversity pays only in a two-sided window, and ~66% of this
   benchmark sits outside it, which is enough to produce the aggregate null by itself.

5. **(added 2026-07-30) The pooling gain is "cast a wider net", not reasoning diversity.** The
   cleanest test we can run on fixed pools: diversity-based *selection* needs no gold, so
   "pool the most diverse 4 of 8" vs "pool a random 4" is an identified comparison of two decision
   rules. Max-Vendi wins by **+0.020 union recall [+0.009, +0.033]** (min-Vendi loses 0.051) — a
   real, deployable, oracle-free effect. But **max-|union| — literally picking the 4 samples that
   name the most distinct files — wins by the same +0.020**, and "just emit more files per sample"
   recovers 87% of it. The structured kernel adds nothing over `len(union(preds))`. So the effect
   is a coverage-counting fact, not a reasoning fact, and it is trivially gameable because pooled
   recall does not penalize precision.

6. **(added 2026-07-31) The metric is not yet reward-safe — 58% of its diversity score is buyable
   for free.** V6, set level: four paraphrases of a single solution score 2.37 where four genuinely
   distinct solutions score 3.36 (4 identical traces = 1.00 by construction). Held-out problems:
   74% hackable. This is a null *about our own construct*, and it is the one that blocks the RL
   application — but the same table is a strong positive for the framing, since every off-shelf
   baseline is 88–113% hackable. Also a scope note: the attack set is four fixed rewrite styles,
   so 58% is a **lower bound** on what an adaptive attacker would extract.

The first two are publishable framing, not failures: "the metric works; on math the answer
multiset is already a sufficient statistic — measured, not assumed; claims are enough, edges are
not needed." The third is the methodological punchline: **every diversity→performance result we
have is observational and mediator-ambiguous**, which is what motivates the interventional
design below.

## 6. Caveats

- Held-out-problem performance on the hardest style (restructure+rename) is the weak spot for
  every variant (0.75–0.79); the ≥0.93-per-style claim is full-set. Likely bottleneck:
  supervision volume (150 positive pairs from 25 problems), not architecture.
- Single domain (competition math, one generator model) so far.
- V6 ran 2026-07-31 (below). The weak spot above is exactly what V6 quantifies: the held-out
  restructure+rename style is also the strongest free attack.

## 7. Next steps (in value order)

1. **Interventional matched-accuracy test** (promoted to #1 by the 07-30 result). Generate two
   matched pools per problem — paraphrase-resamples of one strategy vs forced method-switch
   samples — with per-sample accuracy matched by construction, and measure coverage. This is the
   only design that breaks the mediator ambiguity that defeated V5 on both math and localization.
   Estimated ~$10–20; needs cost sign-off per the spending rule. **Sample problems into the mid
   band** (per-sample accuracy 0.5–0.85, no shared blind spot, small gold set) per the 07-30
   moderator result — a uniform draw spends most of the budget in regimes where the effect cannot
   exist by construction, and the blind-spot rate caps what is achievable.
2. ~~V6 gameability probe~~ — **done 2026-07-31, `scripts/22_logdist_v6.py`, $0. Partial fail.**
   Set-level (k=4 packs, Vendi; 4 identical traces = 1.00): a pack of 4 paraphrases of *one*
   solution vs 4 genuinely distinct solutions. Every cheap baseline is ~fully gameable —
   token-Jaccard 97% hackable, mpnet 93%, raw chunk 88%, raw Qwen 113% (style moves it *more*
   than a method switch). **That contrast is the anti-GCPO result and it is now quantitative.**
   But the adopted ensemble is only R = 1.72 [1.41, 2.02] → **58% of the reward is free**
   (74% on held-out problems; R = 1.47 [1.15, 1.87] against the held-out restructure+rename
   style). Padding (trace ++ paraphrase) does *not* work as an attack — Chamfer is
   duplication-invariant — so the SWE-bench "wider net" failure does not transfer here.
   **Reading: passes as a measurement instrument, fails as a standalone dense RL reward.**
   Caveat: static probe with 4 fixed styles ⇒ a *lower bound* on hackability; n = 18 matched /
   6 held-out problems.
3. **Harden the metric against style attack, then re-run V6 as the acceptance test** (promoted
   from #4 by the V6 result). Adversarial style mining — more, and deliberately held-out, rewrite
   styles as training positives — is the cheap move (~$1–2, same budget as the supervision-scaling
   item it subsumes). Target R_set ≥ 4–5 (≤25% hackable) against unseen styles before wiring
   anything into GRPO as a dense reward (scripts/10). Threshold is post-hoc, not pre-registered.

Operational lesson from 07-30: any future generation run must **persist full completions**, not a
truncated tail — the E3 traces are unrecoverable without paying to regenerate them.
