# Logical Distance Between Reasoning Traces — Construct Design Brainstorm

2026-07-20. New direction: a metric of "logical distance" between reasoning traces, aggregated
into a set-level diversity measure, used to (1) test diversity→pass@k causally, (2) serve as a
non-gameable diversity reward for in-context RL / GRPO post-training (the F1 lever, done right).

## 0. What job must the construct do? (design constraints from our own results)

- **C1 — beat the answer multiset.** CGE's gate achieves 90% precision from the *answer multiset
  alone*. On final-answer tasks, answer entropy is a cheap, strong diversity statistic. Logical
  distance must add predictive/causal value *conditional on* answer agreement — else it's an
  expensive proxy. Corollary: the construct's natural home is regimes with no discrete answer
  (proofs, code, agentic traces) or *mid-trace* use (dense RL reward, early stopping).
- **C2 — style-invariance is the whole point.** F0a/F2 showed persona/style manipulation moves
  surface form without moving competence. A metric that scores paraphrase ≈ method-switch is
  measuring style, and we already know style is inert. This is the lab's home dissociation.
- **C3 — ESS, not mean distance.** M1's diagnosis is "8 draws → ~1 effective draw." The set-level
  aggregate should be an effective-number-of-distinct-strategies (kernel ESS / Vendi-score-style
  exp-entropy of the similarity matrix eigenvalues), not mean pairwise distance — mean distance
  is insensitive to cluster structure (7 clones + 1 outlier vs 2 balanced strategies).
- **C4 — non-gameable under optimization.** If used as an RL reward, the model will maximize it
  the cheapest way. Embedding/distinct-n distance → style shifts (free). The metric's value as a
  reward is exactly the degree to which the cheapest way to increase it is a genuine strategy
  change. Gameability must be a measured axis, not an assumption.

Decompose the construct into two orthogonal choices:
**(A) a pairwise representation + kernel** d(trace_i, trace_j), and
**(B) a set aggregator** D({t_1..t_k}) — default: Vendi/ESS on the kernel matrix.
Most of the design work is (A); (B) is nearly settled by C3.

## 1. Candidate designs for the pairwise kernel (ranked by cost)

- **L0 — answer/outcome multiset entropy.** Already have. The deflationary baseline's first half.
- **L1 — off-the-shelf embeddings** (whole-trace + step-chunked mean-OT). Known failure mode:
  style dominates the top principal directions. Baseline's second half.
- **L1.5 — style-projected / contrastively learned metric (recommended first real candidate).**
  Supervision is free to generate: positives = LLM paraphrase/persona-rewrite of the same trace
  (logic identical, distance target ≈ 0); negatives = known-distinct solutions to the same
  problem (different method, distance target high). Then either (a) estimate a style subspace
  from positive-pair difference vectors and project it out of frozen embeddings, or (b) train a
  small metric head contrastively. Cheap, dense, differentiable-ish, directly optimizes the C2
  dissociation. Risk: learns "paraphrase detector" shortcuts; mitigate with diverse rewrite
  styles + held-out rewrite prompts.
- **L2 — claim-set distance.** LLM extracts intermediate claims/results per trace (canonicalize
  math via sympy-normalization); d = soft-Jaccard / optimal transport over claim embeddings.
  Interpretable ("these two traces share 7/9 lemmas"), medium parse cost. Natural unit for
  math/proofs; for agentic traces the "claims" become actions/observations (already discrete!).
- **L3 — reasoning-graph distance.** Nodes = claims, edges = derivation dependencies;
  d = WL-kernel or fused Gromov-Wasserstein over embedded nodes. Highest fidelity claim, highest
  parse-noise risk. Gate: parse–reparse reliability (V3 below) must be high before the structure
  can add signal over the L2 bag-of-claims; if extraction ICC is low, edges add noise.
- **L4 — model-internal embedding (white-box, fits the repo).** Encode traces with an open
  model's own residual stream (Qwen3-8B, mean-pool mid layers, CUDA_VISIBLE_DEVICES=3).
  Hypothesis: the generator-family's internal representation separates *strategy* better than
  off-shelf text embedders trained for topical similarity. Encoder ≠ generator is fine (we can
  encode gpt-5.5 traces with Qwen). Also the only candidate that later gives a *steering* handle
  (diversity in a discovered strategy-subspace), reconnecting to the persona-vector program.
- **L5 — behavioral/counterfactual distance (appendix tier).** Divergence-point depth (where do
  two traces first become logically incompatible), or continuation-KL (condition the model on
  prefix of A vs B, compare continuation distributions). Elegant, expensive, hard to validate.

## 2. Validation battery (pre-registered, before any downstream use)

- **V1 style-invariance:** paraphrase/persona-rewrite pairs must score near-zero. Metric: AUC
  separating {paraphrase pairs} from {distinct-solution pairs}. Target ≥ 0.9.
- **V2 logic-sensitivity:** known-distinct solutions score high. We have labeled cases: q70
  Mantel-template vs stars-decomposition, the template-basin flips generally; plus forced
  method-switch resamples as synthetic negatives.
- **V3 reliability:** parse–reparse / re-embedding stability (ICC) for any LLM-parsed variant.
- **V4 known-groups (free, existing data):** the 11 CGE flip traces have hand-labeled classes.
  Prediction: template-basin problems (q28/q62/q70/q39) show LOW within-pool logical diversity
  across their 8 low samples; scattered-execution-noise problems (q58/q91/...) show higher.
  A metric that can't recover this ordering is dead on arrival.
- **V5 criterion validity (the pass@k TODO, free on existing raw):** across the 100 BeyondAIME
  problems × 8 neutral traces, does set-level logical diversity predict coverage@8 / pass@k
  *after controlling for answer entropy* (partial correlation / nested regression)? Must beat
  the deflationary baseline: {answer entropy + raw-embedding Vendi}. This is the falsifier.
- **V6 gameability probe (if RL-bound):** measure Δmetric from a pure style intervention vs a
  genuine method switch at matched edit cost. Reward-worthiness = ratio.

Deflationary hypothesis to kill or accept: *logical distance is answer entropy + style-polluted
embedding noise; it adds no predictive value and no non-gameable reward signal.* If V5/V6 can't
reject this on math, pivot the construct to the no-discrete-answer regime (E3 SWE-bench traces,
proofs) where L0 doesn't exist and the construct has no cheap competitor.

## 3. Existing assets (Stage 0 is ~free)

- `runs/infdiv-f2main-BeyondAIME-neutral-k8-s0/raw.jsonl` — 100 problems × 8 traces with
  per-problem coverage@8 and answer multisets → V4/V5 testbed.
- `runs/cge-BeyondAIME/flipped_traces/` — 11 flips with class labels → V4 known-groups.
- F2 static/adaptive arms — persona-styled traces of the *same* problems → natural V1 pairs.
- E3 lens outputs (scripts/14) — second domain, no discrete answer, Link-A re-measurement.
- `deeppersona/diversity_metrics.py` — L0/L1 baselines already implemented.
- Paraphrase-pair generation: ~$3–5 of gpt-4.1-mini rewrites (est. before running, per cost rule).

## 4. Staged plan

- **Stage 0 ($0 + ~$4):** assemble testbed; generate paraphrase/rewrite pairs.
- **Stage 1 (~$0, local GPU):** L0 + L1 + L4 through the battery. L4-vs-L1 on V1/V2 is itself a
  publishable micro-result (do model internals beat text embedders at strategy separation?).
- **Stage 2 (~$5–10):** L1.5 contrastive metric + L2 claim-sets; battery again.
- **Stage 3 (conditional):** L3 graphs only if L2 fails V2 sensitivity while V3 reliability is
  high enough to support structure.
- **Adoption rule:** cheapest level passing V1∧V2 (AUC ≥ 0.9) *and* beating the deflationary
  baseline on V5. Then: (i) re-run E3 Link-A with the validated metric; (ii) wire as GRPO
  diversity reward in scripts/10 (F1 revisited with a non-gameable lever).

## 5. Related work (S2 sweep 2026-07-20 — full map in related_work_map.md §6)

Sweep verdict: **framing partially preempted, all four metric designs unclaimed.**

- **2606.29985 "Are We Measuring Strategy or Phrasing?"** owns the motivating negatives (surface
  metrics unreliable; diversity-RLVR gains are ~80% within-approach; LLM-judge diversity reward
  gets hacked) and poses a reliable directly-optimizable approach-level metric as an OPEN
  problem. Position: we answer their open problem; their judge-hacking result argues *for* a
  learned/structural (non-judge) metric.
- **RPD (2510.26122)** — step-aligned CoT divergence for SFT curation: closest existing metric,
  becomes the published baseline for L2 and the fastest collision risk (their next iteration =
  RPD-as-reward). Argues for moving fast on L2 + V-battery.
- **Rewarding the Rare (2601.08763)** — judge-clustered strategy rarity as GRPO advantage: the
  RL-use baseline to beat on cost + hackability.
- **GCPO (2605.11461)** — DPP volume over off-shelf embeddings inside GRPO: our validated
  kernel is a drop-in embedding-swap ablation there (clean downstream experiment).
- **2606.03883** — claim-DAG extraction from traces (verifiable puzzles): reuse as L3 parser.
- **Kernel Language Entropy (2405.20003)** — formal machinery for the aggregator (von Neumann
  entropy of a kernel); cite alongside Vendi.

Design implications absorbed: (1) drop LLM-judge-as-metric from the candidate list (documented
hackable); (2) V6 gameability probe is now *required*, not optional — it is the headline
differentiator; (3) the incremental-validity regression (V5, conditional on answer entropy) is
confirmed unclaimed — no published paper runs it; (4) L4's hidden-vs-text-embedder comparison is
also unclaimed and publishable standalone.
