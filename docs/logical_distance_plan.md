# Logical Distance (LD) — Plan & Progress

Living doc. Design rationale in `docs/logical_distance_brainstorm.md`; related work in
`docs/related_work_map.md` §6. Started 2026-07-20.

## Construct (frozen 2026-07-20)

- **Pairwise kernel** d(t_i, t_j) between reasoning traces: style-invariant, logic-sensitive.
- **Set aggregator**: Vendi / kernel-ESS (exp von-Neumann-entropy of the k×k similarity matrix)
  = "effective number of distinct strategies" (M1 was an ESS statement).
- Candidates: L0 answer entropy (baseline) · L1 off-shelf embeddings (baseline) ·
  **L1.5 contrastive metric** (paraphrase-positives / distinct-answer-negatives) ·
  **L2 claim-set OT** · L3 reasoning-graph kernel (gated on parse reliability) ·
  **L4 hidden-state embedding** (Qwen3-8B residual stream as encoder).
- **Deflationary baseline to beat**: answer-multiset entropy + raw-embedding Vendi.
- Validation battery V1–V6 (brainstorm doc §2). Adoption rule: cheapest level with
  V1∧V2 AUC ≥ 0.9 that beats the deflationary baseline on V5.

Positioning (from S2 sweep 2026-07-20): we answer the open problem posed by 2606.29985
(surface metrics gamed; judge rewards hacked); L2 baseline = RPD 2510.26122; RL-use baselines =
Rewarding-the-Rare 2601.08763, GCPO 2605.11461 (embedding-swap ablation target). All four
metric designs + the V5 conditional regression are unclaimed as of the sweep.

## Stages

- **Stage 0 — testbed** (this stage): assemble labeled trace corpus from existing raw + generate
  paraphrase pairs. `scripts/15_logdist_testbed.py` → `runs/logdist-testbed/`.
- **Stage 1 — baselines + L4** (local GPU, $0): L0/L1/L4 through V1–V5.
- **Stage 2 — L1.5 + L2** (~$5–10 parse/train cost): battery again; adopt or escalate.
- **Stage 3 — L3 graphs** (conditional on L2 failing V2 with V3 reliability high).
- **Downstream**: (a) V5 = diversity→coverage@8 conditional on answer entropy (TODO item 3);
  (b) re-measure E3 Link-A with validated metric; (c) GRPO diversity reward in scripts/10.

## Stage 0 spec

Testbed sources (all existing raw, BeyondAIME n=100):
- `infdiv-f2main-BeyondAIME-neutral-k8-s0` — 800 traces, core corpus (per-problem coverage@8,
  answer multisets → labels).
- `infdiv-f2main-BeyondAIME-{static,adaptive}-k8-s0` — persona-styled arms (secondary style
  contrasts; NOT V1 positives — they are resamples, not paraphrases).
- `cge-BeyondAIME/raw.jsonl` — 112 medium-effort escalation traces (roles med0/med1/reinterp/
  verify) on the 28 escalated problems.
- Flip-class labels (hand-labeled, CGE analysis): template-basin {28, 39, 62, 70};
  scattered-execution {7, 26, 49, 58, 74, 81, 91}; q62 dual-labeled (also
  verification-at-depth failure).

Generated:
- **Paraphrase positives**: 50 problems × 1 trace × 4 rewrite styles (concise-formal /
  verbose-pedagogical / casual-student / restructured+renamed; style D held out from any metric
  training). Constraint: preserve every step, case split, and \boxed answer; change only
  wording/format/notation. Fidelity filter: boxed answer of rewrite ≡ original.
  Model gpt-4.1-mini. Est. cost ≈ $0.7 (200 rewrites × ~2k-tok traces).
- **Distinct-logic negatives** (free): within-problem neutral-k8 pairs with *different extracted
  answers* — guaranteed logically divergent.
- **Same-answer pairs** (unlabeled tier): within-problem same-answer pairs — method-identity
  unknown; used descriptively, not for training.

## Progress log

- **2026-07-20** — Direction opened (user). Brainstorm doc written; construct decomposed into
  pairwise kernel + ESS aggregator; battery V1–V6 defined.
- **2026-07-20** — S2 sweep (subagent, 15 queries): framing partially preempted by 2606.29985,
  all four metric designs + V5 regression unclaimed; must-cites recorded in related_work_map §6.
  Consequences: LLM-judge-as-metric dropped; V6 promoted to required.
- **2026-07-20** — Stage 0 started: `scripts/15_logdist_testbed.py` (assemble + paraphrase).
- **2026-07-20** — **Stage 0 DONE.** `runs/logdist-testbed/`: 2512 traces (neutral/static/
  adaptive k8 + 112 CGE escalation), problem_labels.json (coverage@8 = 90/100 reproduces known
  0.900; answer entropy; flip classes 4 basin + 7 scattered), 588 distinct-answer negative
  pairs, 2212 same-answer pairs, 200/200 paraphrase positives (50 problems × 4 styles,
  gpt-4.1-mini, $0.35; fidelity via normalized boxed match — strict string match had 9 false
  rejects from thousands-separators/\text spacing). Style D (restructured+renamed) reserved as
  held-out style for L1.5 training. Next: Stage 1 — L0/L1/L4 kernels through V1/V2/V4/V5 on
  local GPU (CUDA_VISIBLE_DEVICES=3).
- **2026-07-20** — **Stage 1 DONE** (`scripts/16_logdist_stage1.py`, metrics in
  `runs/logdist-testbed/stage1_metrics.json`). All zero-shot encoders FAIL the battery:
  - **L1** (Qwen3-Embedding-0.6B) V1V2 AUC 0.772; **l1chunk** (mpnet) 0.803 — style dominates
    (pedagogical/restructured rewrites ≈ as distant as logically-distinct solutions); median
    d(paraphrase) ≈ d(same-answer resample) — cannot tell rewrite from resample.
  - **L4 raw** (Qwen3-8B mean-pooled hidden states, layers 9/18/27/36) INVERTED: AUC 0.19–0.51
    — register moves the representation more than logic. But within-register ordering is
    correct (d same-answer < d distinct-answer), i.e., big style axis + real content axis.
  - **Style-subspace projection probe** (fit SVD of paraphrase-diff vectors on styles A/B/C ×
    25 train problems; test held-out): rescues L4-L18 on *seen* styles (B 0.11→0.79,
    A→0.94, C→0.94 at k=20) — the style axis is low-dim and linearly removable from hidden
    states (NOT from L1 text embeddings, which barely move). But held-out style D
    (restructured+renamed) stays ≤0.59 for every encoder: register projection does not
    transfer to reorder/rename invariance.
  - V4/V5: no encoder's Vendi separates flip classes or adds signal over answer entropy;
    Vendi↔coverage raw correlation is NEGATIVE (difficulty confounds both — V5 must stay
    conditional).
  - **Verdict**: deflationary baseline confirmed weak → good for positioning; the D-failure is
    the empirical motivation for Stage 2: (i) L1.5 contrastive training with a broader style
    support, and especially (ii) **L2 claim-set distance**, where reorder/rename invariance
    holds *by construction* (extraction + canonicalization). Stage 2 est. cost ~$5–10.
- **2026-07-20** — **Stage 2 DONE** (`scripts/17_logdist_stage2.py`; claim extraction $0.89,
  total stage spend < $1.50). **Pairwise kernel: solved. Criterion payoff on math: absent —
  the C1 prediction confirmed.**
  - **L2 claim-set Chamfer** (gpt-4.1-mini extraction, ~14 claims/trace, Qwen3-Emb claims):
    V1V2 AUC **0.933** overall (A .958 / B .957 / C .984 / D .833); paraphrase median 0.12 vs
    distinct-logic 0.28. First kernel past the 0.9 bar.
  - **L1.5 contrastive** (256-d linear head, 150 pos pairs, styles A/B/C × 25 train problems):
    on l4_L18 base generalizes to held-out style D **0.829** (l1 base only 0.633) — model
    internals contain a linearly-extractable logic representation that text embeddings don't.
  - **Ensemble** (z-avg of the two): overall **0.954**, D 0.893. L2 and L1.5 fail on
    *different* pairs (L2 better on register shifts, L1.5-hidden better on restructure/rename).
  - **V4** still null (basin 2.42 vs scattered 2.56, right direction, p=0.39 — n=4v7
    underpowered). **V5 within-problem subset test** (the decisive form; 31 mixed problems ×
    C(8,4)=70 subsets): subset chamfer-Vendi→coverage@4 mean ρ=0.235 (74% positive) BUT subset
    answer-entropy does better (0.384), and **stratified on (problem × answer entropy) the
    Vendi effect vanishes: mean −0.04, 95% CI [−0.26, 0.17]** (24 strata). On neutral-prompt
    math pools, logical diversity→coverage is fully mediated by the answer multiset.
  - **Adoption verdict per pre-registered rule**: L2 (and ensemble) pass V1∧V2 but FAIL V5 on
    math — per brainstorm §2, the construct pivots to regimes without a discrete answer
    (E3 SWE-bench localization traces; proofs) and to the RL-reward use where the value is
    non-gameability (V6), not coverage prediction. Honest headline: "the metric works; on
    final-answer math the answer multiset is already a sufficient statistic — measured, not
    assumed."
  - Next candidates: (a) V6 gameability probe (style-attack vs method-switch deltas — free,
    uses existing pairs); (b) L2 Link-A re-measurement on E3 lens traces (free, locdiv runs);
    (c) wire ensemble kernel as GRPO diversity reward in scripts/10 (F1 revisit).
- **2026-07-23** — **Negative-pair cleaning** (found via example export: some "distinct-answer"
  pairs were formatting artifacts — `28357376` vs `28,357,376`, two phrasings of "no such k").
  Pair construction + answer entropy now use normalized answer-equivalence classes
  (comma/space-strip + sympy grader both directions; `canon_classes` in scripts/15); labels
  store per-sample `answer_class`. 588 → 571 negatives (17 reclassified). **Corrected battery
  (these supersede the Stage-2 numbers above):** L2 chamfer overall **0.946** (A .969 / B .973
  / C .991 / D .852); L1.5-l4_L18 held-out A .907 / B .832 / C .887 / **D .796**; ensemble
  overall **0.964**, D **0.904** — every kernel improved (artifact pairs were depressing AUCs);
  ensemble now ≥0.9 on all four styles (full-set). V5 subset-mediation null UNCHANGED
  (stratified vendi→cov mean −0.04, CI [−0.26, 0.17]) — the pivot verdict stands. NB: 16-eval
  full-set rows for `l15_*` embeddings include train pairs (leaky); cite only the held-out
  numbers from 17 --train-l15.
- **2026-07-23** — **Stage 3 (L3 graphs) DONE — informative null** (`scripts/18_logdist_l3.py`,
  edges-only extraction over FROZEN L2 claims isolates the structure contribution; parses a
  (temp 0) + b (temp 0.7), $1.16 total; POT installed for FGW).
  - **V3 reliability HIGH**: distance-level parse-reparse pearson 0.99 (edge-Chamfer) / 0.91
    (FGW); raw edge-set Jaccard mean 0.85. Edge extraction is stable — the gate condition for
    trusting L3's signal is met.
  - **Battery**: l3edge overall 0.941 (D .856) ≈ L2's 0.946 (D .852) — statistically a tie;
    l3fgw 0.895 (D .780) — the GW structure term *hurts*. 3-way ensemble (L2+L1.5+l3edge)
    0.958 < 2-way 0.964 — l3edge is collinear with L2 (same claim embeddings), adds nothing.
  - **Verdict (clean, because V3 is high)**: derivation-graph structure carries no
    discriminative signal beyond the bag of claims on this testbed — a real property, not
    parse-noise attenuation. L3 CLOSED for math traces. Adopted construct: **2-way ensemble
    (L2 claim-Chamfer + L1.5 hidden-state head), overall 0.964, all styles ≥0.90 full-set.**
    Publishable framing: "claims are enough; edges are stable but redundant" — contra the
    reasoning-graph literature's implicit assumption (2606.03883 et al.) that structure is
    where the signal lives.
- **2026-07-23** — **Stage 4 (L4 completed: LLM-free chunked hidden-state Chamfer) DONE**
  (`scripts/19_logdist_l4chunk.py`, $0, local GPU). Question: is L2's win the LLM claim
  extraction or just set-of-parts + Chamfer — and is there a $0 differentiable kernel good
  enough for GRPO reward use? 128-tok chunks (~5/trace; visible text is only ~550 emb-tokens —
  gpt-5.5 compl_tok mostly hidden reasoning), Qwen3-8B L18 vs mpnet chunk bases.
  - **Aggregation does most of L2's work**: raw mpnet chunk-Chamfer 0.875 full (vs 0.817
    same-embedder whole-trace, L2 0.946) — untrained, no LLM. Raw Qwen-chunk Chamfer still
    register-dominated (0.698, B inverted 0.325) — chunking alone doesn't clean hidden states.
  - **Chunk-level contrastive head** (same recipe as L1.5, + early stopping on 5 val problems —
    without it the Chamfer max memorizes the train problems, loss→0.001): Qwen base heldout
    A .862 / B .738 / C .836 / D .728 — below trace-level L1.5 (D .796); mpnet base early-stops
    at step 0 (training never helps a text-embedding base — replicates the Stage-2 l1 result).
  - **But the chunk kernel is complementary, unlike L3 edges**: 3-way ensemble
    (L2 + l15-trace + l4chunk-head) **full 0.972 (D .932), heldout 0.929 (D .793)** — beats the
    adopted 2-way on both views (2-way heldout 0.917 / D .749). l2+l4ch ≈ l2+l15 (0.917
    heldout) — the chunk head can substitute for the trace head.
  - **$0 LLM-free candidate** (l15 + l4chunk, no L2): full 0.948 / D .914 but heldout 0.837 —
    fails the 0.9 bar out-of-problem. Bottleneck is supervision volume (150 pos pairs,
    25 problems), not architecture.
  - **Honesty note**: held-out-problem D is the weak spot for every combo (.75–.79); the
    "all styles ≥0.90" claim for the adopted ensemble is full-set only. New adopted construct:
    **3-way ensemble (L2 claim-Chamfer + L1.5 trace head + l4chunk head), full 0.972,
    heldout 0.929.** For the GRPO-reward path, next lever = scale paraphrase supervision
    (~$1–2 for 50 more problems × more styles) to push the LLM-free pair past 0.9 heldout.
- **2026-07-30** — **E3 Link B re-measurement (SWE-bench localization) — UNIDENTIFIED, not a
  positive** (`scripts/20_locdiv_linkb.py`, $0, `runs/locdiv-swebv-n50-k8`). Goal: re-run the V5
  criterion-validity test in the regime with no discrete answer, where the answer-multiset
  mediator that killed V5 on math cannot exist by construction.
  - **Blocker found first**: the locdiv run stored only `text_tail` = last 500 chars of each
    completion (the file list). **No reasoning traces exist on disk for E3**, so the validated
    logical-distance kernel cannot be applied here at all. This is an *outcome-level* analysis
    (diversity of predicted file sets), not a logical-distance analysis. Gold sets were
    recovered offline from the cached SWE-bench_Verified patches (per-sample recall reproduces
    exactly, asserted in-script).
  - Design: within each (problem × arm) stratum, all C(8,4)=70 4-subsets; y = union recall,
    x = Vendi over the Jaccard kernel of prediction sets, controls = mean per-sample recall and
    mean |preds|; cluster-bootstrap on problem (50 clusters).
  - **Total effect is positive**: div = **+0.033 union recall per SD** [+0.017, +0.048];
    survives over-control for |union| (+0.024 [+0.010,+0.036]); placebo (div permuted within
    stratum) −0.002. Comparable to the accuracy control (+0.044/SD).
  - **But it is entirely absorbed by per-sample quality dispersion**: adding sd(per-sample
    recall) drops div to **+0.005 [−0.002, +0.010]** while sd(acc) itself takes +0.051.
    Within-stratum corr(div, sd_acc) = 0.55.
  - **Observational data cannot break the tie.** sd(acc) is equally readable as (a) a *mediator*
    — genuine strategy spread makes some samples hit and others miss, which is exactly how
    diversity would help a max-like pooled metric, so conditioning on it blocks the mechanism
    and +0.033 is the correct total effect; or (b) a *confounder* — exogenous sloppiness
    produces both dispersion and different predictions. The design that separates them
    (quality-matched subsets: sd(acc) exactly 0, diversity varying) has **50 subsets in 2 cells
    from 1 problem** — no identification.
  - **Why the testbed is too weak**: gold sets are 2–6 files so recall is coarse (n_gold=2 →
    recall ∈ {0,.5,1}); 52% of 4-subsets sit at union recall 1.0; only **17/50 problems** have
    any within-stratum variance to explain.
  - **Verdict**: the "change regime to escape the L0 mediator" move does not by itself rescue
    the diversity→performance claim — the mediator just changes identity (answer multiset on
    math → quality dispersion here). Both nulls are *observational* nulls with a mediator
    ambiguity, which is now the strongest argument for the **interventional** design
    (matched-accuracy pools: paraphrase-resample vs forced method-switch). Any future E3-style
    run must **persist full completions**, not `text_tail`.
- **2026-07-30 (cont.)** — **E3 moderator analysis: the aggregate null hides a narrow regime
  where pooling pays** (`scripts/21_locdiv_moderators.py`, $0). Prompted by "17/50 informative —
  is that about the nature of the problems?" Answer: yes, and the 33 uninformative ones are not
  the hard ones.
  - **Why strata are constant** (100 = 50 problems × 2 arms): 38 sit at the ceiling (every
    4-subset recovers all gold), **37 are constant at an intermediate value** — stuck at mean
    recall 0.50, and **89% of those have a permanent blind spot** (gold no sample ever proposes)
    — 2 at floor, 23 variable. So "uninformative" = *the model does not vary*, not *the problem
    is hard*: uninformative problems have **higher** per-sample recall (0.710 vs 0.625) but
    **lower** pooled ceiling (0.727 vs 0.922) and higher self-similarity (0.840 vs 0.769).
  - **Inverted-U in difficulty** (total effect, per SD of Vendi): hard (per-sample acc < 0.5)
    +0.008 · mid (0.5–0.85) **+0.049** · easy (≥0.85) +0.003. The linear div×acc interaction is
    null (+0.003 [−0.012,+0.010]) *because* the relation is non-monotonic — do not read that as
    "difficulty doesn't moderate".
  - **A shared blind spot kills it**: +0.004 with a blind spot vs **+0.053** without
    (div×blind = −0.021 [−0.032,−0.012]). Consistent with E3 Link A's 23.6% of gold files missed
    by all 16 samples. Diversity within the sampling distribution cannot reach what the
    distribution never proposes.
  - Smaller effect on larger gold sets (n_gold≥3 +0.009 vs n_gold=2 +0.045;
    div×n_gold = −0.013 [−0.036,−0.005]).
  - **Headroom dominates but is partly definitional**: headroom>0.15 → +0.073 vs +0.001.
    headroom := (pooled recall over all 16) − (mean per-sample recall), so selecting on it partly
    selects on the outcome. **Flag, do not report as a discovery.**
  - **The one estimate that survives the strict control**: within headroom>0.15,
    div | sd(acc) = **+0.019 [+0.002, +0.033]** (n=17 problems) — everywhere else the
    dispersion-controlled effect is indistinguishable from zero (no-blind-spot +0.010
    [−0.006,+0.020]; mid-difficulty +0.010 [−0.001,+0.020]). Same definitional caveat applies.
  - **Reading**: diversity pays only in a two-sided window — the model must vary at all (else
    nothing to pool), the gold must be reachable by the sampling distribution (no blind spot),
    and the problem must not already be saturated. ~66% of this benchmark sits outside that
    window, which is enough to produce the aggregate null on its own.
  - **Design consequence for the interventional run**: sample problems *into the mid band*
    (per-sample accuracy 0.5–0.85, no shared blind spot, small gold set) rather than uniformly —
    a uniform draw spends most of the budget where the effect cannot exist by construction. Also
    pre-screen and report the blind-spot rate, since it caps the achievable effect.
- **2026-07-30 (cont. 2)** — **Selection test: the pooling gain is "wider net", not diversity.**
  Prompted by "doesn't the moderator result support Link B?" Sharpest available test: diversity-
  based *selection* needs no gold, so "pool the most diverse 4 of 8" is a deployable rule, and
  comparing it to "pool a random 4" within the same fixed pool is cleanly identified (two decision
  rules on the same pool — no post-treatment conditioning, no mediator ambiguity).
  - **Max-Vendi-4 beats random-4 by +0.0199 union recall [+0.0090, +0.0325]** (paired by problem,
    n=50); min-Vendi-4 *loses* 0.051. On the 17 informative problems the gain is +0.059
    [+0.033, +0.086]. So there is a real, oracle-free, deployable effect — this part of the
    optimistic reading survives.
  - **But it is entirely non-specific.** Oracle-free ablations against the same baseline:
    max-Vendi **+0.0199** · **max |union| (pick the 4 samples naming the most distinct files)
    +0.0199 [+0.0091,+0.0333]** · max mean |preds| (just name more files) +0.0173.
    corr(Vendi, |union|) = 0.67. The structured kernel adds **nothing** over `len(union(preds))`,
    and 87% of the gain is recoverable by simply emitting more guesses per sample.
  - Max-Vendi selection also coincides with the *oracle* max-accuracy selection (identical means
    to 4 dp; the argmax sets intersect in 12/23 informative pools) — i.e. in this task Vendi is
    behaving as a proxy for "this subset contains the complementary hits", downstream of accuracy,
    not as an independent lever.
  - **Verdict on Link B**: not supported *as a reasoning-diversity claim*. Supported only as a
    coverage-counting claim — pooling more distinct candidates raises pooled recall, which is
    nearly tautological and is trivially gameable (inflate |preds|; pooled recall does not
    penalize precision). This is the **third incarnation of the same deflationary baseline**:
    answer-multiset entropy (math V5) → per-sample quality dispersion (localization Link B) →
    distinct-candidate count (this test). C1 from the brainstorm doc ("beat the cheap statistic")
    remains unbeaten in every regime tested.
  - Consequence: an intervention that raises measured diversity is *not* thereby expected to raise
    performance — on this task the cheapest way to raise both is to widen the net, which is
    exactly the gameability failure mode V6 was designed to detect. Strengthens the case that V6
    should run *before* any sampler-side optimization.
