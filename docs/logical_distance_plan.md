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
- **2026-07-31 — V6 gameability probe (`scripts/22_logdist_v6.py`, $0): PARTIAL FAIL.**
  The last unrun item in the pre-registered battery, and the one that decides whether the metric
  can be an RL reward. V1/V2 asked whether the metric can *classify* paraphrase vs distinct-solution
  pairs (ensemble AUC 0.972 / 0.929 held-out). V6 asks the different question that matters under
  optimization pressure: **how much of the reward is obtainable for free?**
  Design: 18 problems carrying both arms (50 paraphrased × 38 with distinct-answer pairs);
  style arm = 4 rewrite styles of one trace (logic identical), method arm = distinct-answer pairs
  on the same problem, padding arm = trace ++ paraphrase(trace) (logic identical, ~2× length,
  re-encoded on GPU). Surface change = token-set Jaccard distance, which doubles as the pure-surface
  baseline metric, so that baseline scores R = 1.00 by construction (built-in calibration).
  Ensemble is the adopted 3-way, scaled by negative-pair sd but *not* mean-centred — V6 needs
  ratios, which need a real zero (d(a,a)=0); an affine transform, so all AUCs are unchanged.
  - **The deflationary baselines are ~fully gameable, quantitatively.** At set level (k=4 packs,
    Vendi over exp(−d/τ), 4 identical traces = 1.00 by construction), a pack of **4 paraphrases of
    a single solution** scores essentially the same as **4 genuinely distinct solutions**:
    tok_jaccard 3.27 vs 3.34 (**97% hackable**), mpnet-chunk 3.09 vs 3.24 (93%), raw chunk 3.05 vs
    3.31 (88%), raw Qwen L18 3.01 vs 2.78 (**113%** — style moves it *more* than a method switch).
    This is the first direct quantitative case against embedding-based diversity rewards
    (GCPO 2605.11461 runs DPP over off-shelf embeddings) and it is the headline differentiator §5
    of the brainstorm doc asked for.
  - **The trained metric is much better but not safe.** Ensemble set-level R = (Vm−1)/(Vs−1) =
    **1.72 [1.41, 2.02] → 58% of the achievable diversity reward is buyable with zero change in
    logic**; on held-out problems R = 1.36 (74% hackable). Pairwise held-out ensemble
    R = 1.78 [1.48, 2.00]. Against the **held-out D_restructured style** (never seen in training,
    also renames variables — the strongest free attack) held-out R = **1.47 [1.15, 1.87]**.
  - **Component split is informative and non-obvious.** l15_trace is the least hackable at set level
    (16%) but by far the most overfit (matched pairwise R = 34.4 → held-out 3.60, wide CI).
    l2_claim is the most *stable* across the split (2.05 → 1.80 pairwise) — it is LLM claim
    extraction, not a trained head, so there is no head to overfit — but weakest in absolute terms
    (70% hackable at set level). l4_chunk sits between (41%, held-out R_set 1.53). The ensemble's
    robustness comes from l2's stability, its ceiling from l15's discrimination.
  - **The padding attack fails to land — genuine good news.** trace ++ paraphrase(trace) moves
    l4_chunk *less* than a plain style rewrite (d = 0.083 vs 0.108); R = 5.62 [3.89, 7.60] pairwise,
    R_set = 4.64. Chamfer matching is max-over-chunks, hence inherently duplication-invariant.
    So the SWE-bench "wider net" failure mode does **not** transfer to this kernel by mere
    length/repetition inflation. Caveat: this tests *duplication*, not "emit more distinct
    candidates" — the latter is arguably a real diversity increase, not an attack.
  - **Verdict.** Passes as a *measurement instrument* (large, robust margin over every cheap
    baseline; C1 is beaten here, the first regime where it is). Fails as a *standalone dense RL
    reward* at current strength: if ~60–75% of the reward is free, and style edits are cheaper and
    lower-risk for the policy than genuine method switches (which can break correctness), the
    expected equilibrium is style drift — the F0a/F2 inert-variation failure re-entering through
    the reward. A rough target for reward-safety is R_set ≥ 4–5 (≤ 25% hackable) against held-out
    attack styles; **note this threshold is post-hoc, not pre-registered.**
  - **Caveats.** (i) Static probe with four fixed rewrite styles → a **lower bound** on
    hackability; an adaptive attacker optimizing a rewriter against the metric would do strictly
    better, so the true numbers are worse, not better. (ii) n is small: 18 matched problems,
    6 held-out, 7 set-level packs (3 held-out) — held-out point estimates are fragile.
    (iii) The method arm is distinct-*answer* pairs; genuine same-answer-different-method switches
    are excluded, and the same-answer arm sits in between (ensemble 1.51 vs style 0.92, method 2.77),
    consistent with it being a mixture.
  - **Consequences for sequencing.** (a) Report the metric as an analysis instrument and as the
    quantitative anti-GCPO result — both are supported now. (b) Do **not** wire it into GRPO as a
    dense reward without first hardening it: adversarial style mining (more and held-out rewrite
    styles as training positives) is the cheap next move, and V6 is now the standing acceptance
    test to re-run after any such change. (c) The interventional matched-accuracy experiment is
    unaffected — it does not use the metric as a reward.

## 2026-08-05 — V6b: order-aware / OT / Gromov-Wasserstein kernels (`scripts/23_logdist_seqot.py`) — CHAMFER SURVIVES; one small win

- **Question.** Are the adopted Chamfer matchings leaving signal on the table? Three alternative
  pairwise kernels over the *same* precomputed banks (claim embeddings, L4-chunk raw + head,
  mpnet chunks), motivated by trajectory-distance literature: **DTW** (strict temporal alignment —
  order matters), **Wasserstein/EMD** (mass-conserving point-cloud match — Chamfer's relaxation
  tightened), **Gromov-Wasserstein** (internal relative geometry only — coordinate-free).
  $0, CPU-only, ~5 s: full V1/V2 AUC battery + V6 gameability probe + padding attack for all
  4 banks x 4 kernels plus ensemble variants.
- **Predictions stated before running:** DTW helps iff method identity lives in step order, but is
  more exposed to D_restructured (which reorders); OT ≈ Chamfer; GW weak (coordinate invariance
  solves a problem we don't have — both traces share one encoder — while discarding absolute
  semantics).
- **Results (AUC full / held-out problems; R_set with hackable %):**

  | kernel | claim AUC | l4head AUC | claim R_set | l4head R_set |
  |---|---|---|---|---|
  | chamfer (adopted) | .946 / .901 | .926 / .791 | 1.42 (70%) | **2.43 (41%)** |
  | dtw | .940 / .890 | .877 / .696 | **1.56 (64%)** | 1.97 (51%) |
  | ot | .935 / .893 | .889 / .731 | 1.36 (74%) | 1.80 (56%) |
  | gw | .652 / .663 | .713 / .584 | 1.12 (89%) | 1.35 (74%) |

  - **GW is near-chance, as predicted** (held-out AUC .46–.66 across banks; R_set ≤ 1.35). Ruled out.
  - **OT never beats Chamfer** — mass conservation slightly *loosens* discrimination here, and it is
    less padding-robust (R_pad 3.67 vs Chamfer 7.17 on l4head): b ++ paraphrase shifts half the
    transported mass, whereas max-matching ignores it.
  - **DTW at chunk level loses everywhere**, and the padding prediction confirmed: duplication breaks
    the alignment path, so padding moves l4head:dtw *more* than a style rewrite (d_pad .124 >
    d_style .080; R_pad 2.71 vs Chamfer's 7.17). Order-awareness at chunk granularity = attack surface.
  - **The one win: DTW at the *claim* level.** claim:dtw dominates claim:chamfer on gameability
    (R_set 1.56 vs 1.42; held-out 1.44 vs 1.29) at ~0.01 AUC cost. Claims are LLM-parsed in reading
    order; a style rewrite preserves claim order, a method switch does not — so order is anti-hack
    signal *at the right granularity*. Swapping it into the ensemble:
    **ENS[claim:dtw, l15, l4head:chamfer] = R_set 1.87 [1.50, 2.19] (53% hackable), held-out R 1.51**
    vs adopted 1.72 (58%) / 1.36. Cost: AUC_D held-out .793 → .765. A real but modest improvement —
    nowhere near the R_set ≥ 4–5 reward-safety target, so it does not change the V6 verdict or the
    sequencing: adversarial style mining remains the main hardening path, with V6 (now incl. V6b
    kernels) as the standing acceptance test.
- **Interpretation.** The Chamfer max-matching convention is doing real work: it is simultaneously
  the best discriminator and the most padding-robust, and its known relaxation (one chunk matching
  many) does not measurably hurt. The trajectory-alignment framing pays off only where the sequence
  elements are semantically parsed units (claims), not fixed-width token windows.
- Artifacts: `runs/logdist-testbed/v6b_seqot.json`, `manifest_v6b.json`. Seed 20260805, NBOOT 2000.

## 2026-08-05 — N1 dry run (`scripts/24_n1_dryrun.py`, $0) — observational NULL beyond answer entropy

- **Question.** Before spending $10–20 on the interventional N1: does logical diversity
  (ENS[cl-dtw] Vendi over each problem's existing k=8 neutral gpt-5.5 pack) predict correctness
  on the 100 BeyondAIME problems? Uses the newly checkpointed heads (seeded retrain, drift within
  noise; `head_l15_*.pt` now on disk).
- **Across problems: diversity is a symptom of being lost, not a cause of success.**
  vendi ↔ ncorr rho **−0.646**; vendi ↔ pass@8 −0.24. Hard problems produce diverse wrong
  attempts. Circularity confirmed: vendi ↔ n_unique +0.653. Held within n_unique strata the
  association stays *negative* (weighted mean −0.23) — including n_unique=1 packs (62/100),
  where answer entropy is degenerate but logically scattered convergence to one answer predicts
  that answer being *wrong* — convergent-and-similar reasoning is the good sign observationally.
- **Within problems (difficulty controlled): the effect exists but is fully absorbed by the
  cheap statistic.** Over C(8,4) subsets of the 31 mixed-correctness problems, raw within-problem
  spearman(vendi, cov@4) = +0.256 (74% positive) — but within (problem × subset-answer-entropy)
  strata it collapses to **+0.029, 95% CI [−0.205, +0.258]** (n=24 strata). Same verdict as V5's
  deflationary test, now with the full upgraded ensemble: the metric adds no *selection* value
  over answer entropy on natural rollouts.
- **Implications for N1 interventional.** (i) The observational route cannot justify the spend —
  selection on natural packs shows nothing beyond n_unique. (ii) The interventional question
  (does *inducing* logical diversity at matched accuracy lift pass@k?) is not answered by this —
  natural diversity is confounded with confusion, which is exactly why matched accuracy matters —
  but the prior is lowered and the detectable-effect budget is small (CI up to ~+0.26 within
  strata). (iii) If N1 runs, target the regime where the metric could matter: n_unique=1 packs,
  where answer entropy carries zero information. Hold the sign-off request until N2/N3 settle
  whether the metric itself improves.
- Artifacts: `runs/logdist-testbed/n1_dryrun.json`. Seed 20260805.

## 2026-08-05 — N2: adversarial style mining (`scripts/25_style_mining.py`, $1.12) — MAJOR HARDENING; 2-way ensemble reaches the target point-estimate

- **Design.** 9 new rewrite styles of the same 50 base traces (gpt-4.1-mini, 450 rewrites,
  448/450 boxed_ok, $0.66; claims extracted for all, 14.1/trace, $0.46). Six TRAIN styles isolate
  single attack axes (E_reordered, F_renamed, G_formal, H_dialogue, I_answerfirst, J_symbolic);
  three HELD-OUT styles compound them (K_compound = restructure+rename+casual, L_maxlex = ≤2
  shared content words/sentence + free reorder, M_narrative). Both heads retrained with the
  expanded positive cliques (150 → 1125 pairs), seeded; v1 heads kept, so v1-vs-v2 is a clean
  comparison on identical texts (the task-#1 checkpoint fix is what makes v1 scoreable on v2 texts).
  Acceptance probe: V6 set-level with an UNSEEN-style pack {D_restructured, K, L, M}.
- **Component hardening is large.** On unseen-style packs: l15 16% → **9%** hackable
  (R 6.15 → 11.13), and its order-attack AUC holes close (D .78→.87, E .78→.88, I .68→.86);
  l4chunk 44% → **32%** (R 2.29 → 3.10). The order-attacking styles (D, E, I) remain the weakest
  AUC region for every metric — order attacks are the persistent frontier.
- **The 3-way ensemble barely moves (54% → 52%): the untrained claim:dtw component (64%
  hackable, no head to harden) now dilutes the hardened heads.** Dropping it:
  **ENS[l15_v2, l4head_v2] = R_set 4.08 [2.57, 5.82] on unseen styles — 25% hackable, meeting
  the R_set ≥ 4–5 target on the point estimate** (CI still dips to 2.6; n = 7 packs). AUC cost:
  per-style held-out AUC .79–.97 (weakest B/E ~.79) vs the 3-way's .74–.99 — the 3-way is better
  on average but has deeper holes (I .74, D .78); the 2-way's *minimum* is actually higher.
- **Proposed split of uses.** Measurement instrument (U1): keep the 3-way ENSv2 (best average
  discrimination). Reward candidate (U2): the 2-way ENSv2 — least hackable — with l15_v2 solo as
  a dark horse (9% hackable, all per-style AUC ≥ .86, but a single trained head with no ensemble
  redundancy; its V6-era overfitting appears cured by 9-style training, unverified beyond this
  testbed).
- **Caveats.** (i) "Unseen" styles K/L/M come from the same generator (gpt-4.1-mini, same
  PARA_SYS template) as the train styles — milder than an adaptive attacker; the static-probe
  lower-bound caveat stands. (ii) n = 7 set-level packs; CIs are wide. (iii) The R ≥ 4–5 bar
  remains post-hoc.
- Artifacts: `paraphrases_v2.jsonl`, `claims_v2.jsonl`, `chunk_v2_l4_L18.npz`,
  `emb_v2_l4_L18.npz`, `claim_v2_embs.npz`, `head_l15_l4_L18_v2.pt`, `head_l4_L18_v2.pt`,
  `chunk_l4_L18_v2_head.npz`, `n2_eval.json`, `manifest_n2.json`. Total spend $1.12.

## 2026-08-06 — Qwen transfer testbed (`scripts/26_qwen_transfer.py`, $1.70) — TRANSFER FAILS; the construct itself is weak in the GRPO regime

- **Setup.** New testbed `runs/logdist-qwen/`: Qwen3-8B **non-thinking**, MATH levels 4–5,
  1024-token cap, temp 0.7 — the F1/N4 GRPO regime. 100 problems × k=8 ($0, GPU 3), acc 0.484
  (good mid-band). 7 eval-only styles (A–D + K/L/M compound attacks, $0.58; 73% boxed_ok —
  truncated/boxless base traces add positive-pair noise), claims for the 50-problem subset ($0.57).
- **Regime fact #1: answer diversity collapses.** Mean n_unique 1.69 (vs the rich spread on
  BeyondAIME); only 22/100 problems have any distinct-answer pair, 7 have ≥4. The environment a
  diversity reward would operate in is already convergence-dominated.
- **Zero-shot transfer of the v2 heads FAILS.** Overall AUC: 2-way ensemble 0.630, l4head_v2
  0.556, l15_v2 0.707 (A_concise and D_restructured at/below chance). Set-level: **every
  ensemble at R_set ≈ 1.0 — 100% hackable** — a pack of 4 rewordings scores as diverse as 4
  distinct-answer solutions. N4 with these heads would have been style-drift by construction.
- **In-domain retraining (A/B/C cliques, 25 train problems, pack problems held out) only
  partially recovers.** AUC 0.80 (l4_in) / 0.78 (ENS-in) — but R_set 1.22 [0.80, 1.71],
  **~83% hackable**, CI includes 1. Data poverty is part of it (55 neg pairs, noisy positives),
  but the sharper diagnosis is **regime fact #2: V_method ≈ 1.4 even by the in-domain metric**
  (vs ~3.2 on BeyondAIME) — on MATH L4–5 non-thinking, distinct answers are mostly the *same
  method with an arithmetic slip*, so the distinct-answer proxy for "method switch" is largely
  degenerate here. The ground truth, not just the metric, is thin.
- **Verdict for the pipeline.** (i) Metric validity is **regime-bound**: validated on gpt-5.5
  BeyondAIME traces; neither transfers zero-shot nor trivially retrains onto the F1 regime.
  (ii) **N4 stays blocked** — and the blocker is now deeper than metric hackability: in the
  cheap GRPO regime there is little genuine method diversity to reward, consistent with N1-dry
  ("diversity is a symptom of being lost") and the F0a/F2 inert-variation series. (iii) Paths
  forward, in rough order of value: (a) method-labeled ground truth (LLM-judged method
  annotation instead of the distinct-answer proxy) to check whether genuine method switches
  exist but are invisible to answer-based pairing; (b) a bigger in-domain testbed (more
  problems/pairs, thinking-mode traces later); (c) reconsider the reward-target: rewarding
  logical diversity may only make sense in regimes where methods actually diverge.
- Artifacts: `runs/logdist-qwen/` (traces, pairs, labels, paraphrases, claims, banks,
  `head_*_indom.pt`, `transfer_eval.json`, manifest). ~$0.55 additionally lost to a
  crash-before-write in the first paraphrase run (fixed: raw output now persisted pre-grading).

## 2026-08-06 — Method-labeled ground truth (`scripts/27_method_labels.py`, $1.78, gpt-4.1 judge) — PROXY WAS BROKEN, NOT (MOSTLY) THE METRIC

- **Judge validity.** Self-agreement on pair-level same/diff-method: 92% (20-problem repeat at
  temp 0.7). Hidden paraphrase controls (D_restructured + L_maxlex appended to the trace set):
  91% assigned to their base's cluster — the judge is largely style-blind. Trustworthy enough.
- **Construct presence: method diversity exists but is thin.** Mean 1.47 methods/problem;
  38% of problems have ≥2 methods; only 2% have ≥4. So a k=4 method-diverse pack essentially
  does not occur naturally in this regime — set-level Vendi probes built from distinct answers
  were measuring noise. A GRPO reward here could meaningfully push 1→2 methods, not 4-way packs.
- **The distinct-answer proxy fails in BOTH directions.**
  P(diff-method | diff-answer) = **0.32** — 68% of "method switch" pairs in the old negative
  arm are the same method with arithmetic slips. And 364/531 ≈ **69% of genuine method switches
  produce the same answer** — invisible to answer-based pairing. Every Qwen-testbed number
  computed against the proxy (the transfer AUCs, the in-domain retrain) had ~2/3-corrupted labels.
- **Metric vs judge labels (proxy-free AUC, same-method = 0 / diff-method = 1):**
  l15_v2 zero-shot **0.810**, raw_chunk 0.818, l4_v2 0.738; the in-domain heads are WORSE
  (0.66–0.76) — they were trained against the corrupted proxy. So the "transfer failure" was
  substantially the proxy's failure: the v2 trace head transfers far better than the
  answer-proxy eval suggested. Note the deflationary echo: raw chunk embeddings match the
  trained head on *method discrimination* here — the heads' real value-add is style-invariance
  (raw_chunk is 88–97% style-hackable), not method detection per se.
- **Correctness.** pass@8 = 0.42 on ≥2-method problems vs 0.74 on single-method — the
  familiar "method exploration is a symptom of difficulty" signature, consistent with N1-dry.
- **Consequences.** (a) Retire the distinct-answer proxy in this regime; judge labels are the
  ground truth for any further Qwen-side training/eval (and cheap: ~$1.8/100 problems).
  (b) Redo the gameability probe in-regime with judge-labeled method pairs vs paraphrase pairs
  (pairwise R; set-level k=4 packs don't exist here — 2% incidence). (c) The reward question
  sharpens to: can a reward push convergence-dominated sampling from 1 method toward 2+ at
  matched accuracy? That is the N4-relevant experiment, and l15_v2 (zero-shot!) is currently
  the best candidate signal (judge-AUC 0.81, style-blind by construction).
- Artifacts: `runs/logdist-qwen/method_labels.jsonl`, `method_labels_summary.json`,
  `manifest_methods.json`.

## 2026-08-12 — Regime probe (`scripts/28_regime_probe.py`, judge $7.24) — HARDNESS, NOT THINKING, BUYS METHOD DIVERSITY; nonthink-hard is the N4 candidate regime

**Question.** N-methods left U2 blocked on a regime decision: the F1/GRPO regime (Qwen3-8B
non-thinking, MATH L4–5, 1024 tok) has too little genuine method diversity to reward. Which
candidate regime actually contains the construct? Candidates isolate **mode** (thinking) vs
**hardness** (AIME+AMC), all Qwen3-8B rev b968826, k=8, temp 0.7 / top_p 0.95, seed 0, 50
problems each. Judge = scripts/27 method-clustering prompt on **gpt-5.5-2026-04-23**
(user-approved ≤$25; low reasoning effort), incl. a same-judge re-label of the logdist-qwen
baseline so the table is judge-controlled.

| regime | acc | pass@8 | trunc | methods/prob | V_method | ≥2m | ≥4m | p8 multi vs single | self-agree |
|---|---|---|---|---|---|---|---|---|---|
| nonthink-l45 (baseline) | 0.455 | 0.58 | – | 1.66 | 1.53 | 32% | 8% | 0.31 / 0.71 | 98% |
| think-l45 (12288 tok) | 0.750 | 0.88 | 16% | 1.56 | 1.41 | 34% | 4% | 0.82 / 0.91 | 99% |
| think-hard (16384 tok) | 0.680 | 0.86 | 27% | 1.88 | 1.66 | 52% | 10% | 0.77 / 0.96 | 93% |
| **nonthink-hard (2048 tok)** | 0.372 | 0.54 | 0% | **2.32** | **2.02** | **72%** | **18%** | 0.47 / 0.71 | 91% |

- **Thinking is NOT the lever.** Same problems, mode flipped: methods/prob 1.66→1.56. Thinking
  mode converges on one canonical method and polishes it (rumination = within-method
  self-checking, not method exploration).
- **Hardness IS the lever.** AIME+AMC non-thinking: 2.32 methods/prob, 72% of problems ≥2
  methods, 18% ≥4 (vs 8%/4%/10% elsewhere) — enough incidence for set-level (k=4 pack) V6
  probes, which were impossible in the old regime (2%).
- **The winning regime is also the best GRPO regime and the cheapest**: acc 0.37 / pass@8 0.54
  (real headroom, nonzero reward signal), 2048-token rollouts, zero truncation. Same
  regime family as F1 — only the problem source changes.
- **Diversity-as-symptom persists in every regime** (multi-method problems have lower pass@8
  throughout, e.g. 0.47 vs 0.71 in-regime). Same observational confound as N1-dry — hardness
  drives both. The N1 interventional design remains the only way to test the causal claim.
- **Judge robustness.** gpt-5.5 vs gpt-4.1 on identical baseline traces: 87.7% pairwise
  agreement; 5.5 finds slightly more methods (1.66 vs 1.38). N-methods conclusions stand.
- **Caveats.** Truncated think-traces judged from thinking-tail (16–27%); judge reads the
  post-</think> writeup; 50 problems/regime; AIME+AMC mix ~half AMC (acc 0.37 is mostly AMC).
- **Consequence.** Target regime for U2 = **Qwen3-8B non-thinking, AIME+AMC, 2048 tok**.
  Next: N3-in-target-regime — judge-labeled method pairs + paraphrase pairs there, retrain
  heads on judge labels (never the answer proxy), run the V6/V6b gameability grid.
- Artifacts: `runs/regime-probe/` (traces_*.jsonl, method_labels_*.jsonl, summary.json);
  generation via vLLM 0.27.1 in `.venv-vllm` (py3.12; HF generate OOMs at 8–16k tok).

## 2026-08-12 — N3b: in-target-regime gameability (`scripts/29_n3b_ingame.py`, $2.48) — **GATE PASSED**: R_set 6.46, 15% hackable; U2 unblocks

**Setup.** Target regime traces (`runs/regime-probe/traces_nonthink-hard.jsonl`, 50×8) +
650 paraphrases (13 styles A–M) + claims. Method arm = **gpt-5.5 judge labels everywhere**
(eval pairs, set packs, head-training negatives) — the distinct-answer proxy is retired.
In-regime heads: positives = style cliques {base + A,B,C,E–J} on 25 train problems;
negatives = 177 judge diff-method pairs; **D/K/L/M and all 9 pack problems (≥4 judge
methods) fully unseen**. Eval on the 25 test problems.

Set-level (method pack = 4 distinct-judge-method traces; style pack = base ++ D/K/L/M):

| metric | V_style | V_meth | R_set [95% CI] | hackable |
|---|---|---|---|---|
| **ENSinreg (l15+l4, in-regime)** | 1.23 | 2.50 | **6.46 [5.42,7.68]** | **15%** |
| l15_inreg / l4_inreg alone | 1.19 / 1.26 | 2.21 / 2.61 | 6.40 / 6.26 | 16% |
| l15_v2 zero-shot | 1.86 | 2.65 | 1.91 [1.54,2.51] | 52% |
| ENSv2-2way zero-shot | 2.95 | 3.11 | 1.08 | 93% |
| claim:dtw / raw_chunk | 2.82 / 3.00 | 3.25 / 3.29 | 1.24 / 1.14 | 81% / 88% |

- **First full pass of the pre-registered R_set ≥ 4–5 bar on unseen styles** — and in the
  regime N4 would actually train in. The anti-GCPO contrast survives in-regime: every
  untrained metric is 81–101% style-hackable (l4head_v2 zero-shot at 101%: style buys
  MORE reward than a method switch).
- **In-regime training is what does it** (zero-shot v2 heads fail here, confirming
  N-transfer); judge-labeled negatives + 9-style cliques on 2048-tok traces give
  V_style ≈ 1.2 while keeping V_meth ≈ 2.5.
- **Trade-off to carry into N4:** in-regime heads pay for style-invariance with
  discrimination — AUC_D 0.70 (ENSinreg) vs 0.79 (l15_v2) / 0.81 (raw_chunk). Per-style
  AUC_G never drops below 0.62 for in-regime heads (no winning style attack), while
  zero-shot l4head_v2 has styles at 0.31–0.45 (style >> method).
- **Adding untrained claim:dtw dilutes** (ENSinreg3 → 1.54, 65%), same as N2. Reward
  candidate = **ENSinreg 2-way** (`head_l15_inreg.pt` + `head_l4_inreg.pt`, scaled by
  judge-diff-pair sd, no mean-centering).
- Caveats: 9 packs; styles share one generator (gpt-4.1-mini); static probe ⇒ hackability
  is a lower bound; paraphrase boxed-answer preservation 571/650.
- Artifacts: `runs/regime-probe/` (paraphrases.jsonl, claims.jsonl, emb_*/chunk_* npz,
  head_l15_inreg.pt, head_l4_inreg.pt, n3b_eval.json, manifest_n3b.json).
