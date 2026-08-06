# Experiment tracker — reasoning diversity → task performance

Living doc. One row per experiment; details live in the linked per-project docs. Update the
status column when an experiment starts/lands; add new experiments to §4 as they are conceived.

**Overarching question:** what is the relation between the *diversity of reasoning* and task
performance? Not "does sampling more help" — whether making a model reason in genuinely
different ways is a lever that buys competence, and how to operationalize it (as a measurement,
and ultimately as an RL training signal).

## 1. The chain of reasoning so far

1. Surface/persona diversity is **inert**: it changes style, not which problems get solved
   (F0a, F2, F3 Link A — three independent nulls across scales and task types).
2. Therefore the interesting construct is **logical diversity** — different solution *methods* —
   and nothing off-the-shelf measures it (embeddings, token overlap, answer entropy all
   confound style with substance; V6 now quantifies this: 88–113% style-hackable).
3. So the current project builds a **logical-distance metric** (pairwise kernel over reasoning
   traces + Vendi set aggregator) with two intended uses:
   - **(U1) measurement instrument** — re-ask the diversity→performance question with a metric
     that style cannot fool;
   - **(U2) dense RL reward** — plug into diversity-GRPO (`scripts/10_diversity_grpo.py`) so
     training pressure pushes toward genuinely different methods.
4. A pre-registered validation battery V1–V6 gates those uses
   (`docs/logical_distance_brainstorm.md`; running log `docs/logical_distance_plan.md`).

## 2. Completed experiments

| id | question | result | doc |
|---|---|---|---|
| F0a | Do 12 personas lift pass@k? (Qwen3-8B, MATH) | **NULL** — style variation, 0 unique coverage | `q1_progress.md` |
| F2 | Does inference-time prompt diversity help at the frontier? (gpt-5.5, $200) | **NULL** — neutral ≈ static ≈ adaptive | `frontier_inference_diversity.md` |
| F3/E3 | Do lens prompts diversify + pool on SWE-bench localization? | **NULL at Link A** — outputs don't even diversify; Link B effect weak, narrow regime (scripts/20–21) | `f3_diversity_regime_design.md` |
| V1/V2 | Can the logdist ensemble discriminate method switches? | **PASS** — AUC 0.972 full / 0.929 held-out | `logical_distance_plan.md` |
| V6 | Is the metric gameable by pure style edits? (`scripts/22`) | **PARTIAL FAIL** — 58% of set-level reward free via style (74% held-out); but all cheap baselines 88–113% hackable (the anti-GCPO result); padding attack fails (Chamfer duplication-invariant, R=5.6) | `logical_distance_plan.md`, meeting note §5 null #6 |
| N2 | Does adversarial style mining harden the metric? (`scripts/25`, $1.12) | **MAJOR component hardening** — 9 new styles (6 train / 3 held-out compound attacks); l15 16%→9% hackable, l4chunk 44%→32% on unseen-style packs; 3-way ensemble stuck at 52% (untrained claim:dtw dilutes) but **2-way ENS[l15_v2, l4head_v2] = R_set 4.08 [2.57,5.82], 25% hackable — meets the ≥4–5 target on the point estimate**. Caveat: unseen styles share the generator; n=7 packs. | `logical_distance_plan.md` (2026-08-05) |
| N1-dry | Does pack-level logical diversity (ENS[cl-dtw] Vendi, k=8 neutral packs) predict correctness observationally? (`scripts/24`, $0) | **NULL beyond answer entropy** — across problems vendi↔ncorr rho −0.65 (diversity = symptom of being lost); within-problem subset effect +0.26 raw but collapses to +0.03 [−0.21,+0.26] inside answer-entropy strata. No selection value over the cheap statistic on natural rollouts. | `logical_distance_plan.md` (2026-08-05) |
| V6b | Do alternative kernels (DTW / Wasserstein-OT / Gromov-Wasserstein) over the same banks beat Chamfer? (`scripts/23`, $0) | **CHAMFER SURVIVES** — GW near-chance; OT never beats Chamfer and is less padding-robust; chunk-DTW loses everywhere and padding becomes a working attack against it. One win: **claim-level DTW** (style rewrites preserve claim order, method switches don't) — swapping it into the ensemble lifts R_set 1.72→1.87 (58%→53% hackable, held-out R 1.36→1.51) at ~0.03 AUC_D cost. Verdict unchanged. | `logical_distance_plan.md` (2026-08-05 entry) |

Adopted metric: 3-way ensemble (L2 claim-Chamfer + L1.5 trace head + L4 chunk head), components
scaled on negative-pair moments. **New default after V6b: swap the claim slot to claim-DTW**
(ENS[claim:dtw, l15, l4head:chamfer]) — best gameability at small AUC cost. Testbed:
`runs/logdist-testbed/` (100 BeyondAIME problems, 800 neutral traces, 200 paraphrases;
reruns $0/CPU except new-text encoding).

**Standing verdict:** metric validated as an instrument (U1 open); blocked as a reward (U2)
until it survives V6 on unseen styles.

## 3. Gates

- **U1 (measurement)**: open — use the 3-way ENSv2 (claim:dtw + l15_v2 + l4head_v2; best
  average discrimination).
- **U2 (RL reward)**: candidate identified after N2 — the **2-way ENS[l15_v2, l4head_v2]**
  (25% hackable on unseen styles, meets the R_set ≥ 4–5 target on the point estimate; CI dips
  to 2.6). *Target is post-hoc, not pre-registered — say so wherever it is reported.* Still
  required before N4: Qwen3-8B transfer validation (the reward must work on the trace
  distribution GRPO actually produces), and the caveat that "unseen" styles shared the
  generator stands.

## 4. Experiments to come

| id | experiment | use | est. cost | status | notes |
|---|---|---|---|---|---|
| N1 | Interventional matched-accuracy test: does *logical* diversity of a rollout pack predict pass@k where surface diversity didn't? Sample into the mid band. | U1 | ~$10–20 (needs cost sign-off, >$10 rule) | prior lowered by N1-dry | Doesn't use the metric as a reward. N1-dry (2026-08-05): observational selection value is NULL beyond answer entropy; if run, target n_unique=1 packs where answer entropy is degenerate. Hold sign-off until after N2/N3. |
| N2 | Adversarial style mining: more + deliberately held-out rewrite styles as training positives; retrain heads. | U2 hardening | $1.12 actual | **done 2026-08-05** | See §2. Follow-up frontier: order attacks (D/E/I remain the weakest AUC region for every metric); truly independent attack generator (different model/template) for the next round. |
| N3 | Re-run V6 as acceptance test after N2. | U2 gate | $0 (existing pairs + one GPU encode pass) | blocked on N2 | Pass = R_set ≥ 4–5 on unseen styles. Run the full V6b kernel grid (`scripts/23`), not just the adopted ensemble. |
| N4 | Wire ensemble into diversity-GRPO as dense reward. | U2 | GPU time | blocked on N3 | Do **not** start before N3 passes — predicted failure mode is style drift (F0a/F2 inert variation re-entering through the reward). |

## 5. Known gotchas (cost-saving, do not relearn)

- V6 ratios need a real zero: scale ensemble components by negative-pair **sd without
  mean-centering** (affine transform of the adopted form; AUCs unchanged).
- ~~`train_l15` never saves its projection matrix~~ **fixed 2026-08-05**: training is now seeded
  (`torch.manual_seed(0)`) and saves `head_l15_l1.pt` / `head_l15_l4_L18.pt`; banks regenerated,
  drift within noise (ENS[chamfer] R_set 1.72→1.74, AUC_ho 0.929→0.931; conclusions unchanged).
  Old banks kept as `emb_l15_*.npz.pre-head-fix`.
- V6 is a static probe against 4 fixed styles ⇒ every hackability number is a **lower bound**;
  n = 18 matched problems / 6 held-out.
- The V6 method arm is distinct-*answer* pairs; same-answer-different-method switches sit in
  between (1.51 vs style 0.92 / method 2.77) — consistent with a mixture.
- Kernel choices settled by V6b — don't revisit without new evidence: Chamfer max-matching is
  simultaneously the best discriminator and the most padding-robust; order-awareness pays only
  over semantically parsed units (claims), never fixed token windows (chunk-DTW is
  padding-attackable: duplication breaks the alignment path); GW is near-chance (shared encoder
  makes coordinate-invariance useless); OT's mass conservation loosens, not tightens.
- GPU: always `CUDA_VISIBLE_DEVICES=3` (cuda:0 corrupts fp32 matmul); uv venv
  (`.venv/bin/python`); estimate $ before API spend, ask if > $10.
