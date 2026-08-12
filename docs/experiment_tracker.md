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
| N-methods | Is method diversity absent in the GRPO regime, or just mislabeled by the distinct-answer proxy? (`scripts/27`, $1.78, gpt-4.1 judge; self-agreement 92%, style-blindness 91%) | **PROXY BROKEN** — P(diff-method \| diff-ans) = 0.32 (68% of the old "method arm" was same-method slips) and 69% of real method switches share an answer (invisible to the proxy). Method diversity exists but thin: 38% of problems ≥2 methods, 2% ≥4. Proxy-free AUC vs judge labels: l15_v2 zero-shot **0.81** (transfer was much better than the proxy eval implied); in-domain heads worse (trained on corrupted labels); raw_chunk 0.82 (heads' value-add = style-invariance, not method detection). pass@8 0.42 multi-method vs 0.74 single-method. | `logical_distance_plan.md` (2026-08-06) |
| N-transfer | Do the hardened heads transfer to the GRPO regime (Qwen3-8B non-thinking, MATH L4–5)? (`scripts/26`, $1.70) | **FAIL** — zero-shot AUC 0.63 / R_set≈1.0 (100% hackable); in-domain retrain recovers AUC to only 0.80 and R 1.22 (83% hackable). Deeper: mean n_unique 1.69 and V_method≈1.4 — in this regime distinct answers ≈ same method + arithmetic slip; the method-diversity construct itself is thin. | `logical_distance_plan.md` (2026-08-06) |
| N2 | Does adversarial style mining harden the metric? (`scripts/25`, $1.12) | **MAJOR component hardening** — 9 new styles (6 train / 3 held-out compound attacks); l15 16%→9% hackable, l4chunk 44%→32% on unseen-style packs; 3-way ensemble stuck at 52% (untrained claim:dtw dilutes) but **2-way ENS[l15_v2, l4head_v2] = R_set 4.08 [2.57,5.82], 25% hackable — meets the ≥4–5 target on the point estimate**. Caveat: unseen styles share the generator; n=7 packs. | `logical_distance_plan.md` (2026-08-05) |
| N1-dry | Does pack-level logical diversity (ENS[cl-dtw] Vendi, k=8 neutral packs) predict correctness observationally? (`scripts/24`, $0) | **NULL beyond answer entropy** — across problems vendi↔ncorr rho −0.65 (diversity = symptom of being lost); within-problem subset effect +0.26 raw but collapses to +0.03 [−0.21,+0.26] inside answer-entropy strata. No selection value over the cheap statistic on natural rollouts. | `logical_distance_plan.md` (2026-08-05) |
| N-regime | Which regime contains genuine method diversity — is thinking (mode) or AIME+AMC (hardness) the lever? (`scripts/28`, gpt-5.5 judge $7.24, judge-controlled incl. baseline re-label) | **HARDNESS, NOT THINKING** — think-l45 1.56 methods/prob (thinking polishes one method); **nonthink-hard (AIME+AMC, 2048 tok): 2.32 methods/prob, V_method 2.02, 72% ≥2 methods, 18% ≥4** — with GRPO-friendly headroom (acc 0.37, pass@8 0.54) and the cheapest rollouts. Diversity-as-symptom persists everywhere (p8 multi 0.47 vs single 0.71). Cross-judge 5.5-vs-4.1 agreement 87.7%. | `logical_distance_plan.md` (2026-08-12) |
| N3b | Does the metric pass V6 gameability in the target regime (nonthink AIME+AMC 2048 tok), with judge-labeled method pairs and in-regime head training? (`scripts/29`, $2.48) | **PASS — U2 gate cleared 2026-08-12.** In-regime 2-way ensemble (l15+l4 heads, judge-diff negatives, 9-style cliques): **R_set 6.46 [5.42,7.68] = 15% hackable** on unseen-style packs (D/K/L/M, pack problems excluded from training) — beats the ≥4–5 bar. Zero-shot v2 heads fail in-regime (1.08–1.91); all untrained baselines 81–101% hackable (anti-GCPO holds). Trade-off: AUC_D 0.70 vs 0.79–0.81 for un-hardened metrics. Caveats: 9 packs, one style generator, static probe. | `logical_distance_plan.md` (2026-08-12 N3b) |
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
- **U2 (RL reward)**: **UNBLOCKED 2026-08-12** — N-regime fixed the target regime (Qwen3-8B
  non-thinking, AIME+AMC, 2048 tok: V_method 2.02, 72% multi-method, acc 0.37), and N3b
  passed the gameability gate there: in-regime ENSinreg R_set 6.46 [5.42,7.68], 15% hackable
  on unseen styles (bar was ≥4–5). Reward candidate = ENSinreg 2-way
  (`runs/regime-probe/head_l15_inreg.pt` + `head_l4_inreg.pt`, scaled by judge-diff-pair sd,
  no mean-centering; do NOT add untrained claim:dtw — it dilutes to 65% hackable). N4 may
  proceed. Residual risks for N4: static-probe lower bound (RL will search styles the probe
  didn't), single style generator, AUC_D 0.70 discrimination trade-off.

## 4. Experiments to come

| id | experiment | use | est. cost | status | notes |
|---|---|---|---|---|---|
| N1 | Interventional matched-accuracy test: does *logical* diversity of a rollout pack predict pass@k where surface diversity didn't? Sample into the mid band. | U1 | ~$10–20 (needs cost sign-off, >$10 rule) | prior lowered by N1-dry | Doesn't use the metric as a reward. N1-dry (2026-08-05): observational selection value is NULL beyond answer entropy; if run, target n_unique=1 packs where answer entropy is degenerate. Hold sign-off until after N2/N3. |
| N2 | Adversarial style mining: more + deliberately held-out rewrite styles as training positives; retrain heads. | U2 hardening | $1.12 actual | **done 2026-08-05** | See §2. Follow-up frontier: order attacks (D/E/I remain the weakest AUC region for every metric); truly independent attack generator (different model/template) for the next round. |
| N3 | Re-run V6 as acceptance test after N2. | U2 gate | $0 (existing pairs + one GPU encode pass) | **passed in source domain 2026-08-05** (2-way ENS R_set 4.08, 25% hackable — meets ≥4–5 target), but **superseded by N-transfer FAIL** in the old GRPO regime | New gate for U2 is passing V6 **in the target regime**, which N-regime has now fixed as nonthink AIME+AMC 2048 tok — see N3b. |
| N3b | In-target-regime gameability with judge labels. | U2 gate | $2.48 actual | **PASSED 2026-08-12** — see §2 | R_set 6.46, 15% hackable; moved to §2. |
| N4 | Wire ENSinreg into diversity-GRPO as dense reward, in the target regime (Qwen3-8B non-thinking, AIME+AMC, 2048 tok). | U2 | GPU time | **unblocked — next up** | Use `scripts/10_diversity_grpo.py` + ENSinreg heads (see §3). Monitor style drift online: RL searches styles the static probe didn't — track V_style-proxy and judge-audit samples during training. Success = method count/V_method rises at matched accuracy; failure mode = reward climbs while judge-audited method count doesn't (style drift, the F0a/F2 signature). |

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
