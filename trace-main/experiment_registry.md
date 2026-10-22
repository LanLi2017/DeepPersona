# Experiment registry

## G1-DATA-001 — Existing rollout audit and seed selection

- **Question:** Are existing ALFWorld and tau-bench rollouts sufficient to seed
  the controlled behavioral-equivalence benchmark?
- **Dataset:** Two source pools recorded in `data/g1/manifests/manifest.json`.
- **Controls:** Deterministic selection; source hashes; malformed-episode filter;
  all trajectories retained for every selected task.
- **Success criterion:** At least 16 valid tasks per benchmark, all ALFWorld task
  families represented, nontrivial within-task sequence variation, no missing
  selected episodes, and lossless parsing of declared tau-bench tool calls.
- **Result:** Passed. 16 ALFWorld tasks/96 trajectories and 16 tau-bench
  tasks/64 trajectories were selected. All six ALFWorld families are present;
  selected tasks average 3.75 and 4.0 unique exact sequences respectively; all
  462 declared tau-bench tool-call messages match parsed call names. Every one
  of the 96 ALFWorld action sequences was replayed against the copied task
  assets; all step observations and final rewards exactly matched the source.
- **Decision:** Proceed to G1-P1 controlled triplet construction and manual
  adjudication. Do not collect new rollouts yet.

## G1-P1 — Controlled behavioral-equivalence pilot

- **Question:** Does semantic-transition distance rank E0/E1 variants closer
  than outcome-matched E2 variants?
- **Calibration dataset:** Eight train-split anchors (four ALFWorld and four
  tau-bench retail), each paired with one deterministic E0 transformation and
  two real same-task, same-outcome trajectories. This yields 24 blind
  comparisons. Both benchmarks include successful and failed anchors.
- **Full pilot target:** 32 anchors and 128 evaluated pairs after the rubric and
  candidate-selection policy pass calibration.
- **Metrics:** Triplet ranking accuracy (primary), AUROC and transformation-wise
  accuracy (secondary).
- **Baselines:** Raw-text similarity, action edit distance, exact transition
  Jaccard, semantic action edit, LLM judge, semantic occupancy distance.
- **Controls:** Group split and bootstrap by source task; match reward/final
  effect and approximate length; freeze transformations before evaluation.
- **Status:** Calibration package ready; human adjudication pending. No metric
  result has been computed and the full pilot has not started.

### G1-P1T — Tau-bench success-first calibration

- **Dataset:** Retail train tasks 16, 28, 69, and 75; four anchors, twelve
  comparisons. Every source has reward 1, a non-vacuous DB target, no recorded
  tool error, and no human transfer.
- **Presentation:** Aligned structured tool-call diff by default; complete
  conversations are available but collapsed.
- **Pre-annotation:** GPT-5.6, Sonnet, and Opus independently judged blind
  packets. Labels were unanimous on 10/12 comparisons (Fleiss' kappa 0.788),
  including 12/12 correct votes on four mechanically known E0 comparisons.
- **Quality finding:** Three screening rounds rejected 7/17 reward-positive
  candidates. Every retained anchor and natural trajectory passed by majority
  in the final panel, but individual model disagreement remains.
- **Strategy finding:** Natural-pair majority labels were seven E1, one E0, and
  zero E2. Existing trajectories do not supply the intended strategy contrast.
- **Status:** LLM-vetted candidate batch frozen for independent human
  calibration. Targeted successful rollout collection is the next experiment;
  these are calibration statistics, not metric performance.

## G1-PREV-001 — Natural E2 prevalence in public AppWorld / tau2-telecom baselines

- **Question:** How often are two successful same-task trajectories genuinely strategy-different (E2) and both high quality?
- **Data:** Public test-split baseline outputs. These are used for prevalence only and never enter G1 calibration.
  AppWorld: 13 agent configs (GPT-4o, GPT-4-Turbo, Llama3, DeepSeek-Coder × 4 methods), with responses recovered by replay (`scripts/replay_appworld.py`).
  tau2 telecom: Claude 3.7 Sonnet, GPT-4.1, o4-mini, GPT-4.1-mini, 4 trials each.
- **Design:** Per domain, 24 pairs that the structural proxy flags as E2 plus 6 proxy-E0/E1 controls, at most one pair per task, seed 0.
  Each pair got a blind pairwise judge using the frozen decision-relevance boundary (`scripts/build_prevalence_audit.py`, `run_prevalence_judge.py`).
  Judges: Opus via CLI, Kimi K2 Thinking and Haiku 4.5 via Bedrock. GPT-5.6 and Sonnet have no Bedrock access on this account.
- **Result (`data/g1/prevalence_audit/summary.json`):**
  - Haiku is unreliable. It agrees with the other judges on 40% of AppWorld labels and calls 5/6 AppWorld controls E2. Opus–Kimi label agreement is 0.80 (AppWorld) and 0.63 (telecom).
  - AppWorld proxy-E2 pairs: Opus and Kimi both say E2 on 7/24. Of those, 1/24 also has both trajectories high quality by both judges (aw-08: one-step `cart_to_wish_list` vs DELETE+POST). The 3-judge majority gives 3/24.
  - Telecom proxy-E2 pairs: Opus and Kimi both say E2 on 13/24, but 0/24 is E2 with both trajectories high quality. Opus rates 42/60 telecom trajectories low quality (unneeded resets or payments, invented facts). Telecom "E2" is mostly a bad-vs-good contrast.
  - Implied natural high-quality E2 rate: AppWorld ≈ 0.2–0.7% of successful pairs; telecom ≈ 0.
- **Decision:** Natural high-quality E2 is rare in every domain audited, so targeted generation is required.
  AppWorld is the primary domain. Its genuine E2s are *alternative API routes* to the same end state.
  Telecom is deprioritised: its success check is too lenient, and the quality gate would discard most rollouts.
  Haiku is dropped from the judge panel. Human adjudication is required for all E2 candidates (kappa 0.1–0.3).
- **Update 2026-10-07: GPT-5.6 judge added (OpenAI API), replacing Haiku.** Panel = GPT-5.6 + Kimi + Opus (`summary_gpt56_kimi_opus.json`).
  - AppWorld: 4/24 proxy-E2 pairs are majority E2 with both trajectories majority high quality (aw-08, aw-19, aw-20, aw-22). GPT-5.6 and Opus jointly confirm 2 of them (aw-08, aw-20). Implied natural HQ-E2 rate ≈ 0.5–0.9% of successful pairs.
  - Telecom: 0/24. GPT-5.6 rates 53/60 telecom trajectories low quality.
  - Agreement stays low (label kappa 0.27 AppWorld, 0.20 telecom). GPT-5.6 is E2-liberal: 20/24 proxy-E2 pairs and 3/6 AppWorld controls. LLM judges are far from ceiling on E1/E2, so human gold is required.
- **Route scan (`ground_truth/api_calls.json`, train+dev, 147 tasks):** 12 tasks (4 scenarios) contain a step with a known equivalent API route: file move vs copy+delete, note append vs rewrite, queue add-album vs per-song. Route-based E2 alone is too narrow. Information-source and recovery plans are needed.

## G1-GEN-001 — Plan-conditioned AppWorld rollouts, batch 1 (12 route tasks)

- **Design:** For each of 12 train/dev tasks, GPT-5.6 proposed up to 3 decision-distinct plans (30 plans in total). Each plan, plus a no-plan control, was run by GPT-5.6 and Haiku 4.5 using a minimal ReAct code agent (`scripts/appworld_targeted_rollout.py`). Total: 84 rollouts, seed 0, max 30 turns. Outputs are in `outputs/appworld_targeted/pilot/`.
- **Success:** GPT-5.6 succeeded on 12/12 no-plan and 29/30 plan rollouts. Haiku succeeded on 9/12 no-plan and 21/30 plan rollouts. Cost ≈ $16 at list price.
- **Structural proxy (different write-endpoint sets between successful rollouts of the same task):**
  - no-plan vs no-plan: 0/9 pairs differ, which replicates the natural-rarity finding.
  - same plan, different model: 2/21 differ. Plans are followed consistently across the two model families.
  - different plans: 64/160 pairs differ, in 6/12 tasks. The differences are file move vs copy+delete (34d9492 ×3) and in-place rename+move vs a staging directory (68ee2c9 ×3).
  - Information-source plans (directory listing vs per-file details) never changed the writes. Under the frozen boundary they are probably E1.
  - In 396c5a2 and cf6abd2, the plans did not change the writes.
- **Status:** These are structural candidates. Judge and human confirmation of E2 and of quality is pending. Batch 2 (12 non-route tasks) has not been run.
- **Update 2026-10-20: human spot-check of the consensus judge** (`data/g1/review_digest/judge_spotcheck_Lan.jsonl`, 12 stratified pairs, 8 generated and 4 natural). The reviewer saw the consensus verdict and Chinese digests, so these labels may be anchored.
  - The human agreed with the consensus label on 12/12 pairs. Quality: 2 flips (aw-20 A → low, because the trajectory hit an error; gn-07 B → low) and 1 unsure.
  - Per-judge agreement with the human, label / quality: Opus 12/12 / 19/22, Kimi 9/12 / 18/22, GPT-5.6 8/12 / 18/22.
  - GPT-5.6 called all 4 "E1" pairs E2. That confirms it is E2-liberal.
  - aw-20 drops out of HQ-E2. The natural AppWorld HQ-E2 count is now 1 (aw-08).
  - gn-08 is human-confirmed E2 (move with retain_dates vs without), but the write-endpoint proxy scored it E1. The proxy misses differences that live only in the parameters.
  - Proposed (not yet approved): use Opus as the single pre-screen judge. The human checks every Opus-E2 / both-HQ pair plus a random 10% of the rest.

## G1-GEN-002 — Plan-conditioned AppWorld rollouts, batch 2 (12 non-route tasks, v2 plan axes)

- **Design:** Same 12 non-route tasks as the batch-1 plan file, but plans regenerated with `--axes v2`: route / subgoal_order / write_parameters / recovery. Information-source plans were dropped because in batch 1 they never changed the writes. GPT-5.6 + Haiku 4.5, seed 0, max 30 turns, 64 rollouts. Outputs are in `outputs/appworld_targeted/batch2/`. The structural proxy now also compares effective boolean args, with API defaults filled in from `api_docs/standard` (motivated by gn-08).
- **Plans:** GPT-5.6 judged 8/12 tasks to have a single reasonable strategy, so only 4 tasks got 3 plans.
- **Success:** GPT-5.6 10/12 no-plan, 14/20 plan. Haiku 3/12 no-plan, 10/20 plan. No model solved 76f2c72_3 or b0a8eae_2 (in b0a8eae the write_parameters plans picked different playlists, which changes the answer). Cost ≈ $15.7 for rollouts plus ≈ $4.5 for judges.
- **Proxy:** no-plan vs no-plan 0/3 differ; plan vs other 24/50 differ, in only 2 tasks: 6c2c621 (in-place write vs delete+recreate vs staging dir) and 60d0b5b (private vs public refund).
- **Judges (GPT-5.6 / Kimi / Opus majority, 17 pairs, `data/g1/generated_audit_b2/`):** 10 pairs are majority E2 with both trajectories HQ.
  - The proxy's E2 calls were confirmed on 8/11 pairs.
  - Missed by the proxy: 1 pair with a different plan (g2-08, update vs delete+create, a latent difference) and 1 no-plan control (g2-12, aa8502b: follow only artists not already followed vs follow all). Endpoint-level proxies miss decision differences that do not change the write set.
  - 6ea6792 subgoal_order plans were judged E1 (independent steps), as expected.
- **Cumulative (batches 1+2):** 16 majority-HQ-E2 pairs from 8 tasks, but only 5 scenarios. 12/16 are file-system route variants (move vs copy+delete, staging dir, in-place vs recreate). Yield is concentrated and the diversity of E2 *kinds* is low.
