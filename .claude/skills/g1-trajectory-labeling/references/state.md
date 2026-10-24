# State of the work (snapshot 2026-10-21)

The authoritative logs are `trace-main/experiment_registry.md` (G1-PREV-001, G1-GEN-001 and
G1-GEN-002), `trace-main/decision_log.md` and `trace-main/reports/G1_E2_GENERATION_SUMMARY.md`.
They live on branch `lan/traj-diversity-methods`, commits 28fe72b and 6fcd7e2. If you are on
another branch, read them with `git show lan/traj-diversity-methods:trace-main/<file>`.

## Big picture

- **The lab's question:** under a fixed rollout budget, does semantic-behavioral novelty predict
  and improve learning value?
- **What is active now:** only G1 is active. The question there is whether a semantic-transition
  distance ranks an E0/E1 variant closer to its anchor than a matched E2 variant, more accurately
  than text or exact-action baselines.
- **Advance criteria:**
  - at least +0.05 AUROC over the best non-LLM baseline, with a task-cluster bootstrap CI above 0;
  - more than 70% triplet accuracy overall and more than 60% in each perturbation family;
  - the same direction in two domains.
- **Domains:** the original contract named ALFWorld and tau retail. AppWorld is now the primary
  domain (contract updated 2026-10-21). The second domain is still open.

## Findings

1. **Natural HQ-E2 is rare.**
   - AppWorld: about 0.5–0.9% of successful same-task pairs.
   - tau retail: 0.
   - tau2 telecom: 0. Its success check is lenient, and most of its "E2" pairs are good-vs-bad
     contrasts.
   - Naturally, almost everything is E1. E0 was never measured separately.
2. **Plan-conditioned generation works, but only on a few tasks.**
   - Batch 1 ran 12 route tasks (84 rollouts, about $16): 6 HQ-E2 pairs.
   - Batch 2 ran 12 other tasks with v2 axes (64 rollouts, about $16 for rollouts plus about $4.5
     for judges): 10 HQ-E2 pairs. For 8 of these 12 tasks, GPT-5.6 said only one reasonable
     strategy exists.
   - No-plan vs no-plan pairs almost never differ.
3. **Cumulative total: 19 HQ-E2 pairs (majority vote) from 11 tasks and 8 scenarios.**
   - 3 natural pairs: aw-08, aw-19 and aw-22. aw-19 and aw-22 are borderline. Under the stricter
     GPT-5.6+Opus criterion, only aw-08 survives.
   - 12 of the 16 generated pairs are file-system route variants: move vs copy+delete, a staging
     directory, in-place write vs delete+recreate.
   - So the *kinds* of E2 are narrow. Bootstrap by task.
4. **The structural proxy is a pre-filter only.**
   - About 73% of its E2 flags are confirmed.
   - It misses differences that leave the write set unchanged. Example: g2-12, follow only the
     not-yet-followed artists vs follow all of them.
5. **Judges.** The majority vote agrees with the human. Opus is the best single judge, and
   GPT-5.6 is E2-liberal. Haiku was dropped as a judge because it is unreliable. Kappa among the
   judges was 0.2–0.27 before the spot-check.

## Human labels

- `data/g1/review_digest/judge_spotcheck_Lan.jsonl` holds 12 pairs.
- The human agreed with every label.
- Quality flips: aw-20 A was set to low quality, which removes aw-20 from HQ-E2, and gn-07 B
  was set to low quality.
- gn-08, a `retain_dates` difference, is human-confirmed E2. That case led to adding boolean args
  to the proxy.

## Team (meeting 2026-10-21)

- **Lunyiu:** unified trajectory format (pending), Terminal Bench, and the existing distance
  metrics, including the tree-structure method.
- **Lan:** AppWorld and "automation bench". That name came out garbled in the transcript, so
  confirm which benchmark it is. Lan also shares the AI-labeling scripts with the team.
- **Liri:** deep research.
- **Yiren:** IT bench and "harvey lab".
- **Target:** at least 100 labeled trajectories per domain by about 2026-10-28. Labels come from
  AI labeling with cross-model cross-checks, followed by human review. Each person works on a
  branch of a shared repo; ask the user whether that repo is public before putting AppWorld data
  in it.
- **GPU access** (8 nodes, 64 cards) ends in late January 2027.

## Open questions to ask the user when relevant

- Which benchmark is "automation bench"?
- Does "100 条" mean trajectories, pairs or triplets?
- Which shared repo will be used, and is it public or private?
