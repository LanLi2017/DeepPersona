# Research contract

## Central question

Under a fixed rollout, token, or environment-interaction budget, does
semantic-behavioral novelty among agent trajectories predict and improve
marginal learning value?

## Current scope

Only **G1: controlled behavioral equivalence** is active. ALFWorld is the
development domain and tau-bench retail is the cross-domain check.

Domain update (2026-10-21): natural successful E2 is near zero in tau retail and
telecom, and below 1% in AppWorld (G1-PREV-001). E2 is now generated with
plan-conditioned rollouts, and AppWorld is the primary domain (G1-GEN-001).
The second domain for advance criterion 4 is not yet fixed. BESS,
pass@k prediction, post-training selection, learned metrics, and active
acquisition are out of scope until G1 passes.

## Operational target for G1

G1 distinguishes three levels instead of treating all equivalence as binary:

- **E0 — surface equivalent:** only the textual/serialization carrier changes;
  the grounded environment transitions are identical.
- **E1 — strategy equivalent:** low-level realization may differ, but ordered
  subgoals, decision-relevant information, meaningful state effects, and outcome agree.
- **E2 — strategy different:** the trajectories differ in a *decision-relevant*
  way: different write/state-changing actions, a decision grounded in different
  key information, a different dependency order between dependent subgoals, or a
  different recovery branch.

Boundary rule (frozen 2026-10-07): a difference counts as E2 only if it changes,
or could change, what the agent decides. These count as E1:
- extra or missing read-only lookups whose results are not used in a decision;
- reordering independent steps;
- choosing an equivalent authentication route (e.g. email vs name+zip).

Reference cases:

| Case | Label |
|---|---|
| tau 69-B: two extra unrelated `get_order_details` calls | E1 |
| Check status, then cancel *vs* attempt return, get rejected, then cancel | E2 |

Task identity, reward, and final task effect should be matched between E1 and E2
whenever possible. Otherwise a metric can succeed by detecting outcome rather
than strategy. No-op insertion is E1 only when replay verifies no state change
and no new information; otherwise it is an explicit boundary case.

## Primary G1 question

Does semantic-transition distance rank an E0/E1 variant closer to its anchor
than a matched E2 variant more accurately than raw text and exact action metrics?

## Evaluation unit and leakage control

- Primary unit: `(anchor, positive, negative)` triplet.
- Primary metric: triplet ranking accuracy.
- Secondary metrics: pairwise AUROC and perturbation-wise accuracy.
- Split and bootstrap by source task/anchor, never by derived pair.
- Canonicalization rules must be frozen before viewing test labels.

## Advance criterion

Advance from G1 only if the semantic metric:

1. improves over the strongest non-LLM baseline by at least 0.05 AUROC;
2. has a task-cluster-bootstrap 95% confidence interval for the improvement
   above zero;
3. exceeds 70% triplet accuracy overall and 60% in each transformation family;
4. shows the same direction of improvement on ALFWorld and tau-bench.

If an LLM judge is already near ceiling, retain it as a ceiling/cost baseline
and redirect novelty claims toward downstream experience valuation.
