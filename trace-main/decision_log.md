# Decision log

## 2026-09-16 — Reuse existing trajectories for G1

**Decision:** Use ALFWorld as the development domain and tau-bench retail as an
early cross-domain check. Reuse compact existing rollout pools rather than
collecting new rollouts.

**Evidence:** The audited sources contain 720 ALFWorld and 200 tau-bench retail
trajectories. After deterministic quality/diversity selection, the repository
contains 96 and 64 trajectories respectively. All selected tau-bench tasks have
mixed outcomes and four unique exact tool-name sequences. All 462 selected
tau-bench messages declaring tool calls were parsed with exact name agreement.

**Alternatives considered:** Copy all upstream outputs; collect a new 8–16
rollout/task pool; use tau-bench as the sole development domain.

**Reason:** Full outputs are multi-gigabyte and unnecessary for G1. ALFWorld has
cleaner grounded transitions, while tau-bench supplies a stronger semantic and
cross-domain stress test.

**What would reverse this decision:** Replay cannot reconstruct ALFWorld state
effects, tau-bench compact records omit information needed for confirmation or
tool semantics, or the selected pool lacks enough reviewable E1/E2 cases.

## 2026-09-17 — Separate deterministic E0 generation from E1/E2 adjudication

**Decision:** Generate E0 with benchmark-specific deterministic rules, and use
real same-task, same-outcome rollouts as unlabeled E1/E2 candidates. Do not use
an LLM to create or pre-label the core calibration examples.

**Evidence:** ALFWorld supports consistent entity renaming plus fixed action and
observation aliases while retaining the exact grounded event skeleton.
Tau-bench supports consistent identifier anonymization plus an alternate tool
serialization while retaining tool names, call order, and source grounding.
The existing pools also contain at least three distinct action sequences with
the same outcome for every selected anchor task.

**Reason:** E0 is a carrier-level intervention and should be mechanically
auditable. Whether two naturally occurring paths are E1 or E2 is the construct
being calibrated, so assigning that distinction from edit distance or an LLM
would bake a disputed label into the benchmark. Action distance is used only
to sample a near and far candidate for review.

**Additional control:** The blind file contains no roles, expected labels, or
source trajectory IDs. Hidden provenance is stored separately. Both domains
use train tasks only; tau-bench retail test tasks remain untouched.

**What would reverse this decision:** Human calibration shows that deterministic
E0 variants alter perceived semantics, or that the real-rollout pool cannot
produce agreed E1/E2 cases. LLM paraphrases may then be added as a separately
named stress-test family, never as untracked ground truth.

## 2026-09-17 — Make successful strategy diversity the primary G1 track

**Decision:** Develop the revised calibration first on tau-bench retail using
reward-positive candidate trajectories, followed by an explicit quality gate.
Analyze successful strategy equivalence and
failure-pattern equivalence as separate strata; never pool them into a result
that a metric can win by detecting binary reward.

**Evidence:** Across the complete existing ALFWorld pool, only two tasks have
three distinct successful action sequences, and one of those differs mainly by
minor reordering. In contrast, merging four existing tau-bench rollout pools
provides eight train tasks with at least three successful tool sequences after
requiring non-vacuous evaluation, no tool error, and no human transfer. Four
tasks were selected for the initial revision.

**Quality correction:** Tau retail task 57 was removed from the success track.
Its task specification has no expected actions or NL assertions, so unchanged
database state receives reward 1 even for behavior that does not resolve the
stated request.

**Annotation policy:** Deterministic E0 remains program-labelled. Multiple LLMs
may independently pre-annotate natural candidates, but their consensus is weak
supervision rather than ground truth. Human labels remain necessary for a gold
subset and every model disagreement.

**What would reverse this decision:** If successful tau trajectories provide
only superficial path variation after human review, collect targeted rollouts
before scaling the metric experiment. ALFWorld cross-domain evaluation resumes
after additional diverse successful trajectories are collected and replayed.

## 2026-09-17 — Freeze the vetted tau calibration and collect targeted E2 runs

**Decision:** Freeze the 16/28/69/75 candidate batch for independent human
calibration, but do not treat LLM consensus as gold. Begin targeted rollout
collection for genuine successful-strategy contrasts rather than screening
more of the same existing pool.

**Evidence:** Three blind quality screens rejected 7/17 reward-positive
candidates. In the final frozen panel, every natural candidate passed quality
by majority, all 12 deterministic-control votes were E0, label agreement was
10/12 unanimous with Fleiss' kappa 0.788, and the eight natural majority labels
were seven E1, one E0, and zero E2.

**Reason:** Existing runs validate surface-versus-low-level equivalence and the
quality gate, but do not exhibit the strategy diversity required by the
proposal. More passive pool filtering has diminishing value.

**Collection policy:** Use independent LLM rollouts conditioned on distinct
high-level plans (information coverage, subgoal ordering, and recovery), then
execute them in the benchmark. Do not use transcript paraphrase as a strategy
generator. Retain only runs that pass environment success, mechanical checks,
and multi-judge quality screening before blind E1/E2 annotation.

## 2026-10-07 — Freeze the E1/E2 boundary; add tau2 telecom and AppWorld; mixed agent models

**Decision:** E2 requires a decision-relevant difference (see `research_contract.md`
boundary rule). The primary successful-strategy collection moves to tau2-bench
telecom and AppWorld, run in parallel. tau retail remains a cross-domain check.
Rollouts use several agent models, both to raise the success rate and to measure
how often E2 occurs naturally across models.

**Evidence:**
- A structural audit of all 160 seed trajectories found no genuine successful E2.
  Every flagged pair was an artifact: hallucinated IDs, a human transfer scored as
  success, the vacuous task 57, or junk ALFWorld pickups.
- Retail reward pins the final DB state, so writes are nearly fixed.
- The rubric listed "extra query / reorder independent reads" as E1 while also
  listing "information coverage" as E2. Judges split on exactly this case (69-B).

**Gate change:** "no tool error" becomes "no *unrecovered* tool error". Otherwise
recovery-branch E2 is filtered out by construction.

**What would reverse this decision:** human calibration shows the boundary is not
reproducible, or the new domains also yield no successful E2 under plan-conditioned
rollouts.
