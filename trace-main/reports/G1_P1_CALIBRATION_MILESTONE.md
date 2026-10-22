# G1-P1 calibration-package milestone

Date: 2026-09-17

## Evidence

- Built eight deterministic blind triplets from the validated G1 seed pool:
  four ALFWorld anchors and four tau-bench retail anchors.
- Every anchor has three candidates: one mechanically regenerated E0 surface
  transformation and two distinct natural trajectories from the same task and
  outcome class. This produces 24 comparisons for calibration.
- ALFWorld covers clean, cool, examine, and pick-two families. The cool anchor
  is successful; the other three are failures. The original heat candidate was
  rejected during manual audit because all viable trajectories differed only
  near the tail of otherwise identical 40-step traces.
- Tau-bench retail uses train task IDs 14, 30, 47, and 57. Task 57 is successful
  and the other three are failures. The copied split metadata verifies that
  none is in the retail test split.
- Natural near/far candidates are chosen from a deterministic medoid anchor by
  normalized action-sequence edit distance. Distance only controls sampling;
  it does not provide an E1/E2 label.
- E0 construction is non-LLM and benchmark-specific:
  - ALFWorld: consistent entity renaming and fixed action/observation aliases.
  - tau-bench: consistent identifier anonymization and alternate tool-call
    serialization.
- The validator reconstructs every visible anchor and natural candidate from
  its normalized source, independently regenerates each E0 candidate, checks
  same-task/same-outcome constraints, confirms balanced hidden E0 positions,
  verifies train/test separation, compares the browser payload byte-for-data
  with the blind JSONL, and verifies artifact checksums.
- The local annotation page exposes the operational E0/E1/E2 definitions,
  records five component judgments plus overall label and confidence,
  autosaves locally, and exports versioned JSONL with an annotator ID.

## Interpretation

The package is sufficient for rubric calibration, not for evaluating a
trajectory metric. E0 has controlled provenance, while the natural candidates
deliberately remain unlabeled: edit-distance-near is not assumed to mean E1 and
edit-distance-far is not assumed to mean E2. Human review must decide whether
information acquisition, meaningful state effects, ordered subgoals, and
recovery behavior agree.

Outcome matching removes a simple success/failure shortcut, but does not by
itself establish semantic equivalence. In particular, failed traces can fail
through materially different strategies, which is useful E2 evidence but must
be adjudicated explicitly.

## Decision

Freeze this eight-anchor package as the first calibration batch. Keep LLM
rewrites out of the core labels. An LLM paraphrase family can be introduced
later as an explicitly separate robustness test after deterministic E0
examples establish the rubric boundary.

## Next experiment

1. Have at least two annotators independently label all 24 blind comparisons
   using `annotation/g1_calibration.html` without opening hidden provenance.
2. Measure exact E0 recovery, E1/E2 agreement, component-level disagreement,
   and confidence. Adjudicate disagreements with written rationales.
3. Freeze the rubric and candidate-selection policy only if deterministic E0
   is consistently recognized and E1/E2 distinctions are reproducible.
4. Expand to the registered 32-anchor/128-pair pilot, then implement and freeze
   metric baselines before viewing held-out labels.

No scientific performance result is claimed at this milestone.
