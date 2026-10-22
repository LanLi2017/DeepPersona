# G1 data-foundation milestone

Date: 2026-09-16

## Evidence

- Audited source pools:
  - ALFWorld: 120 tasks, 720 trajectories, 221 successful trajectories.
  - tau-bench retail: 50 tasks, 200 trajectories, 41 successful trajectories.
- Excluded tau-bench task IDs `2`, `29`, `44`, and `76` because at least one
  episode had no messages.
- Deterministically selected:
  - 16 ALFWorld tasks / 96 trajectories, covering all six task families;
  - 16 tau-bench retail tasks / 64 trajectories, all mixed-outcome.
- Selected tasks average 3.75 ALFWorld action sequences and 4.0 tau-bench tool
  sequences per task.
- Preserved 48 ALFWorld replay files and four tau-bench retail interpretation
  assets (selected task specs, database, policy, and split metadata).
- Parsed 462/462 tau-bench messages declaring tool calls with exact tool-name
  agreement, recovering 470 structured calls in total.
- Replayed all 96 ALFWorld trajectories against the copied assets on SLURM CPU
  nodes. Every step observation and every final reward matched exactly.
- Re-running data preparation produced byte-identical hashes for source
  snapshots, normalized trajectories, manifests, and benchmark assets.
- No absolute home paths or credential-like strings were found in committed
  data or documentation.

## Interpretation

The existing trajectory pools are sufficient to seed G1. The selected data is
portable, reproducible, grounded in executable ALFWorld environments, and rich
enough to expose invalid actions, redundant actions, tool errors, and recovery
behavior. New rollout collection is not currently justified.

Selection is intentionally information-rich rather than distributionally
representative, so results from the controlled-equivalence pilot must not be
reported as benchmark-wide prevalence estimates.

Replay cost is highly non-uniform: some 40-step ALFWorld failures require tens
of CPU minutes. Controlled-variant replay should therefore use task/rollout
SLURM arrays and record interaction count separately from wall-clock cost.

## Decision

Mark G1-DATA-001 complete and advance to G1-P1: controlled triplet construction.
Do not collect new rollouts, compute BESS, or train a learned metric yet.

## Next experiment

Construct a small adjudication batch before scaling to 128 pairs:

1. choose four anchors per benchmark;
2. create one E0 surface-equivalent, one E1 strategy-equivalent, and one
   outcome-matched E2 strategy-different candidate per anchor;
3. replay every executable ALFWorld variant;
4. manually adjudicate information set, meaningful state effect, subgoal order,
   recovery behavior, and final outcome;
5. revise the rubric before generating the full G1-P1 dataset.
