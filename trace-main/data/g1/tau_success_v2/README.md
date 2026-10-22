# Tau-bench success-strategy candidate calibration v2

This batch is the success-first replacement for the tau-bench half of the
initial G1-P1 calibration package. It uses retail **train** tasks 16, 28, 69,
and 75. Test tasks remain untouched.

Every selected source trajectory satisfies the following mechanical execution
gate:

- recorded `reward >= 1` and `won = true`;
- task evaluation contains a non-vacuous database-write target;
- all compacted messages are present;
- every declared tool call is parsed with its arguments;
- no tool result is marked as an error;
- no transfer to a human agent occurs.

This gate does **not** establish policy compliance or complete task quality.
Three blind quality-screen rounds assess those separately; rejected candidates
do not enter this frozen candidate set.

For each task, the builder selects three explicitly approved, independently
generated trajectories. Their medoid becomes the anchor. The two other real
successful trajectories are blind natural candidates; neither receives an
automatic E1 or E2 label. A deterministic identifier/serialization
transformation of the anchor provides the mechanically verifiable E0 candidate.

## Files

- `source_trajectories.jsonl`: the 12 normalized source trajectories copied
  from four existing upstream rollout pools.
- `task_specs.json`: the four upstream tau-bench task specifications.
- `calibration_triplets.jsonl`: blind annotation data.
- `annotations.template.jsonl`: empty versioned annotation rows.
- `judge_packets.jsonl`: compact blind tool paths and conversation evidence for
  independent LLM pre-annotation.
- `provenance.json`: hidden source IDs, candidate roles, source hashes, and
  selection evidence. Do not inspect this while annotating.
- `checksums.sha256`: hashes for all generated artifacts and browser data.
- `quality_screen*/`: blind packets, raw three-model assessments, provenance,
  consensus, summaries, and checksums for the candidate-screening rounds.
- `quality_screens_summary.json`: compact index of all screening rounds.

Build and validate with:

```bash
python scripts/build_tau_success_calibration.py
python scripts/validate_tau_success_calibration.py
python scripts/summarize_tau_quality_screens.py
python scripts/summarize_tau_success_judges.py
```

Open `annotation/tau_success_calibration.html` for the compact tool-diff view.
Full conversations remain available under collapsed disclosure panels.
