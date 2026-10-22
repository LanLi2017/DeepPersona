# G1-P1 calibration pilot

This directory contains the first blind calibration batch for the behavioral
equivalence rubric.

- `calibration_triplets.jsonl`: eight blind items, four from ALFWorld and four
  from tau-bench retail. Each item has one anchor and candidates A/B/C.
- `annotations.template.jsonl`: empty annotation rows for every comparison.
- `provenance.json`: generation details and hidden candidate provenance. Do not
  inspect this file while annotating.
- `checksums.sha256`: hashes for the generated blind data, template,
  provenance, and browser payload.

Each triplet contains exactly one deterministic surface-equivalent transform
and two real, same-task, outcome-matched trajectories. The two natural
trajectories are selected by normalized action-sequence distance, but neither
is assigned an E1/E2 ground-truth label before human adjudication.
The four ALFWorld anchors cover clean, cool, examine, and pick-two task
families; the uninformative heat group was excluded because its available
same-outcome candidates differed only at the tail of otherwise identical
40-step traces.
All eight anchors come from benchmark training data; test tasks remain untouched
for later evaluation. Each benchmark includes both successful and failed
anchors so that calibration is not confounded with a single outcome class.

Open `annotation/g1_calibration.html` in a browser to annotate. The page stores
drafts in browser local storage and exports completed annotations as JSONL.
Use a stable, non-identifying annotator ID; it and the frozen rubric version are
included in every exported row.

Rebuild and validate:

```bash
python scripts/build_g1_calibration.py
python scripts/validate_g1_calibration.py
```
