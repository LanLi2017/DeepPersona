# trace

Research code and curated data for validating semantic behavioral trajectory
representations for interactive LLM agents.

## Project slides

Open [`slides/roadmap.html`](slides/roadmap.html) for a self-contained overview
of the research plan, roadmap, current tau-bench evidence, and next milestone.
The deck supports arrow-key navigation, fullscreen mode, and browser printing.

The active milestone is **G1: controlled behavioral equivalence**. Existing
ALFWorld and tau-bench rollouts are reused as seed trajectories; large upstream
experiment directories are not copied into this repository.

## Prepare the G1 seed data

From this repository:

```bash
python scripts/prepare_g1_data.py
python scripts/validate_g1_data.py
# Requires the existing sibling icrl checkout and ALFWorld environment:
PYTHONPATH=../icrl ../alfworld-venv/bin/python scripts/validate_alfworld_replay.py
# Or submit the full replay check to the CPU partition:
bash scripts/submit_alfworld_replay.sh
```

The preparation script expects the existing sibling projects at `../npm` and
records source hashes and selection rules in `data/g1/manifests/manifest.json`.
See `data/g1/README.md` for the data layout and intended use.

The completed data-foundation audit is recorded in
`reports/G1_DATA_MILESTONE.md`.

## Run the G1-P1 calibration

Build and validate the deterministic 4+4-anchor calibration package:

```bash
python scripts/build_g1_calibration.py
python scripts/validate_g1_calibration.py
```

Then open `annotation/g1_calibration.html` locally. It presents 24 blind
anchor–candidate comparisons, autosaves answers in browser local storage, and
exports one JSONL row per comparison. Do not inspect
`data/g1/pilot/provenance.json` before annotation because it contains the
hidden E0 slot and natural-trajectory source IDs.

The calibration package and its limitations are recorded in
`reports/G1_P1_CALIBRATION_MILESTONE.md`.

## Inspect the tau-bench success-first calibration

The revised tau-only candidate batch uses four retail train tasks with
reward-positive, non-vacuously evaluated trajectories:

```bash
python scripts/build_tau_success_calibration.py
python scripts/validate_tau_success_calibration.py
python scripts/summarize_tau_quality_screens.py
python scripts/summarize_tau_success_judges.py
```

Open `annotation/tau_success_calibration.html` for an aligned tool-call diff;
full conversations are collapsed by default. Three independent LLM judge
outputs are stored separately under `data/g1/tau_success_v2/judgments/` and
must not be opened before independent human annotation. See
`reports/G1_TAU_SUCCESS_V2.md` for scope and current evidence.

The LLM-vetted candidate batch is frozen for independent human calibration.
The final panel recognized all E0 controls and classified the eight natural
pairs as seven E1 and one E0, with no E2. This supports the pipeline but shows
that targeted successful rollout collection is required for strategy diversity.
