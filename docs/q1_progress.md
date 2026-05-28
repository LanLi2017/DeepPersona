# Q1 Progress

> Status as of 2026-05-28. Scope: Phase 0 (scaffolding) + Phase 1 (persona-prompt effect) on GSM8K. See `q1_experimental_design.md` for the full plan.

## What's built

White-box HF pipeline under `uv` (`pyproject.toml`, cache pinned to `/scratch`):

- `deeppersona/` — shared utils: `config.py` (`RunConfig`), `data.py` (GSM8K loader + frozen splits), `personas.py` (12 math personas + 3 neutral templates), `generate.py` (batched greedy, matched token budget), `verifiers.py` (boxed/####/last-number extraction + exact-match), `manifest.py` (git SHA, lib versions, full config + raw per-item JSONL).
- `scripts/01_baseline.py` — C-none / C-prompt runner with per-batch progress + bootstrap CI.
- `scripts/02_select_persona.py` — sweeps all 12 personas (+ C-none) on val, writes `selection.json`.
- `configs/qwen25_7b.yaml` — model + decoding config.

## Setup / decisions

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, revision pinned `a09a3545…` (not floating on `main`).
- **Decoding:** greedy, bf16, `max_new_tokens=512`, matched across conditions. Confirmed byte-identical on re-run (deterministic).
- **Splits (frozen, seed 0):** calib 200 + val 200 from GSM8K *train* (disjoint), test = full 1319. Selection on val, reporting on test — no contamination.
- **Hardware:** GPU 3 only (L40S 46 GB; GPU 0 is faulty). ~5 min per 200-item condition.

## Phase 1 result — persona-prompt selection (GSM8K val, n=200)

| rank | persona | acc |
|---|---|---|
| 1 | **#10** "step-by-step reasoning model; show all work, verify" | **0.965** |
| 2 | #1 tutor / #5 professor / #6 number-theory | 0.960 |
| — | **C-none baseline** | 0.950 |
| last | #7 chess-grandmaster / #11 contest grader | 0.940 |

- Parse-fail = 0 across all 13 conditions → extraction is clean.
- **Effect is within noise.** Full spread is 0.940–0.965 (a 5-item window / 200). At n=200 near 95%, the 95% CI half-width is ≈±0.03, so no gap is significant. GSM8K is near-ceiling for this model → little headroom for a persona to help. This is the "no/weak effect" scenario the design doc anticipates (§4).
- Best persona (#10) is locked for any future C-prompt-on-test reporting.

## Pending / next options

1. **GSM8K test readout** — C-none vs C-prompt(#10) on the 1319-item test split + McNemar/bootstrap. Proper-hygiene headline number; likely confirms a null on GSM8K (~70 min GPU).
2. **Add MATH** — harder, more headroom; the task where a dissociation is more likely to appear. Needs a MATH loader + boxed-answer normalization, then persona-select on MATH val.
3. **Phase 2 — vector extraction** — `nnsight` residual-stream diff-of-means over calib, layer sweep on val, C-vec steering.

*Open decision:* whether to invest GPU in the (likely-null) GSM8K test run, or pivot to MATH where the prompt-level effect has room to show.
