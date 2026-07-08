# Q1 Progress

> Status as of 2026-05-28. Scope: Phase 0 (scaffolding) + Phase 1 (persona-prompt effect) on GSM8K and CommonsenseQA (CSQA), plus a new **persona-scaffolding** axis (basic vs structured personas) and HPC parallelization. See `q1_experimental_design.md` for the full plan and `proposal_one_pager.md` for framing.

## 1. What's built

White-box HF pipeline under `uv`. Two tasks (GSM8K, CSQA) behind a small task dispatch — adding a task touches only `data.py` / `verifiers.py` / `personas.py`, not the scripts.

- `deeppersona/`
  - `config.py` — `RunConfig` (+ `persona_level` for scaffolding), CLI overrides.
  - `data.py` — GSM8K + CSQA loaders with frozen calib/val/test splits; `load_items(task, …)` dispatch.
  - `personas.py` — per-task persona pools (`MATH_SPECS`, `CSQA_SPECS`) at **two scaffolding levels** (basic / structured); `system_message(task, idx, neutral, level)`, `num_personas(task)`.
  - `verifiers.py` — GSM8K numeric extraction + CSQA single-letter (A–E) extraction; `extract_pred/score(task, …)` dispatch.
  - `generate.py` — batched greedy, matched token budget; `scaffolding_stats.py` — exact McNemar + paired bootstrap + `contrast_summary`; `manifest.py` — git SHA, lib versions, full config + raw per-item JSONL.
- `scripts/`
  - `01_baseline.py` — C-none / C-prompt single cell (task-aware), bootstrap CI.
  - `02_select_persona.py` — persona sweep on val → `selection.json` (runs at one scaffolding level).
  - `03_scaffolding.py` — single-process sweep: none + basic×12 + structured×12, with per-identity Δ + pooled McNemar.
  - `04_aggregate_scaffolding.py` — recombine per-cell HPC runs (glob by `EXP_ID`) into the same contrast.
- HPC (NCSA Delta): `activate_hpc.sh`, `scripts/scaffolding_cell.sbatch` (one cell = one `01_baseline` run), `scripts/submit_scaffolding.sh` (per-cell fan-out, prefetch-then-`afterok` to avoid concurrent weight downloads, `--dry`/`--split`/subset flags).
- `configs/qwen25_7b.yaml` (GSM8K), `configs/qwen25_7b_csqa.yaml` (CSQA).

## 2. Setup / decisions

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, revision pinned `a09a3545…` (not floating on `main`).
- **Decoding:** greedy, bf16, `max_new_tokens=512`, matched across conditions. Deterministic on re-run.
- **GSM8K splits (frozen, seed 0):** calib 200 + val 200 from *train* (disjoint), test = full 1319.
- **CSQA splits (frozen, seed 0):** calib 200 + val 200 from *train* (9741); eval split = the **validation** set (1221) since test labels are hidden — same convention as SRPS (arXiv:2506.07335). 5-way MC, scored by exact-match on the boxed letter.
- **Cluster:** Delta `gpuA100x4` / `bbrz-delta-gpu`; env via `module load miniforge3-python` + `.venv`. HF cache at `/u/yirenl2/.cache/huggingface` (note: `/scratch` is **not** mounted here despite README).

## 3. Benchmark selection — why CSQA

GSM8K Phase 1 came back a near-ceiling null (§5.1), and the dissociation needs a task where a persona effect actually exists. Literature survey (2026-05-28):

- Expert **prompt** personas do **not** reliably help verifiable hard QA and often hurt — Wharton "Playing Pretend" (arXiv:2512.05858) on GPQA-Diamond + MMLU-Pro; Zheng et al. (EMNLP 2024).
- Effects that survive live in **commonsense reasoning** + role **vectors** — SRPS (arXiv:2506.07335) reports role-vector gains on CSQA/SVAMP for 7–9B models.

→ CSQA chosen as the "documented-effect" target. Agentic QA (GAIA / BrowseComp) deferred (infra-heavy, 7B agents floor). Full rationale in `memory/project_benchmark_selection.md`.

## 4. New axis — persona *scaffolding* (basic vs structured)

Adapts Wharton's presence/absence contrast into a **richness** contrast. For each persona identity we hold the identity fixed and vary only the scaffolding:

- **basic** — the original one-line framing (e.g. *"You are a careful everyday reasoner…"*).
- **structured** — the same identity as a labeled block: role → background → numbered method → bulleted principles.

Holding identity fixed isolates *structure* from *which persona*. The **output token budget is matched**, so only the *input* system prompt grows — a structured effect can't be a verbosity artifact. Conditions: **none / basic / structured**. Readout: per-identity Δ(structured−basic), pooled McNemar, paired bootstrap CI.

## 5. Results

### 5.1 GSM8K — persona-prompt selection (val, n=200)

| rank | persona | acc |
|---|---|---|
| 1 | #10 "step-by-step reasoning; show all work, verify" | 0.965 |
| — | C-none baseline | 0.950 |
| last | #7 / #11 | 0.940 |

Parse-fail = 0. **Null within noise** (spread 0.940–0.965; 95% CI half-width ≈±0.03 at n=200 near ceiling). GSM8K is near-ceiling for this model → little headroom. Matches the "weak/no effect" scenario the design doc anticipates (§4).

### 5.2 CSQA — persona scaffolding (val, n=200), matched 512-token budget

| condition | acc |
|---|---|
| none (no persona) | 0.810 |
| basic persona (mean of 12) | 0.8175 |
| structured persona (mean of 12) | 0.7958 |

**Δ(structured − basic) = −0.0217, 95% CI [−0.0346, −0.0088]; pooled McNemar b/c = 150/98, p = 0.0012.**

- Per-identity Δ: **9 of 12 personas drop, 2 flat, 1 up.** Largest drop is the *best* basic persona (#1 "careful everyday reasoner") 0.845 → 0.790 (−0.055, the only individually significant cell, McNemar p=0.013).
- **Parse-fail = 0 in all 25 cells** → the drop is wrong answers, not broken letter-formatting.

**Read:** (i) basic persona ≈ none (0.8175 vs 0.810) — the weak-effect pattern again; (ii) **adding structure *hurts***, consistently across 12 independent persona texts, enough to push structured below the no-persona baseline. Consistent with Wharton (expert personas can hurt) and SRPS's note that small models "struggle to follow complex instructions." Supports the style-not-competence thesis: the extra structure is framing that carries no competence and even crowds out the answer.

**Caveats:**
- The pooled McNemar p is **anti-conservative** — it treats 12×200 paired observations as independent, but the 200 items recur across personas (correlated). Trust the cross-persona consistency (9/12 down) and the bootstrap CI over the p-value; per-identity tests are individually n.s. except #1.
- This is the **val selection slice (n=200), not the held-out test (1221)**. Confirm on test before headlining.
- Verbosity (matched budget) and formatting (pf=0) are controlled, so neither confounds the effect.

Raw: `runs/scaffolding-scaff-20260528-160216.json` (per-cell + per-item correctness).

## 6. Pending / next

1. **CSQA test-split confirmation** — rerun the scaffolding sweep on `--split test` (1221 × 25 cells) for a headline number + tighter CIs; the val effect is small and its pooled p is shaky.
2. **GSM8K test readout** (optional) — C-none vs C-prompt(#10) on 1319 for a clean proper-hygiene null.
3. **Phase 2 — vector extraction** — `nnsight` diff-of-means on calib, layer/α sweep on val, C-vec steering; CSQA is the candidate task since it has a documented role-vector effect.
4. **Why does structure hurt?** — compare output lengths and eyeball structured→basic flips (over-thinking vs distraction).

*Open decision:* invest GPU in CSQA test confirmation of the scaffolding penalty (cheap, sharpens the negative result) vs. move to Phase 2 vector extraction on CSQA.
