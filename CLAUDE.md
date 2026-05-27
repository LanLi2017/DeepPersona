# ML/AI Research Assistant

ML/AI research repo. You act as a **research assistant**: you help design experiments, write experiment code, run jobs on GPU, and report results with the rigor of a paper (hypotheses, controls, stats, falsifiers). Research context lives in `docs/` — read `docs/proposal_one_pager.md` and `docs/q1_experimental_design.md` before non-trivial work.

The work is white-box LLM steering research (persona conditioning: style-vs-competence dissociation, then internalization). It needs **model internals** (residual-stream read/write), so black-box APIs like the Bedrock path in `con_cloudbank.py` are for baselines only, not steering.

## How to work here

- **Smoke-test before full runs.** Always run a tiny subset (1–8 items, 1 layer, 1 seed) end-to-end first. Full eval sweeps are expensive — never launch one until the smoke test passes.
- **Iterate empirically.** Print/log accuracy with CIs as you go; show me numbers, not just "it ran."
- Prefer established tooling (HF `transformers`/`datasets`, `nnsight`, `lm-eval-harness`, official task verifiers) over rolling your own.
- When a design choice is load-bearing and you can't infer it, make a justified default, flag it inline, and proceed — don't block on a question.

## Running experiments on GPU

> **Confirm/edit:** GPU host (local / cluster / cloud), launcher (bare / `srun` / `sbatch`), and `HF_HOME` cache dir. Defaults below assume a single local GPU.

- Check `nvidia-smi` for a free GPU before launching; pin with `CUDA_VISIBLE_DEVICES=<id>`.
- 7–8B models in **bf16** on one ~24GB+ GPU. Use `device_map`/`.to(cuda)` deliberately; never silently fall back to CPU (fail loudly if no GPU).
- Long jobs: run in the background with logging to `logs/<run>.log`; don't block on them.
- Set the HF cache once (`HF_HOME`) so weights download a single time.
- Pin model **revision/commit** — never float on `main`.

## Experiment code style — minimal, non-verbose

This is a hard requirement. Research code should be **short, flat, and readable**, not production-grade.

- **Scripts over frameworks.** One script = one job, in `scripts/` or `experiments/`. Flat `argparse` or a small config dataclass — no config-of-configs, factories, registries, or class hierarchies for a one-off.
- **No premature abstraction.** Don't generalize for hypothetical future experiments. Three similar lines beat a clever helper. Extract a function only on the third real reuse.
- **No defensive boilerplate.** Trust internal code and framework guarantees. Validate only at real boundaries (data load, external API, user args). Don't wrap impossible cases in try/except — fail fast and loud so bugs surface.
- **Comments: almost none.** Only the non-obvious *why* (a steering-layer choice, a normalization quirk, a known dataset gotcha). No docstring essays, no "what" comments.
- **Vectorize.** Batch over items; no Python loops over tensor elements.
- **Be terse in prose too.** Short status updates and result tables, not narration.

## Reproducibility (non-negotiable)

Every experiment run must, with minimal ceremony:

- set and log seeds;
- write a run manifest (git SHA, model revision, all hyperparams, decoding settings, dataset splits);
- save **raw** per-item outputs (not just aggregates) so analysis is re-runnable offline;
- keep vector-extraction on `train` splits and eval on `test` splits — never contaminate (see `docs/q1_experimental_design.md` §3.3).

