# API spending tracker

One bullet per spend, newest last. Log every API-spend event here at launch time (estimate)
and correct to actual when the run finishes. GPU-local runs are $0 and not logged.
Standing rule: estimate $ before any API spend; ask before anything > $10.
gpt-5.5 ≈ $2.5/M in, $25/M out; gpt-4.1-mini ≈ $0.4/M in, $1.6/M out.

## Backfill (from docs/plan-doc records; * = estimate, actuals not recorded at the time)

- 2026-06-17 — **F0a** personas/strategies (GSM8K, MATH): $0 (local Qwen3-8B; Tinker variant on credits).
- 2026-06-17..18 — **E-SPL** axis evolution (AIME/AMC/BeyondAIME): gpt-4.1 mutation calls, small, not recorded (*~$2–5).
- 2026-06-23..30 — **F1** diversity-GRPO sweeps (MATH l4–5): $0 (local GPU).
- 2026-07-05 — **F2** frontier inference-diversity, BeyondAIME + HMMT probe (gpt-5.5, ~7.1M compl tok): **~$200**.
- 2026-07-08 — **F2b/CGE** consensus-gated escalation, BeyondAIME (gpt-5.5, ~1.3M tok incl. smoke): **~$40**.
- 2026-07-15 — **logdist testbed** v1 paraphrases, 200 rewrites (gpt-4.1-mini): **$0.35**.
- 2026-07-20 — **logdist stage 2** claim extraction, 1000 traces (gpt-4.1-mini): **$0.89** (stage total < $1.50).
- 2026-07-22 — **logdist L3** edge extraction a+b (gpt-4.1-mini): **$1.16**.
- 2026-07-28 — **F3/E3** SWE-bench localization probe n30×k2 (gpt-5.5): **$3.6** (computed from raw.jsonl tokens).
- 2026-07-30 — **F3/E3** main n50×k8×2 arms (gpt-5.5, 15.4M in / 0.28M out): **$45.4** (computed from raw.jsonl tokens).
- 2026-07-31 — **V6** gameability probe: $0 (existing pairs, local GPU encode).
- 2026-08-05 — **V6b** kernel comparison + **N1 dry run** (scripts/23, 24): $0 (precomputed banks, CPU).

**Backfilled total ≈ $295** (dominated by F2 $200 + E3 $49 + CGE $40).

## Live log

- 2026-08-05 — **N2 style mining, v2 paraphrase generation** (scripts/25 --generate, 450 jobs, gpt-4.1-mini): est **~$1.8**; actual TBD (running).
- 2026-08-05 — **N2 claims extraction v2** (scripts/25 --extract, 450 texts, gpt-4.1-mini): est **~$0.7**; not yet launched.
- 2026-08-12 — **N-regime** probe, 4×60 judge jobs (gpt-5.5-2026-04-23, low effort, incl. baseline re-label + smoke): **$7.24** (approved ≤$25).
- 2026-08-12 — **N3b** in-regime gameability: paraphrases 650 + claims 1050 (gpt-4.1-mini): **$2.48**. (Day total with N-regime probe: $9.72.)
