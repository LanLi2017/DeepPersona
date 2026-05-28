---
name: project-benchmark-selection
description: Findings on which verifiable benchmarks show a persona/role conditioning effect worth dissociating (for Q1 study); steers task choice away from GSM8K/hard factual QA.
metadata:
  type: project
---

For the Q1 style-vs-competence dissociation study, the user explored moving off GSM8K (near-ceiling null) toward harder/cross-domain or agentic QA that might "benefit more from persona prompting." Literature survey (2026-05-28) found the premise is risky:

- **Expert PROMPT personas do NOT reliably help verifiable knowledge QA.** Wharton "Playing Pretend" (arXiv:2512.05858, Dec 2025) tested in-domain expert personas on **GPQA-Diamond + MMLU-Pro** across 6 models → no reliable gain, often hurts. Zheng et al. EMNLP 2024 (162 roles) and Kang et al. 2025 (occupational personas; bio/chem respond a bit, physics doesn't) agree.
- **Where effects DO appear:** (1) role-PLAY prompting (Kong et al. 2023) gains on 12 reasoning benchmarks — but their own claim is it works as an *implicit CoT trigger*, which the matched-token-budget control would neutralize; (2) role/persona STEERING VECTORS — SRPS (arXiv:2506.07335) Llama3.1-8B CSQA 31.9→39.8, Gemma2-9B SVAMP 37.5→45.1; Potertì domain-matched math.
- **Conclusion:** the effect to dissociate lives at the VECTOR level on commonsense/domain-matched reasoning, not in expert prompts on hard factual QA.

**Recommended task targets (Qwen2.5-7B-Instruct):** CommonsenseQA / SVAMP / StrategyQA (strongest documented vector effect, small/cheap) as the safe "effect exists" target; MMLU-Pro w/ domain-matched personas as the novel cross-domain angle (framed as "does the vector-level competence component survive where prompt personas fail per Wharton?"). GPQA-Diamond too small (198) + 7B floors → supporting only. **Agentic QA (GAIA/AssistantBench/BrowseComp) DEFERRED** — infra-heavy, 7B agents floor, steering through tool loops is a separate paper.

Note: Qwen2.5-7B-Instruct is strong (MMLU 74.2, MATH 75.5) so MATH has less headroom than the progress doc assumed.

**Status (2026-05-28):** CSQA is now wired in as the second task (`task=csqa`): loader/verifier/personas/`configs/qwen25_7b_csqa.yaml`. Per SRPS, **eval is on the CSQA *validation* split (1221)** since test labels are hidden; calib/val = 200-slices of train. Persona 0 mirrors the SRPS "general-knowledge quiz contestant" role. Note: SRPS/Kong apply the role as 2-stage role-PLAY + "Let's think step by step"; our pipeline applies it as a single system-prompt persona at matched token budget — a deliberate difference (we're testing prompt-level scaffolding, not reproducing their steering).

**Why:** Choosing a task with no measurable persona effect makes Phase 4 (decompose the effect) vacuous. **How to apply:** when picking the next task family, prefer one with a documented role-vector effect on 7-8B; treat the Wharton null as motivation, not a blocker. See [[q1-progress-snapshot]].
