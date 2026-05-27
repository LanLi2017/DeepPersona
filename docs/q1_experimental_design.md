# Q1 Experimental Design — Dissociating Style from Competence in Persona Conditioning

> **Status:** Draft v0.1. Scope: Question 1 only (the dissociation test) from `proposal_one_pager.md`.
> **Flagged decisions (change freely):** primary model = Qwen-2.5-7B-Instruct; second model = Llama-3.1-8B-Instruct (chosen to match PAS exactly); white-box stack = HuggingFace Transformers + `nnsight` for activation read/write. These are defaults justified below, not commitments.

---

## 1. The question and the hypotheses

**Q1.** When a persona prompt or a role/steering vector raises task accuracy, is the gain a **task-competence** effect or merely a **style** effect (formality, verbosity, register, confidence)?

We extract the persona signal as a single activation-space direction, decompose it into a **style component** (inside a deliberately constructed style subspace) and a **competence residual** (everything orthogonal to that subspace), and inject each separately.

- **H0 (style-only):** the accuracy gain is carried entirely by the style component. Injecting the competence residual alone yields no significant accuracy gain over baseline.
- **H1 (real competence):** the competence residual alone yields a significant accuracy gain, while being style-neutral by independent judgment.

The result is decision-relevant for the whole program: H1 → Q2 (internalization) proceeds as written; H0 → Q2 reframes into "teach the model to write in the correctness-correlated style," a different and easier project. (See `proposal_one_pager.md`, items 2 and 5.)

---

## 2. The dissociation logic in one picture

```
                         persona vector  v  (layer ℓ)
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                                 │
   project onto style subspace S            orthogonal complement
          │                                                 │
     v∥ = S Sᵀ v   (STYLE component)        v⊥ = v − v∥   (COMPETENCE residual)
          │                                                 │
     inject v∥ alone                            inject v⊥ alone
          │                                                 │
   Δacc_style                                        Δacc_comp
                                  │
                 decision:  where does the accuracy gain live?
```

The experiment is a controlled accuracy attribution: we already know (from the literature) that the *full* signal sometimes helps; the novelty is splitting it and asking **which part** helps.

---

## 3. Setup

### 3.1 Models
- **Primary:** `Qwen2.5-7B-Instruct`. Sits at the intersection of the role-vector steering literature (Potertì et al., 2025 use Qwen-7B) and the reasoning-RL lineage that Q2 builds on (Satori uses Qwen-2.5-Math-7B). Strong math/code, clean HF weights, well-supported by hook libraries.
- **Second (replication + direct rebuttal):** `Llama-3.1-8B-Instruct`. This is the *exact* model PAS (Cui & Chen, 2025) reports no reasoning gain on; a positive Q1 result here is a direct on-their-turf rebuttal.
- *(Optional third for robustness:* `Gemma-2-9B-it`, the common third model in this subfield.)*

All steering work requires **white-box access** (residual-stream read/write via forward hooks). The current `con_cloudbank.py` Bedrock path is black-box and cannot do this — Phase 2–4 must run on locally-hosted HF weights.

### 3.2 Tasks (verifiably scored)
- **Math:** GSM8K (1,319 test), MATH (use a fixed 1,000-item stratified subset across levels for cost; full 5,000 optional). Metric: exact-match on the final boxed answer with standard normalization.
- **Code:** HumanEval (164), MBPP (sanitized, ~427 test). Metric: `pass@1` via the official unit-test harness in a sandboxed runner.

All four are persona-sensitive and verifier-scored, so accuracy is unambiguous and style cannot leak into the score.

### 3.3 Data hygiene (contamination control)
- **Vector-extraction / calibration set:** drawn from each task's **train** split (disjoint from eval). 200 items per task family.
- **Persona-selection (val) set:** a second disjoint 200-item slice of train.
- **Eval set:** the **test** splits above, untouched during any tuning or vector construction.

### 3.4 Decoding & reproducibility
- Main accuracy: **greedy** decoding (deterministic) at a **matched max-new-tokens budget** across all conditions, so verbosity cannot confound accuracy.
- Robustness pass: temperature 0.7, `n=5` samples, 3 seeds → for variance bands and a `pass@1` mean with CIs.
- Pin: model revision/commit, tokenizer, `transformers`/`nnsight` versions, dtype (bf16), seeds. Log everything to a run manifest.

### 3.5 Layer / hook configuration
- Read/write at the **residual stream** output of each decoder block.
- Layer sweep over the middle third of the network (where role/steering directions are typically most effective); pick the extraction/injection layer ℓ on the **val** set, not test.

---

## 4. Phase 1 — Establish that there is an effect to dissociate

Before splitting the signal, confirm a signal exists on this slice.

**Conditions (per model × task):**
1. **C-none:** no persona, plain task prompt.
2. **C-prompt:** best persona prompt (selected as below).
3. **C-vec:** role/persona-vector steering with the *full* vector `v` (built in Phase 2).

**Persona prompt pool & selection.** Assemble ~12 expert-framing personas per task family (e.g., "expert competition mathematician," "careful step-by-step tutor," "senior software engineer who writes tested code"). Select the single best persona **on the val set**, report it **on test**. This removes cherry-picking: selection and reporting use disjoint data.

**Read-out.** If neither C-prompt nor C-vec beats C-none significantly (Section 8), record it: there may be nothing to dissociate on this model/task, which is itself a finding aligned with PAS. The dissociation (Phase 4) is still run, but interpreted as "no effect, hence no competence component."

---

## 5. Phase 2 — Extract the persona vector `v`

Use the standard **difference-in-means** construction (a.k.a. ActAdd / CAA-style diff-of-means), which is what Potertì et al. and the steering literature use.

1. For each calibration item, build a **paired prompt**: persona-framed vs. neutral, *identical task content*.
2. Run both; capture the residual-stream activation at layer ℓ. Aggregate per item (mean over response tokens, plus a last-prompt-token variant — report both).
3. `v = mean(act_persona) − mean(act_neutral)` over the calibration set. Normalize to a unit vector `v̂`.
4. **Layer & coefficient sweep** on val: choose ℓ and steering coefficient α maximizing val accuracy under C-vec.

This `v̂` is the object we decompose. Also retain the **directional-ablation** check (project `v̂` *out* of activations) as a sanity probe — removing a real competence direction should *hurt* accuracy.

---

## 6. Phase 3 — Build and validate the style subspace `S`

This is the delicate, novel core. The style basis must capture *style* while being as **content-free** as possible.

### 6.1 Style axes (initial set, extensible)
`formality`, `verbosity`, `confidence/hedging`, `register/technical-jargon`.

### 6.2 Content-controlled contrastive pairs
For each axis, construct pairs that **hold the answer/content fixed** and vary **only** that stylistic dimension — e.g., the same correct solution written formally vs. casually; terse vs. verbose; hedged vs. confident. Crucially these pairs are built on **non-task** or answer-fixed text so the direction cannot encode "how to solve it." Extract one direction per axis via diff-of-means at layer ℓ.

### 6.3 Orthonormalize
Stack the axis directions and orthonormalize (QR / Gram-Schmidt) → `S ∈ R^{d×k}` whose columns span the **style subspace**.

### 6.4 Validate the style subspace (gate before Phase 4)
- **Sufficiency:** steering with each style direction should visibly shift that attribute. Confirm with (a) an LLM-judge rubric scoring formality/verbosity/confidence, and/or (b) a lightweight trained probe.
- **Content-neutrality:** style-only steering should **not** materially change accuracy on its own beyond what's attributable to verbosity (which we've budget-matched). If a "style" direction strongly moves accuracy, it is contaminated — rebuild the pairs.
- Report the validation; do not proceed to Phase 4 until the style directions behave as style.

> Caveat to document: style and competence may not be perfectly linearly separable (cf. StyliTruth / Shen et al., who find style–truth coupling localized in attention heads and use orthogonal deflation). Linear projection is the v0 instrument; Section 9 lists the upgrade path if it proves insufficient.

---

## 7. Phase 4 — Decompose and inject the competence residual

**Decomposition** of the persona vector at layer ℓ:
```
v∥ = S Sᵀ v        # style component (projection onto style subspace)
v⊥ = v − v∥        # competence residual (orthogonal complement)
```

**Injection conditions** (per model × task), as residual-stream addition `h ← h + α·(unit dir)`:
- **C0:** none (baseline).
- **C1:** full `v̂`.
- **C2:** style-only `v̂∥`.
- **C3:** competence residual `v̂⊥`.   ← the load-bearing condition
- **C4 (optional):** ablate style from activations (project `S` out), leaving competence in place.

**Norm/coefficient control.** Inject all of C1–C3 as **unit vectors at a matched α** to rule out "bigger nudge = bigger gain" confounds; *also* report each with its own val-tuned α. Report both so a reviewer can't attribute results to step-size.

**Style-neutrality audit of C3.** Run the Phase-6.4 judge/probe on C3 outputs: the competence residual should leave style **indistinguishable from baseline** while (if H1) raising accuracy. That conjunction — accuracy up, style flat — is the paper's central evidence.

---

## 8. Metrics and statistics

- **Primary metric:** accuracy (GSM8K/MATH exact-match; HumanEval/MBPP `pass@1`).
- **Paired significance:** same items across conditions → **McNemar's test** on paired binary correctness for each contrast (C3 vs C0, C2 vs C0, C1 vs C0, C3 vs C2).
- **Effect size & CIs:** bootstrap 95% CIs on accuracy *differences* (10k resamples); report absolute Δacc and the per-item flip matrix.
- **Multiple comparisons:** Holm–Bonferroni across the contrast family per model×task.
- **Power note:** HumanEval (n=164) is underpowered for small effects — treat it as supporting, lean on GSM8K/MBPP/MATH for the headline claim. State minimum detectable effect per dataset.
- **Robustness:** repeat headline contrasts across the temperature/seed sweep; report variance bands and layer-sweep sensitivity.

**Decision rule.**
| Outcome | Condition | Interpretation |
|---|---|---|
| **H1 (competence real)** | C3 ≫ C0 (sig.) **and** C3 style ≈ C0 style | Competence component is real and isolable → **Q2 proceeds** |
| **H0 (style only)** | C3 ≈ C0, while C2 ≈ C1 ≫ C0 | Gain is stylistic → **Q2 reframes** to style-internalization |
| **Mixed** | Both C2 and C3 contribute | Report the decomposition of the effect; Q2 targets the competence share |
| **No effect** | C1 ≈ C0 (Phase 1 null) | Nothing to dissociate on this slice → align with PAS, publish the null |

---

## 9. Threats to validity & mitigations

1. **Incomplete style basis** (style has > k dimensions). → Start with 4 axes; add axes if C2 fails to recover most of the C1 effect; report residual unexplained gain.
2. **Non-linear style/competence coupling** (StyliTruth's finding). → If linear projection leaves a large unexplained residual, upgrade to attention-head-localized orthogonal deflation per Shen et al. (documented as the v0.2 instrument).
3. **Norm/step-size confound.** → Matched-α injection (Section 7).
4. **Verbosity confound.** → Matched token budget (Section 3.4) + verbosity as an explicit style axis.
5. **Persona-prompt cherry-picking.** → Val/test split for selection vs. reporting (Section 4).
6. **Data contamination.** → Train-only extraction, test-only eval (Section 3.3).
7. **Layer-choice overfitting.** → Layer/coefficient chosen on val; report full sweep on test as sensitivity.
8. **Single-prompt-template fragility.** → Report across ≥3 neutral templates.

---

## Appendix A — Persona prompt pool (seed examples)
*Math:* "You are an expert competition mathematician; reason carefully and verify each step." / "You are a meticulous math tutor." *Code:* "You are a senior software engineer who writes correct, tested code." / "You are a careful algorithms specialist." *(Full pool: 12 per family, finalized in Week 1.)*

## Appendix B — Style contrastive-pair recipe
Fix a correct solution; produce two renderings differing only on one axis (formal↔casual, terse↔verbose, hedged↔confident, plain↔jargon). ~50 pairs per axis on answer-fixed text.

## Appendix C — Key hyperparameters to pin
model revision, layer ℓ, coefficient α (per condition), token budget, dtype, seeds, decoding params, library versions.

## References (from the prior-art survey)
- Cui & Chen, 2025 — *Painless Activation Steering* (Llama-3.1-8B-Instruct; no reasoning gain). arXiv:2509.22739
- Potertì, Seveso & Mercorio, 2025 — *Designing Role Vectors* (Llama-3.1-8B, Qwen-7B, Gemma-2). arXiv:2502.12055 / EMNLP Findings 2025
- Shen et al., 2025 — *Balancing Stylization and Truth via Disentangled Representation Steering* (StyliTruth; Qwen-1.5-14B-Chat). arXiv:2508.04530
- *Persona is a Double-edged Sword* (role-play hurts zero-shot reasoning). arXiv:2408.08631
- Satori, ICML 2025 — *RL with Chain-of-Action-Thought* (Qwen-2.5-Math-7B), template for Q2. arXiv:2502.02508
