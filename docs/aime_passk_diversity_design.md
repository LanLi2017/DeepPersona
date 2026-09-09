# AIME24 Diversity ↔ Pass@k Correlation Design

> **Status:** Draft v0.1 (branch `aime-persona`). Extends **N1** from `docs/experiment_tracker.md`
> — the interventional test blocked on N2/N3 sign-off — with a design that removes the
> ceiling-effect confound that killed every prior correlation attempt (F0a, F2, Exp1/Exp2 in
> `docs/traj_diversity_report.md`). Flagged decisions below are defaults, not commitments.

---

## 1. Assumptions under test

1. Better learning signal → better downstream task performance after post-training. *(not
   tested here — taken as given)*
2. More diverse rollouts/trajectories in a group → better learning signal, proxied by pass@k.
   *(not tested here — taken as given, per the user's simplification)*
3. **This experiment.** There exists a metric that fairly measures — and could guide generation
   of — diverse rollouts. Operationalized as: **does a candidate diversity score, computed over a
   group of trajectories, correlate with that group's pass@k?**

**Falsifier.** If neither candidate metric (§4.4) correlates with pass@k once the ceiling-effect
confound is removed, Assumption 3 is further disconfirmed — consistent with N1-dry's null — and
metric-guided diversity sampling should not be pursued as an RL reward without a different metric
family.

---

## 2. Models (AWS Bedrock, `us-east-2`, via the `con_cloudbank.py` connector pattern)

Confirmed against `bedrock.list_foundation_models()` on 2026-09-01:

| role | model id | notes |
|---|---|---|
| Qwen | `qwen.qwen3-32b-v1:0` | **Flag:** no 8B or 27B Qwen3 hosted on Bedrock; 32B (dense) is the closest match to the requested "Qwen 3.8-27B." `qwen.qwen3-235b-a22b-2507-v1:0` (MoE, ~22B active) is the alternative if active-param scale matters more than dense-param scale. |
| GPT | `openai.gpt-5.6-sol` | exact match to "GPT 5.6 Sol" |
| GPT | `openai.gpt-5.6-terra` | exact match to "GPT 5.6 Terra" |
| Claude | `anthropic.claude-opus-5` | latest Opus |
| Claude | `anthropic.claude-sonnet-5` | latest Sonnet |

All five go through Bedrock's unified `converse_stream` API (same call shape already used in
`scripts/22_aime_bedrock.py` for `moonshot.kimi-k2-thinking`), so one rollout function serves all
five models modulo response-format quirks (reasoning/thinking content blocks).

---

## 3. Task & data

AIME 2024, 30 problems, `AI-MO/aimo-validation-aime` filtered to `"2024" in url` — identical
source/loader to `scripts/22_aime_bedrock.py::load_aime2024()`.

---

## 4. Procedure

### 4.1 Step 1 — raw rollout generation with guaranteed pass/fail per item

For each (model, item): draw rollouts at temperature 1.0, **one at a time**, stopping as soon as
both ≥1 correct and ≥1 incorrect have been observed, capped at `N_max` (default **12**). If an
item is still degenerate (all-correct or all-incorrect) at `N_max`, **drop it** from that model's
pool and record the drop as a diagnostic — a model dropping most items because it's "too good" or
"too weak" for AIME24 is itself informative (mirrors Exp2's 90% pass@1 problem in the prior
report).

This is the direct fix for the failure mode in every prior run: GSM8K pass@k=1.000 across all
conditions (Exp1) and AIME vanilla pass@1=90% with only 2 hard items (Exp2) both left zero
variance in the dependent variable, making Pearson r undefined.

`N_max=12` default justified by: Kimi K2's AIME screen resolved 27/30 items within 1 draw: most
items should hit "mixed" within a handful of draws; the cap only bites on the model's hardest
items, bounding cost there.

### 4.2 Step 2 — augmentation via LLM revision (N → N·r)

For each raw trajectory, generate `r-1` LLM-revised variants: a cheap paraphrase-style rewrite
(temp 0.7, same underlying solution path) via `gpt-4.1-mini`, reusing the `PARA_SYS` prompt
pattern from `scripts/15_logdist_testbed.py` / `scripts/25_style_mining.py`.

**Each revision is re-graded independently** (re-extract the boxed/final answer, re-compare to
gold) rather than inheriting the parent's correctness label — a rewrite can introduce an
arithmetic slip or accidentally fix one, so label inheritance would silently bias pass@k upward.
Default `r=3`.

**Manipulation check (gate before Step 3).** The augmentation is only valid as a "style-only"
expansion of the pool if paraphrasing genuinely leaves the logic untouched. Verify this
empirically rather than assuming the prompt does what it says:

- Per revision, compare its re-graded label (above) to its parent's label. Compute the
  **correctness flip rate** (parent correct/incorrect ≠ revision correct/incorrect) over the
  full augmented set.
- **Expected: flip rate ≈ 0.** This must be checked at the per-trajectory level, not as an
  aggregate accuracy delta — flips in both directions can cancel out and hide the problem in a
  mean.
- **If flip rate is non-trivial:** the `PARA_SYS` rewrite prompt is not logic-preserving on this
  data (it's altering steps/formulas, not just wording) and its output should not be treated as a
  "style-only" variant of the parent. Fix by tightening the rewrite prompt (e.g. explicitly
  require identical derivation steps/formulas, vary only phrasing/register) and re-run the check
  before trusting any downstream diversity score computed over these revisions.
- This check only validates the *augmentation mechanism* (Step 2). It does not, by itself, say
  anything about whether a genuine method change should flip correctness — it shouldn't, in
  general: two different valid methods can both land on the right answer, and two different
  flawed methods can both land on the wrong one. That's a group-level, not per-trajectory, claim
  — see §4.4's metric validation criteria below.

### 4.3 Step 3 — group construction (no extra rollout cost)

From each item's augmented pool (size `N·r`), draw groups of size `k` (default `k ∈ {4, 8}`):

- **Random:** uniform random subset, multiple draws per (item, k) for statistical power.
- **Controlled:** stratified by *answer entropy* (number of distinct final numeric answers among
  the k members, regardless of correctness) — **not** by number-correct-in-group. Stratifying by
  number-correct is a dead end: pass@k = 1{≥1 correct} is a deterministic function of that count,
  so every group in a fixed-number-correct stratum would have identical pass@k and zero
  within-stratum variance to correlate against. Answer-entropy avoids this: fixing "k distinct
  answer values appear" still leaves open *which* values those are, so pass@k (does the correct
  value happen to be among them) still varies within the stratum. This is the interventional
  version of the entropy-matched controls in F0a/F4 and directly answers what N1-dry could only
  ask observationally: does the diversity metric predict pass@k *beyond* the answer-entropy
  baseline?

Pass@k per group = any-correct-in-group, read directly off Step 1/2 labels — no extra API calls,
per the user's design note.

### 4.4 Step 4 — diversity scoring, **both metric families** (per user decision)

**(a) Simple** — `scripts/09_traj_diversity.py`'s graph/DTW/Wasserstein over `nomic-embed-text`
step embeddings. This is the same instrument that produced the ceiling-effect nulls in the prior
report — now tested on data with guaranteed pass@k variance.

**(b) Validated ensemble** — adapt the zero-shot cross-model pattern from
`scripts/26_qwen_transfer.py`: encode *all* trajectory text (regardless of which of the 5 models
generated it) through the **frozen Qwen3-8B encoder** (residual stream, layer 18) + the trained
projection heads (`head_l15_l4_L18_v2.pt`, `head_l4_L18_v2.pt`), plus claim-DTW (claims extracted
via `gpt-4.1-mini`, embedded with `Qwen3-Embedding-0.6B`).

> **Caveat to carry through the write-up:** the heads were trained on Qwen3-8B's *own*
> generations. Scoring GPT-5.6/Claude/Qwen3-32B text through them is a further cross-model
> extrapolation beyond the existing N4-prerequisite Qwen3-8B transfer test. Treat ensemble numbers
> on non-Qwen text as exploratory until we check the ensemble still discriminates style-vs-method
> on this new text distribution (mirrors what `scripts/26 --eval` does for Qwen3-8B specifically).

**Metric validation criteria (gate before trusting either metric's correlation number).** Step
1/2 already hands us labeled pairs for exactly this check, at no extra cost: (parent, revision)
pairs are known **style-only** (same logic, by the §4.2 manipulation check), and (item,
distinct-answer-class rollout) pairs from Step 1 are a proxy for **method-changed**. A candidate
metric must satisfy both directions before its group-level correlation is trustworthy:

| pair type | expected diversity score |
|---|---|
| parent ↔ its own LLM revision (style-only, same logic) | **low** |
| two raw rollouts with different answer classes (logic/method differs) | **high** |

This is the same test V1/V2/V6/V6b already ran to adopt metric (b) (AUC ≈0.97 discriminating
style-rewrite from method-switch pairs) and to reject metric (a) as the reward candidate (58–113%
style-hackable). Re-running it here, on this AIME/5-model pool specifically, is cheap (reuses
pairs already produced by Steps 1–2) and catches two failure modes at once: a broken augmentation
prompt (§4.2) *and* a metric that can't tell the two apart on this particular text distribution —
before either one has a chance to contaminate the §4.5 correlation result.

### 4.5 Step 5 — correlation

Per (model, metric, k, sampling-strategy): Spearman **and** Pearson r between group diversity
score and group pass@k, bootstrap 95% CI (repo convention, e.g. N1-dry, V6). Report per-model
*and* pooled-across-model (Fisher-z pooled) — a single model's variance/ceiling profile
dominating a naive pool would be misleading.

Baseline to beat (per N1-dry): answer-entropy of the group. If a diversity metric's correlation
collapses once conditioned on answer-entropy, that's the same null N1-dry already found
observationally — this design lets us check it interventionally instead.

---

## 5. Decision rule

| Outcome | Condition | Interpretation |
|---|---|---|
| **PASS** | CI excludes 0 (positive) for ≥1 metric, consistent across a majority of models, at multiple `k` | Assumption 3 supported for that metric → candidate for U2 (RL reward) pipeline |
| **NULL** | CI crosses 0, or metric adds nothing beyond answer-entropy | Consistent with N1-dry — interventional confirmation of the observational null |
| **Mixed** | Metric-dependent (e.g. ensemble correlates, simple DTW/Wasserstein doesn't) | Informative for the U1 gate question already open in the tracker |

---

## 6. Cost & compute

- **Step 1** (5 models × 30 items × up to 12 rollouts, oversample-until-mixed): unknown per-model
  cost until a 3-item smoke test runs — Kimi K2's AIME average was ~$0.07/rollout (22K out
  tokens), but reasoning-heavy Claude Opus-5 / GPT-5.6 may cost more per call.
- **Step 2** (gpt-4.1-mini paraphrase): cheap by precedent — scripts/25/26 priced similar jobs at
  ~$0.01–0.05/trajectory.
- **Step 4b** (encoding): local GPU (Qwen3-8B + Qwen3-Embedding-0.6B, `CUDA_VISIBLE_DEVICES=3`
  per the repo's fp32-matmul gotcha), $0 API cost except claims extraction (~$0.01/trace by
  scripts/26 precedent: $0.8 for ~450 claim jobs).
- **Repo convention:** ask before spending >$10 (`docs/experiment_tracker.md` §5).

---

## 7. Open flags (confirm/adjust freely)

1. Qwen substitution (32B dense vs the requested "27B"/"8B" — nothing that size is hosted on
   Bedrock).
2. `N_max=12`, `r=3`, `k∈{4,8}` are defaults, not validated — cheap to sweep once the pipeline
   exists and Step 1 costs are known.
3. Ensemble cross-model scoring is an unvalidated extrapolation (§4.4) — worth a cheap sanity
   check (does it still separate style from method on non-Qwen text?) before trusting its
   correlation number over the simple metric's.
4. This design runs N1 (interventional test) and a metric bake-off in one pass rather than
   sequentially through N1→N2→N3 as originally planned — the tracker's N1 row should be updated
   to point here once Step 1 lands.
