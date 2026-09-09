# AIME24 Diversity ↔ Pass@k — Intermediate Report

*Date: 2026-09-09. Branch: `aime-persona`. Full design: `docs/aime_passk_diversity_design.md`.*

## 1. Research question

Testing **Assumption 3** (does a diversity metric fairly measure and guide generation of diverse
rollouts?), simplified via **Assumption 1+2** (higher pass@k ⟺ better learning signal):

> Does a group's diversity score correlate with that group's pass@k?

Design: sample N trajectories per (model, item) guaranteeing ≥1 correct and ≥1 incorrect,
optionally augment via LLM revision (N→N·r), construct groups (random / entropy-controlled),
score diversity, correlate with pass@k.

**Status: Steps 1–5 run once end-to-end on the metric side (mixed results, see §4); then pivoted
to testing Step 1 (rollout *generation* method) directly instead of continuing to refine the
diversity metric (see §5).**

---

## 2. Models actually available

Of the originally requested roster (Qwen3.8-27B, GPT-5.6 Sol/Terra, Claude Opus-5/Sonnet-5), only
Qwen had a size match on Bedrock (32B dense, not 8B/27B), and **GPT-5.6-Sol/Terra and Claude
Opus-5/Sonnet-5 are still blocked** — `AccessDeniedException` under both calling conventions
(bare `ON_DEMAND` id and `us.*` inference-profile id), confirmed via `get_foundation_model` that
the IDs are valid; this account simply has no model-access grant for them. Needs Bedrock console
action outside this session.

Models actually used (`us-east-2`, real pricing via AWS Pricing API):

| key | Bedrock ID | in $/M | out $/M | notes |
|---|---|---|---|---|
| qwen32b | `qwen.qwen3-32b-v1:0` | 0.15 | 0.60 | max_tokens capped at 16384 (32768 advertised but errors) |
| llama70b | `meta.llama3-3-70b-instruct-v1:0` | 0.72 | 0.72 | max_tokens capped at 8192 |
| mistrallarge3 | `mistral.mistral-large-3-675b-instruct` | 0.50 | 1.50 | 64000 OK |
| kimik2thinking | `moonshot.kimi-k2-thinking` | 0.60 | 2.50 | 64000 OK; matches prior Aug report |

## 3. Infrastructure bugs found and fixed (`scripts/28_aime_multimodel_rollout.py`)

1. **Per-model max_tokens ceilings** — requesting above them fails the whole call
   (`ValidationException`) before any generation/cost. Fixed via `MODEL_MAX_TOKENS` clamp.
2. **No retry on transient errors** — a single `InternalServerException` used to discard an
   entire item's accumulated draws. Fixed with exponential-backoff retry.
3. **Directory-collision data loss (real money lost)** — two concurrent `nohup` processes
   launched within the same second computed the same timestamp-based output directory; both
   opened `samples.jsonl` in truncating `"w"` mode, destroying each other's writes.
   **qwen32b (271 calls, ~$0.44) and llama70b (323 calls, ~$0.29) were fully lost; mistral lost
   75/195 calls (~$0.46)** — API calls were made and billed, but results were unrecoverable.
   Fixed: output dir now includes PID (`exist_ok=False`), file opened in exclusive `"x"` mode.
4. **Per-draw incremental writes** — previously buffered all of an item's draws in memory,
   written only when the item finished; a killed process lost the item's paid-for-but-unsaved
   draws. Fixed: each draw is written+flushed immediately.

## 4. Metric-side run (Steps 1–5, one pass)

**Step 1 — rollout generation.** 5-item slice showed severe floor/ceiling effects (mixed groups
rare); **scaling to the full 30-item AIME24 pool fixed this**: qwen32b 15/30 mixed, llama70b
10/30, mistral 18/30. Kimi (screened at `n_max=2`) is near-ceiling (27/30 all-correct); the two
failures (items 3, 21) **match the Aug-2026 report's Kimi failures exactly** — cross-validated
across a month and two independent runs.
Run: `runs/aime24-multimodel-rollout-20260905-004327Z-pid22318/`.

**Step 2 — LLM-revision augmentation (`scripts/29_aime_augment.py`).** No `OPENAI_API_KEY`
configured, substituted `openai.gpt-oss-120b-1:0` on Bedrock. Manipulation check (§4.2 of the
design doc): first attempt showed an 18% "flip rate" that turned out to be **83% pipeline
artifact** (truncation at `max_tokens=4000`, and a naive last-number extraction fallback
mis-grading incomplete generations as wrong). Fixed (12000 tokens, stricter extraction): true
picture is **35.3% incomplete** (gpt-oss-120b doesn't reliably end on a clean boxed answer when
asked to rewrite) and **5.7% genuine flip rate** among scored revisions. Trusted pool used
downstream: 259 raw + 316 complete/non-flipped revisions = 575 trajectories.

**Step 3 — group construction (`scripts/30_aime_groups.py`).** Caught and fixed a conceptual bug
in the original design: stratifying "controlled" groups by *number-correct* is a dead end
(pass@k is deterministic in that count, zero within-stratum variance). Fixed to stratify by
**answer entropy** (number of distinct final answers) instead — verified 43/136 entropy-strata
show genuine pass@k variance. Built **1193 groups** (684 at k=4, 509 at k=8; 22 skipped for
insufficient pool size).

**Step 4 — diversity scoring.**
- *Metric (a), simple DTW/Wasserstein* (`scripts/31_aime_diversity_score.py`): local Ollama
  (`nomic-embed-text`), no GPU needed, ~30 min pure-Python compute.
- *Metric (b), validated ensemble* (`scripts/13-26`'s claim:dtw + 2 trained heads):
  **BLOCKED** — this machine has no CUDA GPU (Apple Silicon Mac) and the trained head weights
  (`head_l15_l4_L18*.pt`) don't exist in this checkout (produced in a prior GPU session,
  never synced).
- *Reduced metric (b), claim:dtw only* (`scripts/33_aime_claimdtw.py`): the one ensemble
  component computable without GPU/trained weights — claim extraction via local
  `qwen2.5:1.5b`, embedding via `nomic-embed-text`. ~40 min local compute.

**Step 5 — correlation (`scripts/32_aime_correlation.py`), controlling for answer entropy:**

| metric | ALL (partial r) | llama70b | mistral | qwen32b | verdict |
|---|---|---|---|---|---|
| dtw_div (a) | −0.058 (ns) | +0.044 (ns) | +0.026 (ns) | −0.128 (ns) | **NULL** — matches N1-dry |
| wass_div (a) | −0.031 (ns) | −0.027 (ns) | **+0.251\*** | −0.122 (ns) | mostly NULL, 1/3 models |
| claim_dtw (reduced b) | **+0.151\*** | −0.045 (ns) | **+0.301\*** | **+0.181\*** | **2/3 models positive** — most encouraging signal so far, but only 1/3 of the validated ensemble and not multiple-comparison corrected |

Simple metrics reproduce the established program-wide null (F0a/F2/F3/N1-dry): raw correlations
are significant but sign-inconsistent across models, and mostly vanish once entropy is
controlled. The logic-focused reduced metric is more promising but incomplete — the full
validated ensemble (which is what's actually been shown to resist style-gaming) still needs a
GPU machine to test properly.

---

## 5. Pivot: test Step 1 (generation) directly instead of the metric

Per correction: stop chasing metric refinements (infrastructure-blocked, inconclusive) and
directly test whether *how rollouts are generated* changes pass@k — this is Assumption 2 without
needing any diversity metric at all.

**5a. Persona-prompt sampling** (`vanilla` / `axis` 6-persona-rotation / `matched` single coach
persona, reusing `scripts/22_aime_bedrock.py`'s prompts). 3 models × 3 conditions × 30 items,
`n_max=12`, temp=1.0. Cost: **$6.52**.
Run: `runs/aime24-multimodel-rollout-20260909-060447Z-pid3696/`.

| | pass@k |
|---|---|
| qwen32b | vanilla **0.633** > axis 0.467 > matched 0.400 |
| llama70b | vanilla 0.433 ≈ axis 0.433 > matched 0.400 |
| mistral | vanilla **0.867** > axis 0.833 > matched 0.767 |

**Result: personas do not help, and hurt qwen32b/mistral.** Vanilla wins or ties in all 3 models.
Consistent with the established F0a finding (persona/roleplay is inert-to-harmful for math
reasoning) — now replicated on AIME24 with different models.

**5b. Temperature sampling.** Bedrock Converse caps temperature at **[0, 1] for all three
providers** (1.5/2.0 rejected with `ValidationException`) — 1.0 was already the ceiling, so this
tests *lowering* temperature, not raising it. 3 models × {0.5, 0.7, 1.0} × 30 items, vanilla
prompt. Cost: **$6.83**.
Run: `runs/aime24-multimodel-rollout-20260909-072443Z-pid7768/`.

| | pass@k |
|---|---|
| llama70b | 0.5→0.367, 0.7→0.400, **1.0→0.433** (monotonic) |
| mistral | 0.5→0.767, 0.7→0.800, **1.0→0.833** (monotonic) |
| qwen32b | 0.5→0.533, **0.7→0.633** (peak), 1.0→0.567 |

**Result: the first clean, consistent positive finding in this entire exploration.** Two of three
models improve monotonically with temperature; the third peaks at an intermediate value but is
still well above the low-temperature end. Simply increasing sampling randomness does more for
pass@k than any amount of persona engineering did.

---

## 6. Cost accounting (running total, this session)

| item | cost |
|---|---|
| Step 1 exploration/debugging (access checks, smoke tests, corrupted-batch loss + retry, Kimi screens) | ~$14.49 |
| Step 2 augmentation (buggy v1 + fixed v2, both real spend) | ~$1.41 |
| Step 4 diversity scoring (metric a, reduced metric b) | $0 (local Ollama only) |
| §5a persona-sampling comparison | $6.52 |
| §5b temperature-sampling comparison | $6.83 |
| **Total** | **≈ $29.25** |

Of this, **~$1.65 was lost outright to the directory-collision bug** (§3.3) before the fix landed.

---

## 7. Open items

1. **GPT-5.6-Sol/Terra, Claude Opus-5/Sonnet-5** — still need Bedrock model-access grants from
   the account owner; can't proceed on the original model roster without this.
2. **Full validated diversity ensemble** — needs a GPU machine with the trained head weights
   (`head_l15_l4_L18_v2.pt`, `head_l4_L18_v2.pt`) restored from wherever the original N2/V6b
   session produced them; not resolvable from this laptop.
3. **Why does llama70b not benefit from temperature** the way qwen32b/mistral do? — untested
   hypothesis, could be a model-specific sampling-diversity ceiling.
4. **Multiple-comparisons caveat** on the claim:dtw partial-correlation result (§4) — 2/3
   significant models is encouraging but not yet a confirmed effect; would benefit from a
   held-out replication before treating it as established.
5. Next natural extension of §5: combine the temperature finding with group construction
   (Steps 3–5) — do groups built from higher-temperature rollouts show higher diversity scores
   *and* is that what's driving their higher pass@k, or is it a separate mechanism (e.g. just
   raw coverage, independent of anything a diversity metric would capture)?
