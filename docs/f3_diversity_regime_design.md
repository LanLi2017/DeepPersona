# F3: Where Does Subagent Prompt-Diversity Pay? A Three-Mechanism Regime Test

## Motivation

F0a/F1/F2 established that on competition math, system-prompt diversity across rollouts/subagents
is inert (static) or harmful (adaptive), and CGE's diagnosis decomposed *why* into three
mechanisms: **(M2)** no inference-time oracle, so aggregation is voting and diverse wrong answers
abandon winners; **(A)** diversity never even raised coverage@K, because math failures are
execution-depth, not strategy choice; **(M1/interpretation)** the one framing-sensitive channel
(wrong problem reading) was rare on math. The claim worth publishing is not "diversity fails" but
a *regime characterization*: prompt diversity converts to accuracy only where (a) an oracle
filters candidates, or (b) aggregation pools instead of votes, or (c) failures are basin-locked
interpretations. F3 tests each clause on the benchmark family built for it, reusing the F2
machinery (budget-matched paired arms, K=8, unbiased pass@k, paired bootstrap). Each experiment is
informative at any sign; together they either validate the criterion or replace it.

Same mediation logic as F1: **Link A** (diversity → coverage@K: does exploration widen?) and
**Link B** (coverage → selected answer: does the aggregator harvest it?). Math broke Link A *and*
had no oracle for Link B. The three experiments pull these apart.

## E1 — Code generation with an execution oracle (LiveCodeBench) — tests clause (a)

**Venue.** LiveCodeBench code_generation_lite, newest release available at run time; prefer
problems dated after the model's cutoff; hard+medium difficulty. Venue probe first (neutral arm,
30 problems × K=8): require pass@1 → coverage@8 gap ≥ 0.10, else drop to hard-only. Contamination
caveat: the arm contrast is paired, so contamination shifts regime (saturation) rather than
biasing Δ; the probe guards the regime.

**Design: 3 sampling arms × 2 selection rules, factorial on the SAME samples.**
Arms (identical to scripts/12, K=8 budget-matched, gpt-5.5 pinned, effort=low):

1. *neutral* — same prompt ×8 (internal sampling only; temperature unavailable, as in F2);
2. *static* — 8 distinct algorithmic-strategy directives (DP / greedy / math-closed-form /
  two-pointers / graph reduction / brute-force-with-pruning / invariant hunting / data-structure
   choice) — the code analogue of STRATEGY_AXES;
3. *adaptive* — sequential; gpt-5-mini orchestrator sees prior attempts' public-test results
  (pass/fail per test, no code) and writes the next directive. Note: on code the orchestrator
   has a *partial oracle* (public tests), unlike math — this is the strongest honest version of
   "spawning subagents".

Selection rules, applied offline to saved samples:

- *vote* — majority over normalized outputs on public-test inputs (math-like, no oracle);
- *filter* — discard samples failing public tests, then earliest/majority among survivors
(the AlphaCode rule).

**Metrics.** Hidden-test pass of the selected sample (primary, per selection rule); coverage@8 on
hidden tests (Link A); unbiased pass@k; per-arm public-test survival rate; paired bootstrap on the
same fixed problem set across all arms.

**Predictions (pre-registered).** (i) If M2 was the harm: adaptive/static ≈ neutral under
*filter* even if worse under *vote* — the sign flip is the headline. (ii) If diversity is to ever
pay: Δcoverage@8 > 0 for static — algorithms is the canonical multi-strategy space; if coverage
is flat here too, Link A is dead in general, and the F1/F2 conclusion generalizes beyond math.
(iii) The directive tax (attempt-0-vs-rest within adaptive) transfers or doesn't — per-sample
accuracy comparison, same within-arm control as F2.

**Falsifier for the criterion:** static coverage@8 gain > 0.05 that *filter* fails to convert.

## E2 — Text-to-SQL (BIRD dev) — tests clause (c), interpretation-bound failures

**Venue.** BIRD dev (official zip, SQLite DBs local), moderate+challenging difficulty labels,
n=100 fixed subset (seeded), evidence strings included. Execution accuracy vs gold result set —
fully verifiable, runs locally. Venue probe: neutral K=8 on 30 questions; need result-set
*disagreement* on ≥25% of questions (the analogue of headroom).

**Design.** Failure mode here is misreading schema/question — exactly the basin-locked
interpretation channel that was too rare on math to help (reinterp was CGE's weakest role at
0.714). Three arms, K=8:

1. *neutral* ×8;
2. *reinterp-diverse* — 8 prompts each forcing a different reading discipline (paraphrase the
  question first / enumerate join-path candidates / flag ambiguous column matches / write the
   answer schema before the SQL / list plausible misreadings then commit / …) — diversity aimed
   at the *reading*, not the SQL strategy;
3. *CGE-SQL* — 4 probes; gate = execution-*result* agreement (free local oracle: run candidate
  SQL on the DB, compare result sets); on disagreement escalate 4 samples (2 neutral +
   2 reinterp) — direct port of scripts/13 with the gate signal upgraded from string match to
   execution equivalence.

Selection: execution-result majority vote (standard for the task; invalid-SQL candidates drop
out — a weak built-in filter). Note candidate execution is against the DB only, never the gold
result — no leakage.

**Metrics.** Execution accuracy of selected answer; coverage@8; distinct result-sets per question
(Link A for interpretations); per-bin gate table as in CGE.

**Predictions.** Reinterp-diverse > neutral here (opposite of math) if clause (c) is right —
the reinterp *content* finally matches the failure mode. CGE-SQL ≈ reinterp-diverse at lower
cost. Falsifier: reinterp-diverse ≤ neutral ⇒ interpretation-diversity doesn't pay even where
failures are interpretive ⇒ clause (c) is wrong and the criterion loses a clause.

## E3 — Fault localization pooling (SWE-bench Verified) — tests clause (b)

**Venue.** SWE-bench Verified (HF), n=50 seeded instances. Task: given issue text + repo file
tree (+ on-demand file reads capped at a fixed token budget), output a ranked list of ≤5 files
(and functions) that must change. Ground truth = files touched by the gold patch — verifiable by
construction, no execution environment needed.

**Design.** Aggregation is a *union* (pooling), so M2 cannot exist: a wrong lens costs budget,
never votes. Two arms, K=8 subagents each, budget-matched:

1. *neutral* ×8 — same instruction, independent samples;
2. *lens-diverse* — 8 lenses: trace-the-error-string / data-flow-from-inputs / recent-API-misuse /
  test-file-first / config-and-defaults / concurrency-state / dependency-boundary /
   docs-vs-behavior.

**Metrics.** Union recall@K over gold files (primary — the pooling analogue of coverage@K);
per-sample precision (does the lens tax individual quality, M3's analogue); marginal-gain curves
(recall vs k, k=1..8) — diversity should show a *flatter decay* of marginal gain if lenses are
complementary; paired bootstrap over instances.

**Predictions.** Lens-diverse union recall > neutral — this is the clause with the strongest
prior (attention, not capability, is the bottleneck). Falsifier: if even here diversity adds no
unique true findings, "spawn diverse subagents" has no validated home in any of our regimes.

## Shared protocol

Everything from F1/F2 discipline: fixed seeded problem subsets shared across arms (paired); K=8
budget parity; raw per-sample JSONL + manifest (git SHA, model pin `gpt-5.5-2026-04-23`, prompts,
splits); grading in main thread where signal-based timeouts apply; smoke test (3 items end-to-end)
before every live phase; all falsifiers and selection rules fixed before spend.

## Cost (estimates to be recalibrated on smoke actuals — F2b lesson: hard problems balloon ~4×)


| Phase                                    | Est. output tokens          | Est. cost (gpt-5.5 @ ~$25/M out) |
| ---------------------------------------- | --------------------------- | -------------------------------- |
| E1 probe (30×8 neutral)                  | ~0.7M                       | ~$18                             |
| E1 full (100×8×3 arms)                   | ~7M                         | ~$150–180                        |
| E2 probe + full (100×8×3, short outputs) | ~2.5M + input-heavy schemas | ~$60–90                          |
| E3 (50×8×2, short outputs, big inputs)   | ~0.5M out                   | ~$15–25                          |


Total ≈ $250–320 if all run at gpt-5.5. Staging plan: E1 probe → checkpoint → E1 two arms
(neutral, static) → adaptive only if static moves coverage; E3 is the cheapest and highest-prior —
run first. Every phase >$10 gets an explicit go/no-go with smoke-calibrated numbers per the cost
rule. A gpt-5-mini pilot of E1 (~$10–15) is available if we want the shape before frontier spend.

## E3 Results (2026-07-06, run: `runs/locdiv-swebv-n50-k8`, ~$15)

**Null — and the failure is at Link A, not the aggregator.** On 50 multi-file-gold instances
(venue probe confirmed headroom: per-sample recall 0.706, union@2 growth +0.05), lens-diverse and
neutral arms are statistically indistinguishable: per-sample recall 0.681 vs 0.681 (no directive
tax on this task, unlike math), union recall@8 0.770 vs 0.773, paired Δ = −0.003 [−0.043, +0.033].
The mechanism is visible in the outputs: lens samples are *more* similar to each other (mean
pairwise Jaccard 0.827) than plain resamples (0.808), and per-lens recall is flat (0.655–0.695).
One-line approach directives do not change what the model attends to at all — the issue text pins
the hypothesis. Contrast with F2: on math, directives *did* reshape trajectories (Link A held,
Link B failed); on localization Link A itself fails. The residual misses are shared-blind-spot:
30/127 gold files (23.6%) are found by none of the 16 samples across both arms, while cross-arm
unique finds are symmetric noise (2 lens-only vs 3 neutral-only); pooling all 16 samples buys
0.773 → 0.793, i.e. more samples of *any* kind, nothing lens-specific.

**Caveat on operationalization.** E3 tests the weakest form of "diverse subagents": one-line
system-prompt suffixes on a single-call task. Agentic subagents that explore *differently*
(different files read, different tools) may still pool usefully; whrally most favorable aggregation
regime. Combined with F1/F2: prompt-level diversity either fat this kills is the cheap
version — prompt-only persona/lens variation — in its structuails to alter behavior (E3) or alters
behavior without altering accuracy (F1, F2-static) or harms it (F2-adaptive).

## What each outcome does to the paper

The three experiments are the columns of a 3-clause criterion table; F0a/F1/F2 fill the math row
with "none of the clauses hold and diversity fails." Any clause that validates gives the paper a
positive result and a deployment rule; any that falsifies sharpens the negative into "diversity
does not pay even in its best-case regime," which is stronger than the math-only claim.