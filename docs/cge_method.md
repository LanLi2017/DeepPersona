# Consensus-Gated Escalation: Diagnosis-Driven Orchestration for Frontier-Model Reasoning

## Motivation

Multi-attempt orchestration — spawning sub-agents with varied instructions and aggregating their
answers — is an appealing pattern for hard reasoning tasks, but our controlled study
(`frontier_inference_diversity.md`) found the naive version is *significantly harmful*: an
orchestrator that reads prior attempts and steers each next attempt toward a "genuinely different
strategy" cost −0.085 pass@1 and −0.080 coverage@8 against a same-budget neutral-resampling
baseline. Trace-level diagnosis attributed the damage to three mechanisms. **(M1) Interpretation
anchoring:** directives conditioned on prior derivations correlate all attempts — on marginal
problems the orchestrator reads majority-wrong work and launders that wrong reading into every
subsequent attempt (per-try accuracy collapses 0.232 → 0.071; in traces, a directive states the
wrong setup as fact and seven "different methods" all derive the same wrong answer). **(M2)
Abandoning winners:** with no correctness oracle, the orchestrator diversifies away from correct
approaches too (P(wrong | gold already found) = 0.195 vs 0.131 for neutral). **(M3) An
unconditional directive tax:** forcing off-default methods lowers the per-try hit rate even after
failures (0.203 vs 0.253), because failure in this regime is execution *depth*, not strategy
choice. Consensus-Gated Escalation (CGE) is the constructive counterpart: an orchestration method
whose design makes each mechanism structurally impossible.

## Method

CGE converts the three diagnosed mechanisms into three design rules and derives the procedure from
them. *Rule 1 — information diet:* no attempt ever sees another attempt's reasoning; the
orchestration layer observes final answers only, so no wrong derivation can propagate (kills M1).
*Rule 2 — never re-open a consensus:* agreement among independent attempts is a strong correctness
proxy, so unanimous problems are settled immediately and never disturbed (kills M2). *Rule 3 —
escalate effort, not strategy:* additional compute goes into deeper reasoning on the model's own
approach, never into "use a different method" directives (kills M3).

The procedure, per problem: **(1) Probe.** Draw four blind, independent samples at low reasoning
effort under a neutral prompt. **(2) Gate** on the answer multiset alone: if all four agree, accept
the answer and stop — banking the unspent budget. If not, escalate. **(3) Escalate.** Spend the
banked budget on four blind medium-effort calls: two plain neutral samples (pure execution-depth
escalation); one *re-interpretation* attempt ("restate the problem formally from scratch, flag
possible misreadings, then solve") — targeting wrong-reading failure basins while remaining fully
blind; and one *verifier* that receives the problem plus the candidate answer multiset only —
answers-as-verification-targets is the single permitted cross-attempt information flow. **(4)
Select** by a pre-registered weighted vote: escalated attempts count double, and the verifier's
answer breaks ties. The gate makes the method adaptive: cheap consensus routes expensive reasoning
to only the ~28% of problems that need it.

## Experimental design

We evaluate on the venue and data of the parent study (BeyondAIME, 100 problems, gpt-5.5 at low
reasoning effort), reusing its neutral run so comparisons are paired per problem. The probe round
*is* the first four samples of the existing neutral run — which yields a free, exact ablation:
"escalate with four more low-effort samples" is precisely samples 5–8 of the same run, i.e. the
majority-vote-of-8 baseline. Any CGE gain over that baseline therefore isolates the *content* of
the escalation round, not the extra draws. The primary metric is final-answer accuracy against
this deployable baseline (maj@8 = 0.770 — deliberately stronger than pass@1 = 0.744), with paired
bootstrap CIs over problems; secondary metrics are coverage, tokens/problem, and a per-bin
decomposition by gate outcome. Offline simulation on existing data fixed the gate design before
any spend: unanimous-of-4 fires on 72/100 problems at 90% precision, while 3-of-4 consensus is a
coin flip (6/12 wrong) — so only unanimity stops the pipeline. Everything (gate thresholds, vote
weights, falsifiers) was fixed before the escalation round ran.

## Results

**CGE reaches 0.880 final-answer accuracy versus 0.770 for the budget-comparable baseline — a
paired improvement of +0.110 [+0.050, +0.180] — at 18.9k tokens/problem versus 15.2k.** The gains
land exactly where the design predicts: the unanimous bin is untouched (0.903, locked by the
gate), the weak-consensus bin rises 0.500 → 0.833, and the split bin 0.375 → 0.812. Coverage
matches the baseline (0.900). Against the brute-force alternative — running everything at medium
effort (maj@8 0.900 on the 30-problem subset where that baseline exists, at 44k tokens/problem) —
CGE recovers ~80% of the gain at 39% of the cost. The correct claim is therefore not that CGE
beats uniform escalation on accuracy, but that a two-cent consensus signal allocates expensive
reasoning nearly as well as spending it everywhere.

Two observations bound the method. First, its errors are now almost entirely the gate's accepted
loss: 7/100 problems are unanimously *wrong* at the probe stage and are locked in by design
(accuracy ceiling ≈ 0.93 here); detecting confidently-wrong consensus requires a verification
signal stronger than agreement and is the natural next problem. Second — the finding that closes
the research line — *within* CGE's escalation round the plain medium-effort neutral samples are
the strongest members (per-attempt accuracy 0.82/0.79 on the hardest problems, versus 0.07 for the
naive orchestrator's directed attempts on comparable problems), while the diversity-flavored roles
are the weakest (re-interpretation 0.71, verifier 0.71). Even inside an orchestration method that
works, the lift comes from *more thinking on the right problems*, not from *thinking differently*
— consistent with every experiment in this line, from RL rollout diversity to inference-time
prompting: on competition math, the binding constraint is capability, and orchestration pays
exactly insofar as it allocates capability, not variety.

## Implementation details

Model gpt-5.5 (`2026-04-23`); probe = low reasoning effort, escalation = medium; temperature not
settable on this model class. Verifier prompt exposes the candidate answer multiset as
`{answer: count}` with an instruction to solve independently first, then reconcile. Vote weights:
probe 1, escalated 2, verifier tie-break; answers compared after `\boxed{}` extraction with
sympy-equivalence grading (signal-based timeout, main thread only). Code: `scripts/13_cge.py`
(single flat script; `--simulate` replays the gate and baselines offline with zero API calls,
`--smoke` runs three split problems live, the default runs the full escalation). Round-1 reuse
draws from `runs/infdiv-f2main-BeyondAIME-neutral-k8-s0`. Escalation cost: 1.11M completion tokens
(~$40 total including smoke); a practical note — medium-effort reasoning tokens balloon ~4× on
exactly the problems the gate selects, so cost estimates must be calibrated on smoke-run actuals
for hard problems, not corpus-average tokens/call.
