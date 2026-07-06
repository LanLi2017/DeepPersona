# Does Prompt Diversity Help a Frontier Model at Inference Time? A Training-Free Companion Study

## Motivation

Our RL study (`diversity_rl_design.md`) found that rollout-input diversity is a *real but inert*
lever for an 8B policy: conditioning rollouts on distinct strategy prompts reliably diversifies
trajectories (Link 1) but never moves held-out accuracy (Link 2), suggesting the regime is
capability-bound rather than exploration-bound. That conclusion admits an obvious rejoinder: an 8B
model may simply lack the latent ability that diversity is supposed to unlock. A frontier reasoning
model plausibly *has* multiple viable strategies latent in its policy, so the same lever might
finally pay. We therefore ask the training-free version of the question at frontier scale: holding
the sample budget *K* fixed, does manually injecting system-prompt diversity — either **up front**
or **adaptively during inference** — improve pass@*k* / coverage@*K*? Two features of frontier
reasoning APIs sharpen the design. First, these models reject a sampling-temperature control, so
prompt diversity is one of the *only* diversity levers available — the neutral baseline is pure
internal sampling. Second, mid-generation prompt rewriting is impossible through the API, so the
"modify the prompt during inference" condition is realized as sequential orchestration: an
orchestrator model reads earlier attempts and steers later ones — operationally, spawning
sub-agents with fresh directives.

## Experimental design

Three budget-matched arms, each drawing exactly *K* = 8 samples per problem from the same frontier
model on the same fixed problem set (paired comparison). **Neutral:** the same neutral prompt for
all *K* samples — the strongest natural baseline, since all diversity comes from the model's own
stochastic reasoning. **Static:** *K* distinct strategy directives (the same fixed pool of generic
method prescriptions used in the RL study — "reason backwards", "enumerate cases", …), one sample
each; diversity injected before any evidence about the problem exists. **Adaptive:** attempts are
sequential; after each attempt, an orchestrator sees the previous attempts' visible solutions and
final answers — *no oracle correctness signal* — and writes a short directive steering the next
attempt toward a strategy not yet tried. The adaptive arm is the interesting one: unlike static, it
can react to the actual distribution of attempts, and it operationalizes the popular
"spawn diverse sub-agents" pattern. Attempt 0 in the adaptive arm uses the neutral prompt, which
yields a built-in within-arm control: any gap between attempt 0 and attempts 1..K−1 isolates the
causal effect of the directive itself.

Venue selection reuses the pre-registered probe logic from the RL study: diversity can only pay
where coverage@*K* substantially exceeds pass@1. Frontier models saturate standard competition sets,
so we probed candidates with the neutral arm first and, when both candidate benchmarks saturated at
the model's default reasoning effort, opened headroom by lowering the reasoning-effort setting — a
deliberate regime choice that also carries an interpretation: can prompt diversity buy back what
less internal reasoning loses?

## Findings

**Venue.** At medium reasoning effort the frontier model saturates both candidate sets (BeyondAIME:
pass@1 0.879 → coverage@8 0.933; HMMT-Nov-2025: 0.900 → 0.933 — gaps ≤ 0.05, and samples nearly
always agree). At low effort, BeyondAIME opens genuine headroom (pass@1 0.688 → coverage@8 0.833)
and became the venue: full 100-problem set, all arms paired.

**Static diversity is inert — the RL result reproduces with zero training.** Eight distinct strategy
directives raise output diversity (mean distinct answers per problem 1.81 → 1.95) but leave
performance flat: Δpass@1 = −0.019 [−0.044, +0.005], Δcoverage@8 = −0.010 [−0.060, +0.040] (paired
bootstrap). Static diversity solves 3 problems the neutral arm misses while losing 4 it gets — a
wash. Link 1 without Link 2, exactly as in the RL study, now at frontier scale.

**Adaptive orchestration actively hurts — the first confidently-signed effect in this line.**
The adaptive arm produces the most diverse answers (2.32 distinct per problem) and the worst
performance: Δpass@1 = −0.085 [−0.117, −0.056], Δcoverage@8 = −0.080 [−0.140, −0.020] — both CIs
exclude zero. The within-arm control pins the mechanism: attempt 0 (neutral prompt) scores 0.73,
matching the neutral arm's 0.744, while orchestrated attempts 1–7 average ≈ 0.65 — the directive
itself costs ~9 points of per-sample accuracy. Because the model's default approach is correct ~74%
of the time and the orchestrator has no correctness signal, "try something genuinely different"
mostly means abandoning a winning strategy: the adaptive arm recovers exactly 1 problem the neutral
arm misses while forfeiting 9 it solves.

**The residual failures are capability-bound.** Pooling all 24 samples across all three arms solves
94/100 problems — only 4 more than the neutral arm alone — and 6 problems resist every condition.
This caps the achievable upside of *any* diversity-injection policy at this budget at ≈ +0.04
coverage, including the natural "diversify only when attempts disagree" refinement, which can at
best remove the harm, not create a win.

**Interpretation.** Across the RL study and this one — an 8B under GRPO in two regimes, and a
frontier model training-free — the picture is uniform: prompt diversity reliably reshapes
trajectories and never converts to accuracy on competition math; the binding constraint is solving
capability, not exploration. The new, actionable finding here is that forced diversification has a
measurable *price* when the base policy is usually right: naively spawning sub-agents with
"different approach" directives is not merely useless but harmful, at a quantified −8 points of
pass@1/coverage under a matched budget. Caveats: a single venue and effort setting, one run per arm
(the paired 100-problem bootstrap is the operative uncertainty), and an orchestrator that
diversifies unconditionally rather than only on disagreement.

## Follow-up: Consensus-Gated Escalation (CGE) — orchestration designed against the failure modes

The diagnosis above implies three design rules: never share derivations between attempts (the
anchoring channel), never re-open a consensus (the abandoned-winners channel), and escalate
*effort* rather than prescribe *strategies* (the directive-tax channel). CGE implements them: draw
4 blind neutral low-effort samples; if their answers are unanimous, accept and stop (on our data
this gate fires on 72/100 problems at 90% precision); otherwise spend the banked budget on four
blind medium-effort calls — two plain neutral, one "restate the problem from scratch, then solve"
(targets wrong-reading basins without leakage), and one verifier that sees the *candidate answer
multiset only*, never derivations. Selection is a pre-registered weighted vote (escalated
attempts ×2, verifier breaks ties).

**Result.** Against the deployable baseline (majority vote over 8 neutral samples, 0.770), CGE
reaches **0.880** — a paired improvement of **+0.110 [+0.050, +0.180]** — with gains exactly where
designed: the unanimous bin is untouched (0.903), the weak-consensus bin rises 0.500 → 0.833 and
the split bin 0.375 → 0.812. Cost is 18.9k tokens/problem versus the baseline's 15.2k. On the
30-problem subset where a uniform medium-effort baseline exists, CGE recovers most of its gain
(0.867 vs 0.900) at 39% of its cost (17k vs 44k tokens/problem) — so CGE's contribution is
*adaptive compute allocation*: a cheap consensus signal routes expensive reasoning to the ~28% of
problems that need it, rather than beating brute-force escalation outright.

**The closing irony.** Within CGE's escalation round, the plain neutral medium-effort samples are
the *strongest* members (per-attempt accuracy 0.82/0.79 on the hardest problems — versus 0.07 for
the naive orchestrator's directed attempts on comparable problems), while the diversity-flavored
roles are the weakest (re-interpretation 0.71, verifier 0.71). Even inside the orchestration
method that works, the lift comes from *more thinking on the right problems*, not from *thinking
differently* — fully consistent with the capability-bound conclusion of the entire line. The 7
unanimous-but-wrong problems remain CGE's accepted loss (its accuracy ceiling here is ≈0.93);
detecting confidently-wrong consensus is the natural next problem, and by construction requires a
verification signal stronger than agreement.

## Implementation details

Model: `gpt-5.5-2026-04-23` (newest available), `reasoning_effort=low` for the main runs (medium for
the saturation probes), `max_completion_tokens` 48k; temperature is not settable on this model
class. Orchestrator: `gpt-5-mini-2025-08-07` at low effort, prompted with the problem plus each
prior attempt's final answer and a 600-character solution excerpt, emitting one ≤40-word imperative
directive. Datasets: ByteDance-Seed/BeyondAIME (100 problems, full set) and MathArena
HMMT-Nov-2025 (probe only); fixed problem order (seeded shuffle) shared across arms. Grading:
`\boxed{}` extraction + sympy-equivalence with a signal-based timeout (main thread only; sampling
fans out over a thread pool, grading runs afterwards). Metrics: pass@1, empirical coverage@8,
unbiased pass@k (Chen et al.), answer-format rate, mean distinct final answers per problem; arm
contrasts use a paired bootstrap (10k resamples) over per-problem differences. Code:
`scripts/12_frontier_infdiv.py` (runner, one arm per invocation, raw per-sample JSONL + manifest)
and `scripts/analyze_infdiv.py` (paired analysis). CGE: `scripts/13_cge.py` — round 1 reuses the
first 4 samples of the existing neutral run (paired across methods; the "4 more low samples"
ablation is samples 5–8 of the same run, ≙ maj@8); escalation = 4 medium-effort calls on the 28
gated problems; offline `--simulate` mode reproduces the gate and baseline exactly before any
spend. Total cost ≈ 7.1M completion tokens (~$200) for F2 + ≈1.3M (~$40) for CGE; future runs gate
any >$10 estimate on explicit approval.
