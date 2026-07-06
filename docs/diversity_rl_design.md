# Does Rollout-Input Diversity Help RL for Reasoning? A Controlled Dose–Response Study

## Motivation

Reinforcement learning from verifiable rewards (RLVR) is bottlenecked by exploration: a policy can
only be reinforced on solutions it already samples, and continued training tends to concentrate the
rollout distribution, eroding the strategy diversity that drives pass@k. Prior work counteracts this
by injecting diversity at the *output* level (entropy bonuses, semantic-cluster advantage reshaping)
or at *inference* time (diverse prompting, self-consistency). We instead ask a more basic, controlled
question: if diversity is introduced at the **input** level — by conditioning the rollouts of a group
on different system prompts during RL — does it causally change *what the policy learns*? We frame
this as a mediation chain and measure both links: **(Link 1)** does input prompt-diversity raise the
*output trajectory diversity* of a rollout group, and **(Link 2)** does it change *downstream task
performance*? The contribution is the controlled causal decomposition, which is informative at any
sign; a null is as publishable as a positive, because it localizes *where* the chain breaks.

## Experimental design

We run group-relative policy optimization (GRPO) on a verifiable reasoning task. The independent
variable is **rollout-input diversity**: holding the per-problem rollout budget *G* fixed, we vary how
many *distinct* system prompts the *G* rollouts of a group are conditioned on, sweeping from a single
shared prompt (zero input diversity, i.e. standard GRPO) up to one distinct prompt per rollout. The
distinct prompts are drawn, nested, from a fixed pool of generic *strategy directives* (e.g. "reason
backwards", "enumerate cases", "exploit symmetry"), so only the *number* of distinct prompts varies
across levels, not their content. This yields a dose–response over a single, cleanly manipulated knob.

We run two complementary arms. In the **within-group** arm, the *G* rollouts of one group are split
across the distinct prompts and advantages are centered against a *shared per-problem baseline*, so
the system prompt acts as an exploration variable rather than part of the task state. Crucially, the
policy is *trained with* the diverse-prompt scaffold but *evaluated with a neutral prompt*: any gain
must have **internalized into the weights** rather than relying on a privileged test-time prompt. In
the **across-group** arm (a control), each group stays homogeneous — a textbook-valid GRPO baseline —
while the prompt varies *across* problems, isolating batch-level population diversity from within-group
credit assignment; agreement between the two arms shows the within-group baseline choice is not
driving the result. We add a **temperature-matched control**: a neutral-prompt arm sampled at a
temperature calibrated so its output diversity matches the most-diverse arm's, ruling out the confound
that input diversity merely proxies higher sampling temperature. Any positive Link-2 claim must clear
this control.

## Venue selection

Reasoning diversity can only help where the bottleneck is *finding* a solution strategy, not
*executing* a known one — that is, where pass@*k* substantially exceeds pass@1 (exploration headroom),
the solution space is genuinely multi-strategy, the reward is verifiable, and the task is tractable
but unsaturated. Because our initial reasoning regime proved near-ceiling (a small pass@1→pass@*k*
gap, leaving no room for a diversity effect to surface), we treat venue choice as an explicit,
pre-registered step: a cheap sampling-only probe measures the base model's pass@1 versus pass@*k*
across candidate benchmarks and selects the venue with the largest headroom within a trainable
accuracy band (neither floored, indicating a capability wall, nor saturated). We additionally log the
fraction of samples that emit a parseable answer, to separate genuine capability from answer-format
non-compliance when interpreting the gap.

## Analysis

We report a **dose–response curve**: each output-diversity metric and each outcome metric against the
diversity level, with confidence intervals over multiple training seeds evaluated on a *fixed* held-out
set (so seed variance reflects training stochasticity, not eval-subset resampling). We pre-register an
inverted-U expectation — too little diversity invites premature convergence, too much pushes rollouts
off-distribution and weakens credit assignment — and test for both monotone and quadratic trends. For
**mediation**, we first confirm Link 1 (diversity level → realized output diversity), then test whether
any Link-2 effect attenuates once we condition on realized output diversity. The temperature-matched
control gates the causal attribution to *input* diversity specifically.

## Findings

**Venue selection.** The sampling-only probe cleanly separated the candidates. A strong code model
*saturated* HumanEval (pass@1 ≈ 0.83, pass@16 ≈ 0.90 — a 0.07 gap, no headroom), while the
competition-math sets were both hard and *format-limited*: only a minority of non-thinking samples
emitted a parseable final answer, so their apparent gaps were partly answer-format artifacts rather
than exploration headroom — a confound our format-rate diagnostic was essential for catching. Level-5
Hendrycks MATH offered the cleanest trainable headroom and became the primary venue: at an adequate
generation budget the base policy reaches pass@1 ≈ 0.60 and pass@8 ≈ 0.78, a genuine ~0.18 gap well
inside the unsaturated band.

**Link 1 holds, and saturates early.** Conditioning rollouts on more distinct prompts reliably raises
output-trajectory diversity, and the effect is large relative to its seed variance: distinct-4 rises
from 0.557 under a shared prompt to ~0.61 at four prompts, self-BLEU falls from 0.67 to 0.62, and
final-answer entropy rises in step — all with cross-seed standard deviations near 0.002. The effect is
essentially saturated by four prompts; eight distinct prompts add no further diversity. This pattern
replicates across both the initial near-ceiling regime and the higher-headroom venue.

**Link 2 is null — and not a ceiling artifact.** Despite reliably more diverse rollouts, held-out
accuracy is flat across the entire diversity sweep. On the higher-headroom venue, neutral-prompt pass@1
is 0.595 / 0.601 / 0.598 for one / four / eight prompts — indistinguishable within seed noise — and the
within-training improvement is small and non-monotone (+0.012 / −0.005 / +0.013). Decisively, this null
*reproduces* the one observed in the near-ceiling regime (pass@1 ≈ 0.71 at every diversity level there):
moving to a venue with a real pass@1-to-pass@k gap did not surface any diversity effect. The mediation
chain thus breaks cleanly at the second link — input diversity propagates fully to output diversity but
not at all to downstream performance.

**A persistent but non-significant whisper.** The one directional signal that recurs across regimes is
at pass@8, where the diverse-prompt arms sit marginally above the shared-prompt arm (0.800 / 0.783 vs
0.775), consistent with diversity slightly widening solution *coverage* rather than lifting single-shot
accuracy. It lies within seed noise in every run, so we report but do not claim it; it is the natural
thing to chase with larger seed counts and higher k.

**Interpretation and caveats.** Across two regimes the picture is consistent: input-prompt diversity is
a *real but inert* lever for RLVR on these tasks — the bottleneck is solving capability, not the variety
of approaches the model samples, so reshaping the rollout distribution changes its surface form without
changing what is learned. Two caveats bound the strength of the negative. First, level-5 problems are
*bimodal per item* — a group's rollouts tend to be uniformly correct or uniformly wrong — producing high
constant-reward skip rates and weak gradient signal, so RL moved every arm only slightly; the clean
signal is the flatness of the *endpoint across diversity levels*, not a large within-arm change. Second,
the results to date cover the within-group arm under non-thinking decoding; the across-group control and
a thinking-mode confirmation remain to be run. A training-free companion study at frontier scale
(`frontier_inference_diversity.md`) reproduces the static null with zero training and finds that
*adaptive* diversity injection actively hurts — strengthening the capability-bound interpretation.

## Implementation details

Policy: a single open-weight ~8B instruction model in its non-thinking decoding mode (chosen so a
multi-cell sweep is affordable and the diversity lives in the visible solution text); the cost-heavier
thinking mode is reserved as a confirmatory follow-up. GRPO with group size *G*=8, group-relative
advantages, no KL or entropy regularizer; the within-group arm uses a per-problem marginalized
baseline and the across-group arm the standard per-group baseline. Diversity levels span 1/4/8 distinct
prompts (plus the neutral and temperature-matched controls), each run for tens of optimization steps
across ≥3 seeds. Reward is binary answer-correctness via the task's verifier; the held-out metric is
unbiased pass@*k* (Chen et al. estimator) under a fixed neutral prompt. Output-diversity metrics are
distinct-*n*, self-BLEU, final-answer entropy, and mean per-token sampled-logprob surprisal. The venue
probe samples N problems × *K* completions per benchmark and reports pass@1, pass@8, pass@*K*, the gap,
and the answer-format rate. Confidence intervals use a paired bootstrap over per-item differences.