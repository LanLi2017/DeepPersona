# Research Playbook: Semantic Behavioral Experience for LLM Agents

> **Purpose**
>
> This document is an execution guide for a research agent studying semantic/behavioral representations of interactive LLM-agent trajectories, non-redundant experience, pass@k, and post-training data efficiency.
>
> It integrates the trajectory-diversity research brief, the literature/novelty audit, and the staged MVP plan into one research program with explicit hypotheses, experiments, decision criteria, and deliverables.
>
> The literature is moving quickly in 2026. Re-verify titles, versions, venues, and exact empirical claims whenever preparing a paper draft or novelty statement.

---

# 0. Executive Direction

The broad starting idea is:

\[
\text{learned prompt/task generator}
\rightarrow
\text{diverse trajectories}
\rightarrow
\text{higher pass@k / better post-training}.
\]

The sharper research program is:

\[
\boxed{\text{Semantic Behavioral Experience for LLM Agents}}
\]

with the central question:

> **What makes two interactive agent trajectories genuinely different from the perspective of behavior and learning value?**

The near-term project centers on a **semantic behavioral trajectory representation/kernel**. Once this quantity is validated, the same signal can support data selection, experience valuation, and eventually learned exploration.

A strong final thesis is:

> **Agent training data should be valued by the new, learnable behavior it contributes, rather than by trajectory count, surface diversity, or reward alone.**

A metric-focused thesis is:

> **Surface-different trajectories can be behaviorally redundant, while surface-similar trajectories can visit meaningfully different information and decision states. A useful measure of agent experience therefore operates over semantic behavioral transitions rather than raw text or exact action identity.**

The research progression is:

\[
\boxed{\text{Measure} \rightarrow \text{Validate} \rightarrow \text{Explain} \rightarrow \text{Select} \rightarrow \text{Acquire}}
\]

Each stage earns the next one by establishing a stronger empirical claim.

---

# 1. Research Contract

## 1.1 Primary scientific question

> **Under a fixed rollout, token, or environment-interaction budget, does semantic-behavioral novelty among agent trajectories predict and improve marginal learning value?**

This question anchors method design, experiments, and literature review.

## 1.2 Core hypotheses

### H1 — Surface diversity overestimates behavioral diversity

Repeated rollouts can look lexically different while following the same semantic strategy:

\[
D_{\text{text}} \not\approx D_{\text{behavior}}.
\]

### H2 — Semantic behavioral trajectory representations recover strategy equivalence

A structured metric over semantic transitions should agree better with human/functional judgments than:

- whole-trajectory text embeddings;
- exact action identity;
- action edit distance;
- exact state-transition overlap;
- generic LLM-as-a-judge similarity.

### H3 — Behavioral diversity explains pass@k beyond pass@1

For fixed model/task conditions:

\[
D(\tau_{1:k}) \uparrow
\Rightarrow
\operatorname{pass@k}-\operatorname{pass@1} \uparrow
\]

after controlling for task difficulty, rollout count, token budget, temperature, and trajectory length.

### H4 — Behavioral diversity predicts marginal post-training value

For matched training subsets:

\[
D_{\text{behavior}}(S)
\]

should predict:

\[
\Delta \text{downstream performance}(S)
\]

better than surface diversity or reward alone.

### H5 — Long-term target: useful experience is novelty conditioned on learnability

The more fundamental quantity is:

\[
V(\tau\mid\theta,\mathcal D),
\]

where \(\theta\) is the current learner and \(\mathcal D\) is what it has already seen.

A candidate factorization is:

\[
V(\tau\mid\theta,\mathcal D)
\approx
N_{\text{behavior}}(\tau\mid\mathcal D)
\times
L(\tau\mid\theta).
\]

---

# 2. Contribution Positioning

The literature already provides several important building blocks. The project should build directly on them and place the contribution at their intersection.

## 2.1 Established building blocks

### Occupancy measures in language agents

DMPO already uses state-action occupancy measures in multi-turn language-agent optimization.

**Project position:** use occupancy ideas to define a reusable representation of **behavioral redundancy between complete interactive trajectories**.

### Occupancy-based behavioral descriptors

AutoQD maps policy occupancy measures into MMD-style behavioral descriptors.

**Project position:** make the state/action geometry appropriate for language agents by introducing **semantic transition equivalence**, then validate the resulting kernel on agent-specific downstream quantities.

### Semantic state tracking

WebStep constructs semantic MDP/state-transition traces for web agents.

**Project position:** turn semantic process representations into a **general pairwise and set-level behavioral similarity measure**, and test transfer beyond one manually specified semantic MDP.

### Diverse state-transition exploration

DPEPO explicitly rewards diverse actions and state transitions.

**Project position:** compare exact transition identity with **semantic equivalence-aware transition diversity** and test which better predicts useful exploration and training value.

### Behavioral equivalence / bisimulation

BiPACE uses actor hidden-state similarity as a proxy for behavioral equivalence among agent states.

**Project position:** extend the notion from step-level credit assignment to **whole-trajectory and set-level experience redundancy**.

### Fixed-budget trajectory diversity

TDScaling demonstrates the value of broad trajectory diversity for code/tool-agent training.

**Project position:** move from dataset-level domain/mode breadth to **instance-level redundancy between concrete realized trajectories**.

### State coverage and training value

Recent on-policy distillation work directly connects induced state coverage to data efficiency.

**Project position:** study **environment-grounded semantic behavioral coverage**, make it reference-free, and connect it to interactive-agent post-training.

## 2.2 Core novelty target

The strongest working contribution is:

> **A general, reference-free trajectory kernel for interactive LLM agents that compares complete experiences through semantically equivalent environment transitions, and validates the resulting notion of behavioral non-redundancy against strategy equivalence, pass@k, and marginal post-training value.**

Compactly:

\[
\boxed{
\text{semantic transition equivalence}
\rightarrow
\text{trajectory behavioral kernel}
\rightarrow
\text{set-level non-redundancy}
\rightarrow
\text{learning value}
}
\]

---

# 3. Closest Prior Work and Required Comparisons

These papers define the strongest baselines and the clearest differentiation tests.

## 3.1 AutoQD — occupancy → MMD behavioral descriptors

**AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization**  
ICLR 2026  
https://proceedings.iclr.cc/paper_files/paper/2026/hash/a137760f1bc695ce695c328919b553de-Abstract-Conference.html

Core overlap:

\[
\text{occupancy measure}
\rightarrow
\text{MMD embedding}
\rightarrow
\text{behavior descriptor}.
\]

### Research comparison

Implement generic embedding + MMD as a primary baseline. Show where language-agent semantic equivalence requires more structure than native/raw state-action geometry.

---

## 3.2 DMPO — occupancy in language-agent optimization

**Direct Multi-Turn Preference Optimization for Language Agents**  
EMNLP 2024  
https://aclanthology.org/2024.emnlp-main.138/

### Research comparison

Use DMPO to motivate that occupancy is a natural language-agent object, then focus the contribution on **behavioral representation and redundancy**, rather than preference-optimization derivation.

---

## 3.3 WebStep — semantic MDP/state tracking

**Where Did It Go Wrong? Process-Level Evaluation of Web Agents with Semantic State Tracking**  
COLM 2026  
https://arxiv.org/abs/2606.15673

### Research comparison

Use WebStep as a high-quality semantic-state benchmark when possible. Demonstrate that the proposed kernel supports pairwise/set-level comparison and transfers to another environment.

---

## 3.4 DPEPO — diverse action/state-transition reward

**DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents**  
ACL 2026  
https://aclanthology.org/2026.acl-long.2151/

### Research comparison

Implement exact action/transition diversity as a baseline. Construct equivalence cases where two distinct raw transitions implement the same semantic operation, and test whether semantic transition diversity is more predictive of useful coverage.

---

## 3.5 BiPACE — behavioral equivalence for LLM-agent states

**BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents**  
2026  
https://arxiv.org/abs/2606.25556

### Research comparison

Include actor hidden-state similarity as a baseline where feasible. Evaluate whether whole-trajectory semantic occupancy captures redundancy missed by local state similarity.

---

## 3.6 TDScaling — fixed-budget trajectory diversity

**Beyond Quantity: Trajectory Diversity Scaling for Code Agents**  
Findings of ACL 2026  
https://aclanthology.org/2026.findings-acl.768/

Uses:

- Domain Entropy;
- Reasoning Mode Entropy;
- Cumulative Action Complexity.

### Research comparison

Compare coarse dataset breadth against pairwise/set-level semantic behavioral redundancy. The key question is whether instance-level redundancy offers predictive information beyond domain/mode coverage.

---

## 3.7 One Training Example — state coverage → training value

**Rethinking On-Policy Distillation of Large Language Models II: One Training Example**  
2026  
https://arxiv.org/abs/2609.04172

### Research comparison

Treat this as strong evidence that state coverage can explain data value. Test whether **environment-grounded semantic behavioral coverage** provides an analogous or stronger signal for interactive agents.

---

# 4. Additional Required Literature

## Reasoning diversity / pass@k

- Representation-Based Exploration for Language Models: From Test-Time to Post-Training — ICLR 2026  
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/c420aca62da1d668e8e79e0bcae24e6b-Abstract-Conference.html
- Reasoning Path Divergence (RPD)  
  https://arxiv.org/abs/2510.26122
- SetPO  
  https://arxiv.org/abs/2602.01062
- DQO — ICLR 2026  
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/cfeb1fa78c5fb06be27c0980aa8ee581-Abstract-Conference.html
- UpSkill  
  https://arxiv.org/abs/2602.22296
- Global Forking Tokens — ICLR 2026  
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/88af3540325dd0b70617a9ab605f294d-Abstract-Conference.html
- R-Diverse  
  https://arxiv.org/abs/2602.13103

## Generic RL / behavior diversity

- TrajeDi — ICML 2021  
  https://proceedings.mlr.press/v139/lupu21a.html
- Maximum State Entropy with Parallel Agents — ICML 2025  
  https://proceedings.mlr.press/v267/de-paola25a.html
- QDHF — ICML 2024  
  https://proceedings.mlr.press/v235/ding24h.html
- Policy-Based Trajectory Clustering — UAI 2026  
  https://proceedings.mlr.press/v337/wang26a.html
- Temporal Dependency-Aware Trajectory-Level Behavioural Metric — 2026  
  https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70109

## Data selection / learning value

- NovelSum — ACL 2025  
  https://aclanthology.org/2025.acl-long.908/
- Rank–Surprisal Ratio — ACL 2026  
  https://aclanthology.org/2026.acl-long.1950/
- BOOST  
  https://arxiv.org/abs/2605.24743
- The Coverage Principle — ICLR 2026  
  https://arxiv.org/abs/2510.15020
- Post-Training is About States, Not Tokens  
  https://arxiv.org/abs/2605.22731

---

# 5. Technical Object: Semantic Transition Representation

A raw trajectory is:

\[
\tau=[(s_1,a_1,o_1),\ldots,(s_T,a_T,o_T)].
\]

For many language agents:

- \(s_t\): text/UI/code/environment state;
- \(a_t\): tool/API/UI action;
- \(o_t\): textual or structured observation.

The project represents behavior at the level of **semantic transitions**.

## 5.1 Transition-centered representation

Use:

\[
z_t=\phi(s_t,a_t,s_{t+1}).
\]

This captures the semantic meaning of the action together with its effect.

Example raw action:

```text
click(element_341)
```

Canonical transition:

```text
INSPECT_CANDIDATE(
  object=hotel_A,
  information_revealed=[price, reviews]
)
```

## 5.2 Candidate canonical schema

\[
z_t=(g_t,a_t,\theta_t,i_t,\Delta s_t,r_t)
\]

where:

- \(g_t\): current subgoal;
- \(a_t\): semantic action category;
- \(\theta_t\): semantic arguments/entities;
- \(i_t\): information acquired;
- \(\Delta s_t\): meaningful state change;
- \(r_t\): role in strategy.

This factorization supports interpretable ablations.

---

# 6. Behavioral Occupancy Kernel

Given transition embeddings:

\[
e_t=f_\psi(z_t),
\]

define trajectory occupancy:

\[
\mu_\tau=\frac{1}{Z}\sum_{t=1}^{T}w_t\delta_{e_t}.
\]

Weights can be uniform or discounted.

Define pairwise distance:

\[
D_{\text{occ}}(\tau_i,\tau_j)
=
D_{\mathcal P}(\mu_{\tau_i},\mu_{\tau_j}).
\]

Candidate distribution distances:

- MMD;
- Wasserstein / optimal transport;
- Sinkhorn distance;
- energy distance;
- kernel mean embedding distance.

The scientific contribution should concentrate on the semantic equivalence representation \(\phi\) and its downstream validity. Standard distribution distances provide the comparison machinery.

---

# 7. Temporal Structure

Trajectory strategy can depend on ordering. The project should therefore compare several temporal representations.

### Variant A — Transition occupancy

\[
(s_t,a_t,s_{t+1})
\]

### Variant B — Semantic 2-gram occupancy

\[
(z_t,z_{t+1})
\]

### Variant C — Position-aware occupancy

\[
(z_t,t/T)
\]

### Variant D — Dependency-edge occupancy

Represent causal/subgoal relationships between semantic steps.

### Variant E — Semantic sequence alignment

Use semantic edit distance, sequence kernels, or temporally regularized optimal transport.

The experiment should identify the simplest temporal structure that materially improves behavioral-equivalence and downstream-utility prediction.

---

# 8. Pairwise Metric Baselines

Every metric experiment should include a serious baseline suite.

| Metric | Semantic? | Temporal? | State-aware? | Whole trajectory? |
|---|---:|---:|---:|---:|
| Raw text embedding cosine | weak | no | no | yes |
| Exact action overlap | no | partial | no | yes |
| Action edit distance | no | yes | no | yes |
| Semantic action edit | yes | yes | weak | yes |
| RPD-style alignment | yes | weak | no | yes |
| Actor hidden-state similarity | learned | weak | partial | local/aggregate |
| LLM judge | yes | potentially | potentially | yes |
| Graph/dependency edit | depends | causal | partial | yes |
| Semantic transition occupancy | yes | first-order | yes | yes |
| Higher-order occupancy | yes | yes | yes | yes |

Priority comparisons:

- generic text embedding + MMD;
- DPEPO-style exact transition diversity;
- BiPACE-inspired actor-state similarity;
- RPD-style semantic alignment;
- LLM judge.

---

# 9. Set-Level Diversity and Behavioral Effective Sample Size

The set-level object is:

\[
D_{\text{set}}(\{\tau_1,\ldots,\tau_K\}).
\]

Candidate aggregators:

- mean pairwise distance;
- Vendi Score / effective rank;
- DPP determinant;
- SetPO-style marginal contribution;
- facility-location / coverage objectives.

A useful derived statistic is:

\[
\boxed{\operatorname{BESS}(\Omega)=\text{Behavioral Effective Sample Size}}
\]

Interpretation:

> Out of \(K\) nominal rollouts, how many meaningfully non-redundant behavioral modes were obtained?

Example:

\[
K=32,\qquad \operatorname{BESS}=6.4.
\]

BESS becomes scientifically useful when the underlying kernel passes equivalence and downstream-validity tests.

---

# 10. Milestone / Gate Plan

Each gate answers one falsifiable question and produces a concrete deliverable.

## G0 — Novelty map

### Goal

Maintain a current map of direct overlaps around:

1. semantic transition equivalence;
2. pairwise/set-level agent trajectory kernels;
3. trajectory diversity and pass@k;
4. behavioral coverage and post-training utility.

### Deliverable

A compact novelty matrix that states:

- what each close paper contributes;
- which baseline it implies;
- the specific dimension this project adds.

### Current status

**Conditionally complete.** The sharpened semantic-behavioral-kernel formulation remains viable.

---

## G1 — Controlled Behavioral Equivalence MVP

### Question

Can the metric distinguish behavior-preserving surface changes from behavior-changing strategy changes?

### Data

Construct 100–300 trajectory pairs.

### Behavior-preserving transformations

- reasoning paraphrase;
- tool aliases;
- entity renaming;
- equivalent API calls;
- equivalent search engines;
- UI/DOM ID changes;
- irrelevant no-op actions;
- equivalent routes to the same information state.

Expected:

\[
D(\tau,\tilde\tau)\approx 0.
\]

### Behavior-changing transformations

- one source vs cross-check;
- direct selection vs compare-and-verify;
- different information source;
- different explored state region;
- retry vs diagnose-and-recover;
- changed dependency/action order.

Expected:

\[
D(\tau,\tau')\gg 0.
\]

### Baselines

- text cosine;
- action edit distance;
- exact transition overlap;
- semantic action edit;
- actor hidden-state cosine if available;
- LLM judge;
- semantic transition occupancy.

### Metrics

- AUROC;
- pairwise ranking accuracy;
- effect size;
- robustness by perturbation type;
- sensitivity to length.

### Success criterion

The semantic representation provides a clear and statistically stable improvement over generic embedding + MMD and identity-based metrics.

### Decision criterion

If generic embeddings or an LLM judge already achieve near-ceiling performance, shift effort toward downstream experience valuation rather than further metric complexity.

---

## G2 — Real Trajectory Strategy Similarity

### Question

Does the metric agree with judgments of strategy equivalence on real rollouts?

### Candidate environments

1. ALFWorld;
2. structured tool/API environments;
3. \(\tau^2\)-Bench/BFCL-style traces;
4. WebStep when accessible.

### Annotation rubric

For each pair:

- same strategy;
- mostly same;
- partially different;
- fundamentally different.

Optional operational labels:

- same information sources?;
- same subgoal structure?;
- same recovery behavior?;
- same failure mode?;
- interchangeable as training data?

### Pilot

Annotate 50–100 pairs first and measure inter-annotator agreement.

### Success criterion

The metric outperforms naive text/action metrics and remains stable across model families and task types.

### Decision criterion

If broad “strategy similarity” produces weak annotator agreement, refine the construct into operational functional-equivalence labels and continue with those.

---

## G3 — Behavioral Effective Sample Size and pass@k

### Question

Does semantic behavioral diversity explain multi-rollout success beyond single-rollout quality?

### Setup

For each task, sample:

\[
K\in\{2,4,8,16,32\}.
\]

Vary:

- temperature;
- sampling seed;
- prompt variant;
- model when feasible.

Compute:

- text diversity;
- exact action diversity;
- exact transition diversity;
- semantic BESS;
- pass@1;
- pass@k.

### Primary regression

\[
\operatorname{pass@k}
=
\beta_0
+\beta_1\operatorname{pass@1}
+\beta_2\operatorname{BESS}
+\beta_3\text{difficulty}
+\beta_4\text{length}
+\epsilon.
\]

Primary test:

\[
\beta_2>0.
\]

A stronger target is:

\[
\Delta_k=\operatorname{pass@k}-\operatorname{pass@1}.
\]

### Temperature experiment

Measure whether:

- text diversity keeps rising;
- behavioral diversity saturates;
- pass@k saturates near the behavioral-diversity saturation point.

### Success criterion

Semantic set diversity adds meaningful out-of-sample predictive value beyond pass@1 and naive diversity metrics.

### Decision criterion

If semantic diversity explains little additional pass@k variance, prioritize G4/post-training utility, where the metric may still be valuable as a data-redundancy measure.

---

## G4 — Fixed-Budget Post-Training Utility

### Question

Do behaviorally diverse trajectories causally provide higher training value?

### Setup

Generate one common rollout pool and construct matched subsets with the same:

- trajectory count;
- total tokens;
- task composition;
- approximate success-rate/reward distribution;
- trajectory-length distribution.

### Selection methods

1. random;
2. reward-only;
3. difficulty-only;
4. text diversity;
5. exact transition diversity;
6. RPD-style diversity;
7. semantic occupancy diversity;
8. semantic occupancy × learnability.

### Evaluation

Train identical checkpoints with identical budgets.

Primary plot:

\[
\text{performance}
\quad\text{vs}\quad
\text{training tokens / trajectories}.
\]

Strong result examples:

> 30% behaviorally selected data matches 100% random data.

or:

> The same downstream performance is reached with 2× fewer trajectories.

### Success criterion

A reproducible fixed-budget advantage across seeds and at least one held-out task distribution.

### Decision criterion

If diversity-only gains become small after matching reward/task/token distributions, elevate **experience valuation** as the main thesis and combine novelty with learnability.

---

## G5 — Experience Valuation

### Goal

Move from:

\[
D(\tau,\mathcal D)
\]

to:

\[
V(\tau\mid\theta,\mathcal D).
\]

Candidate formulation:

\[
V=N_{\text{behavior}}\times L_{\text{student}}.
\]

Learnability proxies:

- student NLL;
- RSR-like suitability;
- uncertainty;
- advantage magnitude;
- gradient norm;
- gradient novelty;
- one-step loss reduction;
- influence-function proxy.

### High-impact hypothesis

Learning value may be non-monotonic in novelty:

- familiar experience contributes little new information;
- moderately novel experience offers high learnable value;
- extremely novel/off-policy experience contributes less immediate learning value.

A transferable novelty–learnability–value curve across models and domains could become the central scientific result.

---

## G6 — Active Experience Acquisition / Learned Exploration

### Goal

Use the validated value signal to acquire informative trajectories directly.

Generator:

\[
p_i\sim G_\phi(\cdot\mid x,\mathcal M)
\]

then:

\[
\tau_i\sim\pi_\theta(\cdot\mid x,p_i).
\]

Reward:

\[
R_G
=
\alpha\Delta D_{\text{behavior}}
+\beta L(\tau_i\mid\theta)
+\gamma R_{\text{success}}
-\lambda C.
\]

### Baselines

- temperature sampling;
- diverse personas;
- hand-written strategy prompts;
- latent/control-token exploration;
- difficulty curriculum;
- oversample-then-select.

### Primary efficiency metrics

\[
\boxed{\text{learning gain per environment interaction}}
\]

and:

\[
\boxed{\text{behavioral coverage per rollout}}
\]

A strong result shows the same behavioral coverage or downstream gain with substantially fewer interactions.

---

# 11. Research-Agent Operating Principles

## Principle 1 — One active gate

Keep one milestone as the active research question. Record later-stage ideas in `parking_lot.md` and return to them when the corresponding gate becomes active.

## Principle 2 — Every experiment answers a falsifiable question

Write experiments in the form:

> “Does semantic transition occupancy rank behavior-preserving pairs as more similar than behavior-changing pairs more accurately than text cosine and action edit distance?”

## Principle 3 — Every update uses four sections

### Evidence
What was observed?

### Interpretation
What does it imply for the hypothesis?

### Decision
Which research branch now has the highest expected information value?

### Next experiment
What is the single highest-information next test?

## Principle 4 — Pre-register result interpretations

Before running an experiment, specify what different outcomes imply for:

- representation quality;
- metric quality;
- benchmark suitability;
- existence of a real diversity signal;
- downstream optimization.

This makes negative results useful research evidence.

## Principle 5 — Convert new literature into decisions

For every newly found paper, record its effect on at least one of:

- novelty positioning;
- baseline set;
- hypothesis;
- experiment design;
- claim language.

Papers with no material effect belong in the bibliography rather than the active decision log.

## Principle 6 — Escalate method complexity from diagnosed baseline failures

Start with:

- frozen text embeddings;
- structured/canonical text embeddings;
- exact transition metrics;
- simple semantic canonicalization;
- standard distribution distances.

Introduce learned encoders, graph models, or RL objectives when a measured failure mode identifies the need.

## Principle 7 — Preserve the research sequence

\[
\boxed{\text{Measure} \rightarrow \text{Understand} \rightarrow \text{Select} \rightarrow \text{Generate}}
\]

This keeps each later method grounded in an already validated target quantity.

---

# 12. Required Project Files

## `research_contract.md`

Contains:

- central question;
- current scope;
- current gate;
- current primary success criterion.

## `hypotheses.md`

Suggested schema:

| ID | Hypothesis | Status | Evidence | Confidence | Next test |
|---|---|---|---|---|---|
| H1 | Surface diversity overstates behavioral diversity | untested | — | — | G1 |
| H2 | Structured semantic metric improves equivalence judgments | untested | — | — | G1/G2 |
| H3 | BESS predicts pass@k gain | untested | — | — | G3 |
| H4 | Behavioral diversity improves fixed-budget training | untested | — | — | G4 |

Statuses:

- untested;
- supported;
- weakened;
- falsified.

## `decision_log.md`

```text
Date:
Decision:
Evidence:
Alternatives considered:
What result would reverse this decision:
```

## `parking_lot.md`

Store future-scope ideas such as:

- learned exploration generator;
- QD-RL;
- learned graph metric;
- cross-domain transfer;
- online acquisition;
- theoretical bounds;
- foundation-scale experiments.

## `novelty_matrix.md`

Columns:

- Paper;
- Year/Venue;
- Setting;
- Metric object;
- Semantic equivalence?;
- Whole trajectory?;
- Set-level?;
- Interactive agent?;
- pass@k?;
- post-training?;
- Closest overlap;
- Project distinction.

## `experiment_registry.md`

```text
Experiment ID:
Question:
Hypothesis:
Dataset:
Model:
Metric:
Baselines:
Controls:
Success criterion:
Outcome interpretation:
Result:
Decision:
```

---

# 13. Minimum Viable Implementation Sequence

## MVP-1 — Controlled equivalence pairs

Goal:

\[
\text{textual difference} \neq \text{behavioral difference}.
\]

Use controlled transformations and frozen metrics only.

## MVP-2 — Real strategy labels

Goal:

Show that a simple semantic-transition representation better matches real strategy equivalence than naive metrics.

## MVP-3 — Repeated rollouts and pass@k

Goal:

Show whether BESS adds explanatory power beyond pass@1.

## MVP-4 — Fixed-budget selection

Goal:

Test whether semantic behavioral coverage predicts causal training value under matched budgets.

Use a low-cost proxy first when useful:

- one-step held-out loss reduction;
- gradient novelty;
- short SFT run;
- adapter update.

---

# 14. Candidate Domains

## ALFWorld

Strengths:

- cheap;
- structured;
- interpretable actions/states;
- direct connection to DPEPO/BiPACE-style baselines.

Best use: G1/G2.

## Structured tool/API agents

Strengths:

- clean action semantics;
- easier canonicalization;
- low rollout cost.

Best use: initial cross-domain validation.

## Code agents

Strengths:

- executable success signal;
- rich strategies;
- TDScaling precedent.

Best use: later external-validity study.

## Web agents

Strengths:

- strongest motivation for semantic state/action abstraction;
- rich information-seeking strategies;
- WebStep semantic-state infrastructure.

Best use: later high-fidelity validation after the metric shows signal in a cheaper domain.

---

# 15. Controlled Equivalence Test Suite

For each base trajectory, construct transformed variants.

## Surface-equivalent variants

- tool alias;
- argument paraphrase;
- entity renaming;
- observation paraphrase;
- UI identifier/layout perturbation;
- no-op insertion;
- equivalent route to the same information state.

## Strategy-changing variants

- one source vs cross-check;
- direct commitment vs verify-before-answer;
- single candidate vs multi-candidate comparison;
- retry vs diagnose-and-recover;
- same tools but different information coverage;
- changed dependency/order.

The test suite should become a reusable benchmark artifact if it proves diagnostic.

---

# 16. Metric Design Space

## 16.1 Canonical semantic text

Use rules or an LLM to emit concise semantic transition descriptions, then embed them.

This is the cheapest structured baseline.

## 16.2 Factorized transition kernel

\[
k(z,z')
=
\alpha k_g
+\beta k_a
+\gamma k_i
+\delta k_{\Delta s}.
\]

Advantages:

- interpretable;
- easy to ablate;
- easy to diagnose.

## 16.3 Optimal transport

Treat a trajectory as a weighted set/process of semantic transitions.

\[
W_p(\mu_\tau,\mu_{\tau'}).
\]

Advantages:

- soft matching;
- variable-length support.

Evaluate temporal augmentations alongside it.

## 16.4 MMD

Use MMD as a strong standard baseline because of AutoQD.

## 16.5 Temporal kernel

Add:

- position;
- n-grams;
- local temporal context;
- dependency edges.

## 16.6 Learned contrastive metric

Introduce a learned metric after simple baselines reveal a systematic gap.

Train:

\[
f_\psi(\tau_i,\tau_j)
\rightarrow
P(\text{same strategy}).
\]

Possible supervision:

- human labels;
- controlled transformation invariances;
- environment equivalence;
- LLM teacher labels.

---

# 17. Downstream Utility Metrics

Use several endpoints because the research question is broader than pass@k.

## pass@k

Measures multi-attempt success.

## pass@k gain beyond pass@1

\[
\operatorname{pass@k}-\operatorname{pass@1}.
\]

This is especially useful for separating single-rollout quality from multi-mode coverage.

## Learning Gain per Environment Interaction

\[
\mathrm{LGEI}
=
\frac{\Delta\text{downstream performance}}
{\#\text{environment interactions}}.
\]

## Learning Gain per Training Token

\[
\mathrm{LGPT}
=
\frac{\Delta\text{performance}}
{\text{training tokens}}.
\]

## Area Under Learning Curve

\[
\mathrm{AULC}
=
\int_0^B \mathrm{Perf}(b)\,db.
\]

These efficiency metrics can become stronger headline quantities than a single final score.

---

# 18. Statistical Controls

Report diversity/performance relationships with controls for:

- pass@1;
- task difficulty;
- rollout count;
- token count;
- trajectory length;
- success rate;
- temperature;
- model identity.

For subset-training experiments, also match:

- task distribution;
- reward distribution;
- token budget;
- success/failure mix.

Use out-of-sample prediction or held-out tasks whenever possible rather than only in-sample correlation.

---

# 19. Differentiation Tests Against Closest Prior Work

## AutoQD comparison

**Question:** Does semantic transition structure add value beyond generic embedding + MMD?

**Test:** Compare raw/text-state occupancy and semantic-transition occupancy on controlled equivalence pairs and downstream utility prediction.

## WebStep comparison

**Question:** Does the metric generalize beyond one manually specified semantic MDP?

**Test:** Validate on WebStep plus at least one independent structured agent environment.

## DPEPO comparison

**Question:** Does semantic transition diversity outperform exact transition diversity?

**Test:** Construct tool/API aliases and semantically equivalent transitions where identity-based counting overestimates diversity.

## BiPACE comparison

**Question:** Does whole-trajectory semantic occupancy capture complementary information beyond local actor-state similarity?

**Test:** Compare actor hidden-state similarity with trajectory-kernel predictions of strategy equivalence and training utility.

## TDScaling comparison

**Question:** Does instance-level behavioral redundancy add explanatory value beyond coarse domain/mode diversity?

**Test:** Include TDScaling-style attributes in regression or subset-selection baselines.

## State-coverage comparison

**Question:** Does environment-grounded semantic coverage explain interactive-agent experience value beyond prefix/state-count coverage?

**Test:** compare exact/reference state coverage with semantic behavioral coverage under matched rollout budgets.

---

# 20. ICLR-Level Target

A strong paper can combine four layers.

## Contribution 1 — Diagnosis

Show that nominal rollout count substantially overestimates effective behavioral sample size.

## Contribution 2 — Metric

Introduce and validate a semantic behavioral trajectory kernel.

## Contribution 3 — Scientific finding

Show that semantic behavioral diversity predicts:

- multi-rollout gain;
- saturation;
- marginal training value.

## Contribution 4 — Causal intervention

Under matched budget, select subsets with different semantic behavioral coverage and observe different post-training outcomes.

## Optional Contribution 5 — Active acquisition

Show the same downstream gain with fewer environment interactions.

The strongest paper-level claim is:

> **Semantic behavioral experience is the effective unit of diversity for agent learning.**

---

# 21. High-Impact Story Template

### Finding 1
LLM-agent rollouts exhibit behavioral mode collapse.

Nominal:

\[
K=32.
\]

Effective:

\[
\operatorname{BESS}\approx 5\text{–}8.
\]

### Finding 2
Text and exact-action diversity systematically overestimate useful behavioral coverage.

### Finding 3
Semantic behavioral coverage predicts pass@k saturation.

### Finding 4
Semantic behavioral coverage predicts post-training gain under fixed budget.

### Method
Behavioral Occupancy Kernel / Semantic Transition Occupancy.

### Application
Experience selection or active acquisition.

---

# 22. Experience Valuation Extension

If downstream evidence indicates that novelty alone explains only part of training value, extend the research object to:

\[
\boxed{V(\tau\mid\theta,\mathcal D)}.
\]

A candidate value model is:

\[
V
=
N_{\text{behavior}}
\times
L_{\text{student}}
\times
R_{\text{relevance}}.
\]

This connects semantic novelty to student suitability and curriculum/data valuation.

Potential outcome patterns:

| Observation | Research interpretation | Next focus |
|---|---|---|
| Structured metric strongly predicts equivalence and pass@k | behavioral coverage is a useful test-time quantity | deepen G3 and G4 |
| Structured metric predicts training gain more than pass@k | metric is especially valuable for data valuation | prioritize G4/G5 |
| Novelty loses signal after reward/task controls | novelty is one factor in value | model novelty × learnability |
| Novelty × learnability strongly predicts gain | experience valuation becomes central | make G5 the main thesis |
| Selection works strongly | offline data efficiency established | proceed to active acquisition |
| Active acquisition reduces rollout cost | full closed-loop story established | emphasize interaction efficiency |

---

# 23. Ongoing Literature Monitoring

Run targeted searches periodically.

## Trajectory similarity

- `"LLM agent" trajectory similarity semantic`
- `"agent trajectory" behavioral equivalence`
- `"trajectory metric" tool agent`
- `"semantic transition" LLM agent`
- `"bisimulation" LLM agent trajectory`
- `"occupancy measure" language agent diversity`

## Data value

- `"agent trajectory" data selection post-training`
- `"state coverage" LLM post-training`
- `"trajectory utility" LLM agent`
- `"experience value" language agent`
- `"rollout selection" agent fine-tuning`

## Exploration

- `"diverse exploration" LLM agent state transition`
- `"quality diversity" LLM agent`
- `"learned exploration" language agent`
- `"behavioral diversity" agent RL LLM`

For each new paper, update the novelty matrix with the concrete baseline, experiment, or positioning change it implies.

---

# 24. Deliverables by Gate

## G1

- controlled equivalence-pair dataset;
- baseline implementations;
- equivalence separation table;
- systematic failure cases.

## G2

- real trajectory-pair dataset;
- annotation guideline;
- inter-annotator agreement;
- metric-human agreement table.

## G3

- repeated-rollout benchmark;
- BESS implementation;
- pass@k regressions;
- temperature/sampling curves.

## G4

- matched training subsets;
- post-training learning curves;
- fixed-budget analysis;
- seed robustness.

## G5

- trajectory-value proxy;
- novelty/learnability/value curve;
- student-specific analysis.

## G6

- active acquisition method;
- rollout-efficiency curves;
- comparison with oversample-then-select.

---

# 25. Research-Agent Execution Prompt

The following block can be given directly to a research agent.

> You are executing a staged research program on **semantic behavioral experience for interactive LLM agents**.
>
> Your objective is to test the following claim as efficiently and rigorously as possible:
>
> **Under a fixed rollout/token/interaction budget, semantically non-redundant agent trajectories provide more multi-rollout success and/or more marginal post-training value than surface-diverse but behaviorally redundant trajectories.**
>
> Work on the currently active gate and optimize each step for information gain.
>
> For every experiment, record:
> - hypothesis;
> - baselines;
> - controls;
> - success criterion;
> - interpretation of each plausible outcome.
>
> For every update, report:
> - **Evidence** — what was observed;
> - **Interpretation** — what it implies;
> - **Decision** — which branch has highest expected research value;
> - **Next experiment** — the single highest-information next test.
>
> Maintain:
> - `hypotheses.md`;
> - `decision_log.md`;
> - `novelty_matrix.md`;
> - `experiment_registry.md`;
> - `parking_lot.md`.
>
> Start with simple baselines and introduce additional modeling complexity when an observed failure mode motivates it.
>
> **Current target:** determine whether a semantic transition representation provides a better measure of behavioral equivalence between LLM-agent trajectories than raw text embeddings, exact action/transition overlap, action edit distance, semantic edit distance, actor hidden-state similarity, and LLM judging.
>
> Once this metric question is resolved, advance to set-level diversity, pass@k, and post-training utility according to the gate plan.

---

# 26. Immediate Action List

1. Choose one cheap structured interactive environment, preferably ALFWorld or a structured tool/API environment.
2. Collect a small rollout pool: roughly 20–50 tasks and 8–16 rollouts per task.
3. Define a canonical semantic-transition schema such as:

```json
{
  "subgoal": "...",
  "action_type": "...",
  "arguments_semantic": "...",
  "information_gain": "...",
  "state_change": "...",
  "strategy_role": "..."
}
```

4. Construct 100–300 controlled trajectory pairs, roughly half behavior-preserving and half behavior-changing.
5. Implement minimum baselines:
   - text cosine;
   - action edit;
   - exact transition Jaccard;
   - LLM judge;
   - semantic transition embedding + MMD/Wasserstein.
6. Run G1 evaluation with AUROC, ranking accuracy, perturbation-wise breakdown, and systematic failure examples.
7. Make an explicit **advance / refine / redirect** decision based on the G1 success criterion.

---

# 27. Final Strategic View

The scientific object is:

\[
\boxed{\text{non-redundant behavioral experience}}
\]

The core unknown is:

\[
\boxed{\text{When does an additional agent trajectory teach us something genuinely new?}}
\]

The research program builds increasingly strong answers to that question:

\[
\boxed{
\text{Measure}
\rightarrow
\text{Validate}
\rightarrow
\text{Explain}
\rightarrow
\text{Select}
\rightarrow
\text{Acquire}
}
\]

The one-sentence thesis is:

> **Instead of counting agent rollouts, measure the amount of new, behaviorally distinct, learnable experience they contain.**

---

# 28. Reference Index

Core papers from the novelty audit:

1. AutoQD — ICLR 2026 — https://proceedings.iclr.cc/paper_files/paper/2026/hash/a137760f1bc695ce695c328919b553de-Abstract-Conference.html
2. DMPO — EMNLP 2024 — https://aclanthology.org/2024.emnlp-main.138/
3. WebStep — COLM 2026 — https://arxiv.org/abs/2606.15673
4. DPEPO — ACL 2026 — https://aclanthology.org/2026.acl-long.2151/
5. BiPACE — 2026 — https://arxiv.org/abs/2606.25556
6. One Training Example — 2026 — https://arxiv.org/abs/2609.04172
7. TDScaling — ACL Findings 2026 — https://aclanthology.org/2026.findings-acl.768/
8. Representation-Based Exploration — ICLR 2026 — https://proceedings.iclr.cc/paper_files/paper/2026/hash/c420aca62da1d668e8e79e0bcae24e6b-Abstract-Conference.html
9. RPD — https://arxiv.org/abs/2510.26122
10. SetPO — https://arxiv.org/abs/2602.01062
11. DQO — ICLR 2026 — https://proceedings.iclr.cc/paper_files/paper/2026/hash/cfeb1fa78c5fb06be27c0980aa8ee581-Abstract-Conference.html
12. UpSkill — https://arxiv.org/abs/2602.22296
13. Global Forking Tokens — ICLR 2026 — https://proceedings.iclr.cc/paper_files/paper/2026/hash/88af3540325dd0b70617a9ab605f294d-Abstract-Conference.html
14. R-Diverse — https://arxiv.org/abs/2602.13103
15. TrajeDi — ICML 2021 — https://proceedings.mlr.press/v139/lupu21a.html
16. Maximum State Entropy with Parallel Agents — ICML 2025 — https://proceedings.mlr.press/v267/de-paola25a.html
17. QDHF — ICML 2024 — https://proceedings.mlr.press/v235/ding24h.html
18. Policy-Based Trajectory Clustering — UAI 2026 — https://proceedings.mlr.press/v337/wang26a.html
19. Temporal Dependency-Aware Trajectory-Level Behavioural Metric — 2026 — https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70109
20. NovelSum — ACL 2025 — https://aclanthology.org/2025.acl-long.908/
21. Rank–Surprisal Ratio — ACL 2026 — https://aclanthology.org/2026.acl-long.1950/
22. BOOST — https://arxiv.org/abs/2605.24743
23. Coverage Principle — ICLR 2026 — https://arxiv.org/abs/2510.15020
24. Post-Training is About States, Not Tokens — https://arxiv.org/abs/2605.22731
