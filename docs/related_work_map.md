# Related-Work Map for the Diversity/Orchestration Line (F0a–F3, CGE)

Compiled 2026-07-07 from S2 surveys (ICML 2026 sweep + broader recent venues). Organized by which
of our claims each cluster touches. arXiv IDs given; ICML-2026-specific items live in the earlier
survey (Entropy-informed Decoding 2605.09745; MARL collaboration 2601.21972; DPEPO 2604.24320;
EMS 2604.02863; diversity-collapse-in-RLVR cluster 2606.15455 / DyJR / DSDR).

## 1. Persona/role-prompt effect studies — external support for our nulls

- **Zheng et al. 2024, "When 'A Helpful Assistant' Is Not Really Helpful"** — large-scale study;
  persona effects highly variable, no guaranteed gains. The standard "near-zero average benefit"
  citation; our F0a/F2-static replicate this at frontier scale with matched budgets.
- **Kim et al., "Persona is a Double-edged Sword" (2408.08631)** — role-play prompts *degrade*
  reasoning on 7/12 datasets (llama3); proposes persona+neutral ensemble with LLM selector. Our
  F2-adaptive quantifies the degradation mechanism they observe.
- **PRISM, "Expert Personas Improve LLM Alignment but Damage Accuracy" (2603.18507, 2026)** —
  systematic when/why personas help or hurt across instruction-tuned vs reasoning models. Cites
  "Chen et al. 2026" for personas increasing *behavioral divergence in multi-agent systems* —
  chase that reference for an E3 Link-A contrast.
- **CHOIR (2510.22475)** — harmonizes persona + neutral outputs for robustness (defensive use of
  personas, not exploratory).
- Optimistic side to contrast: **SPP multi-persona self-collaboration (2307.05300)**, persona
  survey (2404.18231), persona brainstorming (2512.04488 — diversity *is* the metric there,
  consistent with our clause: open-ended generation is where persona diversity trivially pays).

## 2. Multi-agent failure analysis — our diagnosis's qualitative cousins

- **MAST, "Why Do Multi-Agent LLM Systems Fail?" (Cemri et al., 2503.13657)** — 14-mode failure
  taxonomy (specification / inter-agent misalignment / task verification) from 200+ traces.
  Position: MAST is observational taxonomy; our M1–M3 is a *controlled, quantified causal*
  account of one cell (orchestrated diversification), with a constructive fix (CGE).
- "AI Organizations are More Effective but Less Aligned" (2604.10290); "More Capable, Less
  Cooperative" (2604.07821) — scaling doesn't buy coordination.
- Failure attribution tooling: Who&When, GraphTracer (2510.10581), DoVer (2512.06749).

## 3. Adaptive self-consistency / early stopping — CGE's method family (must-cite)

- **ESC, "Early-Stopping Self-Consistency" (2401.10480)** — stop sampling when a window of
  answers is unanimous. CGE's gate *is* an ESC-style unanimity window; the delta is what happens
  after: ESC-family methods stop to **save**, CGE banks the savings and **reinvests in
  higher reasoning effort** on the disagreeing problems, plus the blind-escalation design rules.
- **Adaptive Consistency (Aggarwal et al., 2305.11860)** — Beta-posterior stopping.
- Confidence-guided variants: Self-Calibration (2503.00031), CGES Bayesian stopping (2511.02603),
  RASC (2408.17017), ReASC (2601.02970), prefix consistency (2605.07654), confidence-aware
  selective sampling (2603.08999 — single-trace confidence decides whether to resample: a
  cheaper gate than consensus; candidate CGE ablation).
- **PETS (2602.16745)** — optimal trajectory allocation for SC as a principled budget problem
  (crowdsourcing/Dawid-Skene framing); theory home for CGE-style allocation.
- **Asymmetric verification TTS for deep search (2510.06135)** — pass@k/maj@k scaling for search
  agents; bridge for porting CGE to the research-agent regime.

## 4. Experience/memory/tips — the "distill tips into the system prompt" question

- **ExpeL (2308.10144)** — distills natural-language insights from success/failure trajectories;
  the canonical tips-from-experience method. Our CGE-flip analysis predicts its content becomes
  task knowledge, not strategy diversity.
- **Agent Workflow Memory (2409.07429)** — reusable *workflows* from successful trajectories.
- **ACE, "Agentic Context Engineering" (2510.04618)** — evolving playbook-style contexts;
  identifies *brevity bias* and *context collapse* failure modes of context evolution — directly
  relevant if we ever run an E-SPL-style loop on non-math tasks.
- **PREPING, "Building Agent Memory without Tasks" (2605.13880)** — cold-start procedural memory
  built *before* target tasks exist — literally the "generate tips prior to solving" problem;
  their answer is self-generated practice tasks with executable verification.
- Ecosystem: A-Mem, Memp (procedural memory), AgentFly, SkillX (2604.04804), Memento/TER.
- **RLAD (2510.02263)** — trains models to generate reasoning abstractions that condition
  solvers; finds abstractions only help capable solvers and most configurations fail — the
  published cousin of our tips dichotomy.

## 5. Benchmarks for sub-agent spawning / delegation decisions (S2 sweep 2026-07-09)

Benchmarks whose *object of measurement* is the orchestration decision itself, not end-task skill:

- **DecisionBench (2605.19099, 2026-05)** — emergent delegation substrate: orchestrator gets a
  `call_model(name, subtask, budget)` tool over an 11-model peer pool on GAIA/τ-bench/BFCL;
  metrics include delegation rate, routing fidelity@k, vendor self-preference, and a
  *counterfactual-delegation ceiling*. Key findings: end-task quality statistically flat across
  peer-awareness conditions (quality-only eval misses the orchestration signal — their version of
  our Link A/Link B split), and perfect delegation sits 15–31pp above measured (their ceiling ≡
  our coverage-vs-selected gap).
- **MASBENCH (in MAS-Orchestra, 2601.14652, 2026-01)** — controlled benchmark characterizing
  *when MAS beats single-agent* along five axes (Depth, Horizon, Breadth, Parallel, Robustness);
  claims to be the first benchmark of MAS *benefit* rather than MAS *capability*. Their finding —
  gains depend on task structure + verification protocol, not universal — is the same regime
  thesis as our F3 3-clause criterion, discovered from the RL-orchestration side. Bonus finding:
  reasoning-model orchestrators delegate *less* (solve-it-themselves bias) than instruct models.
- **PerspectiveGap (2606.08878, 2026-06)** — 110 scenarios testing whether models can *write the
  sub-agent prompts* (what each sub-agent needs to know); combined pass rate 14.9% avg,
  GPT-5.5 62%; measures information leakage into sub-agent prompts.
- **ERAB Bench (IJRSI 2026)** — safety-side protocol for *unprompted* sub-agent spawning in
  coding setups (fringe venue, n=16 pilot; cite with caution).

Adjacent with decision-level metrics but not benchmarks per se: **Recognize Your Orchestrator**
(2606.01351) — orchestrator-level metrics (scheduling-similarity LCS-F1) separate from task
success; **Self-Resource Allocation** (2504.02051) — orchestrator-vs-planner allocation scored
against Hungarian-algorithm optimum; **RL-through-orchestration-traces survey** (2605.02801) —
decomposes spawning into when/whom/how-communicate/how-aggregate/when-stop and finds NO published
RL method for the stop decision (CGE's gate is an inference-time answer to exactly that slot);
**WideSeek-R1** (2602.04634, WideSearch bench) + **Kimi K2.5 Agent Swarm** (2602.02276,
critical-steps metric) — width-scaling of parallel subagents; **AIRA_2** (2603.26499) —
compute-optimal subagent count scales as √budget *because* independent subagents buy exploration
diversity under an evolutionary selector (oracle-filtered regime — consistent with clause (a));
**AOrchestra** (2602.03786) — automated sub-agent creation; **Terminus-4B** (2605.03195) —
subagent-utility metrics (redo-rate after subagent calls) in coding agents.

## 6. Logical-distance / reasoning-diversity metrics (S2 sweep 2026-07-20)

For the logical-distance construct (docs/logical_distance_brainstorm.md). Field moved fast in the
last 8 months: the *critique* of surface diversity metrics is published; the *metrics* are not.

Must-cites / position-against:

- **"Are We Measuring Strategy or Phrasing?" (2606.29985, 2026-06, SNU)** — ⚠ flag-level
  overlap. Introduces "approach-level diversity" (strategy variation among correct solutions,
  human-calibrated LLM-judge clustering); shows surface metrics are unreliable proxies; shows
  diversity-RLVR preserves its proxy while approach coverage *declines* (~80% of "gain" is
  within-approach); shows directly optimizing an LLM-judge diversity reward gets reward-hacked.
  Owns our motivating negatives, but poses "reliable, directly-optimizable approach-level
  metric" as an *open problem* — positions us rather than kills us (judge-hacking result is an
  argument FOR a learned/structural metric).
- **Rewarding the Rare (2601.08763, 2026-01)** — LLM-judge strategy clustering of rollouts,
  advantage ∝ 1/cluster-size in RL; improves pass@k AUC without hurting pass@1. Strongest
  baseline for the RL-reward use; explicitly notes entropy/embedding signals miss shared
  strategy.
- **Reasoning Path Divergence (2510.26122, 2025-10)** — step-aligned divergence between
  Long-CoT solutions, used to curate diverse-solution SFT sets (+2.8% pass@16); beats
  whole-solution embeddings at strategic diversity. Closest existing trace *metric* (≈ our L2's
  alignment step); no RL reward, no set-level ESS, no style-invariance validation. **Fastest
  collision risk** — one more iteration (RPD as reward + aggregation) hits our lane.
- **GCPO (2605.11461, 2026-05)** — DPP determinant volume over semantic embeddings as
  cooperative rollout reward inside GRPO. Kernel-set-objective-in-GRPO is taken, but with
  off-shelf (style-gameable) embeddings — our metric is the drop-in embedding-swap ablation.
- **Reasoning Structure of LLMs (2606.03883, 2026-06, ETH)** — traces → verifiable reasoning
  graphs (atomic claims + inference edges) on logic puzzles. L3's extraction step is taken;
  inter-trace graph distance is not.

Style-gaming is now well-recognized: 2509.04784 documents concrete hacking (model appends random
content to inflate embedding diversity); DiScO (2606.08974) defends marker-hacking; G2RL
(2512.15687) and VERL (2509.23808) retreat to gradient/hidden-state novelty signals.

Background/aggregators: Vendi (2210.02410) + conditional Vendi (2411.02817); **Kernel Language
Entropy (2405.20003)** — von Neumann entropy of a semantic kernel, the formal bridge from
semantic entropy (Farquhar/Kuhn, Nature 2024) to kernel-ESS; Alignment Score (2511.06168) —
semantic entropy over CoT steps but vs a reference chain only. Strategy clustering:
Distributional Clarity (2601.06911). Diversity→pass@k evidence (all unconditional): DAPO
Div-Equ↔Potential@k (2505.23433), Solution Multiplicity moderator (2511.19942). L4 neighbors:
"thinking embeddings" = mean hidden states over think-span (Rotate2Think 2606.09873, used for
steering not distance); Tracing the Traces (2510.10494).

**Unclaimed lanes (as of sweep):** L1.5 contrastive trace metric (paraphrase-positives /
distinct-solution-negatives); L2 claim-set OT distance; L3 graph-kernel/GW distance; L4
hidden-state diversity kernel + hidden-vs-text-embedder strategy-separation comparison;
Vendi/ESS over a *logic* kernel; and the incremental-validity regression —
diversity→pass@k *conditional on answer-multiset entropy* — no paper found. The composite open
lane: a metric that is (1) not an LLM judge (not judge-hackable), (2) validated
style-invariant, (3) cheap enough for in-loop RL reward.

## Positioning summary

Nothing found (ICML 2026 or elsewhere) that does either of our two distinctive things:
(1) controlled *input*-diversity mediation (Link A: prompts→behavioral diversity; Link B:
diversity→accuracy) measured across regimes (RL, frontier inference, pooling agents); or
(2) the signed negative that adaptive prompt-diversity orchestration harms, with a mechanism
decomposition (M1 anchoring / M2 abandoned winners / M3 directive tax) each converted into a
design rule (CGE). The crowded neighborhoods to differentiate from: output-diversity RLVR
regularizers (they intervene on sampling, we on inputs), ESC-family adaptive stopping (they save,
we reinvest), persona-effect studies (they report variance, we give mechanism + budget-matched
controls), MAST (taxonomy vs causal quantification).
