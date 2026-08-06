# Meta-Learning the Axes of Rollout Diversity (GEPA-style, RL-time)

Brainstorm + lit-grounded positioning + Tinker experimental design. Follows the 2026-06 meeting
(whiteboard at `tmp/image.png`): a GEPA-ish loop that diversity-samples rollouts along a *best axis A*,
turns reward+reflection into *experience*, and updates both the policy (P→P′) **and the axis set**.

---

## 1. The idea in one paragraph

During RL, don't just sample G i.i.d. rollouts per prompt — condition each rollout on a different
**axis-value** (a dimension of variation: professional role, solution method, level of abstraction, …),
so the group spans the solution space instead of collapsing into one mode. Score rollouts with the
task verifier, then run an **outer reflective loop** (GEPA-style) that *evolves the axis set itself*:
an LLM **axis proposer** reads the traces (axis → rollout → outcome), proposes/mutates axes that target
the policy's current failure modes, and a **quality-diversity selection** keeps axes that contribute
*coverage* (solve problems other axes don't), prunes redundant/decorative ones. The policy weights are
updated by GRPO/DARLING on the conditioned rollouts; the axis archive is updated by reflection.

**The novel object is the *axis space itself* — learned, evolved, and used to drive rollouts.** Diversity-
aware RL for pass@k (DARLING, Uniqueness-Aware RL) and prompt evolution during RL (E-SPL) are both taken
(§3); classic quality-diversity *fixes* its behavior descriptors by hand. The unclaimed seam (§5) is
**learning *which* dimensions of variation to diversify along and adapting them online**, then proactively
*conditioning* rollouts on them rather than reweighting diversity post-hoc. The meeting's headline question
is one test of this: **if hand-crafted heuristic axes help, can an LLM axis-proposer discover and evolve
axes that match or beat them — and adapt them to the policy as it learns?**

---

## 2. The two seed papers, and where they stop

- **DivSampling** (Wang et al. 2025, arXiv 2502.11027): prompt perturbation (Role / Instruction /
  Jabberwocky + Random-Idea-Injection / Query-Rephrase) breaks the "local cluster" trap and lifts
  Pass@k. Theory: error rate drops ~linearly in #diverse prompts. **But: inference-time only, axes are
  fixed/hand-designed, no learning, no weight updates.**
- **DARLING** (Li, Zhang et al., Meta FAIR 2025, arXiv 2509.02534): diversity-aware RL — a *learned
  semantic partition classifier* gives a diversity signal, multiplied into the quality reward
  (`r_darling = r × Norm(div)`); jointly lifts pass@1 **and** pass@k; "explicitly optimizing diversity
  catalyzes exploration." **But: diversity is measured *post-hoc* on free rollouts; there is no
  structured axis driving the diversity, and nothing is evolved.** Also shows the failure mode we must
  guard against: n-gram diversity rewards get *hacked* (switch language, pad with self-reflection).

The gap both leave open: **a structured, *evolving* set of diversity axes that drive RL-time rollouts
and adapt to the policy.** That is the project.

---

## 3. Prior art that crowds the space (must position against)

**Closest neighbor — read first:**
- **E-SPL — Evolutionary System Prompt Learning** (Zhang, Chen, Stadie, 2026-02-16, arXiv 2602.14697).
  Joint **RL on weights + evolutionary search on system prompts**, in the same loop: each RL iteration
  samples trajectories under a *population* of system prompts, applies GRPO to weights and
  mutation/crossover (LLM self-reflection) + rating-based selection to prompts. Declarative knowledge →
  prompts, procedural → weights. AIME→BeyondAIME 38.8→45.1, beats reflective prompt evolution alone.
  **This is the whiteboard, minus the word "diversity."** It validates the *mechanism* (co-evolution
  works, is synergistic) — which de-risks us — but it also means **"evolve prompts during RL" is taken.**
  Our contribution cannot be "co-evolve prompts + weights." It must be the *diversity/axis* angle (§5).
  - *Selection:* TrueSkill rating (`μ+λσ`, UCB optimism over *which prompt is best*) — **not** behavioral/
    coverage diversity. They concede crossover homogenizes the population and leave **niching** (fitness
    sharing, island model) to future work (§4.4) — the diversity gap is admitted but never measured.
  - *Mutation:* greedy refinement of the **single** best prompt from one generic root (Alg. 1; root is a
    bare task instruction — sophisticated workflows "are absent from the initial system prompt", §5). No
    diversity is seeded or enforced; at iteration 1 all `M` prompts are the identical root.
  - *Deployment:* the evolved prompt is **shipped and kept at inference** (declarative/procedural split is
    the thesis; Fig. 10 shows the model citing the prompt verbatim mid-solve), with the test prompt picked
    by TrueSkill rating (Fig. 8). **No "weights-without-prompt" strip ablation is reported** → they never
    test internalization. → our exploration-tool (strip) path is the unmeasured alternative.
  - *Phase-2 collision (see §9):* §5 "Self-Write as an RL Problem" **explicitly** proposes folding the
    prompt-writer into RL as future work — i.e., our self-generation idea. And Fig. 15 shows a **fixed**
    `π_ref` editor beats a **drifting** (policy) editor — a direct stability caution for self-generation.

**Mechanism precedents (de-risk the RL plumbing):**
- **AdaGRPO** (Mixture-of-Visual-Thoughts, Li et al. 2025, arXiv 2509.22746): "prefix-guided mode
  exploration" — fix a mode prefix to force rollouts across 2 reasoning modes, compute **mode-relative
  advantage** so an easy mode doesn't dominate the gradient. This is *exactly* axis-conditioned rollouts
  with the right advantage estimator — but with 2 fixed modes. We generalize to many, evolved axes.
- **DQO** (Chen et al. 2025, arXiv 2509.04784): DPP-determinant (volume) semantic diversity inside RL —
  an alternative diversity signal to DARLING's classifier.

**Diversity-axis discovery (the "LLM proposes axes" question, mostly outside RL):**
- **Persona Hub** (Chan et al. 2024, arXiv 2406.20094): 1B personas as "distributed carriers of
  perspective" for synthetic-data diversity, incl. math/reasoning — the canonical persona-as-axis source.
- **Diverse Prompts: MAP-Elites** (Santos et al. 2025, arXiv 2504.14367): quality-diversity over a
  prompt *phenotype* space (shots, reasoning depth, context) — QD over axes, but no RL, no LLM proposer.
- **SimpleStrat** (Wong et al. 2024, arXiv 2410.09038): stratify the answer space into axes to diversify
  generation. T2I work (DivBench 2507.03015, Varif.ai 2506.19644) shows **LLM-proposed attributes beat
  fixed ones** for under-diversity — supporting evidence the axis-proposer can work, in another modality.

**Adaptive-curriculum / exploration framing:**
- **SEC — Self-Evolving Curriculum** (Chen et al. 2025, arXiv 2505.14970): curriculum as a
  *non-stationary multi-armed bandit* — the template for "axis selection as a bandit."
- **"No More Stale Feedback / Co-Evolving Critics"** (Li et al. 2026, arXiv 2601.06794): empirically,
  **a policy's dominant failure modes drift across training phases.** This is the motivation for
  *adaptive* axes (a fixed axis set goes stale) — our strongest "why evolve" argument.
- **RLVR exploration analysis** (Deng et al. 2025, arXiv 2508.07534): "rollout branching factor" as an
  exploration metric; selecting for it beats strong RL baselines on pass@k. Useful as an axis-fitness
  signal and an evaluation lens.

**Diversity-aware RL for pass@k — the crowded core (the *mechanism* is hot, added after survey):**
- **Rewarding the Rare / Uniqueness-Aware RL** (Hu et al. 2026, arXiv 2601.08763) — **co-closest neighbor,
  must-cite.** An LLM judge clusters rollouts by *high-level solution strategy*; advantage is reweighted
  **inversely with cluster size** so rare correct strategies score higher. Lifts pass@k AUC, preserves
  pass@1. This *is* the "coverage/QD-style advantage" of §4(d) — already done. It does **not** condition/
  drive rollouts on an explicit axis (clusters post-hoc), evolve an axis set, use a proposer, or touch
  personas. → narrows us; doesn't kill us.
- **Representation-Based Exploration** (Tuyls et al. 2025, arXiv 2510.11686): hidden-state diversity bonus
  → pass@k up at inference + post-training, 3× sample efficiency. Strong "deliberate exploration → pass@k".
- **G²RL** (Liang et al. 2025, arXiv 2512.15687) + **outcome-based exploration** (Song et al. 2025, rare-
  outcome bonuses) + **EVOL-RL**, **RESTRAIN** — the broader exploration-vs-collapse cluster.
- **Meta-action DPS** (arXiv 2601.07298): penalize homogeneous solution-strategy trajectories; pass@k.
  **VibeThinker** (arXiv 2511.06221): diversity-first SFT → signal RL.

**Quality-Diversity + RL — a *mature* subfield (don't reinvent it):**
- QD-RL is established: **PGA-MAP-Elites**, **DQS** (arXiv 2304.07425), CMA-ME, **Extrinsic Behavioral
  Curiosity** (arXiv 2410.06151).
- **MAP-Elites w/ Descriptor-Conditioned Gradients + Archive Distillation into a Single Policy** (Faldor
  et al. 2023, arXiv 2303.03832) — **must-cite contrast.** A QD archive distilled into one *descriptor-
  conditioned* policy = the template for "internalize a diverse archive into a single conditionable model."
- **The genuine gap:** classic QD **fixes the behavior descriptors** (expert-set). Nobody evolves the
  *descriptor/axis space itself with an LLM*. ← our strongest surviving novelty (§5).

**Prompt internalization / scaffold-then-strip — also mature (so "strip" is not the novelty):**
- **Learning by Distilling Context** (Snell, Klein, Zhong 2022, arXiv 2209.15189) — foundational scaffold-
  then-strip: condition on instructions+scratchpad → fine-tune to predict the answer *without* them.
- **PromptIntern** (arXiv 2407.02211), **Generative Prompt Internalization** (arXiv 2411.15927), Context
  Distillation / PI-PING. → Phase-2's novelty is *what* is internalized, not the act of internalizing.

**Train-then-strip elicitation — a direct Phase-2 caution:**
- **Inoculation Prompting** (Tan et al. 2025, arXiv 2510.04340; Wichers et al. 2025, arXiv 2510.05024): a
  train-time system prompt that *elicits* a trait makes that trait **suppressed** when stripped at test —
  mechanism: inoculated data is "less surprising → less global update" (**the same story E-SPL tells**).
  ⇒ training with a persona scaffold and stripping it may *suppress* persona-behavior, not internalize the
  benefit. A real failure mode for the strip path (§9, risk 6).
- **Spilling the Beans** (arXiv 2511.06626): a *persona* vs *behavior* system prompt during fine-tuning
  generalizes **differently** — directly informs the persona-vs-strategy fork (§9).

**Persona-for-accuracy — contested, actively studied:**
- **Principled Personas** (Luz de Araujo et al. 2025, arXiv 2508.19764): 9 LLMs × 27 tasks — expert-persona
  effects are *"positive or non-significant,"* with ~30pp drops from irrelevant persona details. ⇒ don't
  claim personas help; let pass@k coverage sidestep the question.

**Honest takeaway (post-survey):** the broad angles — diversity-aware RL for pass@k, QD-RL, prompt
internalization — are all **populated as of early 2026**, and Uniqueness-Aware RL now occupies the
coverage-advantage mechanism. So novelty is *narrower* than "diversity-aware RL," and is **not** "evolve a
prompt during RL" (E-SPL) nor "internalize a prompt" (Context Distillation). What survives is in §5.

---

## 4. The method, as a set of design choices (with recommendations)

Loop, matching the whiteboard:

```
Axis archive A_t  ──sample config c_i per rollout──►  G conditioned rollouts per prompt x
        ▲                                                     │
        │                                              verifier reward r_i
   meta-update (GEPA):                                        │
   reflect on (c→rollout→outcome),               RL update on weights (GRPO/DARLING)
   mutate/crossover axes targeting failures,              θ_t → θ_{t+1}  (P→P′)
   QD-select by coverage  ◄───── experience buffer ───────────┘
```

**(a) Axis representation — structured vs unstructured** *(the meeting's 2nd question; repo already has both)*
- Structured = JSON schema `{role, method, abstraction, ...}` with enumerable values → combinatorial
  coverage, **per-axis credit assignment**, clean permutation, interpretable. `deeppersona/personas.py`
  already renders this (`structured` level).
- Unstructured = free-form NL persona/instruction (DivSampling / E-SPL style) → expressive but credit is
  opaque and dedup is fuzzy. `personas.py` `basic` level ≈ this.
- **Recommendation: structured is the differentiator** (it's what lets you answer "which axes matter");
  run unstructured as the ablation. Note the live tension from `q1_progress.md`: *structured personas
  currently underperform basic ones as a static scaffold*. Reframing them as a **diversity** scaffold
  (you only need *one* axis-value to hit the answer) may flip that — a built-in, repo-grounded sub-question.

**(b) Advantage estimation** — start pooled GRPO (axis only shapes the sampling distribution) for the
smoke test; then **axis-relative normalization (AdaGRPO)** so a single easy axis can't dominate the
gradient; keep DARLING multiplicative as a diversity-reward baseline. This is load-bearing — get it wrong
and the policy just imitates the best-axis distribution and collapses.

**(c) Axis fitness / meta-reward (the heart — the whiteboard's "reflection→experience").**
Fitness of an axis-value should reward **useful diversity**, not diversity per se:
- **marginal coverage**: fraction of prompts where this axis uniquely produced a correct rollout (QD signal);
- gated by a **min reward-lift** vs no-axis (DivSampling-style), and
- a **min semantic diversity-yield** (DARLING classifier / DQO embedding-volume) — *this gate is the
  anti-reward-hacking guard*: an axis whose rollouts are paraphrase-identical or decoratively different
  gets pruned.
- Maintain a **Pareto front over (mean reward, coverage, diversity-yield)** — mirrors GEPA's Pareto design.

**(d) Selection — coverage, not rating; and *evolving* the axis space.** E-SPL selects by *relative
performance rating* (elitist → best single prompt); we select by **coverage contribution** (keep a lower-
mean axis if it illuminates cells nothing else does) — MAP-Elites over the axis archive. **But the coverage
advantage alone is not the novelty:** Uniqueness-Aware RL (§3) already does coverage-style advantage via
post-hoc clustering, and classic QD already does archive selection. The real separation is (i) we **drive**
rollouts with an explicit axis (proactive, not post-hoc clustering) and (ii) the axis/descriptor space is
**evolved by an LLM proposer** (classic QD fixes it; Uniqueness-Aware RL never names reusable axes).

**(e) Deployment: scaffold-then-strip vs internalize.** Two distinct papers:
- *Exploration tool*: axes only diversify rollouts; train on `problem→solution` (strip the axis). Final
  model is axis-free; test whether the coverage gain *transfers* at inference with no conditioning.
- *Internalize* (fuses with the existing Q2 proposal): train on `problem→axis→solution`, then at inference
  the model **self-elicits** the coverage-maximizing axis. Distinct from E-SPL (keeps prompt external)
  and from the current proposal (fixed personas, not evolved/coverage-selected). **Specced in §9** as
  self-generation (the model writes its own persona) — the self-contained, weights-only deliverable.

---

## 5. The contribution that survives prior art

The broad angle is crowded (§3): diversity-aware RL, QD-RL, and prompt internalization are all populated,
and **Uniqueness-Aware RL** (arXiv 2601.08763) already owns the coverage-advantage mechanism. So the
honest, defensible core is narrower than "diversity-aware RL":

1. **Evolving the *axes* (behavior descriptors) along which to diversify — with an LLM proposer.**
   **(strongest)** Classic QD *fixes* descriptors (expert-set); Uniqueness-Aware RL clusters *post-hoc*
   with no named, reusable axes; E-SPL evolves one cumulative prompt, not a diverse axis set. Nobody learns
   *which dimensions of variation* to diversify along and adapts them online. This is the unclaimed seam.
2. **Proactively *conditioning* RL rollouts on the axis set** (driving exploration), vs measuring/reweighting
   diversity *post-hoc* (DARLING, Uniqueness-Aware RL). AdaGRPO does this for 2 fixed modes; we generalize.
3. **Failure-mode-adaptive axes** — axes track the policy's drifting blind spots (Co-Evolving-Critics drift;
   SEC bandit-curriculum is adjacent, not the same). "UED for solution strategies."
4. **(Phase 2, §9) Self-contained deployment** — a weights-only model that self-generates its conditioning,
   vs E-SPL's ship-and-keep-the-prompt. *Caveat (new):* internalization is a mature mechanism (Context
   Distillation lineage), and inoculation work warns stripping may *suppress* — so the Phase-2 claim is the
   *combination* (self-generated, coverage-selected, diversity-preserving), **not** the strip itself.

Repo-native bonus: scoring on **pass@k** sidesteps the persona style-vs-competence confound of Q1 entirely
(you only need *one* axis to hit the answer — and *Principled Personas* shows persona-prompting effects are
often non-significant, so coverage is the right way to dodge that debate). The internalize variant is open Q2.

---

## 6. Experimental design (Tinker)

**Frame.** Inner loop = GRPO LoRA RL via Tinker (`refs/tinker-cookbook` recipes `rl/rollouts.py`,
`rl_loop.py`). Custom env builds axis-conditioned prompts and calls the existing verifier
(`deeppersona/verifiers.py`: GSM8K numeric + CSQA letter). Outer loop = cheap reflection/axis-proposer LM
(Claude/GPT, called once per meta-iteration — not per rollout, à la GEPA's cheap-reflection design).

**Model.** Qwen3-4B-Base for the RL (DARLING's verifiable setup; clean base, fast). Note: repo eval
harness uses Qwen2.5-7B-Instruct — keep that only for offline re-scoring if needed.

**Tasks.** GSM8K for smoke + plumbing (fast verifier); MATH / DeepScaleR-subset for the *real* story
(exploration matters most on hard items — DARLING's biggest gains were on the hardest sets). CSQA optional.

**The headline ladder** (fix RL algorithm + rollout budget across all rows):

| # | Axis source | Question it answers |
|---|---|---|
| 0 | none (vanilla GRPO) | exploration floor |
| 1 | temperature-matched | "is it just more entropy?" (entropy control) |
| 2 | **human heuristic axes, fixed** (DivSampling roles / `personas.py` 12) | the human-axis baseline |
| 3 | LLM-proposed axes, **frozen** | can an LLM match humans *without* evolution? |
| 4 | LLM-proposed axes, **evolved** (GEPA, static target) | does evolution help? |
| 5 | evolved + **failure-mode-adaptive** | the full method |

Critical question ⇒ does 3/4/5 ≥ 2? Marginal value of evolution ⇒ 5 > 4 > 3?

**Metrics.** pass@1 (quality); **pass@k to k=128** (DARLING protocol, the exploration headline);
**reward vs #rollouts** (GEPA's sample-efficiency axis); semantic-distinct + distinct-4 (lexical
anti-hack); **axis analytics** (active-axis count, turnover, coverage curves, do evolved axes converge to
human-interpretable dimensions?); easy→hard transfer (E-SPL's AIME→BeyondAIME style); and, for the strip
variant, **does the axis-free policy keep the coverage gain at inference?**

**Ablations.** structured vs unstructured axis rep (b); pooled vs axis-relative advantage; coverage-QD vs
elitist selection (beating an E-SPL-style selector?); **vs Uniqueness-Aware-RL** (post-hoc strategy
clustering + inverse-cluster-size advantage, arXiv 2601.08763) — the key test of whether *conditioning +
evolving axes* beats *post-hoc reweighting*; reflection-LM strength.

**Falsifiers (run the first one as the smoke test — cheapest kill).**
- **F0:** fixed human-axis conditioning gives *no* pass@k lift over vanilla GRPO at training time on
  GSM8K/MATH (Qwen3-4B). If null → the premise is dead; publish "RL-time persona diversification doesn't
  help verifiable reasoning." **Do this before building any meta-loop.**
- **F1:** LLM axes never beat temperature-matched → "meaningful axes" is just entropy.
- **F2:** evolved ≈ frozen → meta-loop isn't worth its cost.
- **F3:** gains vanish after stripping the scaffold → it's a prompting trick, not internalized (still a
  valid inference-time method, weaker claim).

**Smoke test (repo rule).** 1 axis source, 8 GSM8K prompts, G=4, 1 meta-iteration, on
`CUDA_VISIBLE_DEVICES=3` locally — confirm loop runs + rewards log before any sweep.

---

## 7. Open decisions for the next step

1. **Headline framing:** exploration-tool (strip) vs internalize (self-elicited persona). Different papers;
   determines what we build in Tinker first. *(Recommend: build the exploration-tool path first — it's the
   cleaner pass@k story and the F0 falsifier — then add internalization if F0/F1 pass.)*
2. **Primary task/model:** GSM8K+Qwen3-4B (fast, de-risk) vs MATH (where exploration actually bites).
   *(Recommend: F0 on GSM8K, then move the real experiments to MATH.)*
3. **Relationship to the steering-vector proposal** (`proposal_one_pager.md`, `q1_experimental_design.md`):
   this pivots from white-box residual-stream steering toward black-box axis-conditioning + LoRA RL. Keep
   both threads, or fold steering in (e.g., axis = steering direction instead of a prompt)?

**Decided (2026-06):** headline = exploration-tool (strip) first, then internalize (self-generation, §9);
first build = **F0 only**. Phase 2 is gated on F0/F1 passing.

---

## 8. F0 spec — does fixed human-axis conditioning lift pass@k? (the cheapest kill)

Split into a no-training check that can falsify in hours, then a training check only if it survives.

### F0a — inference-time, no training (~hours, reuses existing harness)
**Claim under test:** on *our* tasks/model, conditioning rollouts on a fixed pool of human axes raises
**pass@k at matched per-problem sample budget** vs i.i.d. sampling (DivSampling's claim, re-tested here).
If even this is null, the RL version is hopeless → publishable negative, stop.

- **Model:** Qwen2.5-7B-Instruct (already wired: `configs/qwen25_7b*.yaml`). Use *instruct*, not base —
  axis/persona conditioning needs instruction-following (base models ignore it; DivSampling used instruct).
- **Axes (zero new authoring):** the 12 `MATH_SPECS` / `CSQA_SPECS` personas in `deeppersona/personas.py`
  as axis-value pool #1; DivSampling Role pool as pool #2 (robustness to axis source).
- **Conditions, all at matched total budget n per problem (e.g. n=64):**
  - `vanilla@T`: n i.i.d. samples, temp T (e.g. 0.8).
  - `vanilla@T_hi`: n i.i.d. at higher temp tuned to match axis-condition distinct-n — **the entropy control**
    (rules out "axes = just more temperature"; this is ladder row 1).
  - `axis@T`: n samples split across the 12 axes (~5 each), same temp T.
- **Tasks:** 200-item subsample of GSM8K test + 200 of MATH (cheap; not training so no contamination risk).
- **Metrics:** unbiased pass@k (DARLING/Chen eq.) for k=1..32; distinct-4 + semantic-distinct to confirm
  axes actually raise diversity; per-axis coverage (which axis uniquely solves what — previews the QD signal).
- **Decision rule:** `axis@T` pass@k beats **both** `vanilla@T` and `vanilla@T_hi` by a bootstrap-CI-separated
  margin at some k≥4 (reuse `deeppersona/scaffolding_stats.py` paired bootstrap). Survives → F0b. Null → stop.
- **Smoke:** 8 GSM8K items, n=8, 2 axes, on `CUDA_VISIBLE_DEVICES=3`, confirm pass@k + diversity log.

### F0b — training-time (Tinker GRPO, ~1–2 days, only if F0a positive)
**Claim:** the benefit persists/amplifies when you *train* with axis-conditioned rollouts (and the gain
**transfers after stripping the axis** — the exploration-tool deployment).
- Inner loop: GRPO LoRA via `refs/tinker-cookbook` (`rl/rollouts.py`, `rl_loop.py`); env builds
  axis-conditioned prompts, reward = `deeppersona/verifiers.py`. Start pooled-GRPO advantage; add
  AdaGRPO axis-relative as the principled variant.
- Three trained policies, matched budget: (i) vanilla-GRPO; (ii) **axis-conditioned-GRPO** (ours, fixed 12
  axes); (iii) **Uniqueness-Aware-RL** (arXiv 2601.08763: post-hoc cluster rollouts by strategy, advantage
  ∝ 1/cluster-size, *no* conditioning) — the strongest same-goal baseline. Eval **axis-free** at inference
  (strip), compare pass@1 + pass@k on held-out test.
- **Decision (a) — internalization:** axis-conditioned ≥ vanilla on pass@k *with no conditioning at test*
  → exploration gain internalized into weights → greenlight the meta-loop (rows 3–5). If the gain needs the
  axis at test → inference trick (F3), weaker but still a paper.
- **Decision (b) — beats post-hoc (the load-bearing contrast):** axis-conditioned ≥ Uniqueness-Aware-RL →
  *driving* rollouts with explicit axes beats *post-hoc* reweighting (the core claim of §5.1–5.2). If **not**,
  the contribution collapses toward a known method → pivot to the **evolving-axis-space** (§5.1) as the sole
  differentiator and decide whether the meta-loop is still worth building before investing in it.

---

## 9. Phase 2 — internalize via self-generated persona (the self-contained model)

**Idea.** Once the exploration-tool path (F0/F1) shows conditioning helps, *internalize the conditioner
into the model*: post-train so the model **emits its own persona/strategy prefix, then role-plays it to
solve** — `problem → self-generated persona → solution`, with the persona span in the gradient path. Ship
**weights only**: no external prompt to maintain, no TrueSkill-style selection at inference. This is the
open Q2 of `proposal_one_pager.md`, now using E-SPL's setup as the mechanism but replacing its external
rated-prompt population with a learned self-elicitation.

**Why it's worth doing**
- **Deployment differentiator vs E-SPL.** Self-contained model. E-SPL must ship+keep the evolved prompt and
  pick one by rating at test (§3, no strip ablation). Ours throws the scaffold away.
- **Unifies the threads.** = persona work × E-SPL setup; the self-generated persona plays E-SPL's
  declarative-prompt role, but is *generated*, not *selected*.
- **Subagent realization (narrative + a free baseline).** Maps to orchestrator-writes-subagent-spec → spawn
  → solve. Gives an **untrained baseline**: prompt the model to write its own persona, spawn a fresh
  context with it, solve; the trained version compresses this into one forward pass. Treat the subagent
  framing as narrative + baseline, *not* load-bearing science.

**Gating: strictly after F0/F1.** If a fixed *external* persona doesn't lift pass@k (F0), teaching the
model to *generate* personas is pointless. Phase 2 is an amplifier on a validated premise, not a shortcut
around it.

**The five risks (design against each):**
1. **Decorative-persona collapse (top risk).** The model controls *both* persona and solution → the easiest
   policy is to emit a persona, ignore it, and solve as it always would → plain CoT with a vestigial
   preamble. This is the one-pager's "persona reasoning is decorative" falsifier, made *easier* to hit.
   **Guard/test:** ablate the self-generated persona at eval (resample the solution without it); if the
   answer distribution is unchanged → decorative, report it.
2. **Diversity loss on self-pick.** A deterministic single self-chosen persona makes all G rollouts collapse
   → reintroduces the exact mode collapse the whole pivot fights. **Guard:** keep a diversity mechanism on
   the *persona-generation* step (sample K distinct personas per problem; or a diversity reward on the
   persona span). "Generate a persona then solve" is not enough on its own.
3. **Persona-vs-strategy confound.** Must beat a generic "pick a strategy first" prefix, else the persona is
   strategy-selection in costume (= the metacognition-CoT control from the one-pager).
4. **Drifting-editor instability.** Policy-as-its-own-prompt-writer is the *drifting* editor E-SPL's Fig. 15
   found **worse** than a fixed `π_ref`. Caveat: per-query persona ≠ E-SPL's cross-iteration cheatsheet
   accumulation (an easier task), so this is yellow, not red — but watch training stability.
5. **Prior-art proximity.** = E-SPL §5 future work + self-meta-prompting / Satori-style self-reasoning +
   mature prompt-internalization (Context Distillation, PromptIntern, GenPI — §3). Novelty must rest on
   **persona + diversity + coverage + evolution**, not "the model writes its own prompt" or "we internalize it."
6. **Inoculation / suppression (new — from §3).** Inoculation Prompting (arXiv 2510.04340 / 2510.05024)
   shows a train-time *eliciting* prompt, stripped at test, **suppresses** the trait ("less surprising →
   less global update"). So training with a persona scaffold and stripping it could **erase** the persona
   effect rather than internalize the benefit — the opposite of what Phase 2 wants. **Test directly:**
   compare keep-persona vs strip-persona eval; if strip underperforms keep by a lot, internalization failed
   (you're in the inoculation regime). Spilling-the-Beans (arXiv 2511.06626) shows *persona* vs *behavior*
   framing flips which way this goes — another reason to run the persona-vs-strategy fork (risk 3) explicitly.

**The fork — what does the model generate?**
- **Persona** (identity → role-play): more novel, DeepPersona-aligned, but more likely decorative (indirect
  elicitation).
- **Strategy/plan** (E-SPL-style cheatsheet): more likely to actually help on verifiable math, but closer
  to E-SPL's future work (less differentiated).
- *Recommend:* persona, **forced to earn its keep** against the strategy control (risk 3); be willing to
  publish "persona = decoration, strategy is what helps" as a finding.

**Build recipe (when F0/F1 pass):**
1. **SFT warm-start** on filtered `problem → persona → correct solution` rollouts (cold-start the behavior).
2. **RL** keeping the persona span in the gradient path (GRPO/DARLING), with a diversity mechanism on
   persona generation (risk 2).
3. **Controls:** generic-strategy prefix (risk 3); persona-ablated eval (risk 1); untrained
   subagent-spawn baseline.
4. **Headline:** self-contained (weights-only, no prompt at test) pass@1 / pass@k vs E-SPL-style
   keep-the-prompt deployment, plus the decorative-persona check.
