# Verifiable-Evaluator Benchmarks for Diversity/Orchestration Follow-ups

Compiled 2026-07-06 (Semantic Scholar survey of 2024–2026 papers). Criterion: programmatic
ground-truth evaluation (execution, exact match, proof checker, state check) — no LLM-as-judge.
Annotated with fit for the F2/CGE question: *where could prompt-diversity of subagents pay?*
Our criterion: diversity converts only where (a) aggregation pools rather than votes, or (b) an
inference-time oracle filters candidates, or (c) failures are basin-locked framings, not depth.

## 1. Code generation with executable tests (oracle at inference time — disarms M2)

| Benchmark | Evaluator | Size / notes | Recent usage |
|---|---|---|---|
| **LiveCodeBench** (v6) | hidden unit tests, pass@k | continuously updated from AtCoder/LeetCode/Codeforces; filter by date > model cutoff for contamination-free subsets (~80–700 problems) | standard in 2025–26 inference-scaling papers (arXiv:2403.07974) |
| **CodeContests / CodeContests+** | executable test suites | AlphaCode's venue; CC+ fixes weak tests (fewer false positives) | Large Language Monkeys, Archon, RL-for-code papers |
| **HumanEval+ / MBPP+ (EvalPlus)** | augmented unit tests | 164 / 378 problems; near-saturated for frontier — headroom check needed | still used for budget-matched sampling studies |
| **APPS** | test cases, 3 difficulty tiers | 10k problems; older, partial contamination | competitive-programming TTS papers |
| **BigCodeBench** | unit tests, library-use tasks | 1,140 problems | 2025 code-agent papers |
| **SWE-bench Verified / Multilingual / Live** | repo test suites (fail→pass) | 500 human-validated / 300 multilingual / auto-updated live variant (contamination-safe) | the default agentic-coding eval; note memorization critiques (SWE-Bench Illusion, arXiv:2506.12286) |

Best fit for a sign-flip test of F2: **LiveCodeBench post-cutoff subset** — same arms as
scripts/12, selection by public-test filtering instead of voting. CodeContests+ as backup.

## 2. Formal theorem proving (perfect verifier — the extreme oracle case)

| Benchmark | Evaluator | Size / notes |
|---|---|---|
| **miniF2F** (Lean 4) | Lean type-checker | 488 problems; ~90% solved by specialized provers, ~49% by general LLMs at pass@16k — headroom depends on model class |
| **PutnamBench** | Lean/Isabelle/Rocq checker | 658+ Lean problems; hard (SOTA prover ~7–13% formal), massive pass@k regimes standard (pass@1024) |
| **ProofNet** (Lean 4 port) | checker | 371 undergrad problems |
| **CombiBench** | Lean checker | combinatorics-specific, 2025 |

Verifier has zero false positives → repeated/diverse sampling is pure coverage; the regime where
diversity levers have the most theoretical room. Heavy compute per attempt.

## 3. Bug / vulnerability detection (set-valued, pooling aggregation — disarms voting)

| Benchmark | Evaluator | Size / notes |
|---|---|---|
| **SecVulEval** | statement-level CVE ground truth | 25,440 C/C++ functions, 5,867 CVEs; best model ~24% F1 — huge headroom |
| **SEC-bench** | automated CVE reproduction: PoC triggers + patch validation (execution) | 200 verified instances; top models 18% PoC / 34% patch |
| **eyeballvul** | CVE ground truth, repo-level, weekly-updated (contamination-safe) | 24k vulns, 6k revisions; recall/precision of *finding lists* — exactly union-coverage |
| **PrimeVul** | deduplicated CVE labels | fixes BigVul label leakage (68%→3% F1); function-level classification |
| **Defects4J** | fail→pass test execution | classic Java bug set; detection + repair |

Caveat: binary-classification variants (PrimeVul) are voting-style; the *detection-in-the-wild*
variants (eyeballvul, SEC-bench) are the pooling-metric ones our criterion favors.

## 4. Deep research / web QA (exact-match short answers)

| Benchmark | Evaluator | Size / notes |
|---|---|---|
| **BrowseComp** | exact string match (answers designed checkable) | 1,266 hard multi-hop questions; papers use 100–260 subsets for cost; pass@k + maj@k already reported in TTS papers (arXiv:2510.06135) |
| **BrowseComp-Plus** | exact match over a *fixed curated corpus* | removes live-web nondeterminism — better for controlled arm comparisons |
| **GAIA** (text-only val) | exact match | 103 questions; 3 difficulty levels; ubiquitous |
| **WebWalkerQA** | exact match | 680 queries, structured web traversal |
| **SEAL-0 / xbench-DeepSearch** | exact match | adversarial-to-retrieval; Chinese/professional variants |
| **Humanity's Last Exam** (text) | exact match | 2,158 text questions; expert-level, mostly knowledge-bound |

Multi-angle search = genuinely different retrieval trajectories; asymmetric verification
(checking a found answer is cheap) — the published TTS-for-search result uses exactly this.

## 5. Text-to-SQL (execution accuracy against gold result)

| Benchmark | Evaluator | Size / notes |
|---|---|---|
| **BIRD** (dev / Mini-Dev) | execution-result match | 1,534 dev questions, 95 DBs; the de-facto standard; note ~annotation-error critiques (arXiv:2601.08778) |
| **Spider 2.0** (-Lite/-Snow) | execution accuracy | 547 enterprise workflows, ~800-col schemas; GPT-4 ≈ 6% — huge headroom; gold public for only 121 of Snow |
| **KaggleDBQA** | execution accuracy | small, real-world web DBs |

Interpretation-dominated failure mode (ambiguous schema/question reading) — the one family where
our q16-style *re-interpretation* diversity attacks the actual bottleneck. Multi-generator
frameworks (XiYan-SQL, R3 consensus multi-agent) already report gains here.

## 6. Program induction / hypothesis search

| Benchmark | Evaluator | Size / notes |
|---|---|---|
| **ARC-AGI-1 / -2** | exact output-grid match; train pairs = free per-hypothesis oracle | ~93% / ~69% current SOTA (huge cost); program-synthesis + sample-many-verify is literally the SOTA recipe |
| **BARC** | unit-test pass % | ARC-derived, used in abstraction-conditioning work (RLAD, arXiv:2510.02263 — closest published cousin of our "tips" question) |

## 7. Agentic environments (final-state checks, no judge)

| Benchmark | Evaluator | Size / notes |
|---|---|---|
| **τ-bench / τ²-bench** | deterministic DB final-state match | customer-service tool use; pass^k (consistency across k trials) is native |
| **AppWorld** | programmatic state assertions (~1.8k unit tests) | 60k-LOC deterministic engine |
| **TerminalBench** | task-specific checks in Docker | pass@k scaling curves already published (arXiv:2602.01244) |
| **OSWorld** | scripted state verification | GUI/computer-use; low SOTA (~12–40%) |
| **MCP-Universe** | execution-based, time-varying ground truth | real MCP servers; explicitly rejects LLM-judge |

## Notes for our next experiment

- Primary candidate: **F2 arms on LiveCodeBench post-cutoff with public-test filtering** —
  prediction: static/adaptive diversity flips from harmful to neutral-or-positive because the
  oracle removes the abandon-winner cost (M2) and voting is replaced by filtering.
- Secondary: **eyeballvul or SEC-bench subset** for the pooling-metric claim (unique true
  findings @ K diverse lenses vs K neutral passes).
- BIRD/Spider-2.0 for the interpretation-diversity claim (reinterp role should finally beat
  plain escalation).
- All are $-estimable up front; per the cost rule, smoke + estimate before any run > $10.
