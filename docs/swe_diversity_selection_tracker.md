# SWE repair diversity — experiment plan and progress

**Latest completed milestone:** the fixed20 evaluation runtime checks are complete:18/20 pass the original whole-command rule, while all20 pass independently audited required-test conditions. The23 supported training sources and actual GPU smoke also pass. A separate required-test control is being prepared, with the original strict failure preserved. Full training and autonomous SWE improvement remain untested; no diversity benefit is established. See §31 and the [revised control plan](swe_required_test_control_plan.md).

Owner: Yiren Liu. Started: 2026-09-09. Status: Paper program active under $200 API cap. Reward extraction was audited; the original selector comparison and the corrected regrouping pilot did not establish a diversity benefit. A stronger supervision control improves supplied-test held-in accuracy9→14/16 and calibration19→20/21, but a frozen stronger-verifier check is flat14→14/17 and fails its gate. A fresh six-task SWE pool yields47/48 successful repairs but one successful mechanism per task under blinded code reviews. A fixed local Qwen one-shot baseline and a separate native-tool baseline each resolve1/6 tasks, with different successful tasks. All six directed short/full-context tool controls pass. The subsequent six assisted-start continuations resolve0/6; the fixed evidence-utility branch is closed, and a separate inventory identifies284candidate source tasks/258metadata families, with two1.11runtime canaries passing. A four-episode teacher debugging smoke yields4/4 required-test successes and passes replay. Final107 remains unscored; human construct audit, official container replication and a working diversity method remain open.

**Current qualification:** The earlier last-layer selection/transfer pilots use the frozen **last-block execution reward**. A first-block sensitivity changes 79/960 training labels and leaves only26/64 original development groups balanced. Those earlier results must not be interpreted as a robust test of functional correctness or semantic diversity. Original artifacts remain unchanged; see §17.

**Question:** Can trajectory diversity select a set of successful SWE repair attempts
more effectively than quality scoring and cheap patch diversity?

This is the working log for the new direction discussed after the Aug 23 team meeting.
Record decisions, failed probes, actual costs, artifacts, and changes of interpretation
here as they happen. Proposed experiments below are not completed results.

## Progress toward the original goal — summary as of 2026-09-09

The original goal is to test whether **a diversity metric can identify groups of SWE
repair trajectories that provide better learning signals**, using pass@k as an initial
proxy before considering post-training. We have not yet demonstrated that benefit.
We have identified concrete reasons why the current metrics are unreliable and built
a better testbed for fixing them.


| Experiment                                             | Result                                                                                                                                                                                                      | Connection to the goal                                                                                                                                                                 |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cheap diversity selection** on 301 held-out issues   | Quality-only selection achieved **37.21% coverage**, versus **33.95%** for random selection. Tuned diversity chose zero weight; a separately frozen nonzero setting reduced coverage by 1 percentage point. | Diversity needs to add value beyond simply choosing higher-quality attempts. We have not established that yet.                                                                         |
| **Semantic diversity** on 100 development issues       | Model-reviewed quality reached **42% coverage**; semantic diversity added no coverage. Oracle coverage was only **44%**.                                                                                    | This pilot had little remaining headroom. It does not establish that semantic diversity is generally useless.                                                                          |
| **Repair-location audit**                              | Claims sometimes omitted locations, misidentified functions, or missed important arguments and retained statements. Identical runtime edits could receive substantial distance.                             | The metric was not reliably measuring the distinctions we intended.                                                                                                                    |
| **Six controlled location contrasts**                  | Claims explicitly named the differing functions in only **one pair**; one pair produced identical claims.                                                                                                   | Some repair information disappeared **before embedding**, so changing the distance function alone could not recover it.                                                                |
| **Source-grounded representation**                     | Preserved the targeted code details, passed auxiliary-edit invariance checks, and distinguished all six cross-function contrasts.                                                                           | We now have inspectable evidence for operations, locations, and guards.                                                                                                                |
| **Initial within-function stress tests**               | An edit before versus after a state update still received distance zero, despite different behavior.                                                                                                        | Exposed order loss in the initial comparison; the follow-up below addresses the tested cases.                                                                                          |
| **Pass@k-to-learning audit**                           | Exact random-group SWE coverage is **36.483%**, but only **29.373%** of groups have a nonzero centered binary-reward term.                                                                                  | Coverage and useful updates are different endpoints; all-correct groups can have maximal coverage and zero centered reward signal.                                                     |
| **Fresh code-learning bridge**                         | **960 rollouts / 120 MBPP train tasks**. Last-block execution:546 passes/22 eligible tasks; first-block sensitivity:625 passes/17 eligible tasks.                                                           | Reward extraction materially changes the learning groups. MBPP remains a mechanism study, not SWE replication.                                                                         |
| **Actual update-transfer pilot**                       | Under last-block rewards, FP32 diversity lift **+3.19×10⁻⁶ NLL/token**, interval **[−7.54,+11.06]×10⁻⁶**.                                                                                                   | No reliable diversity benefit; reward-extraction sensitivity qualifies the group composition (§15/17).                                                                                 |
| **New-source replication**                             | Under the same reward rule, diversity lift remains uncertain; additive calibration utility passes Holm correction (p=.00439).                                                                               | Supports reference-loss utility prediction conditional on the frozen rewards and shared measurement references; not diversity or functional gain (§16).                                |
| **Executable calibration grid**                        | Base and all12 checkpoints score17/21 under frozen extraction. First-block base sensitivity is19/21; four-arm comparison is complete with identical base outputs.                                           | Saved adapters change logits, but no accuracy gain is established. Final107 stays unscored (§16/17).                                                                                   |
| **Exact-multiset regrouping**                          | Completed17 source tasks and a16-task interface sensitivity; neither mixing nor the curvature selector has supported positive lift at either step.                                                          | The proposed nonadditive score failed its pilot; numerical amplification also limits interpretation (§17).                                                                             |
| **Stronger verified-supervision control**              | Held-in accuracy **9→14/16**, calibration **19→20/21**, no regressions; all six gains fix assertion failures.                                                                                               | Establishes a working learner-sensitivity control. Capacity, optimizer, objective and data changed together; this is not a diversity result or final-test evidence (§18).              |
| **Expanded-test learner check**                        | On17 predeclared compatible calibration tasks, original/base-suite accuracy16→17/17, but combined MBPP+ accuracy14→14/17 with no task flips.                                                                | The stronger-verifier learner gate fails; posthoc contract ambiguities also limit interpretation. Hold the proposed selection trial (§19).                                             |
| **Correct-program selection support**                  | 23 tasks support length-matched positive pairs with different AST scores; MBPP+ membership reduces that to at most21 before regrading.                                                                      | Establishes candidate support, not a diversity effect; a stronger-verifier development gate precedes training (§19).                                                                   |
| **SWE local execution bridge**                         | Two SymPy baseline/gold pairs reproduce target failures/fixes with all official regressions preserved.                                                                                                      | Real repository execution is feasible locally; this is not generated repair performance or official container verification (§19).                                                      |
| **Fresh SWE neutral repair pool**                      | 48 fixed GPT-5.6 Terra attempts;47 pass required tests.21 patch texts/ASTs yield only one successful mechanism per task under two blinded reviews.                                                          | Real generated repairs are verified, but the diverse-correct-pair support gate fails. Every K2 group succeeds, so pass@2 supplies no variation (§20).                                  |
| **Local SWE student headroom**                         | Six fixed Qwen3-8B one-shot attempts resolve1/6: two invalid exact edits and three target/regression failures.                                                                                              | Establishes student/interface headroom despite teacher saturation; no learning gain or matched-tool comparison yet. Triton attention precision is qualified (§21).                     |
| **Observed debugging trajectories**                    | Four fixed teacher episodes complete with4/4 required-test successes; first episode replays exactly. Blinded review finds repeated pre-edit diagnostics within each task.                                   | Real trajectories are now available, but complementary evidence and a diversity benefit remain unestablished; test student evidence utility before scale-up (§21).                     |
| **Native-tool student baseline**                       | Six fixed Qwen episodes resolve **1/6**; five produce no patch. All46 model responses and45 tool actions pass the provenance audit.                                                                         | Tool access is operational, but unproductive action selection limits the learner. Check directed tool competence before evidence utility; no diversity or learning gain (§22).         |
| **Directed tool controls**                             | Literal search, bounded read and printed probe each pass in short and full context: **6/6 compliance and execution**,222 output tokens.                                                                     | Explicit tool use works on these controls. Proceed to a bounded evidence-utility check; autonomous feedback use, diversity and learning remain untested (§22).                         |
| **Assisted-start evidence pilot**                      | Two tasks×three arms; **0/6 resolved**. Actual observations produce two patches, but both fail their target test; all required regressions pass.                                                            | The fixed evidence-utility gate fails. Close this hint branch and establish a task-disjoint SWE learning control before further diversity comparisons (§23).                           |
| **SymPy learning-data readiness**                      | Metadata yields **284 source candidates / 258 families** after exclusions; two fixed 1.11 base/gold compatibility checks pass.                                                                              | A bounded route to an ordinary-supervision learner control is available. Candidate quality, broader runtime readiness and post-training gains remain unverified (§24).                 |
| **Compact SWE supervision feasibility**                | All52 public contexts succeed, but only **5/32** source targets meet exact conversion, visibility and completion constraints; minimum16.                                                                    | No training or quality test was launched. This is a data-feasibility failure, not a learning-null result (§25).                                                                        |
| **Source context capacity**                            | Increasing native prompt budget6,144→12,288→24,576 yields **5→6→8/32** prequalified sources.                                                                                                                | Capacity alone does not supply the declared training dose; all64 larger source contexts pass audit (§26).                                                                              |
| **Production-source retrieval**                        | At24,576 tokens, production chunks reaches **9/32** and whole-file preference **12/32**, versus8/32 baseline. Both fail minimum16; all64 contexts pass independent audit.                                   | Whole-file inclusion reduces region misses, but missing changed files dominate remaining failures. No learning or diversity claim follows (§27).                                       |
| **Issue-symbol/dependency localization**               | All32 contexts pass independent audit, but only **8/32** targets qualify, versus12/32 for whole-file preference and minimum16.                                                                              | The final planned retrieval variant fails. Close the static interface branch; next audit joint localization/read/repair supervision, retaining autonomous SWE evaluation (§28).        |
| **Constructed localization/read/repair support**       | **23/32** valid demonstrations, required 16; zero technical failures. Independent audit verifies 85 causal rows and 62 tool observations.                                                                   | Clears a training-data support obstacle. Locations are assistant training labels; autonomous learning and diversity benefits remain untested (§29).                                    |
| **SWE learner technical and source-quality readiness** | Actual multi-turn GPU smoke passes; all **23/23** supported source baseline/gold pairs pass after a strictly reviewed metadata-collector correction.                                                        | Executable supervision and training feasibility are established. The original 20 evaluation tasks still need readiness checks before full training and an autonomous comparison (§30). |
| **Position-sensitive comparison trials**               | Final ordered anchors pass **6/6 structural** and **1/2 invariance** checks in the last separately authored toy suite; a 12-pair human packet is ready and unscored.                                        | Tested placement/order failures are addressed. Normalization covers **0/80 original repair scopes**, so natural semantic validity and utility remain open (§14).                       |




Relative to the meeting's assumptions:

- **Assumption 1 — better learning signals improve downstream performance:** stronger
verified-supervision training now improves held-in and calibration code execution.
This is a supplied-test development control. The stronger-verifier check is flat14→14/17
and exposes further contract ambiguity (§19); final-task and SWE learning replication remain open.
- **Assumption 2 — more diverse groups provide better signals:** still open. Our
selection pilots have not demonstrated an incremental benefit.
- **Assumption 3 — a metric can measure and guide useful diversity:** this is where
we have made the most progress. We have exposed failures and established necessary
checks, but have not validated a metric.

The update-transfer replication, selector comparison, regrouping and stronger learner
control are complete (§§16–18). The stronger-verifier gate failed (§19). A fresh SWE
repair pool now verifies generation end to end, but lacks successful mechanism
variation and pass@2 headroom (§20). The next separate design should establish
student/task learning headroom and assess full diagnostic trajectories at fixed
final repair. A working diversity method, downstream learning evidence and human
construct validation remain required; see the [paper program](swe_paper_program.md).

## 1. Why this experiment

The project has established method discrimination, but has not demonstrated a reliable
performance benefit from inducing method diversity. Relevant prior results:

- F3/E3 localization: prompt lenses did not change useful coverage. Later selection gains
were matched by counting distinct predicted files. The saved records contain only
`text_tail`, not complete reasoning or repair trajectories.
- N4: correctness-gated diversity GRPO did not increase correct-method diversity at the
tested doses; correct solutions offered little method support. No observed hacking.
- G3/G4: claim-Chamfer discriminates code approaches, but remains style-sensitive at set
level; deadband calibration helps robustness at a discrimination cost.
- Aug 26 V1: diversity supplied little predictive value for pass@k beyond accuracy.

Sources: [program tracker](experiment_tracker.md),
[logical-distance log](logical_distance_plan.md),
[prior SWE design](f3_diversity_regime_design.md).

For independent rollouts on a fixed task, true pass@k = 1 - (1 - p)^k, where p is
pass@1. A realized k-group's coverage is simply whether any member succeeds. Correlating
diversity with that indicator does not isolate a diversity benefit; controlling its
correct count eliminates outcome variation. Matching average accuracy across tasks also
does not match the distribution of per-task success probabilities.

Instead, evaluate a frozen selector that cannot access benchmark correctness labels.
This tests **selection utility**, separately from construct validity, generation guidance,
and downstream training value. A good diversity measure need not pass all four tests.

## 2. Staged experiments


| ID  | Experiment                                             | Deliverable / decision                                                             | Status                                                                                                                                                |
| --- | ------------------------------------------------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0  | Audit local infrastructure and public trajectory pools | Reusable assets; counts/headroom; initial schema/provenance inspection             | Support audit complete; broader provenance audit remains                                                                                              |
| S1  | Offline selection pilot on existing repair candidates  | Paired selection gains and available headroom                                      | Cheap pilot complete; semantic development gate failed (§10)                                                                                          |
| S2  | Fresh SWE-bench repair replication                     | Baseline/gold smoke, then separate development and pilot sets                      | Six-task host generation/evaluation complete;47/48 teacher successes,1/6 local student successes. Official container execution remains open (§§20–21) |
| S3  | Budget-matched generation guidance                     | Diverse next hypothesis vs equally informed ordinary retry                         | Conditional on S1/S2 utility                                                                                                                          |
| S4  | Diversity-curated SFT                                  | Same issues, successful-example/token budgets, separate evaluation, multiple seeds | Conditional; not implied by selection results                                                                                                         |


Default order: **S0 → S1 before new paid generation**. Existing public trajectories can
test analysis and reveal failure modes; they do not replace a controlled contemporary
SWE-bench replication. Do not launch a model/metric grid before one useful comparison.

## 3. Selection protocol (draft; freeze before outcome evaluation)



### Pools, splits, and information boundaries

- Fresh-run starting point: one capable model, one pinned agent scaffold, independent
attempts with identical budgets; 16 attempts per issue, select k=4. Three smoke issues,
a separate development set (suggested 10 issues), then 50 pilot issues. These sizes are
feasibility defaults, not a power calculation or an approved spending commitment.
- Freeze issue IDs before generation. Retain all-failure and all-success issues; report
mixed-success issues as a secondary diagnostic. Do not sample until both outcomes occur.
- For public pools, audit counts by **issue × generator/configuration**, not issue alone.
Prefer homogeneous pools. If sources must be mixed, keep source quotas identical for
every selector and use a quota-matched random baseline. A model ID is not proof of
identical decoding, scaffold, or independence; record missing provenance.
- Split by issue; keep retries, revisions, and descendants with their parent. Check
duplicates and shared original issues/base commits across splits. Add repository-held-out
sensitivity when data permits. Never tune on the final evaluation labels.
- Selector inputs: issue, agent-visible context/history, actions, patch, legitimate local
test observations. Exclude reference patches, benchmark grading logs, success flags,
gold-file overlap, and any metadata constructed from those labels. Agent-written success
claims are untrusted visible text, not correctness labels.
- Save immutable raw files; extract selector features separately from evaluator labels.



### Rules to compare


| Rule                            | Purpose                            |
| ------------------------------- | ---------------------------------- |
| Random k                        | Exact within-pool baseline         |
| Quality-only top k              | Individual quality recognition     |
| Diversity-only                  | Whether dispersion alone is useful |
| Quality + semantic diversity    | Primary candidate                  |
| Quality + cheap patch diversity | Required deflationary control      |


Primary contrast: quality + semantic diversity minus quality-only. Require a comparison
against quality + cheap diversity before claiming semantic value. Freeze a development-
selected representative of each family rather than selecting the best test result.

Initial diversity families: (a) canonical repair-hypothesis/action claims + Chamfer;
(b) read/edit artifact sets, retaining investigation and final-repair views separately;
(c) lexical patch Jaccard and exact patch deduplication. Do not port numeric `raw_val`
to SWE as the primary metric. Keep graph topology out of the first experiment.

Initial quality baseline: deterministic visible indicators (nonempty parseable patch,
normal submission, observable local test evidence), followed if needed by a frozen patch
reviewer. Fit any weights or score calibration on development issues only. Patch length,
trajectory length, and touched-file count are required nuisance diagnostics.

For an implementation, use a simple additive objective, e.g. mean development-scaled
quality + lambda × mean pairwise distance. Lambda includes zero and is selected on
development only. At N=16, k=4 there are 1,820 subsets, so exact search is inexpensive.
Use deterministic, label-blind tie breaking. Vendi is a secondary aggregator; if used,
check kernel eigenvalues and document any PSD correction rather than silently treating
an arbitrary distance kernel as PSD.

### Outcomes and statistical units

For issue i and selector s, define Y_is = 1 if any selected patch passes benchmark
evaluation, else 0. Call this **selected-set coverage@4**, not iid pass@4.

For N candidates with c successes, exact random-k expected coverage is
`b_i = 1 - choose(N-c, k) / choose(N, k)`.

- Report mean Y_s, paired gains vs each control, selected mean correctness, and costs.
- Bootstrap over issues (suggested 10,000 resamples); overlapping subsets are not
independent observations. Keep all methods paired on the same issue sample.
- Report candidate-pool coverage `mean(1[c>0])`, random coverage, and maximum possible
selection gain `mean(1[c>0] - b_i)`. This bound identifies support/ceiling limitations
without inventing a selector that uses gold labels.
- For mixed-source quotas, compute the matching random expectation and oracle bound
under those same quotas; the pooled hypergeometric formula is insufficient.
- Set a minimum worthwhile gain before confirmatory evaluation; provisional 5 percentage
points. Use pilot issue-level variability to size a separate confirmatory run. Fifty
issues cannot establish small effects; wide CIs are inconclusive, not evidence of zero.
- Record missing labels and environment failures separately from incorrect patches.
Predeclare retry/exclusion rules; do not silently drop difficult issues.
- Selecting 4 from 16 costs all 16 generations plus scoring. This is not a claim of
inference savings or a deployable top-1 verifier. Benchmark tests score utility offline.



### Construct and negative-control audit

Sample pairs on development issues for blinded labels: same/different repair hypothesis,
same/different investigation, and unclear. Include successful/successful pairs, failures,
and divergent hypotheses leading to the same final patch. Audit judge agreement rather
than treating judge labels as infallible.

Explanation-only rewrites preserve actions and patches and should not change scores
substantially. They are controls, not additional independent candidates. Changed code
requires fresh grading. Keep revision families together in splits. Do not assume that
selective duplication is harmless under a mean-based Chamfer implementation; test it.

## 4. Infrastructure and reusable assets


| Asset                                                   | Observed state                                                                             | Consequence                                                     |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `scripts/14_locdiv.py`                                  | Single-call file prediction; saves last 500 text characters                                | Reuse IDs/loading ideas only; not a repair agent                |
| `runs/locdiv-swebv-n50-k8/raw.jsonl`                    | 800 rows, 50 issues, file predictions/recall/precision/token counts                        | Cannot recover repair trajectories or patch success             |
| `runs/swebench_repos/`                                  | 12 repository directories                                                                  | Existing checkouts; execution dependencies not verified         |
| `scripts/20_locdiv_linkb.py`, `21_locdiv_moderators.py` | Within-pool comparisons and issue bootstrap                                                | Reuse statistical patterns; replace localization outcome        |
| `scripts/34_domain_transfer.py`, `35_hybrid_metric.py`  | Claims, embeddings, Chamfer, deadband                                                      | Adapt extraction to repairs; keep new calibration separate      |
| `.venv`                                                 | NumPy/SciPy/datasets available; SWE-bench, SWE-agent, mini-SWE-agent absent                | Offline analysis possible; agent setup separate                 |
| Docker                                                  | CLI/socket exist; daemon access returns permission denied, including outside Codex sandbox | Local container evaluation not yet feasible; no images verified |
| Alternative runtimes                                    | Podman/Singularity/Apptainer not found on PATH                                             | No verified local fallback                                      |


Use a separate environment for eventual agent/harness dependencies; the research
environment pins Python 3.11 and older Torch/Transformers. Pin agent version, harness
commit, dataset revision, container digest, model snapshot, prompts, and sampling settings.

For fresh runs, persist each trajectory as it completes, including unsuccessful attempts.
Keep unique attempt IDs and grading run IDs so repeated attempts do not overwrite each
other. Manifest + raw trajectory + final patch + separate grade record + usage/cost per
attempt are the minimum artifacts. See the upstream
[mini-SWE-agent output schema](https://mini-swe-agent.com/latest/usage/output_files/).

## 5. Public-data exploration

**First candidate: [Nebius SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories).**
The dataset card reports 80,036 attempts, including 13,389 successes and 66,647 failures,
on SWE-bench-extra and SWE-bench dev issues. Fields include issue ID, model name, trajectory,
generated patch, exit status, target, and evaluation logs. This is a plausible no-generation-
cost development source, not a SWE-bench Verified test result.

Metadata audit completed below: revision pinned, all narrow columns read, counts and
headroom computed, and three raw examples inspected. Broader duplicate/leakage checks
remain. The full trajectory corpus was not downloaded.

**Fallback: [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories).**
The current viewer exposes messages, issue IDs, resolved labels, model, trajectory ID,
and patch. Synthetic repair tasks and multiple formats differ from the intended benchmark;
pool multiplicity and correctness provenance remain unaudited. Not interchangeable with
fresh Verified rollouts.

## 6. Costs and launch decisions

Current new API spend: **$0**. No generation, paid judging, training, or container jobs
launched. Public metadata retrieval and local analysis do not spend model API credits.

Estimate fresh-run cost from measured smoke usage, including input/cache/output tokens,
scoring calls, failed calls, retries, and container compute. Budget all N attempts even
when only k are selected. The previous localization design said ~$15 for its main run;
the [spending log](spending_tracker.md) records **$45.4**, so do not reuse that estimate.

Standing project rule: estimate before API spend and ask before spending over $10.
The historical G1–G3 envelope is not automatically a budget for this new experiment.
Prepare a concrete smoke configuration and cost cap before a launch decision.

## 7. Progress log



### 2026-09-09 — Direction opened; local feasibility checked

- User authorized experiment planning/exploration and a new living tracking document.
- Inspected repository conventions, prior SWE runner/artifacts, metric implementations,
dependency availability, and spending history. Existing unrelated workspace edits preserved.
- Confirmed the 800-row localization pool lacks full trajectories and patches.
- Found the Docker daemon permission problem; this is an account/runtime limitation,
not a failed SWE experiment. No local container smoke attempted.
- Identified a public labeled-trajectory source that may allow S1 before new generation.
Counts and schema above are publisher-reported until the metadata audit completes.
- Scientific decision: separate metric validity, selector utility, generation intervention,
and training benefit; do not repeat the same-group pass@k correlation as the primary test.



### 2026-09-09 — S0 completed: public pools have measurable selection headroom

Added [scripts/38_swe_pool_audit.py](../scripts/38_swe_pool_audit.py). One-shard smoke
passed before the full run. Reads four Parquet columns (issue, model, target, exit status)
and retains source file/row locators. All 12 shards now cached: **80,036 records, 3,591
issues, 13,389 successes**, matching the publisher's totals. Dataset revision pinned to
`68195a1450865274106246d0d0296a1d6807b88e`.


| Population, full pool has >=16 attempts | Pools                | Mixed full pools | All-failure full pools | Expected random-four coverage | Expected random-sixteen pool coverage | Expected oracle gain |
| --------------------------------------- | -------------------- | ---------------- | ---------------------- | ----------------------------- | ------------------------------------- | -------------------- |
| All issue-model groups                  | 1,488 (1,440 issues) | 763              | 708                    | 35.76%                        | 46.77%                                | 11.02 pp             |
| `swe-agent-llama-70b` only              | 1,440 issues         | 731              | 694                    | 35.07%                        | 46.05%                                | 10.98 pp             |
| `swe-agent-llama-8b` only               | 48 issues            | 32               | 14                     | 56.21%                        | 68.37%                                | 12.16 pp             |


There are no 405B issue-model pools with >=16 attempts. Do not compare the model rows as
model-quality rankings: they cover different issue populations. Fifteen 70B pools and two
8B pools are all-success. "Mixed" refers to the entire published pool, not a sampled
16-pack. Expectations average uniform, without-replacement N-subsamples of each full pool;
these are exact corpus summaries, not confidence bounds or selector performance.

**Decision:** use the 70B-labeled pools for the first offline development experiment:
large enough without mixing model names. A sixteen-candidate pool has room for a useful
selector; eight-candidate 70B pools offer only 3.01 pp expected headroom on their different,
broader eligible issue population. Keep N=16/k=4 provisionally. An 11 pp oracle bound
leaves modest room above the provisional 5 pp worthwhile-gain target; do not promise it.

**Important provenance limit discovered.** The publisher's
[collection description](https://nebius.com/blog/posts/scaling-data-collection-for-training-swe-agents)
uses iterative fine-tuning, temperatures 0.3–1.2, and infrastructure retries. The released
metadata does not identify those settings per row. Thus a common `model_name` does **not**
establish a fixed policy or iid rollouts. Treat S1 as retrospective selection over published
pools; retain fresh controlled generation as the test of sampler-level claims.

**Schema sample:** retrieved the first row group (1,000 rows read transiently), saved only
source rows 0/1/2 from shard 0. All are `AnalogJ__lexicon-336`, 70B, failed, context-limited
attempts, with 93/51/59 recorded messages and nonempty patches. These are format examples,
not a representative audit. Reserve this issue for development if it enters a later split.
Actual `trajectory` values are **lists of message dictionaries**, unlike the card's string
schema; fields are `role`/`text`, roles `system`/`user`/`ai`. A first inspection failed by
assuming a JSON string; rerunning on the already-saved samples with list support succeeded.
The separate `target` and `eval_logs` fields must never be selector inputs. Keep `mask`
and other training-format metadata out as well. Context exits retain unfinished histories;
do not infer that every published record ends in an agent submit action.

**Validation:** exact boundary checks passed for all-failure/all-success pools and one
success among 16 (random-four coverage 0.25; oracle gain 0.75). Full cached analysis reruns
without network access. Summary records the analysis script hash; manifest pins dataset
revision and source columns. No success-label inference, code execution, or paid judging
was used. API spend: **$0**.

Artifacts under `runs/swe-diversity-selection/public-audit/`:
`metadata.jsonl`, `manifest.json`, `summary.json`, `schema_samples.json`,
`schema_summary.json`; initial one-shard results in `smoke/`.

Reproduce from cached metadata:

```bash
.venv/bin/python scripts/38_swe_pool_audit.py
```

Retrieve the pinned narrow-column metadata again if needed:

```bash
.venv/bin/python scripts/38_swe_pool_audit.py --fetch --smoke
.venv/bin/python scripts/38_swe_pool_audit.py --fetch
```



### Next S1 implementation slice (proposed)

1. Use the 1,440 eligible 70B-labeled issues. Freeze issue-level development/evaluation
  splits without selecting on correctness; reserve the inspected issue for development.
   Suggested first slice: 100 development, 300 held-out, remaining issues reserved.
   Confirm shared-source-issue/base-commit overlap before freezing these counts.
2. Fetch patches and exit status before full histories. Start with random selection,
  normal-submission/patch-validity quality controls, and patch-Jaccard/dedup selection.
   Materialize a seeded 16-pack per issue independently of success. Keep failed/empty
   candidates in the pool; absence of a patch is not positive diversity.
3. Add semantic claims only after cheap baselines establish what remains to explain.
  Full-history downloads and extraction need a separate size/cost estimate. Audit whether
   final-patch differences and exploration differences carry distinct signals.
4. The primary outcome is gain over quality-only; semantic value additionally requires
  gain over quality plus cheap patch diversity. Describe outcome-correlated metadata and
   source-policy heterogeneity as possible mechanisms, not causal reasoning-diversity gains.
5. If the confidence interval remains wide, use the measured paired variance to size a
  new evaluation. If quality/cheap controls exhaust headroom, stop metric escalation.



### 2026-09-09 — S1 launched: split and analysis rules frozen before evaluation

User authorized proceeding. Added `scripts/39_swe_selection.py` with separate prepare,
patch-fetch, feature, development, and evaluation stages. No model calls. Outcomes are
stored separately from selector inputs; selection accepts only patch-derived features
and exit status. Quality recipes and a small token/file Jaccard lambda grid are fixed in
`runs/swe-diversity-selection/s1/split.json`; development chooses recipes, then writes
`frozen_selectors.json` before held-out scoring. The first pilot tests **cheap** diversity;
semantic claims have not been extracted and are not part of this pilot's primary claim.

Split: **100 development / 301 held-out / 1,039 reserve issues**, with 1,392 issue families
among 1,440 eligible issues. Families share a repo/base commit or normalized issue text.
The held-out set is 301 rather than 300 to keep a family together. The previously inspected
AnalogJ issue is assigned to development. Candidate sampling uses a fixed SHA256 ordering
(seed 20260909), independent of success; 16 per issue = **6,416 selected records**.

Issue metadata initially missed two issues because the current SWE-bench-extra release
removed them. Recovered their repository/base-commit/text from release-era revision
`acdbe5da55313c2d85084def051f2b9b2bb5c60a` rather than excluding them. Other issue metadata
pins: SWE-bench-extra `11dcbfb30e19552df2a2f8030bd764adc95c92a5`, SWE-bench dev
`e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`. These records exclude gold/test patches.

Protocol details: exact search over 1,820 four-subsets; quality recipes = patch syntax,
normal submission, combined `(2*valid + submitted)/3`, and combined with a small preference
for shorter patches. Compare quality-only, quality+dedup, pure cheap diversity, and
quality+cheap diversity. Lambda grid = {0, .05, .1, .25, .5, 1, 2}; choose only on dev
coverage, breaking ties by fixed grid order. Token distance uses changed-line token sets;
file distance uses changed-file sets. Empty/invalid patches receive zero diversity distance
and remain candidates. Normalized duplicate patches are retained within issues. Before
labels are scored, exclude held-out issues sharing a valid normalized patch with dev in
the same repo; preserve the initial split and record these exclusions separately.

Primary cheap-pilot contrast: frozen quality+cheap diversity versus frozen quality-only.
Also compare dedup. Bootstrap 10,000 times over **issue families**, keeping their members
together; report individual selected accuracy alongside coverage. No test-based tuning.

Checks passed before fetching: valid/empty patch parsing, exact subset count, deterministic
tie order, no positive distance for empty patches, and unchanged choices when target labels
are added to otherwise identical feature records. Three-development-issue fetch smoke
launched. Spend remains $0.

### 2026-09-09 — S1 development complete; nonzero secondary frozen before test

Full patch retrieval and feature extraction completed: 6,416 patches, 5,566 syntactically
valid, average 12.14 distinct normalized valid patches per 16-pack. No normalized valid
patch is shared across development and held-out issues within the same repo, so no overlap
exclusions were needed. End-to-end three-development-issue smoke passed before full fetch.

On 100 development issues, quality coverage is patch-validity 38%, submission 39%, combined
39%, short-patch preference 40%. With the winning short recipe, no positive token-diversity
weight exceeds 39%; file-diversity lambda=.05 ties 40%. Fixed tie order chooses **lambda=0**.
This is a development preference for omitting diversity, not a powered held-out null.

Before opening held-out outcomes, add one explicitly secondary comparison: the best
**nonzero** development setting (file Jaccard, lambda=.05) versus the same quality rule.
Keep the primary zero-weight winner unchanged. Preserve the first settings file as
`frozen_selectors_initial.json`, then freeze the amended set including this diagnostic.
This avoids interpreting the mechanically identical primary arms as evidence that all
nonzero diversity settings fail. No held-out labels have been evaluated during this decision.

### 2026-09-09 — S1 RESULT: quality helps; cheap diversity adds no demonstrated benefit

**Data:** 301 held-out issues, 292 issue families, 16 published 70B-labeled attempts per
issue, select four. All splits and selector settings frozen before outcome evaluation.
Published success labels used; no patches executed locally. API spend **$0**.


| Frozen selector                                       | Selected-set coverage@4 | Mean correctness of selected candidates |
| ----------------------------------------------------- | ----------------------- | --------------------------------------- |
| Uniform random four (exact expectation)               | 33.95%                  | 19.52%                                  |
| Quality-only (valid, submitted, shorter)              | **37.21% (112/301)**    | **23.92%**                              |
| Quality + normalized patch deduplication              | 37.87% (114/301)        | 22.09%                                  |
| Quality + tuned cheap diversity                       | 37.21% (112/301)        | 23.92%                                  |
| Nonzero secondary: quality + file Jaccard, lambda=.05 | 36.21% (109/301)        | 21.43%                                  |
| Pure token-Jaccard diversity                          | 32.56% (98/301)         | 15.70%                                  |
| Oracle using all sixteen candidates                   | 44.85% (135/301)        | —                                       |


Paired differences, 95% percentile bootstrap over issue families (10,000 draws):

- **Quality minus random: +3.26 pp [+0.84, +5.68].** A measurable selection benefit from
observable patch quality; this is not a reasoning-diversity effect.
- **Nonzero file diversity minus quality: -1.00 pp [-3.33, +1.33]**, 5 wins / 8 losses.
This tested setting does not demonstrate an improvement; its interval does not support
the provisional +5 pp worthwhile effect in this population. It does not test every
metric or prove that semantic diversity cannot help.
- **Dedup minus quality: +0.66 pp [0.00, +1.67]**, only 2 wins / 0 losses. Fragile evidence,
not an established complementarity effect; the percentile interval's zero lower bound
should not be read as significance from two discordant issues.
- Pure token diversity minus random: -1.40 pp [-4.41, +1.51].
- The primary tuned-diversity selector chose **lambda=0 on development**, making it exactly
the quality selector. Its 0.00 [0.00, 0.00] contrast is mechanical, **not** a powered null
for diversity. The nonzero secondary was frozen and documented before test evaluation.

**Mechanism diagnostic:** diversity selects larger, broader edits. Quality selects 11.21
changed lines / 1.25 files per candidate on average; file-diversity selects 45.01 / 3.45;
pure token diversity selects 86.68 / 3.74. Their lower selected correctness is consistent
with a quality tradeoff from selecting unusual/broad patches. These are descriptive
associations, not proof that size causes failures or that all selected differences are style.

**Remaining support:** 166/301 sixteen-pools contain no labeled success; 11 are all-success.
Quality misses a reachable success on **23 issues**. Its oracle headroom is **7.64 pp
[4.68, 10.67]**. Dedup still leaves 21 reachable issues. Thus cheap controls do not exhaust
headroom, but a +5 pp semantic improvement would need to recover most of the remaining
opportunities without sacrificing current wins. This is a demanding target.

**Label audit:** two issues (`joke2k__faker-828`, `BlueBrain__NeuroM-1008`) assign opposite
labels to byte-identical valid patches. Four retrieved grading logs show different test
outcomes: Faker's `test_name_female` flips; NeuroM's `test_polygon_diameter` flips. This
is consistent with execution/test variability, whose cause cannot be established from
logs alone. NeuroM's successful record still lists four other pytest failures; the
publisher's success flag may concern a required subset, so do not substitute an ad hoc
"all pytest tests pass" grader. Original labels retained; no test-result-based retuning.
Post-hoc sensitivity excluding the two issues gives quality-vs-random +3.22 pp,
dedup-vs-quality +0.67 pp, and nonzero-diversity-vs-quality -1.00 pp: conclusion unchanged.
Neither conflicting-label issue accounts for the two dedup wins.

**Decision:** complete S1-cheap with no semantic-diversity claim. Do not expand the cheap
metric/weight grid on these held-out labels. The useful next gate is whether semantic
repair/trajectory distinctions improve selection on development **beyond the frozen quality
and dedup controls**. Distinguish a patch-only semantic probe from full-history method
measurement. Estimate extraction/download cost first; use untouched reserve families for
any new confirmatory comparison after adapting the method to these findings. Fresh model
rollouts and RL remain premature.

**Reproducibility and checks:**
[runner](../scripts/39_swe_selection.py); artifacts under
`runs/swe-diversity-selection/s1/`: `issue_metadata.jsonl`, `split.json`, `labels.json`,
`patches/`, `features.jsonl`, `feature-audit.json`, `development.json`,
`frozen_selectors_initial.json`, `frozen_selectors.json`, `evaluation_per_issue.json`,
`evaluation.json`, `label_conflict_pairs.json`, `label_conflict_logs.json`, and
`label_conflict_log_summary.json`. Outcome logs remain separate from features. Source
rows/revisions, seed, script hashes, frozen settings, and exact selections are saved.
Independent validation of all 301 saved selections reproduced coverage/correctness and
confirmed quality-only maximizes its individual quality score. Lint passed before the run.

Cached outcome reproduction (does not tune or fetch):

```bash
.venv/bin/python scripts/39_swe_selection.py evaluate
```

Full stage order on a new output copy: `prepare` → `fetch --smoke` → `features --smoke`
→ smoke selection check → `fetch` → `features` → `develop` → `evaluate`. The source issue
metadata is a prerequisite; its source revisions are recorded above. Existing split and
selector files are protected from accidental re-preparation/re-tuning.

## 8. Next actions

- [x] Finish the public metadata audit; save a reproducible summary and source revision.
- [x] Inspect three raw trajectories and identify the parser schema.
- [x] Implement evaluator/selector separation and audit patch duplicates/provenance.
- [x] Freeze development/held-out issue IDs and a compact cheap-selection protocol.
- [x] Run cheap patch/quality baselines and quantify remaining headroom.
- [x] Run a bounded semantic-diversity development gate; failed, so preserve reserve issues.
- [ ] Resolve container access or choose a verified remote runtime for fresh replication.
- [x] Prepare and enforce a measured semantic-extraction budget; fresh generation still conditional.

For every future entry record: question, data/split, exact method/config, result with
uncertainty where applicable, interpretation limits, artifacts, actual cost, and next decision.

## 9. Semantic development gate — 2026-09-09

**Status:** protocol frozen before model extraction and semantic outcome evaluation.
100 existing development issues, 16 candidates each; the old 301-issue test split will
not be reused for tuning or a new confirmatory claim. Artifacts: `s1-semantic/` under
the same run root. Runner: [scripts/40](../scripts/40_swe_semantic.py).

**Qualitative audit:** 20 development issues chosen by a fixed hash; 47 patch examples
(high cheap quality, large token distance, and a successful alternative when available).
The last example type uses labels for diagnosis, so this is not a blind validation set.
`audit_ids.json` and `audit_cases.json` retain IDs, examples, and selection information.
Repeated finding: patch diversity mixes actual repair alternatives with irrelevant
reproduction scripts, logs, tests, and vendored artifacts. Examples:

- Black-4176: hashing a cache key versus truncating it; a reproduction script adds
superficial differences. Pydantic-2170: `__qualname__` repair versus application-only
reproduction; successful alternatives can be exact duplicates.
- Folium-1633: collection handling at `get_bounds` versus `iter_coords`, two labeled
successful repair locations. Urllib3-1692: tilde handling in allowed characters versus
a broader helper change. These are meaningful semantic distinctions to preserve.
- Spidermon-365: local exception fallback versus action-chain restructuring, both failed.
Fffw-64: accepting constructor kwargs versus removing a clone argument, both failed.
- Aiokatcp-8: vendored `.eggs` files dominate one patch; pvlib-1822: a broad function
rewrite complicates interpretation. These examples were not exhaustively code-reviewed.
- Pydantic-2286: two labeled successful edits have different apparent acceptance scopes;
published success is not proof of complete specification correctness.

**Frozen probe:** issue text capped at 768 tokens, patch at 3,072 tokens; remove known
vendored/cache directories and prioritize runtime files over tests/reproductions/docs.
Save filtering/truncation metadata. The LLM sees issue and patch only, never success
labels, grading logs, or audit comments. Extract 1–6 factual repair claims plus a 0–4
quality review from the same view. Empty/no-repair claims have zero distance to all,
and duplicate extraction inputs have zero distance. This is patch semantics, not a
measurement of the full agent reasoning trajectory.

Pinned extractor `gpt-4.1-mini-2025-04-14`, temperature 0, strict JSON, max output 512
tokens; 1,227 unique valid inputs across 1,600 candidates. Qwen3-Embedding-0.6B snapshot
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, local GPU 3, bf16, normalized last-token
embeddings; symmetric mean best-match cosine distance (claim Chamfer).

Quality blend alpha in {0,.25,.5,1}, chosen on development coverage; then exact 4-of-16
selection by mean quality + lambda times mean pair distance, lambda in
{0,.05,.1,.25,.5,1,2}. Controls: cheap quality, reviewer quality, reviewer-quality plus
normalized-patch deduplication, and token Jaccard on the same filtered/clipped patch
view. Grid ties select the first setting. The filtered-token control separates input
cleanup from semantic representation.

**Go/no-go:** best semantic development coverage must strictly exceed both reviewer
quality and reviewer deduplication, with selected accuracy at least as high as both.
Passing this exploratory gate warrants a frozen comparison on untouched reserve
families; it is not itself evidence of generalization or post-training benefit.
Failure stops this probe before reserve spending.

**Cost before launch:** input upper estimate 1,511,469 tokens plus up to 512 output
tokens/job: $1.609746 first attempt, $3.219492 if every job retries once.
[Official model pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini):
$0.40/M input, $0.10/M cached input, $1.60/M output. Eight-job smoke precedes full
extraction and local embedding smoke. Responses and usage saved per attempt, including
parse failures; uncertain interrupted calls retain a conservative cost reservation.
Credential loaded using python-dotenv 1.2.3 from `.tinker_env`; no secret is logged.

**Smoke correction before full launch:** first eight API calls parsed successfully
($0.004124), but qualitative inspection found an inferred implementation for an unseen
function, a missed deleted constant, and redundant/context-only claims. Archived the
original protocol/responses under `smoke-v1/`; revised the prompt to at most three
nonredundant claims, explicitly inspect deletions, and avoid inferred behavior. No
semantic selection outcomes or held-out labels informed this revision. Repeat smoke
before expansion; include the abandoned eight calls in total spend.

**Second smoke / model escalation:** revised mini smoke still described reproduction-only
patches as repairs and, on RxPY-492, treated unchanged scheduling logic as a new repair.
Archived as `smoke-v2/` (9 attempts, 8 valid outputs, $0.0061788). Switch only the model
to pinned `gpt-4.1-2025-04-14`, repeating exactly these eight input views with the same
prompt before full extraction. Model choice is based on extraction fidelity, not
selection coverage. Both abandoned mini batches total $0.0103028.

[GPT-4.1 pricing](https://developers.openai.com/api/docs/models/gpt-4.1) is $2/M input,
$0.50/M cached input, $8/M output. Full development ceiling $8.350572 on first attempts;
hard $9 cap (including mini smokes and any retry) enforced by reserving estimated
maximum call cost before dispatch. At most two attempts per input; never launch a
call exceeding the cap. No full launch until the stronger extractor smoke passes.

**Stronger smoke:** GPT-4.1 correctly abstained on reproduction-only/test-only patches,
but still inferred a scheduler change from unchanged context in RxPY-492. Full diff
inspection confirmed only whitespace and duplicate unreachable returns there. Cost
$0.026174; archive `smoke-v3/`. Added two short demonstrations of unchanged-context and
reproduction-only traps to the prompt, then repeat the same eight views. This is the
last planned prompt revision for this pilot; report residual fidelity errors explicitly.

**Input ablation, prompt held fixed:** the final prompt demonstration did not eliminate
RxPY's unchanged-context error. Archived `smoke-v4/`; $0.029052, cumulative $0.0655288.
Test a changed-lines-only view on the same eight candidate patches: keep file/hunk
headers and added/deleted lines, omit unchanged context, explicitly flag omission.
This reduces opportunities to copy context into claims but weakens quality review
context; both reviewer and semantic selectors receive the same view. No further prompt
revisions are planned. The full raw patch remains available for audit and validity
features; this excerpt is never applied or executed.

**Final smoke decision: proceed with a noisy exploratory measure.** Changed-lines-only
GPT-4.1 returned valid outputs for all eight jobs, abstained on the RxPY context trap
and test/reproduction-only cases, and preserved copy-identity/callback-shield changes.
Residual imprecision remains (DVC remote modification described as removal; RxPY concern
omits duplicate returns). This eight-case set informed development, so it is not an
independent extractor-accuracy estimate. Freeze this model, prompt, and input view; no
more revisions based on selection outcomes. Full-run ceiling **$8.312066**, plus
**$0.0655288** abandoned smoke spend; retries included under the **$9** hard cap.
Input omits unchanged diff context and retains added/deleted lines and file/hunk
headers; the original raw patch is preserved. This supersedes the original mini/full-
context settings above. Local embedding smoke produced finite unit-norm 1,024-vectors.

**Extraction complete / format repair before outcomes:** 1,238 attempts, $3.54394
recorded GPT-4.1 usage; total including abandoned smokes **$3.6094688**. Nine inputs
exceeded the requested three-claim limit on both attempts. Their JSON and quality
fields are valid; deterministically retain the first three claims for every parsed
response, preserve all original text and both attempts, and save this deviation before
any semantic coverage evaluation. No labels inform this repair and no extra calls are
needed. All 1,227 jobs then have usable outputs. Final source views truncate 9/1,600
patches and 192/1,600 issue texts; known vendor files removed on 19 candidate patches.
An additional eight-issue qualitative sample is hash-chosen excluding prior smoke
issues, with no correctness-based sampling (`unseen_fidelity_sample.json`).

## 10. Semantic result and next decision — 2026-09-09

**Result: development gate failed.** 100 development issues, 16 published attempts each,
select four. All 1,227 unique patch inputs processed; 1,627 distinct claim strings encoded
on GPU 3. Total API cost **$3.6094688**, including all abandoned smoke attempts.


| Selector                                               | Coverage@4 | Selected-candidate accuracy | Mean changed lines/candidate |
| ------------------------------------------------------ | ---------- | --------------------------- | ---------------------------- |
| Random four, exact expectation                         | 36.48%     | —                           | —                            |
| Existing cheap quality                                 | 40%        | 27.75%                      | 11.10                        |
| Reviewer + cheap quality, alpha=.25                    | **42%**    | **33.00%**                  | 18.09                        |
| Same quality + normalized-patch dedup                  | 42%        | 31.50%                      | 19.94                        |
| Quality + best semantic setting, lambda=0              | 42%        | 33.00%                      | 18.09                        |
| Quality + smallest nonzero semantic weight, lambda=.05 | 42%        | 32.00%                      | 19.51                        |
| Quality + semantic weight 2                            | 39%        | 21.00%                      | 26.81                        |
| Oracle over all sixteen                                | **44%**    | —                           | —                            |


Both semantic and filtered-token grids choose zero weight. Semantic nonzero coverage
across {.05,.1,.25,.5,1,2} is {42,41,41,40,40,39}%; filtered-token coverage is
{41,39,41,39,38,37}%. No tested semantic weight improves on reviewer quality. At .05,
coverage matches on every individual issue (zero wins, zero losses), while selected
accuracy falls by 1 pp. At weight 2, three existing wins are lost and none gained.
Reviewer quality itself gains two issues over cheap quality and loses none
(`DCC-Lab__RayTracing-180`, `rolepoint__pyseeyou-14`).

**Interpretation limits:** these are development-selected descriptive numbers, not
held-out estimates, so no inferential confidence intervals or generalization claim are
attached. A lambda=0 winner is mechanically identical to quality-only; the nonzero grid
shows this particular representation did not help here. There is little development
headroom: 56 pools are all failures, 2 all successes, and reviewer quality covers 42 of
the 44 pools with any success. Its maximum remaining gain is only **2 pp**. This pilot
cannot rule out a useful effect on a population with more complementary repair support,
nor evaluate the team's post-training assumptions. Source trajectories remain
heterogeneous, retrospectively collected, and outside SWE-bench Verified.

**Measurement diagnostic:** the eight additional hash-chosen cases mostly preserve the
coarse edit/abstention distinction, but reviewer quality remains unreliable. For example,
Pycoin-353 receives quality 3 despite the concern correctly noting that the requested
P2SH prefix is unchanged. Humanize-123's description overstates recursive comma handling.
This is a qualitative audit, not a formal error-rate estimate. One development issue
(`gahjelle__pyplugs-29`) has opposite labels for identical valid patches; no labels were
changed. It is not among the reviewer wins or the two remaining missed issues.

**Post-hoc inspection of both remaining opportunities (no retuning):**

- `dfm__emcee-295`: the successful patch swaps axes at the `lnprobability` compatibility
property. Failed candidates transpose at other access points or change backend layout.
Several receive the same review score; the successful patch also includes a 15-line
reproduction script, making the raw-length tie-break unfavorable. Semantic distance
does not reliably choose the correct repair location.
- `scrapinghub__price-parser-43`: success normalizes non-breaking spaces in
`extract_price_text`; failures normalize them later or change regexes. Many receive
quality 4. Claims often omit the enclosing function, collapsing an important distinction
between similar operations at different pipeline stages.

These cases motivate a **location-aware repair representation**, not an unmeasured
claim that larger diversity rewards will fix the problem. No further weights, prompts,
or input variants were evaluated after these selection outcomes.

**Decision and concrete next experiment:**

1. Close this development probe and preserve all 1,039 reserve issues. Do not launch a
  reserve evaluation, fresh rollouts, or RL on this result. The prior 301-issue test
   results were not reused for the new comparison.
2. Next, build a small repair-pair annotation audit on development data: record file,
  enclosing function, changed condition/operation, and where the value is transformed.
   Compare (a) same repair plus irrelevant edits, (b) same operation at different
   locations, and (c) distinct repair mechanisms. Use blind annotations of actual changes;
   keep correctness labels separate. Outcome-guided diagnostic pairs are a separate
   stratum, not a performance-evaluation sample. Existing artifacts suffice; no new
   model spend is needed to prepare the packet.
3. Require the representation to preserve repair location and resist auxiliary-file
  changes before another selection pilot. Any new efficacy test needs a newly frozen
   population, an estimate of headroom beyond the final quality baseline, and controls
   with the same issue/context access and auxiliary-edit treatment. Budget and selection
   settings must be concrete before spending or opening reserve outcomes.

**Validation and artifacts:** [runner](../scripts/40_swe_semantic.py),
`runs/swe-diversity-selection/s1-semantic/`: `protocol.json`, `jobs.jsonl`, `mapping.json`,
`responses/`, `smoke-v1/` through `smoke-v4/`, `extraction_summary.json`,
`embedding_manifest.json`, `embedding_texts.json`, `embeddings.npy`,
`selection_features.npz`, `development.json`, `development_per_issue.json`,
`validation_and_diagnostics.json`, and fidelity audit files. Independent enumeration
checked every non-dedup objective maximum and recomputed coverage/accuracy for all
**1,900 issue-selector combinations**. Matrices are finite, symmetric, and zero on
identical claim sets; embedding vectors have unit norm and no claim exceeds the 128-token
embedding limit. Ruff passed. Raw attempts preserve extra claims; parsing retains the
first three after the documented, pre-outcome format repair.

## 11. Repair-location audit — 2026-09-09

**Completed:** [20-pair annotation packet](../runs/swe-diversity-selection/location-audit/annotation/README.md),
[annotation rubric](swe_repair_location_audit.md),
[runner](../scripts/41_swe_location_audit.py), source-grounded scope extraction, and
provisional assistant annotations. **New API cost $0**; cumulative semantic-stage cost
remains $3.6094688. No selector weights or reserve outcomes were evaluated.

**Sample frozen before analysis:** 18 issue-disjoint pairs selected without correctness
labels or claim-distance scores, plus the two previously outcome-guided diagnostic
pairs. Six candidates per heuristic stratum: identical runtime diffs with differing
auxiliary files, similar operations/different hunk hints, and different changed-token
sets. Python-only, <=80 runtime changed lines and <=16,000 raw characters. Heuristic
bins are not ground truth. Source metadata/commit hashes and SHA-based sampling are saved.

**Reliable location evidence:** fetched 26 public source files at their exact base
commits; all 45 runtime file instances (including two new files) reconstructed and parsed.
Before/after AST scopes distinguish qualified nested methods. Every existing-file
instance (43) matched the patch's advertised old Git blob hash, and independent `git apply` reconstruction in disposable directories matched the in-memory result. Candidate
code was never executed. No source, hunk-matching, or syntax failure occurred in this sample.

The two diagnostics resolve to:


| Pair              | Patch A location       | Patch B location                | Existing claim distance |
| ----------------- | ---------------------- | ------------------------------- | ----------------------- |
| P05: price-parser | `extract_price_text`   | `parse_number`                  | .2304                   |
| P06: emcee        | `Backend.get_log_prob` | `EnsembleSampler.lnprobability` | .2498                   |


These have related operations at different stages, with additional operation details
that also matter. Claims do not exactly collide; P05 omits both function names, while P06
partly identifies the compatibility property. A file-only control cannot distinguish
P05. A qualified-scope set can distinguish both, but this is only structural information,
not evidence that it measures repair usefulness.

**Observed invariance failure:** two of six pairs with identical runtime diff blocks
have nonzero claim distances: P03 (parameters-validation) **.2815** and P04 (pyvo) **.0667**.
P03 differs by a reproduction script and produces a more specific paraphrase of the same
return operation; its distance is larger than either location diagnostic. The remaining
four distances are zero, including one pair with empty claims and one with identical
extractor inputs after cache-file filtering. Thus these six are not six independent
nontrivial invariance successes/failures. One extraction per input does not disentangle
input sensitivity from model variability; the observed representation is nevertheless
not invariant on this example.

**Fidelity errors beyond naming:** P14 attributes a scheduler edit to the wrong factory;
P19 omits a forced `text=True` value; P20 claims staterror is appended only under a new
guard although an unconditional append remains outside it. P02 describes debug printing
as a repair. These errors show why adding function names alone is insufficient: the
representation must preserve changed operations, guards, and relevant retained context.
Detailed evidence is in `claim_fidelity_notes.json`.

**Sampling correction:** provisional inspection labels the 18 main pairs as 5 same-runtime
changes, 10 different mechanisms, 1 repair-versus-diagnostic-only edit, and 2 with no
relevant repair. There are **zero clean location-focused pairs in the main sample**;
only the two outcome-guided diagnostics get that label. Two of the six different-hunk
candidates even have identical verified scope sets (P18/P19), while others mix scope
and mechanism changes. Neither hunk labels nor scope-set inequality are semantic labels.
Do not compute an AUC against these sampling bins or claim location validity from them.
Assistant labels are provisional, not independent human gold; some issues were inspected
in earlier work and the automatic identity summary was visible during the audit.

**Outcome-hidden review materials:** `annotation/` contains issue text, full A/B patches,
base-source excerpts, and a blank JSONL template. Outcomes, reviews, extracted claims,
metric scores, and sampling strata are omitted. Analyst outputs and provisional notes
are separate. Human reviewers should use the packet before opening those analyst files.
No messages or files were sent to collaborators.

**Next decision:** pause efficacy testing. First obtain independent annotations and add
outcome-independent, operation-matched location contrasts; the current main sample does
not supply them. A candidate representation should retain `(file, qualified scope, removed code, added code, relevant retained context)` with explicit evidence and unknown
fields. Evaluate claims against that evidence, including unchanged-control-flow traps.
Keep file-only and scope-only controls, since scope-only also misses same-function
mechanism changes such as P18/P19. A new utility pilot requires passing those checks and
a newly frozen sample with headroom beyond an equally informed quality baseline.

**Artifacts:** `runs/swe-diversity-selection/location-audit/`: `protocol.json`, `pairs.json`,
`sources/`, `analysis.json`, `contexts.json`, `annotation/`,
`assistant_provisional_annotations.jsonl`, `claim_fidelity_notes.json`,
`structural_controls.json`, `findings.json`, and `validation.json`. Ruff and whitespace
checks passed. `scripts/41 ... report` reproduces descriptive findings after the saved
annotations; no model calls are involved. Existing S1/S1-semantic artifacts stay frozen.

## 12. Location follow-up and second-model review — 2026-09-09

**Completed:** a restricted search for exact small edit deltas found four natural candidate
pairs; inspection rejected them as location contrasts because edits coincide or only
nonfunctional import placement differs. Six controlled pairs now apply one identical
replacement at two distinct functions in cached pinned source (RxPY, urllib3, fixtures,
humanize, fsspec, spidermon). These are explicitly synthetic placement stress tests, not
natural rollouts or verified successful repairs. No correctness labels are assigned.
Operations/sites frozen before any new model judgments or scores.

Second pass: pinned GPT-5.5-2026-04-23, low reasoning, old 20 pairs plus six new pairs,
with no prior labels, extracted claims, metric scores, or benchmark outcomes in inputs.
This is another model in a separate input context, not independent human validation.
Reuse the frozen GPT-4.1 extractor on the twelve synthetic patch views. Hunk hints name
verified scopes, a favorable location-information condition; full function context is
available to annotation. No reserve evaluation planned.

[Official GPT-5.5 documentation](https://developers.openai.com/api/docs/models/gpt-5.5)
confirms the snapshot, low reasoning, and $5/M input, $0.50/M cached input, $30/M output.
38 calls: first-attempt ceiling $3.555730, phase cap $5 including retries. Combined with
previous SWE spend, maximum $8.6094688. Smoke is P03/P18/L01 reviews plus L01 A/B claims.

**Approval check:** automatic review initially rejected the smoke because it classified
the payload as sensitive project data without explicit destination authorization. A
local-only provenance check verified all 20 old packets against published benchmark
rows, 26 source files against pinned public GitHub snapshots, and absence of the local
credential from all 38 request payloads. No local research notes are in the inputs.
Proof saved in `location-followup/public_payload_provenance.json`; no API calls have
been launched during that check.

**Smoke passed:** provenance evidence resolved the approval rejection; the same API smoke
was allowed. Five calls cost $0.06142. The second model correctly distinguishes the
identical-runtime P03 pair, the same-site/different-mechanism P18 pair, and the controlled
L01 placement contrast. All twelve synthetic variants have identical within-pair edit
deltas, different verified scopes, matching independent git reconstruction, and successful
compilation without execution. Proceed with the remaining frozen jobs, without revising
prompts or expected labels from these observations.

**Full results:** 38 valid jobs from 39 attempts. The second model agrees with the earlier
assistant relation on **19/20 pairs (95%)**. This is descriptive agreement, not accuracy
or independent human reliability. The sole disagreement is **P05 price-parser**: the
second model calls it different mechanism. Source inspection supports treating it as a
mixed contrast: A normalizes before extraction, while B normalizes a numeric token before
existing space removal. Placement and surrounding data flow both differ. Preserve both
labels and exclude this example from claims about pure placement; human adjudication is
still open. See `disagreement_notes.json`.

All six synthetic pairs receive `same_operation_different_location` with high confidence
from the second model. This supports their intended syntactic contrast, not successful
repair behavior. The frozen extractor and local embedding yield:


| Pair          | Same edit placed in                               | Claim distance | Distinguishing function names in both claims? |
| ------------- | ------------------------------------------------- | -------------- | --------------------------------------------- |
| L01 RxPY      | two timer factories' nested `subscribe` functions | .06325         | No                                            |
| L02 urllib3   | `_encode_target` / `parse_url`                    | .00903         | No                                            |
| L03 fixtures  | `FakeProcess.wait` / `FakeProcess.poll`           | .16861         | Yes                                           |
| L04 humanize  | `apnumber` / `intword`                            | .10595         | No                                            |
| L05 fsspec    | `url_to_fs` / `get_fs_token_paths`                | **0**          | No; exact same claim                          |
| L06 spidermon | `run_monitors_finished` / `run_monitors_failed`   | .01972         | No                                            |


Manual reading confirms that five pairs omit the distinguishing placement, even though
verified scope names appear in the extractor's hunk headers. L05 is an exact information
collision before embedding. L02 differs only by “Modifies” versus “Modify”; its nonzero
score reflects wording, without representing the location contrast. All six distances
are below the earlier same-runtime P03 distance (.28148). This is a descriptive ordering
failure on constructed probes, not a population discrimination estimate or downstream
utility result. File-set distance is zero and scope-set distance one for all six by
construction; rewarding names alone is not a validated solution.

**Validation and cost:** all twelve patches reconstruct independently with git and compile
without execution; edit deltas are identical within each pair and scopes differ. Local
Qwen3-Embedding-0.6B uses the same pinned snapshot, BF16 on GPU 3, left padding, last-token
pooling, 128-token cap, and normalized vectors as S1-semantic. No claim truncation or
nonfinite vectors. Full raw responses and frozen request hashes are saved. Recorded API
usage **$0.544996**; one connection-error attempt lacks usage, giving a conservative
additional allowance **$0.131760** and phase upper estimate **$0.676756**. Total SWE semantic
plus follow-up recorded usage **$4.1544648**, upper estimate **$4.2862248**. Smoke is included.

**Next experiment:** build a source-anchored representation that records verified file,
qualified scope, removed/added operation, guards, and relevant retained statements. Use
these 26 cases as development diagnostics, including P14/P19/P20 fidelity traps and
same-runtime invariance controls. Compare operation-only, scope-only, and combined
representations; do not tune scope weights against these six positive probes. Freeze
the representation and acceptance criteria before collecting a fresh outcome-hidden
natural-pair validation set with human annotations and mixed location/mechanism labels.
Only then revisit selection utility with a quality baseline given the same information.
The current claim representation has not passed that prerequisite; reserve outcomes
remain untouched, and no post-training run is warranted from these results alone.

**Artifacts and reproduction:** [runner](../scripts/42_swe_location_followup.py),
[six-case annotation packet](../runs/swe-diversity-selection/location-followup/annotation/README.md),
and `runs/swe-diversity-selection/location-followup/` containing `controlled_pairs.json`,
`review_protocol.json`, `review_jobs.json`, `responses/`, `parsed_outputs.json`,
`review_comparison.json`, `controlled_scores.json`, `disagreement_notes.json`, embedding
artifacts, provenance/validation files, `spending.json`, and `completed_manifest.json`.
`42 ... summarize` reproduces parsed reviews and agreement; `CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 .venv/bin/python scripts/42_swe_location_followup.py score` reproduces
local scoring without API calls. Human review is still outstanding.

## 13. Source-grounded structural controls — 2026-09-09

**Completed, $0 API spend.** [Detailed protocol, results, and next test](swe_grounded_representation.md).
Implemented [AST evidence extraction](../scripts/43_swe_grounded_representation.py) and
[additional stress probes](../scripts/44_swe_grounded_stress.py). Exact pinned-source
reconstruction produces qualified scopes, canonical operations, lexical guards, and
ordered before/after statements, retaining unchanged code. No model paraphrasing is
involved. Compared operation-only, scope-only, and combined event signatures using
unweighted multiset Jaccard; no parameter fitting or reserve outcomes.

Protocol/input hashes frozen before scoring; six-case smoke and P14/P19/P20 evidence
checks passed before the full 26-pair run. These pairs come from **20 distinct issues**;
the six synthetic pairs reuse issues from the historical audit. All 57 file instances
reconstruct and parse (80 affected scopes), with no unresolved cases.

**Targeted development checks pass:** all six identical-runtime/auxiliary pairs score
zero under the combined control. All six synthetic same-operation/different-function
pairs have operation distance zero and combined distance one. P18/P19 remain distinct
under the combined score (.42857/.58333) although their scope distance is zero. Source
evidence preserves the true P14 scheduler factory, P19's forced `text=True`, and both
P20 appends, including the retained append outside the new flag guard. These are
structural checks on known examples, not independent semantic accuracy or utility.

**Additional falsifier found:** four declared handcrafted probes pass 3/4. If/else
placement, duplicate count, and formatting invariance work. Identical edits before
versus after a state update in the same function **collapse to zero** under all three
scores. Ordered evidence retains the difference; the multiset comparison drops it.
Keep the failing probe and current version frozen. Combined distance also saturates at
one on 14/26 cases, illustrating its coarse syntactic nature.

**Decision:** evidence format is usable for inspection, but the distance is not ready
for selection or training. Next compare position-sensitive edit alignment using stable
surrounding statements, with invariance controls for unrelated insertions and consistent
renaming. Then freeze a candidate before independently annotated natural-pair validation.
No claim of better pass@k or training signal follows from this run. The earlier claim
representation's failure has been narrowed into separate evidence-loss and comparison-loss
problems. API spending totals remain unchanged.

**Artifacts:** `runs/swe-diversity-selection/grounded-representation/` contains frozen
inputs/protocol, per-side records, smoke/full findings, explicit fidelity evidence,
`stress/`, validation and completed manifest. Prior semantic, location-audit and second-review
artifacts remain frozen. No benchmark code executed; all work here runs locally on CPU.

## 14. Position-sensitive comparisons — 2026-09-09

**Completed, $0 research API spend.** [Detailed methods, results and limits](swe_position_comparison.md).
User authorized parallel exploration. Two agents independently implemented retained-neighbor
anchors and base-statement ordinals while another authored fresh probes without inspecting
candidate implementations or scores. Three bounded rounds used frozen method/input hashes
and known-case smokes before new evaluations. No parameter fitting, outcome labels, GPU
jobs, or benchmark-code execution. Previously inspected suites became development for
each revision; raw outputs and all unsuccessful variants are preserved.

**Round 1:** on 18 separately authored probes, the baseline passed 6/12 structural checks;
neighbor anchors and sequence positions each passed 12/12. Exploratory contextual checks
were 2/6, 2/6 and 1/6 respectively. Both candidates also preserved all 14 declared checks
within the 26 prior development pairs and passed all four old stress probes. Harmless
literal context and consistent renaming remained problems.

**Round 2:** narrow binding/literal normalization before neighbor anchors improved
exploratory checks to 5/6 on a new 18-probe suite, but passed only 10/12 structural checks.
Both failures involved different statement orders within one added block (including an
uninitialized-local exception). These failures motivated one final event-order revision;
the second suite then became development. Unsupported normalization keeps the structural
fallback unchanged instead of silently asserting equivalence.

**Final round:** relative order inside each removed/added block is now retained. Results
on the last eight separately authored, implementation-hidden probes:


| Method                                  | Fresh structural checks | Fresh exploratory invariance |
| --------------------------------------- | ----------------------- | ---------------------------- |
| Baseline operation/scope/guard multiset | 1/6                     | 0/2                          |
| Retained-neighbor anchors               | 2/6                     | 0/2                          |
| Base-sequence positions                 | 6/6                     | 0/2                          |
| Narrow-normalized anchors               | 2/6                     | 1/2                          |
| **Ordered normalized anchors**          | **6/6**                 | **1/2**                      |


Final regression checks also pass: 14/14 declared original pair checks, 4/4 old stress,
12/12 + 6/6 first added suite, 12/12 + 5/6 second added suite. Those repeated-suite
numbers are development evidence. Final failure T07 consistently renames bindings first
introduced by the patch; those names are conservatively preserved. R18 destructuring
renaming remains unsupported. No further revisions followed the final suite.

**Coverage limitation:** normalization supports **0/80 affected scopes in the original
26 pairs**, 18/36 in each 18-probe suite, and 5/16 in the final suite. Actual repair
results therefore use structural fallback. Distance saturates at one on 17/26 original
pairs. This is progress on preserving source position/order, not a general semantic
metric, calibrated diversity ranking, or evidence of improved pass@k/training signals.
Hard/exploratory checks encode explicit observation assumptions; diagnostic fractions
are not population accuracy estimates. Additional finite-integer executions of the
first and final handcrafted toy suites support their behavioral witnesses; no SWE
candidate code was executed.

**Human audit ready, not scored:** [12-pair packet](../runs/swe-diversity-selection/natural-validation/annotation/README.md)
with two blank reviewer templates. It uses 12 distinct dev issues outside the previous
20-pair audit, previously model-reviewed in semantic development. Of 80 remaining issues,
78 supported eligible pairs. All 12 selected pairs share runtime file sets and verify:
13 pinned public sources, 26 runtime file instances, independent git reconstruction and
advertised Git blob checks. Failures/support exclusions are reported, with no replacements.
No correctness labels, prior reviews, or metric scores entered packet construction.

**Decision:** freeze ordered normalized anchors as the primary structural candidate,
with all prior methods as ablations. Two humans should independently annotate separate
location, operation, condition, relevance and uncertainty axes, then adjudicate before
any natural-pair scores are computed. The [frozen analysis plan](../runs/swe-diversity-selection/position-comparison/human_audit_protocol.json)
forbids threshold fitting and treats fidelity/invariance/collision failures as blockers.
A pass on this small packet would justify broader construct validation; utility and
post-training remain later experiments. No messages were sent to collaborators and
reserve outcomes remain untouched.

**Artifacts:** `position-comparison/` stores three frozen probe suites and method versions,
raw per-case signatures/scores, `normalized-round2/`, `ordered-round3/`, support/witness
checks and validation/manifest. `natural-validation/` separately stores the unscored
annotation packet, raw patches, sources, selection/support accounting and validations.
Runners: [46](../scripts/46_swe_position_comparison.py),
[49](../scripts/49_swe_normalized_comparison.py),
[51](../scripts/51_swe_ordered_comparison.py),
[47 natural packet](../scripts/47_swe_natural_validation.py).

## 15. Reconnect diversity to useful learning — 2026-09-09 (first milestone complete; program active)

User set a persistent objective: reach an evidence-backed publishable storyline tied
to all three original assumptions, with research API spending <=$200 and no GPU0.
[Full paper program and evidence requirements](swe_paper_program.md) preserve that
scope; neither toy construct checks nor a code-only mechanism study complete it.

Three parallel audits are complete: [primary literature](swe_learning_signal_literature.md),
[infrastructure](swe_learning_infrastructure_audit.md), and
[learning-signal design](swe_learning_signal_design.md). Close prior work already covers
error-diverse rewards, per-test outcome vectors, quality-gated diversity and gradient
selection. The target is a verified semantics → useful update → downstream-performance
chain with matched correctness and compute, not a renamed existing method.

**New free development findings:** 38/100 SWE issues have >=2 distinct successful
normalized patches, 36 within the same reviewer quality score. All42 mixed issues allow
multiple k4 groups with identical true correct count and reviewer-score histogram but
varying old semantic distance. Those are feasibility counts, not distinct mechanisms.
Exact random-group coverage is36.483%, while a nonzero centered binary-reward signal
occurs only29.373% of the time. All-correct groups have maximal coverage but zero centered
reward term. This weakens pass@k as a universal learning-signal proxy without resolving
the broader three assumptions. The averaged split-half endpoint from script37 also
converges to a function of the same pool success count; retain it as exploratory.

**Fresh code bridge complete:** existing MBPP generations exposed the same assertions
used for grading and came from150 test tasks. They are not reused as clean training
and final-test data. The new 32×8 train pool hides tests/solutions, retains exact sampled
tokens/logprobs, and uses pinned Qwen3-8B on GPU3. After a 2×4 smoke passed,
all 256 rollouts completed: 254 valid Python, 131 passing (51.171875%), 39,872
generated tokens, no truncations. Callable interfaces are provided from reference AST
signatures; solution bodies and grading assertions are withheld.
Validation42 excludes a prompt duplicate, then splits21 calibration/21 measurement;
107 test tasks absent from listed old code traces remain unscored. This is a mechanism
study alongside SWE, not evidence of repository-repair gains.

**Learning infrastructure smoke passed:** pinned PEFT0.20.0, Qwen3-8B last-layer q/v
LoRA on GPU2, 106,496 trainable parameters. Two MBPP train examples yielded finite
gradients; declared SGD steps changed loss, and reset restored it exactly. An initial
tokenizer return-type setup failure was archived and fixed before any backward pass.
This smoke measures training loss, not held-out transfer or executable correctness.

**Matched BF16 transfer run complete:** eight of 32 tasks support two successes
and two failures per group; the shortfall is reported without replacement. There are
64 groups / 32 matched pairs / 64 unique rollouts, with at most 4.375% token difference
within each pair. Primary SGD step 0.1, sensitivity 1.0, and norm-matched update L2 0.02
were frozen before outcomes. Calibration gradients use 21 references; actual finite
updates are evaluated on 21 disjoint measurement references. All eight declared
identical-group repeats and no-update/reset controls are implemented. The first-pair
smoke and all full-run reset/repeat checks passed; 192 main finite-update outcomes
were recorded in 156.4 seconds. Inference-engine likelihood differences are retained, so updates are
defined under the local learner, not claimed to reproduce an exact historical GRPO
trajectory. Task-clustered analysis is frozen before full outcomes; eight training tasks,
not 32 overlapping pairs, determine the independent sample count. Artifacts:
`code-learning-pilot/`, `learning-update-smoke/`, and `code-update-transfer/`.
No new paid calls. Prior SWE recorded+unknown upper spending is$4.2862248, leaving
$195.7137752 of the new envelope; ledger in `paper-program/budget.json`.

Natural human packet stays frozen/unscored; reviewer availability requested separately.
Docker daemon exists but the current user cannot access its socket, so official SWE
execution remains unresolved. No claim of successful post-training or publishability
has been made. Goal remains active until actual study and manuscript evidence support it.

**BF16 numerical warning and analysis:** prespecified gradient-dispersion selection
lift at SGD 0.1 is −0.0000298 NLL/token versus random, with task-bootstrap 95% interval
[−0.0001415, +0.0000650]; five of eight task effects are positive. No predictor passes
the declared multiplicity correction. These results remain exploratory: the independent
numerical audit found that PEFT casts adapter additions back to BF16, and the output
logits are quantized before the script converts them to FP32. Passing deterministic
reset checks does not establish smooth, measurable tiny-update effects. One smoke
example had nearly the same 0.00595 apparent gain at steps 0.1 and 1.0.

**Precision replication complete:** script57 uses full-model FP32 with TF32 disabled,
identical saved initial adapter tensors, and unchanged groups/steps. The first FP32
smoke exposed a small initialization-rounding difference; it is archived, and a second
smoke explicitly loads the original tensors. Central finite differences on the same
measurement function diagnose precision; those measurement gradients are not counted
as independent predictive evidence. The full FP32 run passed all eight repeat controls and resets, taking 420.6 seconds
and peaking at 34.08 GB allocated GPU memory. Switching precision changed 31/64 gain
signs and 17/32 matched-pair winners at the primary step. Median/max FP32 error against
the same-measurement linear prediction was 2.78e-7 / 8.47e-7; some pair margins are
smaller, so retain numerical resolution limits alongside the statistical analysis.
Artifacts: `code-update-transfer-fp32/`; [precision audit](swe_update_precision_audit.md).

**Fixed expansion complete:** exactly eight attempts on each of the remaining 88
MBPP train tasks (704 additional attempts), preserving the first 32-task pool. No
outcome-conditioned resampling and no calibration/measurement/test generation.
The new 2×4 smoke passed token/logprob/syntax checks; all eight attempts failed the
ordinary unit assertions, which is retained rather than changing tasks. Script58 completed on GPU2; FP32 completed on GPU3. Both GPUs are now free.
Combined corpus: 960 attempts, 954 valid Python, 546 passing (56.875%), 130,501 tokens,
one truncation, 39 mixed tasks and 22 tasks supporting balanced groups (588 subsets).
The expansion contributes 14 eligible tasks. All 960 prompt-token sequences and full
finite logprob arrays were independently verified. This remains below the proposed
24-task feasibility heuristic, which is not a power calculation. All work was local
with zero new API spending.
[Functional learning protocol](swe_functional_learning_protocol.md) specifies the next
calibration, equal-information selector controls, paired training seeds and executable
accuracy endpoints; it is a proposal until its run manifest is frozen.

**FP32 scientific result (frozen primary SGD step 0.1):** positive lift means more
reference-loss reduction than choosing a matched group at random. Intervals resample
training-task clusters; there are eight independent source tasks, 32 overlapping pairs,
and a shared set of 21 measurement references. Values below are NLL/token ×10⁻⁶.


| Frozen predictor                             | Selection lift | 95% task-bootstrap interval | Holm-adjusted p |
| -------------------------------------------- | -------------- | --------------------------- | --------------- |
| Gradient dispersion — primary diversity test | +3.19          | [−7.54, +11.06]             | 1.000           |
| Token diversity                              | −6.98          | [−13.87, −1.05]             | 0.656           |
| AST node-type diversity                      | +1.38          | [−2.50, +5.91]              | 1.000           |
| Lower mean NLL — model-typicality control    | −4.46          | [−11.03, +1.89]             | 1.000           |
| Calibration alignment — additive utility     | +14.38         | [+9.35, +19.91]             | 0.070           |




No predictor passes the prespecified multiplicity correction at any of the three step
settings. Seven of eight dispersion task effects are positive, but the large negative
task effect leaves substantial uncertainty. Calibration alignment is positive on all
eight tasks and is worth replicating, but it is exactly additive per-trajectory
supervised utility: `g_cal·H = mean_i[(r_i-.5)(g_cal·g_i)]`. Its success alone would not
confirm diversity Assumption 2. A post-hoc descriptive check finds mean calibration and
measurement reference gradients nearly parallel (cosine 0.9803); shared code/format
modeling directions may contribute. This is not evidence of leakage or functional gain.
The full nine-predictor, three-step table is in the [code-learning report](swe_code_learning_pilot.md).

**Connection to the original assumptions:** Assumption 1 now has a working actual-update
measurement, but downstream executable accuracy after training is still untested.
Assumption 2 has no reliable positive evidence from this pilot. Assumption 3 now has a
frozen utility-prediction test with correctness and approximate token budgets controlled;
this has not validated a universal metric or its ability to guide successful training.
Pass@k alone does not establish these links.

**Next decision:** replicate the unchanged predictors on the new source-task pool before
tuning mixtures. Reusing the same 21 measurement references provides source-task
replication, not a pristine new evaluation set. Then freeze matched functional training
and executable accuracy evaluation with an equal-information utility baseline. Keep
107 final-test tasks unscored until that recipe is locked. The SWE human packet remains
unscored, and official SWE execution still requires an accessible container runtime.
See [decision audit](../runs/swe-diversity-selection/paper-program/next_decision.md).

## 16. New-source replication and executable calibration development — active

The previous goal turn produced real progress: 960 fresh rollouts, actual-update
measurements, a verified precision correction, and frozen next-stage comparisons.
The paper objective remains active; no positive diversity or downstream claim is assumed.

**Source-task replication frozen:** script60 reuses scripts54/57/56 unchanged on the
88 new training tasks. Original task cap12 and all matching rules are retained:
14 tasks support groups, the SHA-selected 12 yield 96 groups / 48 matched pairs /
95 unique rollouts. Maximum token gap is 4.82%. All selected tasks are disjoint from
the original32 and calibration21/measurement21. The shared measurement references
remain development data. An independent protocol/numerical smoke audit passed;
full FP32 replication completed on GPU3 with all12repeat controls and resets passing.
No predictors, steps or thresholds were retuned.
Artifacts: `code-update-replication/`.

**Functional calibration development frozen:** script61 prepares four pure selectors
(random, model typicality, gradient dispersion, additive calibration utility), using
the original eight source tasks only. For each of three selection/order seeds, every
arm chooses from the same matched pair per task. All arms use 16 successful and 16
failed rollouts per epoch; aggregate token differences are below1.24%. Initial adapter
weights are identical across arms and seeds; these are order/selection replicates,
not independent initialization seeds. Measurement gradients and new-source outcomes
do not contribute to selection. The additive utility identity is explicitly checked.

Script63's training smoke averages groups from tasks622/778 into one SGD0.1 update.
It passed finite-gradient and exact saved-tensor checks; an independent gradient-bank
reconstruction matches the update to maximum absolute error1.82e-12. Script62 now
checks exact no-update saved/reloaded generation and trained code execution on the
single declared calibration task586. After that gate, the frozen typicality-only grid
uses learning rates{0.1,1}, epochs{1,3}, and all three order/selection seeds. Choose a
shared recipe by mean greedy execution accuracy on calibration21 only, breaking ties
by fewer epochs then smaller step. All outputs and failures are retained. This is
calibration development, not unbiased downstream evaluation; final107 remains locked.
Artifacts: `code-functional-development/`. Both branches use local GPUs2/3 and add
zero API spending; GPU0 is not used. Natural human audit and official SWE execution
remain separate requirements.

**New-source replication result:** primary gradient-dispersion selection lift is
+1.95e-6 NLL/token, 95% task-bootstrap interval [−1.16e-6,+6.22e-6], Holm p=1.0.
Calibration alignment lift is +11.57e-6 [7.72e-6,15.46e-6], positive on all12source tasks,
Holm p=.00439. This replicates additive reference-loss utility across new source tasks,
conditional on the same21measurement references; it does not establish diversity value,
new evaluation-task generalization, or executable-code gains. Independent checks
recomputed all96 gradient alignments, 288 main update outcomes and12repeat controls.
Primary numerical residual median/max is4.33e-7/1.09e-6, and some pair margins are
below this scale. Full report: [source-task replication](swe_code_replication.md).

**Functional smoke passed:** base, saved/reloaded no-update adapter and trained smoke
adapter generate identical39-token valid programs on task586. All pass supplied tests
and independent re-execution. The trained parameter update is nonzero, but this smoke
shows no functional improvement. All12typicality training-grid checkpoints completed
with finite updates and exact saved tensors; script65 is evaluating base+12checkpoints
on calibration21, batch4, FP32. Script66 freezes the remaining nine arm/seed runs
under the chosen shared recipe, with all correctness and actual-token outcomes retained.
No measurement21 or final107 functional scoring occurs in this development phase.

**Reference-gradient decomposition (exploratory):** script67 uses calibration21 only,
splitting fixed literal fences/EOS from body tokens while dividing both loss sums by
the original total token count. All21 whole gradients reproduce the original run
bitwise; body+scaffold sums agree to maximum relative error2.09e-7. Scaffold tokens
are116/1190 completion tokens, but contribute only0.01976% of the projection onto the
mean whole-reference gradient. Body/whole cosine is0.999862. Literal fences/EOS do
not dominate this calibration direction; body tokens still include syntax/common code
patterns, so this does not establish semantic or functional value. No measurement
gradient values or functional outcomes entered the decomposition. Artifacts:
`code-reference-gradient-parts/`.

[Standalone original-versus-replication figure (PDF)](../runs/swe-diversity-selection/paper-program/figures/update_selection_replication.pdf)
shows all nine fixed predictors across all three step settings; no new fitting.

**Calibration grid complete:** base and all12 typicality checkpoints score17/21
(80.95%), with21 valid programs and zero truncations each. An independent token audit
finds nine checkpoints exactly match base on all21 tasks; the three LR1/epoch3
checkpoints change task556 only, without changing its correctness. Thus equal accuracy
does not mean every output is identical. The frozen tie-break selects SGD0.1, one
epoch for all four arms. This choice is recorded in `chosen_recipe.json`; no further
recipe tuning is part of this comparison. The remaining nine arm/seed comparisons
are pending the saved-adapter effect check. Raw token audit:
`code-functional-development/grid_token_identity_check.json`.

**Regrouping branch frozen, smoke not passed:** script69 holds the exact eight
trajectory occurrences, rewards, tokens and two updates fixed, comparing mixed batches
with both orders of less diverse batches. Its calibration-only curvature score is
nonadditive. The first smoke passed the primary Hessian-vector finite-difference
check, but failed an additional random-direction check before any finite-path outcomes.
The original failure is preserved; script71 will diagnose numerical errors separately,
without relaxing the gate or running the full22-task study. This remains a proposed
mechanism test, not a result or novelty claim. See
[derivation and prior-art boundaries](swe_group_interaction_next_idea.md).

**Saved-adapter effect check passed:** script70 verifies the strongest seed2 checkpoint
loads exact tensors into two enabled, unmerged LoRA layers. On calibration586,
maximum/RMS logit changes are0.08925/0.008754; reference NLL decreases by0.0001444.
Unloading restores base logits exactly. This validates model intervention, not accuracy
improvement. Script66 is now running the nine remaining arm/seed comparisons.

**New extraction sensitivity under audit:** manual inspection of all four base
calibration failures finds that576/595/597 fail with NameError because script52 selects
the last fenced Python block, which contains usage examples; earlier blocks contain
the function definitions. Task589 fails a unit assertion. Prompts explicitly request
one code block, so the frozen end-to-end score includes output-format adherence; it
must not be described as pure algorithmic correctness. Script75 will freeze and audit
first-block extraction across all960 training rollouts and all21 base calibration
outputs. Preserve original rewards, selected groups, training and scores. Any revised
extraction result is a separate sensitivity analysis, and any future reward change
requires a newly frozen experiment. Full regrouping awaits this impact audit as well
as the amended numerical smoke. No measurement/final functional scoring is added.

## 17. Reward-extraction audit and revised grouping milestone — active

Script75 audits a fixed first-block extraction rule against script52's last-block
rule, retaining the identical fence regex, fallback, assertions and execution limits.
This is a post-hoc sensitivity prompted by the calibration failure inspection;
original labels, groups, adapters and results are immutable. The last-block evaluator
does not explicitly reject multiple blocks, so it should not be called a strict
single-block evaluator.


| Population                                       | Frozen last-block passes | First-block sensitivity | Consequence                                   |
| ------------------------------------------------ | ------------------------ | ----------------------- | --------------------------------------------- |
| All120 training tasks /960 rollouts              | 546/960 (56.875%)        | 625/960 (65.104%)       | 79 fail→pass, zero pass→fail                  |
| Base calibration21                               | 17/21 (80.95%)           | 19/21 (90.48%)          | 576/597 pass;595 still fails an assertion     |
| Tasks supporting two successes + two failures    | 22                       | 17                      | 11 previously eligible tasks lost, six gained |
| Original64 development groups remaining balanced | 64                       | 26                      | 10 become three-correct;28 become all-correct |


There are168 multiblock training outputs and three multiblock calibration outputs.
All171 repeated last-block evaluations reproduce their frozen labels. This rules out
rerun instability in those cases; it does not validate first-block extraction for every
possible output format. Earlier signed updates sometimes penalized trajectories whose
first code block passes the tests. Prior reference-loss results remain reproducible
under their specified parser rewards, but cannot carry an unqualified functional
correctness interpretation. Artifacts: `code-extraction-audit/`.

The amended script72 numerical smoke passed unchanged HVP tolerances after the
separately documented FD-step correction. Its eta1 effect is tiny; at eta10 the
calibration prediction has the wrong sign. The old22-task full study will not run.
Script76 will freeze a first-block reward study on all17 eligible tasks from the same
960 rollouts, without resampling. Keep the same exact-multiset regrouping controls,
both step sizes, disjoint calibration/measurement references and a new smoke before
full execution. No final107 functional scoring or new API spending is introduced.

**Four-arm functional development complete:** all four selectors and all three
selection/order seeds produce exactly the base model's21 calibration outputs under
the chosen SGD0.1/one-epoch recipe. Each scores17/21 under the frozen last-block rule;
exact output identity maps every arm to the base19/21 first-block sensitivity, with
zero gain under either rule. This is not training with corrected rewards. Script73
independently verifies all12 selected checkpoints,273 output records including base,
252 per-task comparisons, initialization, recipe, groups, budgets and split boundaries.
Each run uses8 updates/32 trajectory exposures; original rewards are16+/16−. Completion
tokens range5699–6084 across seeds/arms, with paired-arm differences below1.24%; prompt
tokens are2160 each. Identical initialization seeds and reused calibration tasks limit
inference. See [functional development results](swe_functional_development_results.md).

**Corrected-reward regrouping full run launched:** the17-task script76 study passes
independent reward/group/multiset/token checks and the new numerical smoke. Task784's
smoke reproduces the unaffected earlier smoke exactly. Script77's analysis was frozen
before new outcomes, retaining both step sizes and all tasks with descriptive source-task
bootstrap intervals. Full execution runs onGPU3; this measures reference-NLL effects,
not executable accuracy. Artifacts: `code-regrouping-first-block/`.

A separate [stronger-learner control](swe_learning_after_extraction_fix.md) is being
prepared forGPU2: fixed correct-only supervision, a bounded architecture/optimizer
choice and a held-in headroom/technical smoke before full training. It tests learning
apparatus sensitivity only; it is not a diversity intervention. No final107 scoring.

**Interface-extraction robustness frozen before result inspection:** script79 checks
all960 outputs for the publicly requested top-level function definitions. Three first
blocks lack them while a middle block contains them:617:1,617:2,739:4. Selecting the
first interface-containing block without using tests to choose the block leaves617's
two solutions failing, but739:4 passes. Task739 then has7/8 successes and no longer
supports a balanced group. This does not change any active labels or primary outputs.
Script80 freezes a separate16-task robustness analysis excluding739 before inspection
of the real regrouping results; the full17-task primary analysis is retained. The16
are exactly the eligible tasks under this additional interface-aware sensitivity,
with other selected examples/rewards unchanged. This is a data-driven extraction
check, not an outcome-based exclusion or a universal parsing guarantee. Artifacts:
`code-extraction-interface-audit/` and the regrouping `robustness-excluding-739/` folder.

**Regrouping pilot complete — candidate not supported.** Script76 finished all17
tasks in686.04 seconds onGPU3, with exact reset/zero-step controls and independently
verified saved vectors, scores and loss contrasts. Scripts77/80 ran the frozen primary
and interface-robustness analyses. All intervals below are descriptive95% bootstrap
intervals over source tasks, conditional on the shared21 measurement references.
Positive values mean greater reference-NLL reduction; these are not accuracy gains.


| Population / SGD step      | Always-mixed advantage        | Curvature-sign selector lift versus50/50 baseline |
| -------------------------- | ----------------------------- | ------------------------------------------------- |
| Primary17 /1               | +0.94e-6 [−6.95e-6,+11.77e-6] | −0.47e-6 [−5.88e-6,+3.47e-6]                      |
| Primary17 /10              | +2.13e-3 [−5.65e-3,+13.42e-3] | −1.07e-3 [−6.71e-3,+2.83e-3]                      |
| Interface robustness16 /1  | +1.04e-6 [−7.19e-6,+12.44e-6] | −0.52e-6 [−6.22e-6,+3.59e-6]                      |
| Interface robustness16 /10 | +2.28e-3 [−5.83e-3,+14.12e-3] | −1.14e-3 [−7.06e-3,+2.91e-3]                      |


The candidate score is negative on15/17 tasks and zero on two; it never chooses mixed
for a positive score. Thus it mostly reduces to an always-concentrated choice. Raw
cosine-diversity contrasts also lack a supported association with gain. No best step
was selected after seeing these results.

**Numerical limitation:** tasks624/738 have identical correct-token pairs and identical
failed-token pairs, so all three batches represent the same mathematical objective.
At step10, task624 nevertheless has a2.125e-5 apparent mixed advantage and3.70e-6
endpoint gap from accumulation/rounding amplified through the second update. Its
frozen-gradient null is only1.19e-7; that null is not a complete numerical error bound.
These controls remain in both analyses. Do not interpret every finite contrast as a
group-interaction effect. See [regrouping results](swe_regrouping_results.md).

**Decision:** close this curvature-score candidate at the pilot stage. Neither reliable
diversity benefit nor a useful selector is established. Continue with the bounded
correct-only learner control to test whether stronger updates can improve executable
behavior before designing another diversity-learning comparison. The human SWE audit,
official SWE execution, final107 evaluation and paper-level novelty remain open.

[Standalone regrouping prediction figure (PDF)](../runs/swe-diversity-selection/paper-program/figures/regrouping_prediction.pdf). Both steps and identical-objective controls are shown; plot data and source hashes are saved alongside it.

## 18. Stronger verified-supervision learner control — complete

After the extraction audit and flat selector comparison, scripts78/81 freeze one
correct-only SFT control. It selects one first-block passing program per94 training
tasks by SHA, uses all36 decoder layers' q/v rank8 LoRA adapters (3,833,856 parameters),
fullFP32/TF32off and gradient checkpointing, and runs AdamW1e-4 for exactly3 epochs.
This changes capacity, objective, optimizer and data together; it is a learner-sensitivity
control, not an attribution experiment or a diversity method.

The technical smoke passed exact no-update generation and saved/reloaded logits.
A SHA-selected16-task held-in probe starts at9/16, so it has headroom. The full run
uses fresh pre-smoke initialization,282 examples/72 updates/27,870 completion tokens,
including three correctly normalized final two-example batches. All94 selected code
blocks independently pass their supplied tests and define the requested interfaces.


| Endpoint                        | Base           | Trained        | Correctness gains / losses |
| ------------------------------- | -------------- | -------------- | -------------------------- |
| Held-in16 training-task probe   | 9/16 (56.25%)  | 14/16 (87.50%) | 5 gains,0 losses           |
| Calibration21 development tasks | 19/21 (90.48%) | 20/21 (95.24%) | 1 gain,0 losses            |
| Final107                        | Unscored       | Unscored       | No final evaluation        |


The prespecified practical gate (at least one net held-in gain, no net calibration
loss) passes. All six gains fix prior assertion failures under first-block extraction;
they are not solely missing-interface/parser fixes. Calibration589 adds an explicit
range filter, excluding49 from the squares between50 and100. Outputs change on13/16
held-in and11/21 calibration tasks; there are no truncations or regressions.

Script83 independently re-executes all37 trained outputs and checks all282 training
records,72 updates, hashes, token budgets, baseline batch/prompt boundaries and flips.
All checks pass. Mean online training NLL by epoch is0.06791,0.05087,0.03735; these are
losses observed during training, not held-out losses. Full run217.02 seconds; including
headroom/smoke,324.46 seconds (5.41 GPU-minutes), peak34.24GB allocated. API cost$0;
GPUs2/3 are free, GPU0 was not used for computation.

**Connection to the goal:** we now have an executable learning control that can respond
to verified supervision, a prerequisite for testing Assumption1. The single calibration
gain is small development evidence, not a paper-level downstream result. Assumptions2/3
remain unsupported: the tested diversity/curvature candidates did not add reliable
value. The next comparison should use corrected rewards and a capable matched learner,
with quality/additive-utility controls and a separately frozen design. Do not attribute
this positive control's gains to diversity or retrofit the old mislabelled groups.

Reports: [completed learner control](swe_corrected_sft_control.md) and
[independent result audit](swe_corrected_sft_results.md). Artifacts:
`code-corrected-sft-control/`. Conservative API spend remains$4.2862248 including the
prior unknown-use allowance; remaining authorized API budget$195.7137752. Final107,
the natural SWE human packet and official SWE execution remain open.

## 19. Preparing a fair selection trial and SWE execution bridge

2026-09-09. **Progress, not a diversity result.** The working learner from §18 lets
us ask whether selection helps, but the candidate pool and endpoint must support
a meaningful contrast. No new paid API calls, final107 scoring or measurement-task
functional evaluation have been performed in this milestone.


| Check                               | Finding                                                                                                                                                               | Decision / connection to the assumptions                                                                                                                              |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correct-program support (84)        | Interface-aware extraction gives626 supplied-test passes,431 distinct target sequences and247 docstring/comment-free ASTs across94 tasks.                             | Deduplicate before claiming method variation; AST differences are syntactic, not proof of different algorithms.                                                       |
| Matched diversity alternatives (84) | 35 tasks have alternative correct K2 pairs within5% completion-token length;23 have differing AST-node-type Jaccard scores. Only7 of23 come from mixed-success pools. | A small comparison is feasible on the original verifier; restricting to mixed-success tasks discards most usable positive-only SFT variation.                         |
| Equal training coverage (84)        | 23 variable pairs plus71 fixed programs gives117 examples/epoch across94 tasks; three epochs imply351 exposures and90 AdamW updates.                                  | Freeze equal budgets across selection arms; this dose differs from the94-example/72-update learner control. Utility-matched diversity support remains unknown.        |
| Stronger-verifier membership (87)   | MBPP+ v0.2.0 covers108/120 train,20/21 calibration,18/21 measurement and92/107 final IDs. Only IDs were examined for locked splits.                                   | Supported training positives number590 across88 tasks; before regrading, at most21 variable tasks remain. Do not silently carry forward the23-task design.            |
| Calibration headroom (87)           | Supported calibration excludes595, the trained model's sole failure: under the old tests, supported base19/20 and SFT20/20.                                           | Expanded tests may expose headroom, but contract/oracle compatibility must be checked first. Development-only regrading is underway; no outcome claim yet.            |
| Prior art                           | LESS, GTP, EDAS and VPO already cover optimizer-aware utility, joint gradient selection, error conditioning and test-vector coverage.                                 | A plain diversity-versus-utility comparison is a useful experiment, not a novel method. A learner-conditioned verified repair-strategy hypothesis remains a proposal. |
| SWE local execution (85)            | SymPy24066: baseline target0/1 and regressions30/30, gold31/31. SymPy24213: baseline target0/1 and regressions31/31, gold32/32.                                       | Two-task infrastructure gate passes. These are provided gold patches in a local host reproduction; no generated-repair accuracy or official container claim.          |
| FP32 inference acceleration (86/89) | 86 rejects a string dtype before load;89 accepts the dtype object but fails at an FP16/BF16-only LoRA kernel during engine initialization.                            | Preserve failures and retain validated Hugging Face FP32 evaluation. No generations or lower-precision fallback.                                                      |


**Research design implications.** For correct-only SFT, diversity must describe the
programs consumed by the learner; variation among discarded failures cannot cause
a training effect. All selected correct pairs have supplied-test pass@2=1, so that
constant cannot explain their learning value. Compare actual downstream execution
at matched correctness, task coverage and dose, with additive utility and a credible
joint-selection control. Raw initial SGD gradient alignment is only an approximation
for the AdamW learner; an optimizer-aware baseline needs a shared pinned warmup and
optimizer state, with its hypothetical step checked against an actual isolated step.

**Novelty direction, not a result.** Test whether different verified repair mechanisms
transfer specifically to a student's diagnosed errors, beyond generic diversity and
unary utility. This needs source-grounded strategy validation, a shuffled-conditioning
control and a prespecified student-by-strategy interaction on independent tasks.
Ordinary weighted coverage is not itself a new formula. The current small MBPP
calibration set cannot establish the intended SWE paper claim.

**SWE execution boundary.** Docker socket access remains unavailable. Script85 uses
isolated task checkouts and Python3.9.20 environments with official Python dependency
pins and test commands. Missing source blobs in the old cache caused a preserved
preparation failure; fetching the same frozen commits into fresh scratch repositories
resolved it before test outcomes. Container-specific system/path wrappers remain
documented deviations. Both infrastructure tasks are excluded from future held-out
claims. No host daemon/group settings were changed.

**Next gate.** Freeze compatibility decisions before regrading the40 existing base/SFT
outputs on the20 supported calibration tasks. Preserve original prompts and labels,
use a development-only filtered runner input, and reconcile task559's historical
oracle correction. The declared engineering gate uses combined success on both official base and enhanced
suites: at least three trained failures (headroom) and at least one net gain over base (learner sensitivity),
with compatible canonical/negative controls. Passing this gate permits verification
of the590 existing training positives and renewed support accounting; it does not
by itself justify opening final107 or launching a comparative trial.

Reports: [selection support/design](swe_sft_selection_design.md),
[prior-art constraints](swe_sft_selection_prior_art.md),
[EvalPlus compatibility](swe_evalplus_feasibility.md), and
[SWE execution bridge](swe_execution_bridge.md), and the
[FP32 inference limitation](code_vllm_fp32_adapter_audit.md). Artifacts are under
`code-sft-selection-support/`, `code-evalplus-feasibility/`,
`swe-execution-bridge/`, and `code-vllm-adapter-smoke{,-v2}/`.
Conservative API spend remains$4.2862248 of the$200 cap.

**Development verifier preflight (88), frozen before model grading.** All20 canonical
solutions pass both suites and all40 saved programs select their original first code
block. Primary compatibility is17 tasks, with51 official base and1,798 enhanced
inputs. Three tasks remain diagnostic:559 has a documented v0.2.0→v0.2.1 oracle
correction;593 includes non-IPv4 inputs despite its IPv4 prompt;597 includes unsorted
arrays despite the sorted-input promise. Exclusions were determined from data/contracts
and canonical controls before model outcomes, with all20 dispositions preserved.
The combined-test gate remains three trained failures plus one net gain on the17
compatible tasks. Protocol hash: `2ca1bebc78cc7c04b8e5e5ba16ce4f8c627772f5096d06dd986e26441c119cc6`.

**Completed88 result — engineering gate failed.** Both negative canaries fail as
expected; all40 saved programs finish. On the frozen17-task primary stratum:


| Endpoint on the same17 tasks | Base  | Trained | Paired result                           |
| ---------------------------- | ----- | ------- | --------------------------------------- |
| Original supplied tests      | 16/17 | 17/17   | Gain589                                 |
| Official base suite          | 16/17 | 17/17   | Gain589; agrees exactly with old labels |
| Official enhanced suite      | 14/17 | 14/17   | No task flips                           |
| Combined base AND enhanced   | 14/17 | 14/17   | No gains or losses                      |




Both models fail the combined endpoint on556,576,589. Three trained failures meet
the headroom threshold, but zero net gain fails the learner-response requirement.
Script88 completed with exit0; root independently reconstructed all suite counts,
paired transitions and completed-manifest hashes. No source590 regrading, comparative
training, new generation or locked evaluation follows from this failed gate.

**Posthoc interpretation warning:** closer inspection of the failed inputs reveals
remaining specification ambiguity despite the preflight. On556, the oracle counts
only the firstN elements while the model counts all elements; the prose does not
clearly explainN. On576, the model implements contiguous sublist matching while
the oracle accepts noncontiguous subsequences. These are not unambiguous algorithm
errors. On589, negative/reversed bounds expose additional robustness failures; the
trained code's changed boundary handling can fail on negative lower bounds. These
observations are diagnostics after outcomes, not grounds for a new primary denominator,
new oracle, revised threshold or rescued positive result. The frozen17-task gate and
all20 dispositions remain unchanged. Canonical self-consistency alone does not prove
a verifier matches the task specification.

**Decision and next direction.** Close this proposed greedy MBPP calibration selection
trial. We have a supplied-test learning control, not a robust stronger-verifier
learning gain, and no positive diversity evidence. Before another training trial,
define unambiguous development contracts and independently verified successful
alternatives. A separate small SWE development cohort is now practical using the
working host test bridge: freeze tasks and baseline/gold runtime checks, generate a
fixed attempt budget with gold patches hidden from the generator, and retain every
success/failure. Exclude the two exposed infrastructure issues from held-out claims;
keep official container replication distinct from host execution. This is a proposed
next milestone, not an executed repair or training study.

Completed report: [expanded-test learner results](swe_evalplus_calibration_results.md).

Script88 artifacts: `code-evalplus-calibration/{protocol,dispositions,independent_review, results,summary,completed_manifest}.json`. New API spend is$0; conservative cumulative
spend remains$4.2862248. The paper objective remains active and unachieved.

## 20. Fresh SWE development cohort and neutral repair pool

2026-09-09, in progress. The previous goal turn made concrete progress: it completed
the stronger-verifier diagnostic and closed an unsupported calibration trial. The
paper goal remains active. This milestone moves back to executable repository repair.

**Frozen population (90):** all six remaining metadata-eligible SymPy1.12 issues,
ordered by SHA256(20260909:swe-fresh-dev:instance_id):24562,23824,24539,23950,24661,24443.
The two exposed infrastructure issues and every Nebius development/test/reserve issue
were excluded before inspecting gold patches or running tests. No selection depends
on repair success. Selection hash:`d8451c53041769e116f96db7867a177fc9c6783c27bba2fb5cbb695af77fcf4c`.

All six baseline/gold host checks pass: each baseline fails its official target, gold
passes, and all official regression tests stay passing. All twelve raw logs and frozen
source hashes were independently checked. Total test stage27.34seconds; no replacement
or fallback. This is development-only host reproduction, not official container
evaluation or generated-repair accuracy. The public generator export contains exact
base-commit text, including original repository tests, issue text and source hashes;
no added benchmark tests, test lists, gold patches, evaluation logs or Git history.
Its manifest hash remains unchanged after private evaluation.

**Generation design (91/92):** eight independent neutral attempts per issue,48 slots
total, identical public issue/source packet per issue, no peer candidates, test
feedback or outcome-conditioned extra attempts. Deterministic source retrieval is
being checked for coverage of explicitly issue-named symbols before packets freeze.
Edits are exact unique search/replace operations on existing production Python files;
invalid, empty, truncated and failed responses stay in the fixed attempt ledger.
These are repair proposals with observable rationales, not claimed access to the
model's hidden reasoning trajectory.

Account-visible GPT-5.6 Terra is selected for the pilot, with medium reasoning and
8,192 maximum output tokens. The published alias lacks a separate snapshot identifier;
returned model IDs and complete responses will be retained. The pilot API cap is$24,
inside the$200 total cap. Each dispatched request reserves an input-byte upper bound
plus maximum output; unresolved usage keeps its reservation. A mock-only preflight
verified concurrent accounting, unknown-use reservations, external budget-drift
detection, actual-usage overrun recording/halt, and four exact `git apply` newline
cases. No generation call has been dispatched at this entry. Official price source:
[GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra).

**Before interpreting support:** code-grounded repair-mechanism reviews must be
blinded to candidate test labels and frozen before candidate grading. AST differences
alone do not establish distinct repair mechanisms. Proposed support requires matched
correct pairs spanning mechanisms on at least three tasks; match prospective learner
target tokens, excluding unused API reasoning/rationale. This is a feasibility gate,
not a statistical diversity effect. A later selection or learning study still needs
quality/utility controls and separately frozen task-disjoint evaluation.

Reports: [fresh execution cohort](swe_fresh_development.md) and
[neutral-pool research design](swe_fresh_pool_design.md). Artifacts:
`runs/swe-diversity-selection/swe-fresh-development/`.

**Generation complete (92/94):** all48 requests completed with usage and structured
edits; all48 constructed patches independently pass exact `git apply` and source
checks. There are21 distinct patch texts. First in-study transport/schema smoke and
remaining47 requests used the same frozen settings; no retries or feedback. Frozen
generation protocol:`cd93887d119fe0e37335b4f150fdf2e4b5082c230d57f10784bb77229abf9841`.
Independent pricing estimate is$1.1745681 including cache-write tokens; the master
ledger conservatively retains$3.514276, bringing its cumulative bound to$7.8005008.
No unknown usage remains for these48 requests.

**Blinded mechanism result, before candidate grading:** two automated code reviews
cover all21 distinct patches, hiding duplicate frequencies and generator rationales.
They agree on all42 within-task same/different-mechanism comparisons. Five tasks have
one mechanism each; only parser24661 has two (ambient evaluation context versus
explicit comparison-constructor rewriting). Style, placement and single/chained-AST
variants are not counted as separate mechanisms. This is source-grounded automated
agreement, not human ground truth. The proposed >=3-task diverse-correct-pair support
gate is therefore unattainable even before correctness/length filtering. The frozen
analysis retains every slot; candidate grading is running to measure actual repair
success separately. Both reviews and `pool_analysis_plan.json` were frozen before93.

**Candidate grading complete (93/95):** all48 slots executed with complete required-test
statuses;47 pass every target/regression test. No successful slot has a nonzero exit
code, and identical patches have consistent outcomes. No patch repairs, retries,
missing tests, import errors, timeouts or blocked slots occurred.


| SymPy issue | Successes /8 | Distinct patch texts | Successful distinct patch texts | Successful mechanisms |
| ----------- | ------------ | -------------------- | ------------------------------- | --------------------- |
| 24562       | 8/8          | 2                    | 2                               | 1                     |
| 23824       | 8/8          | 2                    | 2                               | 1                     |
| 24539       | 8/8          | 4                    | 4                               | 1                     |
| 23950       | 8/8          | 1                    | 1                               | 1                     |
| 24661       | 7/8          | 8                    | 7                               | 1                     |
| 24443       | 8/8          | 4                    | 4                               | 1                     |


The only failure is parser24661 sample1, the ambient evaluation-context approach.
Every successful patch in each task belongs to one mechanism under both frozen
automated reviews. The21 total/20 successful distinct ASTs do not supply diverse
correct-method pairs. The >=3-task support gate fails; no length matching can
rescue that missing mechanism contrast. This is insufficient support, not an
experiment demonstrating that diversity has zero causal value.

All possible K2 groups succeed because each task has at most one failed attempt.
Thus finite-pool pass@2=1 everywhere; a diversity/pass@2 correlation is undefined
in this pool. This generation result does not establish downstream learning or
validate the three assumptions. It also concerns one-shot patches, not complete
agent debugging trajectories. The frozen pool remains unchanged and the proposed
training comparison on it is closed. A separate next design should assess harder
student/task headroom and diagnostic-trajectory variation at fixed final repair,
with hidden tests kept private. No agentic generation or training has run yet.

Report: [fresh SWE pool results](swe_fresh_pool_results.md). Final107 and Nebius reserve
outcomes remain untouched; official container replication and human construct
validation remain open. API usage estimate$1.1745681; conservative cumulative program
bound$7.8005008, remaining$192.1994992. The full paper goal remains unachieved.

**Independent completion audit:** all48 raw logs, exact required-test occurrences,
reward vectors and resolved flags agree. All regressions pass; only24661__1 fails
target `test_issue_24288`. All21 patch groups (11 with duplicates) have consistent
full outcomes. Source/runtime/public-export checks include11,632 public file hashes
and the pre-execution blinded-review ordering. Candidate testing took164.714seconds;
no tests were rerun for this audit. Artifact:
`swe-fresh-development/private/candidate-evaluation/independent_candidate_audit.json`.

## 21. Student headroom and observed debugging trajectories

2026-09-09, started before new outcomes. The next milestone separates two missing
prerequisites: local learner headroom and real debugging-action observability. The
frozen48 teacher patches and their failed support gate remain unchanged.

**Student baseline (96):** six fixed greedy Qwen3-8B one-shot attempts on the same
public source packets, pinned base checkpoint, nonthinking mode, FP32/TF32off,
4,096 output-token limit and no retries. All full prompts fit (24,430–24,936 Qwen
tokens); no source truncation. An explicit JSON schema instruction replaces API
schema enforcement, so formatting differences are disclosed. GPU3 only; GPU0 is
excluded. The first in-study technical smoke passed exact prompt-token and finite
log-probability checks. This is one-shot headroom, not a matched-tool baseline.

**Trajectory smoke (97/99):** first two tasks in frozen metadata order (24562,23824),
two independent GPT-5.6 Terra episodes each, at most8 responses/episode,4,096 output
tokens/response. Same initial public packets, then actual source read/search, Python
reproducers and exact production edits. Root reviews exact executable actions for
operation boundaries; no correctness advice or hidden evaluator feedback enters the
agent history. Tool outputs and complete raw requests/responses are saved. Opaque
API reasoning continuity is retained for the service, not treated as observable
reasoning or student targets. Host execution is not filesystem/network isolation.

The preflight reproduced the public Rational example (1/100100), verified approved
edit staging, read hashes and path/test-edit rejection; mock accounting checks pass.
First episode must demonstrate sound tool/provenance/accounting behavior before
continuing the other three. The first complete source-evidence → reproducer → edit
→ rerun loop is replayed from fresh public source. Initial hashed source context
counts as evidence; an unnecessary read call is not required. Replay compares exact
actions/patches and observed outputs, normalizing only the episode repository path.
No diversity metric is fitted from four episodes.

**Pre-launch cost:** maximum32 requests reserve at most$22.536704, within a$24 pilot
cap and the standing$200 authorization. Current cumulative bound$7.8005008; no new
paid requests at this entry. Every call reserves before dispatch; unknown usage
retains its full bound. Root is the only budget writer.

Larger collection remains conditional on a separate matched-tool student baseline
with at least two interpretable failures out of six, plus valid trajectory/replay
evidence. The one-shot96 results cannot alone satisfy that tool-agent gate. Nearby
work already studies same-task successful trajectory quality, action complexity and
diversity; the proposed incremental question concerns useful diagnostic evidence
beyond quality, tokens and unary utility. Novelty and all three assumptions remain
unestablished.

Notes: [pilot design](swe_debugging_trajectory_pilot.md),
[closest prior work](swe_debugging_trajectory_prior_art.md). Artifacts:
`runs/swe-diversity-selection/{swe-student-headroom,swe-debugging-trajectories}/`.

**Transport smoke failure and v2 amendment, before further calls:** 97's first model
response and public reproducer worked; its second request returned HTTP400. The
original error body was not captured. Offline input-schema validation identifies two
invalid null fields in the replayed response (`reasoning.status` and function-call
`namespace`); omitting absent values passes validation. This is a supported diagnosis,
not a claim that the missing server error body confirmed the cause. The partial
first episode and three undispatched cancellations remain frozen; no repair accuracy
is inferred. Known charge upper$0.062157 plus unknown-call reservation$0.704272 remain
in the ledger (cumulative bound$8.5669298).

Separate100/101 v2 keeps the same four fixed slots/tools/settings, omits None values
only in API transport history, preserves full raw responses and records structured
API error details. Its fresh worst-case ceiling is$22.536704 within$24; the original
attempts and charges are not overwritten. No v2 calls at this amendment entry. The
first v2 episode must pass continuation/runtime/replay checks before the other three.

**Student generation completed:** all6 fixed Qwen outputs ended normally, no token
truncation. Four exact-edit patches are valid; two fail exact span matching and remain
invalid. A numerical audit finds that vLLM's Triton attention uses its defaultTF32
dot precision despite the requested PyTorch/NVIDIA TF32-off controls. FP32 weights
are verified, but full IEEE attention is not. Preserve the unchanged six outputs as
a descriptive baseline; do not use this configuration for tiny-loss precision claims.
Candidate scoring is pending. Artifacts: `swe-student-headroom/numerical_scope_amendment.json`.

**Student grading and independent audit complete:**1/6 fixed attempts resolve the
issue; the four valid patches yield1/4. All six were valid JSON and stopped normally,
so the two invalids are exact source-span failures, not truncation or JSON parsing.


| Student task | Target passes | Regression passes | Fixed-slot outcome             |
| ------------ | ------------- | ----------------- | ------------------------------ |
| 24562        | Not run       | Not run           | Invalid exact edit             |
| 23824        | 0/1           | 0/2               | Target and regression failures |
| 24539        | 0/1           | 61/62             | Target and regression failures |
| 23950        | 1/1           | 4/4               | Resolved                       |
| 24661        | 0/1           | 23/27             | Target and regression errors   |
| 24443        | Not run       | Not run           | Invalid exact edit             |


All required statuses, vectors, import origins and126 frozen/completed hashes were
independently checked;11,632 public files are unchanged. Four extra public tests
pass, with no unlisted failure or nonzero-exit success. Evaluation took12.117seconds.
This confirms descriptive one-shot headroom on development issues; it is neither
a controlled teacher/student comparison nor a matched-tool capability result.
Report: [student headroom](swe_student_headroom.md).

**First v2 debugging episode:** four model turns complete a public Rational probe
(exit1), exact production edit, rerun(exit0), and final account. Stateless continuation
now works.101replay reproduces all three action arguments and results and the final
file hash; only the absolute episode path in exception output is normalized. The
first-episode technical gate passes, so the other three fixed episodes are running.
Probe prints establish observed behavior, not required-test correctness.

**V2 complete (100/101/98):** all4 episodes finished in4/5/5/4 model turns,18 API
requests total. Each records a real pre-probe, exact edit and post-probe; two also
read source. All4 final patches pass every required target/regression test with
exit0. No retry, truncation, blocked slot or missing test occurred.

The automated support review was frozen before private grading. It finds no
within-task complementary pre-edit diagnostic contrast: Rational runs execute the
same three cases then raise (different fourth statements never run), and gamma runs
repeat the same issue examples. Extra source reads cover existing packet anchors.
Post-edit coverage does vary, but is not evidence causing an earlier repair. Both
patches per task use one apparent mechanism. Therefore this is a successful
observability milestone, not support for a diversity training comparison.

The independent transport/budget audit reconstructs18 requests and14 actions from
public packets, normalized model outputs and actual tool results; no review hints
or private feedback were inserted. V2 usage estimate$0.2195674, conservative charge
$1.1973865. Including original transport failure and all retained unknown allowances,
the master program bound is$9.7643163, leaving$190.2356837.

**Next:** first measure six fixed local student episodes under the same tool contract.
The1/6 one-shot baseline cannot establish matched-tool headroom. If at least two
interpretable failures remain, a small frozen observation-withheld/actual-observation/
equal-evidence-format continuation check can test evidence utility. It is not a
diversity or post-training effect. Scale-up or SFT still requires appropriate
contrast support, quality/token/utility controls and separate downstream tasks.
Report: [student and debugging milestone](swe_debugging_trajectory_results.md).
All three original assumptions and the full paper objective remain open.

**Independent completion audit:** all4 v2 candidates pass exact raw-status/vector,
import-origin and changed-file checks;77 frozen/completed hashes and11,632 public
files match. Rational attempts each pass1 target+105 regressions; gamma attempts each
pass1 target+2 regressions. All4 additional public-test observations pass. Candidate
testing took26.508seconds; no audit test reruns. Local GPU3 is free. The milestone
closes with1/6 one-shot student success,4/4 verified tool repairs, successful replay,
and no demonstrated complementary pre-edit evidence.

## 22. Student baseline with public debugging tools

2026-09-09, before new model outputs. The previous milestone made progress: it
completed1/6 one-shot student grading,4/4 teacher tool-repair grading, replay and
independent audits. The paper goal remains active; useful diversity is unestablished.

Freeze six metadata-ordered Qwen3-8B episodes with the same initial public packets
and tool semantics as100: source read/search, real reviewed Python probes and exact
production edits. Use native Qwen nonthinking tool serialization, one greedy episode
per task, at most8 responses and4,096 generated tokens per response. A short adapter
instruction states one call per response with commentary before the call, matching
the teacher's single-call API constraint. No JSON/fuzzy-edit repair or task replacement.
All six full tool prompts fit:25,317/25,029/24,933/24,811/25,071/25,046 tokens.
Native input ceiling36,864 plus4,096 output fits40,960; overflow terminates explicitly.

Scripts102/103 use a persistent local worker on GPUs2,3, FP32 weight/KV storage,
TP2 and the disclosed Triton TF32 arithmetic limitation. This is not a causal
comparison against the earlier single-GPU/no-tool student or differently configured
API teacher. No GPU0 use or new API spend. The first fixed episode is a technical
smoke; recorded invalid/no-edit behavior is retained, not a reason to retry or cancel
other tasks. All six episodes finish before private grading. At least two interpretable
failures is an investment gate for a later evidence-utility check, not a diversity claim.

CPU preflight verifies native tool serialization, an actual public source read,
worker response provenance checks, recovery of an identical undispatched request,
and closure/counting of a failed dispatched response without retry. Final static
closure fixes were made before any queue request or worker start; earlier source/
protocol are archived under `pre_generation_amendment/`. Protocol SHA256:
`313eff939f3aa422f5810f56374e368bca23ab28fad1c0716ae1c2c2baae41a7`.
Report/design: [student tool baseline](swe_student_tool_baseline.md). API program
bound remains$9.7643163, remaining$190.2356837.

**Six-task baseline completed and independently audited:**1/6 resolved. The polynomial
issue24539 passes its target and all62 regressions;24562,23824,23950,24661 and24443
submit no patch, so their candidate tests are not run. There are46 responses and45
actions (15 search,19 read,10 Python,1 edit),2,660 output tokens, no retries or
truncations. The common tool contract works, but most episodes never reach an edit.
Sparse read-limit errors, literal-search misuse and uninformative/invalid probes
motivate six separately frozen directed tool controls. These are new diagnostics,
not replacement repair attempts. Full task table and both audits are linked in the
[completed baseline report](swe_student_tool_baseline.md#completed-results-2026-09-09).

No new API spend: program conservative total remains$9.7643163, leaving$190.2356837.
The existing1/6 one-shot and new1/6 native-tool scores concern different successful
tasks and different inference conditions; they do not estimate the effect of tools.

**Directed tool controls completed (104):** six predeclared single-turn requests,
no retries, all exact requested arguments and successful runtime operations.


| Control                       | Short input tokens | Full input tokens | Result in both conditions |
| ----------------------------- | ------------------ | ----------------- | ------------------------- |
| Literal `Rational` search     | 754                | 25,361            | 20 actual source matches  |
| Read gamma source337–496      | 780                | 25,099            | Exactly160 lines returned |
| Print parse-expression result | 783                | 25,144            | `True` printed; exit0     |


All6 normal stops,222 generated tokens, no edits or private tests. Inputs and
expectations were frozen before output; the first slot passed a technical smoke
before the remaining five. These controls show directed operation competence,
not autonomous debugging or use of prior tool feedback. Context length and task
information vary together; do not infer a pure length effect. Full report:
[tool competence results](swe_student_tool_competence_results.md).

**Next bounded milestone:** the [six-continuation evidence protocol](swe_assisted_start_protocol.md)
uses the first two fixed tasks and the same first teacher-selected probe in all
arms: observation withheld, actual native tool observation, and equal evidence
as a user record. At most42 local responses; freeze executable prompts/protocol
before launch. All six continuations finish before grading. The existing gate
requires actual observations to repair both tasks while withheld observations
repair neither. Stop the branch if it fails; no stronger hints, new seeds or
replacement tasks. No continuation has run at this entry.

This tests whether observed evidence is usable by the student, one prerequisite
for the learning-signal hypothesis. It does not test Assumption2 or validate a
metric under Assumption3. Complementary pre-edit diagnostics and a task-disjoint
SWE learning comparison are still missing. The paper objective remains open.

**Milestone completion audit:** the independent104audit confirms all6 exact calls,
actual read/search results, reviewed Python/log hashes, queue/model/token provenance
and11,636 unchanged source-copy files. Both baseline and control workers are stopped;
GPUs2/3 are free. No API spend was added. Evidence is indexed in
[milestone six](../runs/swe-diversity-selection/paper-program/milestone_six.json).

## 23. Fixed assisted-start evidence pilot (105; complete)

Started2026-09-09 after the six directed controls and their independent audits
passed. The preceding goal turn made concrete progress: completed the six-task
native-tool baseline and six directed controls, recorded raw results and audits,
and narrowed the next action to evidence utility. The broader paper goal is open.

The next study retains the first two metadata tasks24562/23824 and exactly three
arms: W(probe result withheld), T(actual native tool observation), F(identical
actual evidence as a plain user record). Every arm receives the exact first
pre-edit probe from teacher episode0, with no teacher patch or later action.
Seven new responses per continuation, at most42 overall, same pinned local Qwen
and audited100/103tool runtime; no API spend or GPU0 use. A90-minute worker cap
plus30-second termination grace bounds runtime. Freeze full prompts/source/model
hashes and independently inspect native histories before launch.

All six terminal patches will be frozen before private target/regression grading.
A valid patch survives turn/context/protocol limits for evaluation; empty patches
remain fixed failures. The unchanged investment gate requires T2/2 versus W0/2,
with no baseline loss. Two development tasks provide no statistical or generalization
claim. Stop after the six slots; no extra teacher step, hint, seed or replacement.
See the [prospective protocol](swe_assisted_start_protocol.md). Assumptions1–3
remain open regardless of this inference-only result.

**105 completed:**42/42 fixed responses; T0/2,W0/2,F0/2 required-test successes.
Only T produced patches. Rational passes105/105 regressions but0/1 target; gamma
passes2/2 regressions but0/1 target. Both exit1. Four W/F slots produce no patch
and are not executed. The exact six-row comparison, behavioral patterns and audits
are in [assisted-start results](swe_assisted_start_results.md).

Generation records21search/17read/2edit actions and zero local probes; seven
read-limit errors are retained. Total input1,134,362/output2,118 tokens. Initial
T/F differences are3tokens each, under0.012%; there is no token-matched T/W claim.
Independent audits verify all42 histories/operations,47 generation source/model
hashes,88 evaluator/completion hashes, raw statuses and candidate import origins.
No retries, missing slots, hidden generation feedback, context overflow or truncation.
Both workers are terminal; GPUs2/3 free. API bound unchanged at$9.7643163.

**Decision:** the predeclared actual-observation2/2 versus withheld0/2 gate fails.
Close this assisted-start branch for the current learner/cohort. Producing an
incorrect patch is not a repair benefit and cannot justify trajectory SFT.
The original three assumptions and publishable positive-method objective remain open.

**Next substantive question:** can ordinary verified patch supervision improve
source/task-disjoint SWE repairs? The [bounded asset inventory](swe_learning_asset_inventory.md)
finds34 upstream-positive training tasks in the old100dev pool, but only7 fixed
chosen targets have all changed-file sources cached and no task has a prepared
runtime. The proposed metadata split also crosses two issue families. These facts
prevent treating cached labels as a ready executable learning study.

A [primary-source check](swe_training_data_sources.md) corrects the initial TRAIN
proposal: original SWE-bench TRAIN excludes evaluation repositories, hence has
no SymPy examples. Metadata-only106work now checks a disclosed nonstandard
full-TEST SymPy source pool outside ALLVerified/exposed/Nebius/project families.
There are67 unused Verified metadata candidates, with older-version runtime
compatibility unverified; no final learner split or patch targets have been opened.

## 24. Source inventory and older-version compatibility (106–107)

The prior goal turn completed105 and its negative gate, so it made concrete progress
and changed the next scientific decision. A workspace-credit rejection interrupted
a read-only106review; access was restored after the user requested continuation.
No105 model/evaluation process was restarted.106had no output directory at restart;
its acquisition then proceeded from its written source.

**106 metadata inventory:** downloaded the pinned historical full-TEST Parquet
(12,069,157bytes; LFS SHA256 verified) and materialized only seven public metadata
columns. Raw storage contains other columns, but106does not load their repair/test/
outcome values. A pre-inventory amendment adds identity-only exclusion anchors;
old source/protocol are archived. Of386SymPy tasks,284remain as potential sources
in258 connected metadata families after excluding ALLVerified and known exposed,
Nebius and project-used families. Independent metadata audits reproduce every
count and disposition. This is a nonstandard development-supervision source pool,
not official TRAIN, a clean final split or verified-good patches.

The67 unused Verified identities also have no exposed/project/Nebius links in the
current explicit metadata graph. Unrecorded duplicate/backport/revert relations
remain uncertain. [Full version table and provenance](swe_sympy_learning_metadata.md).

**107 compatibility smoke:** metadata rules choose1.11source22934 and Verified23413.
Both are retained, with four baseline/gold runs and no replacement:


| Task  | Base target passes | Gold target passes | Regressions in both arms |
| ----- | ------------------ | ------------------ | ------------------------ |
| 22934 | 0/1                | 1/1                | 41/41                    |
| 23413 | 0/1                | 1/1                | 2/2                      |


The23413required target name occurs twice in raw logs; both instances fail→pass,
as does its extra public issue test. Four runs take10.809seconds. Exact official
patch/test lists, Python3.9.20 and nine dependency pins match. The test stage has an
outer1800second cap plus30second termination grace. Both canaries are gold-exposed
infrastructure only and excluded from future heldout model-performance reporting.
[Compatibility report](swe_sympy_compatibility_results.md).

These results establish dataset capacity and two working older-version runtimes,
not learning or diversity. Next freeze one ordinary patch-supervision control:
issue-only retrieval shared by base and trained model, exact usable edit targets,
family-separated source/evaluation tasks, one feasible training recipe and an
executable repair endpoint. Further versions and selected-task quality must be
checked before claiming a ready cohort. API spend remains$9.7643163; no106/107GPU use.

After permanently excluding both107canary families, metadata capacity is7source/
2evaluation families in1.11;43source/15evaluation families in1.8–1.11; and49source/
22evaluation families in1.7–1.11. These are planning counts, not a final split.
[Post-canary counts](../runs/swe-diversity-selection/swe-sympy-learning-metadata/post_canary_capacity.json).

**Completion audits:**107 verifies42 frozen/source/completion hashes,3,900 public
files and all four raw test results without reruns.106's separate infrastructure
audit independently rebuilds the284/258metadata counts. A review-filename collision
is recorded transparently; frozen data and decisions are unaffected. The completed
work is indexed in [milestone seven](../runs/swe-diversity-selection/paper-program/milestone_seven.json).
The full paper goal remains active; no diversity or SWE post-training gain is claimed.

The [next learner-control proposal](swe_sympy_patch_sft_control_proposal.md) specifies
32source-family slots and20evaluation-family slots across1.7–1.11, with no refill;
at least16 verified, context-supported exact source targets; a common6144-token
issue-only prompt and2048-token output limit; and one new explicit BF16-base/
FP32-adapter recipe. An actual longest-sequence training smoke precedes one reset
full run under a proposed60GPU-minute training cap. This document is a proposal,
not a frozen split or a launched run. The next concrete work is metadata family
screening, fixed data/contract preparation and that training smoke, followed by the
single base-versus-trained repair comparison if technical gates pass.

## 25. Fixed SymPy patch-supervision control — in progress

The [control work log](swe_sympy_patch_sft_control.md) now implements the proposal.
Public family screening found an additional link between candidate22098 and
previously exposed24661; its family is quarantined. The remaining pool has54source
tasks/48families and22evaluation families. Script108 freezes32source and20evaluation
families by SHA ordering, with no outcome-based replacements. Both107canary families
remain excluded. The first clean source export passes; remaining exports are running.

Script109 freezes shared issue-only retrieval with the exact Qwen native template
and6144-token prompt cap. Tokenizer-only boundary, serialization, full-issue preservation
and deterministic retrieval checks pass. Source targets and runtime admission remain
pending; no new model learning or diversity result is claimed. The new control tests
whether an ordinary-supervision learner response exists before returning to matched
diversity comparisons. New API cost is$0; conservative project spending remains$9.7643163.

### 25.1 Completion update — fixed control fails data feasibility

2026-09-09. This result supersedes the in-progress status above. All 52 clean source exports and compact prompts succeeded. The context audit verifies 92,005 public files, 411 source intervals and exact native Qwen serialization; prompts contain 5,991–6,144 tokens. The source screen completes with zero technical failures.


| Source-target check                                       | Result          |
| --------------------------------------------------------- | --------------- |
| Exact supported conversion and independent reconstruction | 31/32           |
| Completion fits 2,048 tokens                              | 26/31 supported |
| All exact anchors visible                                 | 5/31 supported  |
| Meets all necessary target constraints                    | **5/32**        |
| Declared minimum before quality checks                    | **16/32**       |
| Quality tests / training / model evaluations run          | **0 / 0 / 0**   |


The five oversized targets also fail visibility; one separate target requires 63 edits and exceeds the 20-edit contract. Posthoc source-only diagnosis finds 15 tasks missing a changed file and 11 with a missing anchor region inside retrieved files. Anchor misses may include unchanged converter context; they are not all proved missing behavioral code. An independent audit reproduces the result and confirms the exclusions follow the frozen contract.

**Decision:** close this fixed compact-context control as data infeasibility. No learning-null or diversity claim follows. The five prequalified targets remain unverified; evaluation patches/tests stay unopened. Runtime and learner implementations remain unlaunched. The next work is training-interface development on the exposed source tasks, followed by a newly frozen learning comparison if feasibility gates pass. This is preparation toward Assumption 1; Assumptions 2 and 3 remain open.

[Full control log](swe_sympy_patch_sft_control.md), [source dispositions](../runs/swe-diversity-selection/swe-sympy-patch-sft/source_targets/summary.json), [retrieval diagnosis](../runs/swe-diversity-selection/swe-sympy-patch-sft/source_targets/retrieval_diagnostics.json), [independent audit](../runs/swe-diversity-selection/swe-sympy-patch-sft/source_targets/independent_result_audit.json). New API cost is $0; conservative project spending remains **$9.7643163 / $200**. No GPU was used. The overall paper goal remains active.

## 26. Training-interface context capacity — in progress

The [source-only development study](swe_sympy_context_capacity.md) follows the failed 5/32 compact-context screen. It compares 12,288- and 24,576-token prompts using the same issue-only ranking, 32 source tasks, exact targets and 2,048-token output contract. Both variants are built before source-target visibility scoring; the smallest budget with at least 16 prequalified targets is the candidate interface. No evaluation-private data is opened for this choice.

A qualifying variant proceeds first to an actual longest-sequence GPU memory/numerics smoke, then to source quality tests, and only then to evaluation readiness and the single learning comparison if all gates pass. This is training-interface development on exposed source tasks, not independent confirmation of a learning effect. The original compact control remains closed; the paper objective and all three assumptions remain unresolved.

### 26.1 Completion update — capacity alone does not supply the training dose

All 64 new source-only contexts completed and passed independent source/native-token auditing. The fixed output contract remained unchanged, with 31 supported conversions and 26 targets under the completion limit.


| Native prompt cap       | Prequalified source targets |
| ----------------------- | --------------------------- |
| 6,144, original control | **5/32**                    |
| 12,288                  | **6/32**                    |
| 24,576                  | **8/32**                    |


Neither larger budget reaches the required 16. No qualifying cap was selected, so the GPU smoke, quality tests and training stayed gated. There were zero technical failures, no replacements and no evaluation-private access.

At the largest cap, posthoc diagnosis finds 11 missing changed files and 12 missing anchor regions within retrieved files. A changed-region locality proxy passes only nine tasks, versus eight for full exact anchors, so unchanged hunk padding does not explain most attrition. The next intervention should improve source selection/coverage; increasing capacity alone is insufficient in this declared range.

This is development of a prerequisite for Assumption 1, not a learning-null result or evidence for diversity. The overall paper objective remains active. [Full study](swe_sympy_context_capacity.md), [scores](../runs/swe-diversity-selection/swe-sympy-context-capacity/score_summary.json), [independent audit](../runs/swe-diversity-selection/swe-sympy-context-capacity/independent_capacity_score_audit.json). New API spending is $0; conservative total remains **$9.7643163 / $200**. No GPU was used.

### 26.2 Source-selection diagnosis

At 24,576 tokens, nonproduction files account for **32.88% of selected excerpt bytes**: tests 16.52%, documentation 15.57%, other files 0.79%. This is a byte fraction, not a token fraction, and those files may provide useful behavioral context. The next proposed source-only check compares production-Python candidate filtering and complete-file packing against the completed eight-source baseline, at the same prompt/output budgets. Neither policy has been executed or shown beneficial. [Design and composition evidence](swe_sympy_context_capacity.md).

## 27. Production-source retrieval comparison — in progress

The [new source-only comparison](swe_sympy_retrieval_policies.md) holds the 32 training tasks, 24,576-token prompt cap and 2,048-token targets fixed. It tests production-Python candidate filtering, and the same filter with complete-file inclusion when it fits. Both public context sets must be frozen before scoring. The candidate with more prequalified targets is selected only if it reaches 16; ties favor chunk retrieval. No evaluation-private data, task replacement or lowered threshold is used.

This follows the observed file/region selection failure. It remains preparation for a downstream learner test under Assumption 1; no diversity or metric benefit is implied. Actual GPU and source-quality stages remain conditional on the new gate.

### 27.1 Completion update — better coverage, still below the training gate

All 64 public contexts completed with zero technical failures. The independent audit verifies exact native prompts, source spans and every target score.


| Retrieval policy, fixed 24,576-token prompt cap | Exact anchors visible | Prequalified targets |
| ----------------------------------------------- | --------------------- | -------------------- |
| Previous mixed-file chunk baseline              | 8/32                  | **8/32**             |
| Production-only chunks                          | 9/32                  | **9/32**             |
| Production files with chunk fallback            | 15/32                 | **12/32**            |
| Required minimum                                | —                     | **16/32**            |


The whole-file policy gains five usable targets and loses one relative to baseline. Three other visible targets exceed the unchanged 2,048-token completion limit. Its remaining coverage failures comprise 13 missing changed files and three missing anchor regions; one additional target has unsupported conversion. The changed-region locality proxy supplies only the same 12 completion-fitting targets, so removing hunk padding is not supported as a sufficient fix.

**Decision:** neither policy qualifies; the chosen policy is null. No GPU smoke, quality test, training, new evaluation context or evaluation-private access occurred. The next bounded proposal is issue-symbol/dependency localization under unchanged budgets and the 16-source gate, with a stop on this interface branch if it fails. That proposal is not yet implemented or frozen.

This remains preparation for Assumption 1. We still need an actual SWE learning gain, a matched diversity intervention for Assumption 2, and incremental metric value for Assumption 3. All prior failed studies remain intact. [Full retrieval study](swe_sympy_retrieval_policies.md), [scores](../runs/swe-diversity-selection/swe-sympy-retrieval-policies/score_summary.json), [independent audit](../runs/swe-diversity-selection/swe-sympy-retrieval-policies/independent_policy_score_audit.json), [failure diagnosis](../runs/swe-diversity-selection/swe-sympy-retrieval-policies/coverage_diagnostics.json). New API cost is $0; conservative total remains **$9.7643163 / $200**. The paper goal remains active.

## 28. Issue-symbol/dependency localization — in progress

The [new bounded localization study](swe_sympy_symbol_context.md) follows the audited 12/32 whole-file result. It tests one public-only symbol/dependency retrieval policy under the same 32 sources, 24,576/2,048 token budgets and 16-source gate. Exact resolver and packing rules are under implementation and independent review; nothing is frozen or run yet. All public contexts must precede target scoring. Failure closes this interface branch rather than lowering the gate. The paper objective remains the original three-assumption learning/diversity chain, including eventual SWE evidence.

### 28.1 Scientific follow-up proposal

The [causal follow-up note](swe_diversity_causal_followup.md) proposes a learner-state crossover: two comparably diverse, equally correct training sets should exchange their relative transfer value when the learner's independently diagnosed weakness changes. It requires an executable learner control, outcome-blind strategy/token support audits, predefined warmups and diagnostics, actual Adam-aware marginal-value checks, and both additive and joint-gradient selection baselines. Failed support or manipulation gates stop the strict contrast; they do not authorize relaxed matching or checkpoint selection.

This is a hypothesis, not a working method or novelty claim. The current localization study uses real exposed SymPy sources and only tests data feasibility. An eventual positive paper claim still needs actual functional transfer, a matched diversity intervention and independent real-code/SWE evidence. The old regrouping null and stronger-verifier null remain unchanged.

### 28.2 Completion update — close the static-retrieval branch

The single symbol/dependency policy completes all 32 public contexts with zero technical failures. Exact native/source/graph/packing and source-target auditing passes; the final result is **9/32 anchor-visible and 8/32 prequalified**, below the required 16.


| Fixed source-context interface                    | Prequalified targets |
| ------------------------------------------------- | -------------------- |
| Mixed chunks, 6,144 tokens                        | 5/32                 |
| Mixed chunks, 12,288 tokens                       | 6/32                 |
| Mixed chunks, 24,576 tokens                       | 8/32                 |
| Production chunks, 24,576 tokens                  | 9/32                 |
| Production files, 24,576 tokens                   | 12/32                |
| Issue symbols + one dependency hop, 24,576 tokens | **8/32**             |


The final policy gains one source and loses five relative to the previous best. Its failures include 12 missing changed files, 10 missing anchor regions and one unsupported conversion. One visible target exceeds the completion cap. The locality proxy supplies only 10 completion-fitting cases. The 600-attempt limit is reached on 29 tasks; no posthoc cap/ranking rescue is attempted.

**Decision:** close this static-retrieval branch as planned. The selected policy is null. No GPU, source-quality tests, training, model evaluation or evaluation-private access occurred. An audit-only execution amendment preserved an explicitly interrupted serial attempt and completed the same checks across eight disjoint CPU shards; all 32 tasks passed. This result concerns data support under the fixed exact-target contract, not whether SWE learning or diversity can work.

**Next:** the [joint localization/read/repair support audit](swe_localize_repair_support_plan.md) will investigate constructed maintainer-supervised action targets from the same 32 exposed sources. Initial prompts remain gold-blind; correct locations are training outputs, and evaluation must require self-selected reads. This is a distinct learner control, not a passed old gate or a diversity result. Actual autonomous SWE transfer remains necessary for Assumption 1, followed by matched diversity and metric tests for Assumptions 2 and 3.

[Full study](swe_sympy_symbol_context.md), [scores](../runs/swe-diversity-selection/swe-sympy-symbol-context/score_summary.json), [independent audit](../runs/swe-diversity-selection/swe-sympy-symbol-context/independent_policy_score_audit.json), [diagnosis](../runs/swe-diversity-selection/swe-sympy-symbol-context/coverage_diagnostics.json). New API cost is $0; conservative project total remains **$9.7643163 / $200**. The full paper goal is still active.

## 29. Constructed localization/read/repair support audit (2026-09-09)

**Prospective, before real source construction.** The static-retrieval branch remains closed at 8/32 for the last policy and 12/32 for its best predecessor. The new control teaches localization as assistant output: complete public issue plus all production filepaths → public file outline → exact previously listed source tiles → unchanged canonical repair. These are constructed maintainer-supervised demonstrations, not observed debugging traces. See [implementation plan](swe_localize_repair_support_plan.md).

The fixed cohort remains the same 32 exposed source tasks. Script126 must freeze every public initial record before loading any source-label record. Required support is at least16 complete demonstrations with zero technical failures. Limits are initial12,288 native tokens; every generation prefix24,576 plus2,048 output reservation; causal row26,624; seven total outline/read calls; fixed160-LF-line tiles; each observation4,096 raw tokens and16,384 UTF-8 bytes; action256 and final patch2,048 tokens including EOS. Oversized observations fail without truncation or rescue.

Native chat templates can rewrite historical assistant formatting. Script125 therefore stores separate causal rows, masking the entire online history and supervising only the current assistant completion through EOS. Final patch IDs must equal the existing111 canonical targets. Independent fabricated reviews precede protocol freeze and real construction. No source replacement, post-outcome limit changes, GPU or API calls are planned for this audit. Passing would support a new training technical smoke and baseline/gold source quality study; it would not yet establish Assumption1,2or3. Prior milestone11's45evidence hashes were reverified.

### 29.1 Fixed support result — independent real-data audit pending

The frozen126 protocol yields **23/32 complete valid demonstrations**, above the required16, with **zero technical failures**. All32 initial states were frozen first;31 fit12,288tokens (range9,321–12,413). The first fixed source21567 passes with one outline and two reads. All remaining31 completed without retries.

Accepted demonstrations contain85 assistant rows:62 tool actions (25 outlines and37 reads) plus23 final patches. Longest causal row is19,556tokens, longest prefix18,557, and largest action47tokens. Total supervised tokens are14,396. Twelve accepted sources require2calls, eight3calls, one4calls, and two5calls.

Nine unique exclusions remain: five final-patch overflows (two also exceed action count), two further action-count failures, one original unsupported conversion, and one initial-prompt overflow. No observation or history overflows occur among the23 that reach those checks. Later flags on early-rejected sources are unassessed, so their false values are not separate measured failure rates.

This passes **data support only**. Runtime gold quality, GPU training feasibility, autonomous learner improvement and diversity benefit remain unverified. Independent32-source replay/disposition auditing is underway; new per-turn training smoke and source-quality wrappers are being prepared. No API cost, GPU use or evaluation-private access occurred in this support run. Protocol SHA256: `82acad361ab678cc8688b9568a5c1005a2cf2fdbbf3c20532453a6145be0822e`.

### 29.2 Independent support audit passes

The independent eight-worker audit passes all 32 source slots and reproduces **23 supported demonstrations**, all rejection counts and the gate decision. It verifies 60,261 public file hashes, independently reconstructs 62 outline/read observations, and checks all 85 causal rows against native rendering, masking and exact canonical targets. Elapsed time: 8.69 seconds. No construction/replay/outline helpers from125 or converter calls from110 were used. Original111 independent git reconstruction attestation is checked; that git reconstruction was not rerun.

This completes the support audit. The GPU smoke and source-quality stages remain separate, conditional next controls. [Independent audit](../runs/swe-diversity-selection/swe-localize-repair-support/independent_support_audit.json), [support result](../runs/swe-diversity-selection/swe-localize-repair-support/support_summary.json), [detailed report](swe_localize_repair_support.md). New API spend remains $0; conservative project spending remains $9.7643163 / $200.

## 30. Actual multi-turn SWE training smoke (prospective, 2026-09-09)

Support milestone12 is complete and hash-pinned. New127/129 code and128 source-quality wrapper pass independent review. Six CPU objective checks pass, with zero gradient difference between task/turn means and flattened weights; four bounded launcher tests pass. Before first GPU execution, freeze the actual support corpus, tokenizer/model/software, two extreme source tasks (22236 and23141), all their ten assistant rows, and a shared 3,600-second smoke-plus-future-training budget.

The smoke uses GPU2 only, BF16 frozen Qwen3-8B plus FP32 all-layer q/v LoRA (rank8, alpha16, 3,833,856 parameters), AdamW learning rate1e-4, and FLASH-only SDPA with TF32/autocast disabled. Each current-completion mean NLL is weighted by1/(number of tasks × that task's turn count). There is one accumulated optimizer step, followed by another backward pass with Adam state resident and no second update. Exact base/zero-adapter, standard PEFT save/reload, initial reset and frozen-base byte hashes must pass.

The external launcher checks physical GPU2 availability/UUID, charges elapsed time from process creation, and applies absolute 3,595/3,600-second termination deadlines with process-group cleanup. No failed-run retry or backend fallback. Future-dose estimates use the maximum measured row time across both passes, all85rows per epoch, three epochs, conservative optimizer overhead, a20% reserve and120seconds for load/save. Source-quality admission requires an actual successful smoke plus sufficient budget after charging the greater of internal and external duration. No source-quality assets, full training or evaluation are authorized by a support-only result.

### 30.1 Actual GPU smoke passes; source-quality checks next

The one-attempt127/129 run passes on physical GPU2, session60138 exit0, no timeout. Internal duration212.06seconds; externally charged duration **213.23seconds**. All ten causal rows complete both the initial accumulation and the backward pass with Adam state resident. Exactly one optimizer step changes the adapter (L2 delta0.12134, gradient norm0.93913). The probe changes by up to0.546875, while standard PEFT save/reload and original reset are exact and frozen-base parameter byte hashes remain equal.

Peak allocated memory is25.49GiB and peak reserved35.20GiB. Actual dtype hooks confirm BF16 base/input/output with FP32 LoRA A/B and optimizer state. The conservative maximum three-epoch forecast is **3,248.84seconds**, below the remaining **3,386.77seconds** by137.93seconds. This forecast is not a timing guarantee; the remaining shared hard budget still governs a future run.

This establishes actual training technical feasibility only. No full training or autonomous evaluation occurred, and source runtime quality is still unverified. New128 source-only quality selection is frozen:23 supported sources active, all32 original slots retained, evaluation IDs empty. Official source asset acquisition has started; reviewed baseline/gold checks follow. New API spend remains$0. GPU0 was never used.

### 30.2 Source preparation exposes a metadata collector defect

The original128 preparation completed all23 slots:7 ready and16 bare package-provenance assertion failures. **No baseline/gold tests were dispatched.** All source assets and commands had passed review; original preparation artifacts remain preserved under `source_runtime/`.

Read-only probes across all23 existing baseline environments identify the exact cause. Every environment has the intended direct package pins and imports wheel0.44.0. In16 cases, importing older SymPy exposes `setuptools/_vendor` on `sys.path`; metadata discovery then returns a second wheel0.43.0 record. The inherited90 query builds a last-wins dictionary from all discovered distributions, so vendored metadata overwrites the directly installed version. The7 ready cases do not exhibit that overwrite. This is a collector defect, not evidence of incorrect installed dependencies or failed repairs.

A prospective130/131 correction will use a new namespace, `source_runtime_metadata/`, and preserve the original attempt. It retains the same23 active sources,32 slots, frozen assets, package pins, setup commands and test commands. Only the exact environment-provenance query changes: require a unique direct-site distribution per pin, verify metadata resolution and imported wheel origin/version, retain the complete duplicate inventory, and fail on real ambiguity or mismatch. Strict evidence is saved before admission. There is no package retuning, task replacement, test retry or evaluation-data access. Independent review and bounded interception/negative checks precede the corrected preparation.

The actual GPU smoke independently audits successfully:159 input/artifact hashes, ten initial and ten resident-state rows, and CPU adapter tensors. All72 initial B matrices are zero, all72 updated B matrices are nonzero, and all72 A matrices remain unchanged after the first step, as expected. Live GPU dtype/reset/base-hash statements are explicitly distinguished from independently checked CPU artifacts. [Actual smoke audit](../runs/swe-diversity-selection/swe-localize-repair-support/learner_actual_independent_audit.json).

### 30.3 Collector-only correction frozen and running

All23 read-only baseline probes pass strict direct-site dependency checks. The16 last-wins metadata mismatches match exactly the16 original preparation failures. Independent correction review passes; eight collector falsifiers and seven interception cases pass. Sources130/131 are frozen, and corrected preparation is running in its new namespace. All138 official asset files were copied by verified hash without refetching. Original32 slots/23 active sources, package pins and commands are unchanged.

The cause is older SymPy's `distutils.version` import activating setuptools' shim and appending its vendored directory; the diagnostic did not load `pkg_resources`. The new collector saves duplicate metadata and actual resolved/imported locations rather than discarding version checks. [Correction review](../runs/swe-diversity-selection/swe-localize-repair-support/metadata_correction_independent_review.json). No repair tests have run yet.

### 30.4 Corrected environments pass; runtime verification in progress

Corrected preparation completes23/23 ready, with46 strict environment proofs saved and matched exactly to the corresponding environment records. Exactly46 instances of the old provenance query were replaced; setup/package/test commands remain unchanged. Original failed preparation remains preserved.

The first fixed source21567 passes baseline/gold verification:3 target failures become passes and all57 required regressions pass in both arms. The smoke exits0 in19.89seconds, with no retry. The other22 source pairs are running under the same frozen limits. This is maintainer-target runtime quality, not model repair performance.

The next full-control proposal is recorded in [the autonomous learner plan](swe_localize_learner_control_plan.md): quality-admitted sources, fixed three-epoch task-balanced training, fresh initial adapter, and a separate readiness/inference protocol for the original20 evaluation tasks. Source quality alone does not establish evaluation readiness. Full training and autonomous evaluation have not run.

### 30.5 All23 supported sources pass runtime quality

The corrected first smoke and remaining group both exit0 with no timeout: **23/23 baseline/gold pairs pass**, above the required16. Across the23 task instances, all29 target obligations fail on base and pass with gold, while1,276 regression obligations pass in each arm. These counts are task/test obligations, not globally unique test functions. The46 executions use358.24 externally measured seconds (19.89 first pair +338.35 remaining group), with no test retry or source replacement.

All32 slots remain:23 quality-admitted plus the9 earlier support exclusions. All364 files in the completed runtime manifest were rehashed by root. The independent official-parser/log/environment/disposition audit is underway. Evaluation readiness remains explicitly unassessed; private evaluation assets are still unopened. Full training, autonomous model repair gains and diversity benefits remain unestablished.

[Source-quality summary](../runs/swe-diversity-selection/swe-localize-repair-support/source_runtime_metadata/summary.json), [readiness report](swe_localize_learner_readiness.md), [next control plan](swe_localize_learner_control_plan.md). New API spend remains$0; conservative project spending remains$9.7643163 / $200. GPU use was limited to the213.23-second technical smoke onGPU2.

### 30.6 Independent quality audit passes; source-readiness milestone complete

The independent audit verifies all32 dispositions, all46 saved runtime logs with the pinned official parser, all46 environment/provenance records and their actual metadata/import hashes, and all23 source-gold bindings to111. It reproduces23 quality-admitted sources and9 preserved support exclusions. No tests were rerun. The earlier milestone11 and12 evidence hashes were reverified unchanged.

This completes source learner readiness, with the full paper goal still active. **Next:** freeze the separate20-task evaluation readiness and inference interfaces, then the fixed three-epoch task-balanced training run and autonomous base-versus-trained comparison within remaining compute. No trained checkpoint, autonomous SWE improvement or diversity benefit is claimed. [Independent quality audit](../runs/swe-diversity-selection/swe-localize-repair-support/source_runtime_metadata/independent_quality_audit.json).

## 31. Fixed20 autonomous learner control (prospective, 2026-09-09)

Milestone13's62 evidence hashes were reverified. The next experiment tests whether the23 quality-admitted constructed demonstrations improve autonomous SWE repair, an ordinary-supervision prerequisite for Assumption1. It does not yet test diversity under Assumptions2/3.

Implementation is in progress in three parallel parts:132/133 exact three-epoch task-balanced training and external budget enforcement;134 frozen public initial states and separate runtime readiness for the original20 evaluation IDs;135/136 matched base/trained inference and bounded execution. Frozen125/127/129 and the earlier failed static-retrieval branches remain preserved. The training run must start from the smoke's initial adapter, with3,386.77seconds left in its shared budget, and wait for independently audited evaluation readiness and a frozen inference protocol.

Both models must choose their own outlines and reads. The unchanged125 system contract terminates invalid calls without corrective hints; each attempted call consumes a slot. Repeated valid calls also consume slots. All20 task slots remain in the denominator, including initial overflow and invalid/incomplete episodes. Proposed inference has a separate7,200-second hard outer budget and180-second cooperative per-episode deadline, with overruns explicitly invalid and all unstarted slots retained. No model generations or new private evaluation access have occurred at this entry. API spending is unchanged.

### 31.1 All20 public initial states fit; official Python version mismatch found before setup

All20 new public issue/file-index prompts are frozen and fit12,288tokens (range9,160–10,755). The original cohort/order is unchanged; no private evaluation assets were acquired until that manifest existed. All120 pinned official assets were then acquired and passed the command grammar. Review of seven new setup/release variants found package-list, minimum-Python and release metadata differences; the test runner matches previously reviewed source bytes.

Operator review found that task19495's official Docker specification requiresCPython3.9.21; the other19 specify3.9.20. The inherited90bridge assumes3.9.20. Allnine package pins match. **No environments or baseline/gold tests were started.** The review therefore stops the134 preparation path rather than altering the official asset or misreporting the interpreter version.

A root shell invocation also reached134prepare after its review command had failed to create an authorization file.112 rejected the missing file before any preparation dispatch;134's finally block wrote a zero-query receipt. This pre-dispatch error and the original namespace are retained in `evaluation_readiness/preparation_review_terminal.json`; it was not a runtime/test attempt.

New137 implementation will explicitly match each task's official3.9.20/3.9.21 interpreter, retain exact package pins/commands/all20 slots, and reuse all frozen prompts/assets in a separate namespace. This is a prospective compatibility correction before test outcomes, not task replacement or repair rescue. Full training remains gated on independent runtime readiness.

### 31.2 Correction:19495 also has an official dependency-pin difference

Independent exact Docker parsing found `flake8-comprehensions==3.16.0` for19495, versus the inherited3.15.0. The earlier31.1 statement that allnine package pins match was premature: the root review stopped at that task's Python assertion before checking its remaining pins. No actual preparation or test followed. The independent reviewer is parsing every exact Python/dependency declaration across all20 assets.137 must match explicit official per-task versions for the same dependency names and validate actual provenance; no substring-based acceptance, package-version spoofing or implicit fallback. The original134 artifacts remain preserved.

### 31.3 Official-version bridge frozen; actual evaluation preparation started

The exact20-task configuration audit confirms only19495 differs:Python3.9.21 and flake8-comprehensions3.16.0. Its interpreter was acquired locally, with4,293 installation-file hashes and symlink provenance recorded. Independent137/138 review passes35 parser/pin/probe checks. The new namespace retains the original20 prompts and120 official asset files, and verifies actual interpreter/package origins rather than changing observed version strings.

The137 cohort is now frozen and actual environment preparation has started. No baseline/gold outcome is claimed yet. The autonomous135/136 protocol is also frozen after26 controller and4 real CPU-process launcher checks. Full-training static review passes, including exact23-task/85-row weighting and deadline checks. Full training still waits for all20 runtime checks and their independent audit. API spending remains unchanged.

### 31.4 Evaluation preparation and first runtime pair pass

All20 tasks prepared successfully, with40 strict baseline/gold environment proofs. Root compared each raw proof with its environment record and checked the exact task-specific Python/package map;19495 correctly uses3.9.21 and3.16.0. No preparation failures or fallback occurred.

The first fixed evaluation task19954 passes baseline/gold quality in6.92 externally measured seconds, exit0 and no timeout. The remaining19 pairs are running once under the same fixed commands and combined7,200-second budget. Full training remains unlaunched until complete20-task readiness and independent auditing pass.

### 31.5 Strict runtime gate fails18/20; benchmark grading semantics under independent audit

All40 baseline/gold executions finish without timeout, totaling216.64externally measured seconds. The frozen137 rule admits18/20. Tasks19637 and19346 fail its required gold-exit0 condition: their required target and regression tests all pass under gold, while unlisted preexisting tests fail in both baseline and gold and keep the overall command exit at1. The original strict failures, logs and all20 slots remain intact. No full training or model generation has run.

The pinned official SWE-bench grader resolves instances from their required F2P/P2P test lists; it has an exit/status consistency check rather than a blanket exit0 requirement. Independent auditing is now comparing the exact official grader against all saved logs. Any subsequent benchmark-aligned control must be separately frozen before model execution, retain strict whole-command results as a sensitivity, and preserve the failed original gate. No tests, task substitutions, patch changes or environmental retries are proposed. [Pinned official grading code](https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/swebench/harness/grading.py).

### 31.6 Independent strict and required-test audits complete

The independent audit verifies all20 initial states, exact official environments,40 raw environment records and40 saved logs. It reproduces the strict18/20 result. A separate exact-official-functions audit and explicit baseline/gold pair check validate20/20 required-test pairs without rerunning any test. These audits preserve the original failure and distinguish host-log reconstruction from an official container run.

A new prospective [required-test control](swe_required_test_control_plan.md) keeps the same literal-PASSED requirement for every required test and replaces blanket exit0 with completed-exit/status integrity. It retains whole-command success as a sensitivity and exact official grading as a diagnostic. All20 tasks,23 source demonstrations, model/data/dose/budgets and public prompts remain fixed. The revision occurs after runtime inspection and before any model trial, and will be reported as such. New artifacts use separate required-test namespaces; the original strict control remains unlaunched and preserved.