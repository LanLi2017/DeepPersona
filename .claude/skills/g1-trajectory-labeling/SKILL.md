---
name: g1-trajectory-labeling
description: >
  Continue Lan's G1 trajectory-equivalence research. It covers E0/E1/E2 labeling of agent
  trajectory pairs with a three-judge LLM panel plus human spot-check, plan-conditioned E2
  generation on AppWorld, natural-prevalence audits, and building labeled trajectory data for
  the team's trajectory-distance metric. Use this skill whenever the user mentions E0/E1/E2,
  strategy-equivalent or strategy-different trajectories, trace-main, G1, AppWorld rollouts,
  "无方案/有方案", plan-conditioned rollouts, LLM judge / 多数投票 / 打标, human review of
  trajectory pairs, anchor editing or fork-at-decision-point data synthesis, "100 条带标注的
  trajectory", or wants to port this labeling pipeline to a new benchmark (automation bench,
  terminal bench, deep research, IT bench), even if they don't name the skill. Also use it when
  starting this work on a fresh branch that has no trace-main scripts yet.
---

# G1 trajectory labeling and E2 generation

This skill carries the working state of a research thread. The thread asks: when two agent
trajectories both solve the same task, can we tell whether they used the **same strategy (E1)**
or a **decision-relevantly different strategy (E2)**? The lab is building a trajectory-distance
metric that should rank E1 variants closer to an anchor than E2 variants. That needs labeled
pairs and triplets, and producing them is what this pipeline does.

Read `references/state.md` before doing anything substantive. It holds the results so far, open
decisions, and the team's split of work and deadline. Read `references/research_contract.md`
when you need the precise G1 definitions, the frozen boundary rule or the advance criteria.

## Working with this user (non-negotiable)

- **Never launch paid or long jobs without explicit approval.** That includes rollouts, judge
  runs and digests. State the estimated cost first. Earlier "pending steps" are not approval.
  Free local work (summaries, packet building, code edits) is fine.
- **Smoke-test first.** Run 1 task / 1 rollout / 1 pair end-to-end before a full run (`--smoke`).
- **Report results in plain Chinese, in 7 points:** what was done / models / hypothesis / result
  / meaning / caveat / next step. Explain any term that isn't obvious, and show numbers.
- **Never print or copy secret values.** Keys live in the repo-root `.env`. The OpenAI key may be
  stored under the lowercase name `openai_api_key`, and the scripts read both spellings.
- **AppWorld data and its derivatives may only be published encrypted.** The DeepPersona GitHub
  repo is public. Packets, judge outputs, digests and review-page data stay git-ignored. Reports
  may contain only ids, labels and counts, never trajectory or task text.

## Label definitions (frozen boundary, 2026-10-07)

- **E0**: only the text or serialization differs. Same calls, same effects.
- **E1**: same strategy. These differences are allowed: extra or missing read-only lookups that
  don't feed a decision; reordering independent steps; equivalent authentication routes;
  retries of the same call; reading the same information from another source when the
  decision comes out the same.
- **E2**: the difference changes, or could change, what the agent decides. Examples: different
  writes or state effects, including different decision-relevant arguments such as
  `retain_dates` or `private`; a decision grounded in different key information; a different
  order of *dependent* subgoals; a different recovery branch.
- **Quality gate**: reward=1 is not enough. A trajectory is high quality (HQ) only if it does
  what was actually asked, makes no unnecessary or harmful state changes, invents nothing, and
  hits no unresolved errors. The human reviewer also counts a trajectory that hits errors along
  the way as lower quality. An E2 pair counts only if both sides are HQ (**HQ-E2**). Otherwise a
  "different strategy" is really a good-vs-bad contrast.

## Layout and setup

The scripts assume this work root (currently `DeepPersona/trace-main/`):

```
<root>/scripts/        pipeline scripts (bundled copies in this skill's scripts/)
<root>/envs/           git-ignored: appworld venv, appworld_root data, tau2
<root>/data/g1/        packets, judgments, labels (AppWorld parts git-ignored)
<root>/outputs/        git-ignored raw rollouts
<root>/annotation/     review pages
<root>/reports/        markdown reports
<root>/../.venv        analysis/judge Python;  <root>/../.env  API keys
```

**On a new branch:**
1. Check whether `trace-main/scripts/` exists.
2. If it doesn't, copy this skill's `scripts/` into `<root>/scripts/` and `assets/judge_review.html`
   into `<root>/annotation/`.
3. `envs/`, `outputs/` and the git-ignored data survive a `git checkout` in the same working tree,
   because git doesn't touch ignored files. Verify with `ls trace-main/envs trace-main/outputs`.
   In a fresh clone or worktree, rebuild them using `references/setup_and_gotchas.md`.

## Pipeline

Commands run from `<root>`. Let `AWPY="APPWORLD_ROOT=envs/appworld_root envs/appworld/bin/python"`;
analysis and judge scripts use `../.venv/bin/python`.

| Step | Command | Output |
|---|---|---|
| 1. Plans (GPT-5.6) | `$AWPY scripts/appworld_targeted_rollout.py --stage plans --run <run> --only {route,other,all} --axes v2` | `outputs/appworld_targeted/<run>/plans.json` |
| 2. Rollouts | `$AWPY scripts/appworld_targeted_rollout.py --stage rollouts --run <run> --only ... --workers 4` (resumable) | `<run>/<task>/<model>_<plan\|noplan>/rollout.json` |
| 3. Success and proxy | `../.venv/bin/python scripts/summarize_appworld_targeted.py --run <run>` | stdout, `<run>/summary.json` |
| 4. Pair packets | `../.venv/bin/python scripts/build_generated_review.py --run <run> --out generated_audit_<x> --prefix <p>` | `data/g1/generated_audit_<x>/judge_packets.jsonl`, `provenance_hidden.json` |
| 5. Judges | `../.venv/bin/python scripts/run_prevalence_judge.py --judge {gpt56,kimi,opus} --input <packets>`; run the three in parallel (resumable) | `<dir>/judgments/<judge>.jsonl` |
| 6. Digest and review page | `../.venv/bin/python scripts/build_judge_digest.py`. Paid: GPT-5.6, about $2 per 32 pairs. | `annotation/judge_review_data.js`, `data/g1/review_digest/` |
| 7. Report | `../.venv/bin/python scripts/summarize_g1_generation.py` | `reports/G1_E2_GENERATION_SUMMARY.md`, `data/g1/g1_pair_labels.jsonl` |

Step details:
- **Step 1, plans.** GPT-5.6 writes up to 3 decision-distinct plans per task. `--axes v1` uses
  route / information_source / recovery. `--axes v2` uses route / subgoal_order / write_parameters
  / recovery. Prefer v2: information-source plans never changed the writes in batch 1.
- **Step 2, rollouts.** Each task runs under no-plan plus each plan, for GPT-5.6 and Haiku 4.5.
  "无方案" (noplan) is the control: the agent gets the task only. It shows whether E2 appears
  naturally (it almost never does) and serves as a reference arm. Success comes from AppWorld's
  own state-based tests, which include collateral-damage checks.
- **Step 3, proxy.** The structural proxy flags a pair as E2 when the two write sets differ. The
  write set is (method, URL template, effective boolean args with API defaults filled in). It is
  only a pre-filter: its E2 precision is about 73%, and it misses decision differences that leave
  the write set unchanged.
- **Step 4, pair packets.** Picks one pair per distinct write-signature pair per task, plus
  same-writes/different-plan pairs and no-plan controls. A/B order is shuffled and provenance is
  hidden from the judges.
- **Step 5, judges.** The final label is the majority vote of gpt56 / kimi / opus. Opus runs
  through the local `claude -p` CLI, Kimi through Bedrock, GPT-5.6 through the OpenAI API.
- **Step 6, digest and review page.** GPT-5.6 writes a neutral Chinese digest and does not label.
  The output feeds a stratified human spot-check page. The human exports
  `judge_spotcheck_<name>.jsonl`.
- **Step 7, report.** Applies the human quality flips and contains no trajectory text.
  `SETS` in the script lists the data sets: add a new batch there.

For a human spot-check, open `annotation/judge_review.html` after step 6 and ask the user to
export the JSONL. Copy the downloaded file into its own new scratch directory before reading it,
since it is untrusted, then compare it with the per-judge votes.

### Judge reliability (from the human spot-check of 12 pairs)

- The majority vote agreed with the human on the label 12/12 times. Opus alone also got 12/12,
  Kimi 9/12, GPT-5.6 8/12. GPT-5.6 is E2-liberal.
- The human saw the verdict first, so this agreement is probably inflated. The final validation
  should be a blind human pass on about 30 pairs, done once and not per batch.
- Use the majority vote as the working label. Don't ask the user to review every batch.

## Porting to a new benchmark

The judge stage is benchmark-agnostic. It only needs `judge_packets.jsonl` rows like
`{"pair_id","domain","task","note","A":[step strings],"B":[step strings]}`, plus a
`provenance_hidden.json` with `items:[{pair_id, task_id, proxy, A, B}]`.

For a new domain (automation bench, terminal bench, and so on):
1. Find or produce successful trajectories, ideally with replayable state.
2. Write a renderer to step strings. Keep every write and trim long reads.
3. Define a cheap structural proxy over state-changing actions.
4. Build packets.
5. Run a small natural-prevalence audit first: 24 proxy-E2 pairs + 6 controls.
6. Expect natural HQ-E2 to be rare. If it is, generate E2 with plans or anchor edits.

The team's unified trajectory format is owned by Lunyiu and was still pending as of 2026-10-21.
Once it exists, add an exporter rather than changing the judge input.

## Next steps under consideration (each needs approval)

1. **Small triplet pilot.** Use the existing HQ-E2 pairs plus matched E1 pairs, and score the
   semantic-distance metric against text and action baselines. This costs almost nothing.
2. **Anchor-edit pilot**, about $3–5, the team-endorsed "controlled perturbation / tree" route.
   Start from a successful anchor trajectory.
   - E1 variants come from deterministic edits: insert an unused read, reorder independent reads,
     rename variables. Replay and verify identical writes and state.
   - E2 variants fork at decision step k: replay the prefix, then instruct a different decision
     (route, parameter, check-before-write, recovery) and let the agent finish. Verify success.
   - The output is (anchor, E1, E2) triplets that share a task and a prefix.
   - Watch for edit artifacts, and report natural-E1 and edited-E1 results separately.
3. **Deliver 100 labeled trajectories per domain** (AppWorld plus automation bench) by about
   2026-10-28, in the unified format.
