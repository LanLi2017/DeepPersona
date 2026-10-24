# Environment setup and known gotchas

## Rebuilding envs (only needed in a fresh clone or worktree; otherwise `envs/` is already on disk)

Run all of these from `<root>`, which is `trace-main/`. `envs/` is git-ignored.

```bash
# AppWorld: Python 3.11 venv. Version 0.1.3.post1 was used.
python3.11 -m venv envs/appworld
envs/appworld/bin/pip install appworld==0.1.3.post1 boto3 python-dotenv requests
envs/appworld/bin/appworld install
envs/appworld/bin/appworld download data --root envs/appworld_root                # data version 0.1.0
envs/appworld/bin/appworld download experiment-outputs --root envs/appworld_root  # public baselines, used for the prevalence audit

# tau2-bench, pinned to commit 4ce7c03
git clone https://github.com/sierra-research/tau2-bench envs/tau2-bench-src && git -C envs/tau2-bench-src checkout 4ce7c03
# install it into envs/tau2, adding websockets
```

The analysis and judge scripts use the repo-level `../.venv`. It needs requests, boto3,
python-dotenv, and the `claude` CLI on PATH for the Opus judge. Keys live in the repo-root
`.env`:
- OpenAI: `OPENAI_API_KEY` or lowercase `openai_api_key`.
- Bedrock: the AWS credentials in the environment, region us-east-2.

## Model access (as of 2026-10)

- **GPT-5.6** (`gpt-5.6-sol`): goes through the OpenAI API. Price is $4 per 1M input tokens and
  $20 per 1M output tokens.
- **Bedrock** has Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) and Kimi K2 Thinking
  (`moonshot.kimi-k2-thinking`). Haiku is $1 per 1M input tokens and $5 per 1M output tokens.
  GPT-5.6, Opus 5 and Sonnet 5 return AccessDenied on Bedrock.
- **Opus** runs via `claude -p --model opus --tools '' --output-format json`. It costs about
  $0.17 per pair.
- The Codex login had expired. Don't rely on it.

## AppWorld gotchas (all already handled in the scripts)

- **SIGALRM** — AppWorld uses SIGALRM, so each rollout must run in a process's main thread. Use
  `ProcessPoolExecutor`, never threads.
- **Frozen clock** — AppWorld freezes the clock with freezegun, which breaks AWS SigV4 signing.
  Bedrock calls therefore go through a `--_bedrock` subprocess of the rollout script. The
  "System time is way off" warnings in logs are expected.
- **Closed database** — read `completed` before the world exits; otherwise you hit a
  closed-database error.
- **Replay** (`replay_appworld.py`):
  - Use `requester.<method>`.
  - Swap stale access tokens for the replayed ones.
  - Call `reset_request_count()` on each call to avoid the 1000-request cap.
  - Catch API exceptions into `{'error': ...}`.
  - Then call `apis.supervisor.complete_task()` and `evaluate().to_dict()`.
- **Probe output directories** — these pollute `experiments/outputs`. Delete them, and filter
  by name.
- **Task datetime** — put it in the plan prompt; otherwise "this year" gets misread.

## Agent gotchas

- **Kimi** fails as an agent: it doesn't `print()` and it emits a native tool-call format. It is
  kept as a judge only.
- **Haiku** writes several turns in one reply and hallucinates their outputs. Execute only the
  first code block and truncate the rest of the turn.
- **Agent prompt** — the agent prompt (`AGENT_SYS`) has to require `print()`, show a format
  example, and forbid XML tool calls.

## Proxy gotchas

- **Defaults** — fill in API defaults from `envs/appworld_root/data/api_docs/standard/*.json`.
  Otherwise passing `overwrite=False` explicitly looks like a difference.
- **Normalize IDs** — replace numeric IDs in URLs: `re.sub(r'/\d+', '/{id}', url)`.
- **Exclusions** — leave out `/supervisor`, `/api_docs` and auth/login/token calls.

## Unrelated but in the same repo

The Tinker client returns near-duplicate samples when `SamplingParams(seed=fixed)` is combined
with `num_samples>1`. Don't pass a seed there (see `scripts/34_csqa_collapse_check.py` in
DeepPersona).
