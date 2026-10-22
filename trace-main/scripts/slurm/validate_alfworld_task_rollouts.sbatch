#!/bin/bash
#SBATCH --job-name=trace-g1-replay-rollout
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/g1-replay/logs/%x-%A_%a.out
#SBATCH --error=outputs/g1-replay/logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?submit this job from the trace repository root}"
task_shard="${TRACE_REPLAY_TASK_SHARD:?set TRACE_REPLAY_TASK_SHARD}"
report_run_id="${TRACE_REPLAY_REPORT_RUN_ID:?set TRACE_REPLAY_REPORT_RUN_ID}"
cd "$repo_root"
export PYTHONPATH="$repo_root/../icrl"

"$repo_root/../alfworld-venv/bin/python" \
  scripts/validate_alfworld_replay.py \
  --num-shards 16 \
  --shard-index "$task_shard" \
  --rollout-index "$SLURM_ARRAY_TASK_ID" \
  --report "outputs/g1-replay/reports/${report_run_id}/partials/task_${task_shard}/rollout_${SLURM_ARRAY_TASK_ID}.json"
