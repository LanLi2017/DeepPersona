#!/bin/bash
#SBATCH --job-name=trace-g1-replay
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --array=0-15
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/g1-replay/logs/%x-%A_%a.out
#SBATCH --error=outputs/g1-replay/logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?submit this job from the trace repository root}"
cd "$repo_root"
export PYTHONPATH="$repo_root/../icrl"
report_run_id="${TRACE_REPLAY_REPORT_RUN_ID:-$SLURM_ARRAY_JOB_ID}"

"$repo_root/../alfworld-venv/bin/python" \
  scripts/validate_alfworld_replay.py \
  --num-shards 16 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --report "outputs/g1-replay/reports/${report_run_id}/shard_${SLURM_ARRAY_TASK_ID}.json"
