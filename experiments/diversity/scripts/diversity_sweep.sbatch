#!/bin/bash
# Rollout diversity sweep on Qwen2.5-1.5B-Instruct (CSQA val, n=200).
# Usage:
#   sbatch experiments/diversity/scripts/diversity_sweep.sbatch
#   sbatch --export=ALL,SMOKE=1 experiments/diversity/scripts/diversity_sweep.sbatch
#SBATCH --job-name=div-sweep
#SBATCH -o /projects/bbrz/yirenl2/DeepPersona/experiments/diversity/logs/%x_%j.out
#SBATCH -e /projects/bbrz/yirenl2/DeepPersona/experiments/diversity/logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lanl2@illinois.edu
#SBATCH --nodes=1
#SBATCH --mem=32g
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbrz-delta-gpu
#SBATCH --time=01:30:00
#SBATCH --gpus-per-node=1

set -euo pipefail
REPO_DIR="${REPO_DIR:-/projects/bbrz/yirenl2/DeepPersona}"
cd "${REPO_DIR}"
source activate_hpc.sh

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  HF_HOME=${HF_HOME}"
echo "Start: $(date)"

ARGS="--config experiments/diversity/configs/qwen15b_csqa.yaml"
[[ "${SMOKE:-0}" == "1" ]] && ARGS="${ARGS} --smoke"

python experiments/diversity/scripts/01_rollout_diversity.py ${ARGS}

# Print analysis immediately after
JSONL=$(ls -t "${REPO_DIR}/experiments/diversity/runs"/diversity-*.jsonl | head -1)
echo "=== Analysis: ${JSONL} ==="
python experiments/diversity/scripts/02_analyze.py "${JSONL}"

echo "End: $(date)"
