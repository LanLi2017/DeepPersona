#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# One persona-scaffolding *cell* = one 01_baseline.py run (GPU).
# Parameterized entirely by env vars passed through sbatch --export:
#   CONDITION      none | prompt              (required)
#   PERSONA_IDX    0..11                       (required if CONDITION=prompt)
#   PERSONA_LEVEL  basic | structured          (required if CONDITION=prompt)
#   EXP_ID         shared --run-name tag for the sweep (required)
#   SPLIT          calib | val | test          (default: val)
#   CONFIG         config yaml                  (default: configs/qwen25_7b.yaml)
#   N_ITEMS        limit items                  (optional)
#   SMOKE          1 to force smoke mode        (optional)
#   REPO_DIR       repo root                    (default: this repo)
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=scaff-cell
#SBATCH -o /projects/bbrz/yirenl2/DeepPersona/logs/%x_%j.out
#SBATCH -e /projects/bbrz/yirenl2/DeepPersona/logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yirenl2@illinois.edu
#SBATCH --nodes=1
#SBATCH --mem=48g
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbrz-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1

set -euo pipefail

: "${CONDITION:?CONDITION not set (none|prompt)}"
: "${EXP_ID:?EXP_ID not set}"
REPO_DIR="${REPO_DIR:-/projects/bbrz/yirenl2/DeepPersona}"
CONFIG="${CONFIG:-configs/qwen25_7b.yaml}"
SPLIT="${SPLIT:-val}"

cd "${REPO_DIR}"
source activate_hpc.sh

ARGS="--config ${CONFIG} --condition ${CONDITION} --split ${SPLIT} --run-name ${EXP_ID}"
if [[ "${CONDITION}" == "prompt" ]]; then
    : "${PERSONA_IDX:?PERSONA_IDX required for condition=prompt}"
    : "${PERSONA_LEVEL:?PERSONA_LEVEL required for condition=prompt}"
    ARGS="${ARGS} --persona-idx ${PERSONA_IDX} --persona-level ${PERSONA_LEVEL}"
fi
[[ -n "${N_ITEMS:-}" ]] && ARGS="${ARGS} --n-items ${N_ITEMS}"
[[ "${SMOKE:-0}" == "1" ]] && ARGS="${ARGS} --smoke"

echo "=== scaffolding cell: cond=${CONDITION} pidx=${PERSONA_IDX:-NA} level=${PERSONA_LEVEL:-NA} split=${SPLIT} exp=${EXP_ID} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)  HF_HOME=${HF_HOME}"
echo "Start: $(date)"

python scripts/01_baseline.py ${ARGS}

echo "End: $(date)"
