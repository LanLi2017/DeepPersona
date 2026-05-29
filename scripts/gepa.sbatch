#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# GEPA persona-prompt optimization on one GPU.
#
# Stage 1 (search): serve Qwen2.5-7B-Instruct with vLLM (OpenAI API), then run
#   scripts/05_gepa_optimize.py. Task LM = local vLLM (greedy). Reflection LM =
#   OpenAI (reads OPENAI_API_KEY from the environment).
# Stage 2 (verdict, optional): free the GPU, switch to the core HF env, and run
#   scripts/06_gepa_eval.py on the val split (native greedy harness).
#
# Submit (key must be in your shell; --export=ALL propagates it):
#   export OPENAI_API_KEY=sk-...
#   sbatch --export=ALL scripts/gepa.sbatch                       # defaults
#   sbatch --export=ALL,PERSONA_IDX=1,AUTO=medium scripts/gepa.sbatch
#   sbatch --export=ALL,SMOKE=1 scripts/gepa.sbatch               # tiny end-to-end check
#
# Tunable env vars (with defaults):
#   PERSONA_IDX=1  SEED_LEVEL=structured  AUTO=light  MAX_METRIC_CALLS=
#   REFLECTION_MODEL=openai/gpt-5  N_TRAIN=  N_VAL=  PORT=8000
#   RUN_VERDICT=1  VERDICT_SPLIT=val  SMOKE=0
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=gepa
#SBATCH -o /projects/bbrz/yirenl2/DeepPersona/logs/%x_%j.out
#SBATCH -e /projects/bbrz/yirenl2/DeepPersona/logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yirenl2@illinois.edu
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbrz-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-/projects/bbrz/yirenl2/DeepPersona}"
cd "${REPO_DIR}"
mkdir -p logs runs

: "${OPENAI_API_KEY:?OPENAI_API_KEY not set. Run 'export OPENAI_API_KEY=...' then 'sbatch --export=ALL ...'}"

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_REV="a09a35458c702b33eeacc393d103063234e8bc28"
PERSONA_IDX="${PERSONA_IDX:-1}"
SEED_LEVEL="${SEED_LEVEL:-structured}"
AUTO="${AUTO:-light}"
REFLECTION_MODEL="${REFLECTION_MODEL:-openai/gpt-5}"
PORT="${PORT:-8000}"
RUN_VERDICT="${RUN_VERDICT:-1}"
VERDICT_SPLIT="${VERDICT_SPLIT:-val}"
SMOKE="${SMOKE:-0}"
export HF_HOME="${HF_HOME:-/u/yirenl2/.cache/huggingface}"

# Stable output dir so the verdict stage can find evolved_prompt.txt.
TS="$(date -u +%Y%m%d-%H%M%SZ)"
OUT_DIR="${REPO_DIR}/runs/gepa-${TS}-csqa-p${PERSONA_IDX}-${SEED_LEVEL}"

DRIVER_ARGS=(--task csqa --persona-idx "${PERSONA_IDX}" --seed-level "${SEED_LEVEL}"
             --reflection-model "${REFLECTION_MODEL}" --task-api-base "http://localhost:${PORT}/v1"
             --out-dir "${OUT_DIR}")
if [[ "${SMOKE}" == "1" ]]; then
    DRIVER_ARGS+=(--n-train 8 --n-val 8 --max-metric-calls 12)
else
    [[ -n "${MAX_METRIC_CALLS:-}" ]] && DRIVER_ARGS+=(--max-metric-calls "${MAX_METRIC_CALLS}") \
                                     || DRIVER_ARGS+=(--auto "${AUTO}")
    [[ -n "${N_TRAIN:-}" ]] && DRIVER_ARGS+=(--n-train "${N_TRAIN}")
    [[ -n "${N_VAL:-}" ]]   && DRIVER_ARGS+=(--n-val "${N_VAL}")
fi

echo "=== GEPA job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "persona=${PERSONA_IDX} seed=${SEED_LEVEL} auto=${AUTO} reflect=${REFLECTION_MODEL} smoke=${SMOKE}"
echo "out_dir=${OUT_DIR}"
echo "Start: $(date)"

# ── env: GEPA stage uses the separate .venv-gepa (dspy + vllm) ───────────────
module load miniforge3-python
source "${REPO_DIR}/.venv-gepa/bin/activate"

# ── Stage 1a: launch vLLM OpenAI server in the background ────────────────────
VLLM_LOG="logs/vllm_${SLURM_JOB_ID:-local}.log"
echo "[vllm] starting -> ${VLLM_LOG}"
# --enforce-eager skips torch.compile + CUDA-graph capture, which otherwise adds
# several minutes to a cold first launch (vLLM 0.21 defaults to VLLM_COMPILE).
# Inference is marginally slower but prefix-caching (shared persona prompt) keeps
# the GEPA rollout loop fast. Startup on this Lustre node is slow (~5 min of
# imports alone), so the health window below is generous.
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-1800}"   # seconds
vllm serve "${MODEL_ID}" \
    --revision "${MODEL_REV}" \
    --served-model-name "${MODEL_ID}" \
    --dtype bfloat16 \
    --port "${PORT}" \
    --max-model-len 4096 \
    --enforce-eager \
    --gpu-memory-utilization 0.85 > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[vllm] waiting up to ${VLLM_START_TIMEOUT}s for /health (pid ${VLLM_PID})..."
HEALTHY=0
SECS=0
while [[ ${SECS} -lt ${VLLM_START_TIMEOUT} ]]; do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[vllm] ready after ${SECS}s"; HEALTHY=1; break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[vllm] server process died during startup; tail of ${VLLM_LOG}:"; tail -40 "${VLLM_LOG}"; exit 1
    fi
    [[ $((SECS % 60)) -eq 0 ]] && echo "[vllm] still starting... ${SECS}s elapsed"
    sleep 10; SECS=$((SECS + 10))
done
[[ "${HEALTHY}" == "1" ]] || { echo "[vllm] not healthy after ${VLLM_START_TIMEOUT}s"; tail -60 "${VLLM_LOG}"; exit 1; }

# ── Stage 1b: GEPA search ────────────────────────────────────────────────────
echo "[gepa] python scripts/05_gepa_optimize.py ${DRIVER_ARGS[*]}"
python scripts/05_gepa_optimize.py "${DRIVER_ARGS[@]}"

# ── Stage 2: native-harness verdict (free the GPU first) ─────────────────────
if [[ "${RUN_VERDICT}" == "1" ]]; then
    echo "[verdict] stopping vLLM to free the GPU..."
    cleanup; trap - EXIT; sleep 10
    echo "[verdict] switching to core HF env"
    source "${REPO_DIR}/activate_hpc.sh"
    EVAL_ARGS=(--gepa-prompt-file "${OUT_DIR}/evolved_prompt.txt" --persona-idx "${PERSONA_IDX}" --split "${VERDICT_SPLIT}")
    [[ "${SMOKE}" == "1" ]] && EVAL_ARGS+=(--n-items 16)
    echo "[verdict] python scripts/06_gepa_eval.py ${EVAL_ARGS[*]}"
    python scripts/06_gepa_eval.py "${EVAL_ARGS[@]}"
fi

echo "End: $(date)"
