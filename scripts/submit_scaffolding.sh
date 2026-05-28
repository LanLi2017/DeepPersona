#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Submit the persona-scaffolding sweep as independent per-cell GPU jobs.
#
# Cells = none + basic×{personas} + structured×{personas}  (default 1 + 12 + 12 = 25).
# All cells share --run-name <EXP_ID>; recombine afterward with:
#     python scripts/04_aggregate_scaffolding.py --exp-id <EXP_ID>
#
# Because the model is not yet in the HF cache, a single prefetch job runs
# first and every cell job waits on it (afterok) so 25 jobs don't each try to
# download the weights concurrently. Use --no-prefetch once it's cached.
#
# Usage:
#   bash scripts/submit_scaffolding.sh                       # full val sweep (25 cells)
#   bash scripts/submit_scaffolding.sh --dry                 # print sbatch lines only
#   bash scripts/submit_scaffolding.sh --split test          # test split
#   bash scripts/submit_scaffolding.sh --personas 0 10 --levels structured
#   bash scripts/submit_scaffolding.sh --n-items 8 --smoke   # tiny check on GPU
#   bash scripts/submit_scaffolding.sh --no-prefetch --time 02:00:00
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/scaffolding_cell.sbatch"

ALL_PERSONAS=(0 1 2 3 4 5 6 7 8 9 10 11)
ALL_LEVELS=(basic structured)

DRY_RUN=false
DO_PREFETCH=true
DO_NONE=true
SMOKE=false
SPLIT="val"
CONFIG="configs/qwen25_7b.yaml"
N_ITEMS=""
TIME_LIMIT=""
ACCOUNT=""
PARTITION=""
EXP_ID="scaff-$(date +%Y%m%d-%H%M%S)"
SELECTED_PERSONAS=()
SELECTED_LEVELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)          DRY_RUN=true; shift ;;
        --no-prefetch)  DO_PREFETCH=false; shift ;;
        --no-none)      DO_NONE=false; shift ;;
        --smoke)        SMOKE=true; shift ;;
        --split)        SPLIT="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --n-items)      N_ITEMS="$2"; shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        --account)      ACCOUNT="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --exp-id)       EXP_ID="$2"; shift 2 ;;
        --personas)     shift; while [[ $# -gt 0 && ! "$1" == --* ]]; do SELECTED_PERSONAS+=("$1"); shift; done ;;
        --levels)       shift; while [[ $# -gt 0 && ! "$1" == --* ]]; do SELECTED_LEVELS+=("$1"); shift; done ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ ${#SELECTED_PERSONAS[@]} -eq 0 ]] && SELECTED_PERSONAS=("${ALL_PERSONAS[@]}")
[[ ${#SELECTED_LEVELS[@]} -eq 0 ]] && SELECTED_LEVELS=("${ALL_LEVELS[@]}")

mkdir -p "${REPO_DIR}/logs"

# Extra sbatch flags (override #SBATCH directives in the worker).
SB_OPTS=()
[[ -n "${TIME_LIMIT}" ]] && SB_OPTS+=(--time "${TIME_LIMIT}")
[[ -n "${ACCOUNT}" ]]    && SB_OPTS+=(--account "${ACCOUNT}")
[[ -n "${PARTITION}" ]]  && SB_OPTS+=(--partition "${PARTITION}")

EXPORT_COMMON="ALL,EXP_ID=${EXP_ID},CONFIG=${CONFIG},SPLIT=${SPLIT},REPO_DIR=${REPO_DIR}"
[[ -n "${N_ITEMS}" ]] && EXPORT_COMMON="${EXPORT_COMMON},N_ITEMS=${N_ITEMS}"
$SMOKE && EXPORT_COMMON="${EXPORT_COMMON},SMOKE=1"

echo "EXP_ID=${EXP_ID}  split=${SPLIT}  personas=(${SELECTED_PERSONAS[*]})  levels=(${SELECTED_LEVELS[*]})"
echo ""

# submit <job_name> <extra_export> [extra_sbatch_opts...]; echoes the job id (or DRY)
submit() {
    local name="$1"; local export_extra="$2"; shift 2
    local cmd=(sbatch --parsable --job-name="${name}" "${SB_OPTS[@]}" "$@"
               --export="${EXPORT_COMMON},${export_extra}" "${SBATCH_SCRIPT}")
    if $DRY_RUN; then
        echo "[DRY] ${cmd[*]}" >&2
        echo "DRYID"
    else
        echo "submit ${name}" >&2
        "${cmd[@]}"
    fi
}

# ── prefetch: warm the HF cache once; cells depend on it (afterok) ─────────
# Uses its own EXP_ID=warmcache so its run dir is NOT picked up by the
# aggregator's *${EXP_ID}* glob.
DEP_OPT=()
if $DO_PREFETCH; then
    PF_EXPORT="ALL,CONDITION=none,N_ITEMS=1,EXP_ID=warmcache,CONFIG=${CONFIG},SPLIT=${SPLIT},REPO_DIR=${REPO_DIR}"
    if $DRY_RUN; then
        echo "[DRY] sbatch --parsable --job-name=scaff-prefetch ${SB_OPTS[*]:-} --export=${PF_EXPORT} ${SBATCH_SCRIPT}"
        PF_ID=DRYID
    else
        PF_ID=$(sbatch --parsable --job-name=scaff-prefetch "${SB_OPTS[@]}" --export="${PF_EXPORT}" "${SBATCH_SCRIPT}")
        echo "prefetch job: ${PF_ID}"
        DEP_OPT=(--dependency=afterok:${PF_ID})
    fi
fi

NUM=0
if $DO_NONE; then
    submit "scaff-none" "CONDITION=none" "${DEP_OPT[@]}" >/dev/null
    NUM=$((NUM + 1))
fi
for lvl in "${SELECTED_LEVELS[@]}"; do
    tag=${lvl:0:1}   # b / s
    for p in "${SELECTED_PERSONAS[@]}"; do
        submit "scaff-${tag}${p}" "CONDITION=prompt,PERSONA_IDX=${p},PERSONA_LEVEL=${lvl}" "${DEP_OPT[@]}" >/dev/null
        NUM=$((NUM + 1))
    done
done

echo ""
echo "Submitted ${NUM} cell jobs (+$($DO_PREFETCH && echo 1 || echo 0) prefetch). Monitor: squeue -u \$USER"
echo "Aggregate when done:"
echo "    python scripts/04_aggregate_scaffolding.py --exp-id ${EXP_ID}"
