#!/usr/bin/env bash
# F1 first-look: within-arm rollout-input-diversity dose-response.
# 3 runs (D0 neutral / D2=4 strategy prompts / D3=8 strategy prompts), 20 GRPO steps each,
# Qwen3-8B non-thinking, MATH levels 4-5 train -> MATH-500 neutral-prompt held-out eval.
# Sequential (one Tinker training client at a time). Per-run logs in logs/.
cd /scratch/yirenl2/DeepPersona
set -a; . ./.tinker_env; set +a          # load TINKER_API_KEY (not echoed)
export HF_HOME=/scratch/yirenl2/.cache/huggingface
mkdir -p logs

COMMON="--arm within --group-total 8 --batch-questions 16 --max-tokens 1024 \
--lr 1e-5 --n-steps 20 --eval-every 5 --n-test 40 --test-group-size 8 \
--levels 4,5 --train-size 1000 --seed 0 --tag firstlook"

run () {  # $1=logname  $2...=extra args
  local log="logs/$1.log"; shift
  echo "[driver] start $log $(date -u +%H:%M:%S)"
  .venv/bin/python scripts/10_diversity_grpo.py $COMMON "$@" > "$log" 2>&1 \
    && echo "[driver] done  $log $(date -u +%H:%M:%S)" \
    || echo "[driver] FAIL  $log $(date -u +%H:%M:%S)"
}

run divgrpo_D0_neutral --prompt-source neutral  --n-distinct 1
run divgrpo_D2_nd4      --prompt-source strategy --n-distinct 4
run divgrpo_D3_nd8      --prompt-source strategy --n-distinct 8
echo "[driver] ALL DONE $(date -u +%H:%M:%S)"
