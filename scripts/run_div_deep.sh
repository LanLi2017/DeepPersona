#!/usr/bin/env bash
# F1 deeper within-arm dose-response: 60 GRPO steps x 3 seeds x 3 knob levels (neutral/4/8 prompts).
# Same regime as first-look (Qwen3-8B non-thinking, MATH lvl 4-5, lr 1e-5). Seed-outer so a full
# dose-response (seed 0, all 3 levels) is ready early. Sequential; per-run logs in logs/.
cd /scratch/yirenl2/DeepPersona
set -a; . ./.tinker_env; set +a          # load TINKER_API_KEY (not echoed)
export HF_HOME=/scratch/yirenl2/.cache/huggingface
mkdir -p logs

BASE="--arm within --group-total 8 --batch-questions 16 --max-tokens 1024 --lr 1e-5 \
--n-steps 60 --eval-every 10 --n-test 40 --test-group-size 8 --levels 4,5 --train-size 1000 --tag deep"

run () {  # $1=logname  $2...=extra args
  local log="logs/$1.log"; shift
  echo "[driver] start $log $(date -u +%H:%M:%S)"
  .venv/bin/python scripts/10_diversity_grpo.py $BASE "$@" > "$log" 2>&1 \
    && echo "[driver] done  $log $(date -u +%H:%M:%S)" \
    || echo "[driver] FAIL  $log $(date -u +%H:%M:%S)"
}

for s in 0 1 2; do
  run divgrpo_deep_nd1_s$s --prompt-source neutral  --n-distinct 1 --seed $s
  run divgrpo_deep_nd4_s$s --prompt-source strategy --n-distinct 4 --seed $s
  run divgrpo_deep_nd8_s$s --prompt-source strategy --n-distinct 8 --seed $s
done
echo "[driver] ALL DONE $(date -u +%H:%M:%S)"
