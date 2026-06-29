#!/usr/bin/env bash
# F1 headroom test: within-arm diversity dose-response on the HIGHER-HEADROOM math venue
# (Hendrycks MATH level-5, picked by scripts/11 venue probe: pass@1~0.32, pass@16~0.55).
# Same design as run_div_deep.sh but --levels 5 and --max-tokens 1536 (probe showed 1024 truncated
# 56% of level-5 solutions before \boxed). Eval is level-5 MATH-500 (matched headroom).
# Tests whether the F1 Link-2 null was a near-ceiling artifact. Sequential; logs in logs/.
cd /scratch/yirenl2/DeepPersona
set -a; . ./.tinker_env; set +a          # load TINKER_API_KEY (not echoed)
export HF_HOME=/scratch/yirenl2/.cache/huggingface
mkdir -p logs

BASE="--arm within --group-total 8 --batch-questions 16 --max-tokens 1536 --lr 1e-5 \
--n-steps 60 --eval-every 10 --n-test 40 --test-group-size 8 --levels 5 --train-size 1000 --tag ml5"

run () {  # $1=logname  $2...=extra args
  local log="logs/$1.log"; shift
  echo "[driver] start $log $(date -u +%H:%M:%S)"
  .venv/bin/python scripts/10_diversity_grpo.py $BASE "$@" > "$log" 2>&1 \
    && echo "[driver] done  $log $(date -u +%H:%M:%S)" \
    || echo "[driver] FAIL  $log $(date -u +%H:%M:%S)"
}

for s in 0 1 2; do
  run divgrpo_ml5_nd1_s$s --prompt-source neutral  --n-distinct 1 --seed $s
  run divgrpo_ml5_nd4_s$s --prompt-source strategy --n-distinct 4 --seed $s
  run divgrpo_ml5_nd8_s$s --prompt-source strategy --n-distinct 8 --seed $s
done
echo "[driver] ALL DONE $(date -u +%H:%M:%S)"
