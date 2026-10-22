#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
mkdir -p outputs/g1-replay/logs outputs/g1-replay/reports
exec sbatch scripts/slurm/validate_alfworld_replay.sbatch
