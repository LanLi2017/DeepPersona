#!/bin/bash
# Source this in sbatch jobs to activate the project env on Delta.
# Usage: source activate_hpc.sh
module load miniforge3-python
source /projects/bbrz/yirenl2/DeepPersona/.venv/bin/activate
# HF weights cache. /scratch is not mounted on this cluster, so default to the
# (already-populated) home cache. Override by exporting HF_HOME before sourcing.
export HF_HOME="${HF_HOME:-/u/yirenl2/.cache/huggingface}"
