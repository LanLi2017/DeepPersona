---
name: reference-delta-slurm
description: How to run DeepPersona GPU jobs on the NCSA Delta Slurm cluster — partition/account, env activation, HF cache location, and the /scratch gotcha.
metadata:
  type: reference
---

DeepPersona runs on the **NCSA Delta** Slurm HPC. Conventions (mirrored from `/projects/bbrz/yirenl2/demonstrated-feedback`, the user's other repo):

- **Partition:** `gpuA100x4`  **Account:** `bbrz-delta-gpu`  (1×A100 via `--gpus-per-node=1`, `--mem=48g`).
- **Env activation in jobs:** `module load miniforge3-python` then `source <repo>/.venv/bin/activate` (uv-created venv; no separate cuda module — torch cu124 bundles it). DeepPersona now has `activate_hpc.sh` doing this.
- **HF weights cache:** `/u/yirenl2/.cache/huggingface` (home; 32G, populated). `HF_HOME` is unset by default → HF uses this path anyway.
- **GOTCHA: `/scratch/yirenl2` does NOT exist on this cluster** even though `README.md` / `docs` reference `/scratch`. Use home (`/u/yirenl2`) or `/projects/bbrz/yirenl2`. The README's "cache pinned to /scratch" is stale/aspirational.
- **Submission idiom:** one `sbatch` job per parallel unit via a loop wrapper using `--export=ALL,VAR=...` env vars + a `--dry` preview (not job arrays).

**Scaffolding sweep specifics:** `scripts/submit_scaffolding.sh` submits 25 per-cell jobs (none + basic×12 + structured×12), each = one `01_baseline.py` run sharing `--run-name <EXP_ID>`; recombine with `scripts/04_aggregate_scaffolding.py --exp-id <EXP_ID>`.

**Why prefetch matters:** as of 2026-05-28 the pinned `Qwen/Qwen2.5-7B-Instruct` is NOT in the HF cache (only Llama-3.1-8B / Mistral-7B etc. from the other project are). So the submit wrapper runs a single prefetch job first (`EXP_ID=warmcache`, 1 item) and makes all cells `--dependency=afterok:<jid>` to avoid 25 concurrent 15GB downloads. Drop with `--no-prefetch` once cached. **How to apply:** confirm the model is cached before assuming runs are cheap; keep prefetch on for the first sweep. See [[project-benchmark-selection]].
