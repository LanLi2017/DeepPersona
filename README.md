# DeepPersona

White-box LLM steering research: dissociating **style** from **competence** in persona conditioning (`docs/proposal_one_pager.md`, `docs/q1_experimental_design.md`).

## Setup (uv)

```bash
cd /scratch/yirenl2/DeepPersona
uv sync                                 # installs torch (cu124), transformers, datasets, nnsight, ...
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The uv cache is pinned to `/scratch/yirenl2/.cache/uv` in `pyproject.toml` (home has a quota).

## Phase 1 smoke (GSM8K)

```bash
# 1) C-none, 4 items
CUDA_VISIBLE_DEVICES=0 uv run python scripts/01_baseline.py \
    --config configs/qwen25_7b.yaml --condition none --split val --smoke

# 2) C-prompt with persona 0, 4 items
CUDA_VISIBLE_DEVICES=0 uv run python scripts/01_baseline.py \
    --config configs/qwen25_7b.yaml --condition prompt --persona-idx 0 --split val --smoke
```

Outputs land in `runs/<timestamp>-<git>-<name>/`:
- `manifest.json` — git SHA, library versions, full config.
- `raw/items.jsonl` — per-item prompt, generation, parsed answer, correctness.
- `metrics.json` — accuracy + bootstrap 95% CI + parse-fail rate.

## Persona selection (val only)

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/02_select_persona.py \
    --config configs/qwen25_7b.yaml > logs/select-gsm8k.log 2>&1 &
```

Writes `runs/<...>-persona-select-.../selection.json` with per-persona accuracies and the locked best persona.

## Repo layout

```
deeppersona/   # shared utilities (config, data, generate, verifiers, manifest, personas)
scripts/       # one script = one job (01_baseline, 02_select_persona)
configs/       # per-model YAML with pinned revisions
data/splits/   # frozen 200-item calib/val indices (committed)
runs/          # per-run outputs (gitignored)
logs/          # nohup logs (gitignored)
docs/          # design docs
```

## Conventions

- One script = one job. Flat `argparse` + small `RunConfig` dataclass.
- Greedy decoding for headline accuracy; matched `max_new_tokens` across conditions.
- Pin model revision in config (CLAUDE.md: never float on `main`).
- Calibration + val from train split; test untouched until reporting.
- Every run writes a manifest + raw per-item outputs so analysis is re-runnable offline.
