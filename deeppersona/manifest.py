"""Run manifest: pin everything a reviewer could ask for."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig


def _git(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _version(pkg: str) -> str:
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "")
    except Exception:
        return ""


def make_run_dir(base: Path, cfg: RunConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    sha = _git(["rev-parse", "--short", "HEAD"], base) or "nogit"
    name = f"{ts}-{sha}-{cfg.run_name}-{cfg.task}-{cfg.condition}"
    if cfg.condition == "prompt" and cfg.persona_idx >= 0:
        name += f"-p{cfg.persona_idx}"
    if cfg.smoke:
        name += "-smoke"
    run_dir = base / "runs" / name
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, cfg: RunConfig, repo_root: Path) -> None:
    try:
        import torch
        cuda_dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        cuda_dev = ""
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "sha": _git(["rev-parse", "HEAD"], repo_root),
            "short_sha": _git(["rev-parse", "--short", "HEAD"], repo_root),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "dirty": bool(_git(["status", "--porcelain"], repo_root)),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_device": cuda_dev,
        },
        "versions": {
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "datasets": _version("datasets"),
            "nnsight": _version("nnsight"),
        },
        "config": asdict(cfg),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def append_item(run_dir: Path, record: dict) -> None:
    with (run_dir / "raw" / "items.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
