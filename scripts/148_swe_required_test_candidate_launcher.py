#!/usr/bin/env python3
"""Reuse140 process-group deadlines for the separate147 required-test experiment."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load('required_launcher140', ROOT/'scripts/140_swe_localize_candidate_launcher.py')
s = load('required_launcher147', ROOT/'scripts/147_swe_required_test_candidates.py')
previous.OUT = s.OUT
previous.SOURCE = Path(s.__file__)
previous.s = s
# main's launch/review receipts must bind the actual148 entry point, not140.
previous.__file__ = str(Path(__file__).resolve())
OUT, SOURCE = s.OUT, Path(s.__file__)
sha, read, save = s.sha, s.read, s.save
controlled, main = previous.controlled, previous.main


if __name__ == '__main__':
    main()
