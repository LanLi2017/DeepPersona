#!/usr/bin/env python3
"""Instantiate unchanged131 direct-site attestation with explicit official pins."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strict131versioned", ROOT / "scripts/131_swe_environment_provenance.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
NAMES = [s.split("==")[0] for s in original.PINS]


def pins(packages):
    assert set(packages) == set(NAMES), "Exactly the same nine dependency names required"
    expected = dict(s.split("==") for s in original.PINS)
    assert packages["flake8-comprehensions"] in {"3.15.0", "3.16.0"}
    for name in NAMES:
        if name != "flake8-comprehensions":
            assert packages[name] == expected[name], name
    return [name + "==" + packages[name] for name in NAMES]


def probe(packages):
    """Only substitute the declared PINS assignment, not observed values or checks."""
    old = "PINS = " + repr(original.PINS) + "\n"
    assert original.PROBE.startswith(old)
    return "PINS = " + repr(pins(packages)) + "\n" + original.PROBE[len(old) :]
