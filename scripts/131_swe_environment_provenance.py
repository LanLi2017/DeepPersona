#!/usr/bin/env python3
"""Strict direct-site dependency attestation; self-contained Python3.9 probe, no installs."""

import argparse
import hashlib
import json
from pathlib import Path

PINS = [
    "pip==24.2",
    "setuptools==75.1.0",
    "wheel==0.44.0",
    "mpmath==1.3.0",
    "flake8==7.1.1",
    "mccabe==0.7.0",
    "pycodestyle==2.12.1",
    "pyflakes==3.2.0",
    "flake8-comprehensions==3.15.0",
]

COLLECTOR = r"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import sysconfig
sys.dont_write_bytecode = True

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def validate(provenance):
    errors = []
    prefix = Path(provenance['sys_prefix']).resolve()
    expected = provenance['expected_virtual_env']
    roots = {str(Path(root).resolve()) for root in provenance['direct_site_roots']}
    if not expected or prefix != Path(expected).resolve():
        errors.append('sys_prefix_does_not_match_expected_virtual_env')
    if not roots or any(not Path(root).is_relative_to(prefix) for root in roots):
        errors.append('direct_site_root_outside_virtual_env')
    packages = {}
    for spec in PINS:
        name, pinned = spec.split('==')
        candidates = [row for row in provenance['direct_distributions'] if normalize(row['name']) == normalize(name)]
        if len(candidates) != 1:
            errors.append(name + ':direct_distribution_count_' + str(len(candidates)))
            continue
        direct = candidates[0]
        packages[name] = direct['version']  # Actual observed value, never substitute the pin.
        if direct['version'] != pinned:
            errors.append(name + ':direct_version_mismatch')
        location = str(Path(direct['metadata_path']).resolve())
        if str(Path(location).parent) not in roots:
            errors.append(name + ':metadata_not_direct_site_child')
        resolved = provenance['resolved_distributions'].get(name)
        if resolved is None:
            errors.append(name + ':resolved_distribution_missing')
        elif (str(Path(resolved['metadata_path']).resolve()) != location or resolved['version'] != direct['version'] or normalize(resolved['name']) != normalize(name)):
            errors.append(name + ':resolved_distribution_not_direct_entry')
    wheel_candidates = [row for row in provenance['direct_distributions'] if normalize(row['name']) == 'wheel']
    if len(wheel_candidates) == 1:
        expected_wheel = Path(wheel_candidates[0]['metadata_path']).resolve().parent / 'wheel' / '__init__.py'
        if Path(provenance['wheel_import_file']).resolve() != expected_wheel.resolve():
            errors.append('wheel:import_origin_not_direct_package')
        if provenance['wheel_import_version'] != wheel_candidates[0]['version']:
            errors.append('wheel:import_version_metadata_mismatch')
    if provenance['wheel_import_version'] != '0.44.0':
        errors.append('wheel:import_version_not_pinned')
    return packages, errors

def distribution_record(dist):
    location = Path(dist._path).resolve()
    info = next((location/name for name in ['METADATA', 'PKG-INFO'] if (location/name).is_file()), None)
    return dict(name=dist.metadata['Name'], version=dist.version, metadata_path=str(location),
                metadata_file_sha256=hashlib.sha256(info.read_bytes()).hexdigest() if info else None)

def collect():
    before = list(sys.path)
    import sympy
    import wheel
    roots = sorted({str(Path(sysconfig.get_path(key)).resolve()) for key in ['purelib','platlib']})
    direct = [distribution_record(dist) for dist in metadata.distributions(path=roots)]
    direct.sort(key=lambda row:(normalize(row['name']),row['metadata_path'],row['version']))
    discovered = [dict(discovery_index=index, **distribution_record(dist)) for index,dist in enumerate(metadata.distributions())]
    resolved = {}
    for spec in PINS:
        name = spec.split('==')[0]
        try:
            resolved[name] = distribution_record(metadata.distribution(name))
        except metadata.PackageNotFoundError:
            resolved[name] = None
    counts = {}
    for row in discovered:
        name = normalize(row['name']); counts[name] = counts.get(name,0)+1
    provenance = dict(sys_prefix=sys.prefix, expected_virtual_env=os.environ.get('VIRTUAL_ENV'),
        executable=sys.executable, direct_site_roots=roots, direct_distributions=direct,
        resolved_distributions=resolved, full_discovery=discovered,
        full_discovery_duplicates={name:[row for row in discovered if normalize(row['name'])==name] for name,count in counts.items() if count>1},
        sys_path_before_sympy=before, sys_path_after_sympy=list(sys.path),
        wheel_import_file=wheel.__file__,wheel_import_version=wheel.__version__,
        wheel_import_file_sha256=hashlib.sha256(Path(wheel.__file__).read_bytes()).hexdigest(),
        sympy_import_version=sympy.__version__, bytecode_writes_disabled=sys.dont_write_bytecode,
        policy='Exactly one direct purelib/platlib metadata entry per pin; resolved metadata must be that entry; imported wheel must match its direct package. Full discovery retained but never collapsed last-wins.')
    packages, errors = validate(provenance)
    return dict(python=platform.python_version(),packages=packages,sympy_file=sympy.__file__,
                provenance_all_checks_passed=not errors,diagnosis=errors,provenance=provenance)
"""
PROBE = (
    "PINS = "
    + repr(PINS)
    + "\n"
    + COLLECTOR
    + r"""
try:
    result = collect()
except Exception as error:
    result = dict(python=platform.python_version(),packages={},sympy_file=None,
                  provenance_all_checks_passed=False,diagnosis=['probe_exception:'+type(error).__name__],
                  error=dict(type=type(error).__name__,message=str(error)))
print(json.dumps(result,sort_keys=True))
"""
)


def synthetic(output):
    namespace = {"PINS": PINS}
    exec(COLLECTOR, namespace)
    validate = namespace["validate"]
    root = Path("/tmp/fabricated131/venv/site-packages")
    direct = [
        {
            "name": s.split("==")[0],
            "version": s.split("==")[1],
            "metadata_path": str(root / (s.replace("==", "-") + ".dist-info")),
        }
        for s in PINS
    ]
    base = {
        "sys_prefix": "/tmp/fabricated131/venv",
        "expected_virtual_env": "/tmp/fabricated131/venv",
        "direct_site_roots": [str(root)],
        "direct_distributions": direct,
        "resolved_distributions": {r["name"]: dict(r) for r in direct},
        "wheel_import_file": str(root / "wheel/__init__.py"),
        "wheel_import_version": "0.44.0",
        "full_discovery": direct
        + [
            {
                "name": "wheel",
                "version": "0.43.0",
                "metadata_path": str(root / "setuptools/_vendor/wheel-0.43.0.dist-info"),
            }
        ],
    }
    rows = []

    def check(name, record, passed):
        packages, errors = validate(record)
        observed = not errors
        rows.append(
            {
                "name": name,
                "passed": observed == passed,
                "observed_valid": observed,
                "diagnosis": errors,
                "actual_packages": packages,
            }
        )
        assert observed == passed, (name, errors)

    def clone():
        return json.loads(json.dumps(base))

    check("direct044_vendor043_accepted_without_overwrite", clone(), True)
    wrong = clone()
    wheel = next(r for r in wrong["direct_distributions"] if r["name"] == "wheel")
    wheel["version"] = "0.43.0"
    wrong["resolved_distributions"]["wheel"] = dict(wheel)
    wrong["wheel_import_version"] = "0.43.0"
    check("wrong_direct_version_not_rescued_by_vendor", wrong, False)
    missing = clone()
    missing["direct_distributions"] = [r for r in missing["direct_distributions"] if r["name"] != "wheel"]
    check("vendor_only_not_accepted", missing, False)
    duplicate = clone()
    duplicate["direct_distributions"].append(
        dict(next(r for r in direct if r["name"] == "wheel"), metadata_path=str(root / "Wheel-0.44.0.dist-info"))
    )
    check("duplicate_direct_normalized_name_rejected", duplicate, False)
    imported = clone()
    imported["wheel_import_file"] = str(root / "setuptools/_vendor/wheel/__init__.py")
    check("vendor_import_rejected", imported, False)
    resolved = clone()
    resolved["resolved_distributions"]["wheel"]["metadata_path"] = str(
        root / "setuptools/_vendor/wheel-0.43.0.dist-info"
    )
    check("resolved_vendor_metadata_rejected", resolved, False)
    escaped = clone()
    escaped["direct_site_roots"] = ["/tmp/other_venv/site-packages"]
    check("site_roots_outside_venv_rejected", escaped, False)
    wrongprefix = clone()
    wrongprefix["expected_virtual_env"] = "/tmp/different_venv"
    check("unexpected_prefix_rejected", wrongprefix, False)
    report = {
        "all_checks_passed": all(r["passed"] for r in rows),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(PROBE.encode()).hexdigest(),
        "checks": rows,
        "scope": "Fabricated distribution records only; no environments mutated, packages installed, tests or model work.",
    }
    with Path(output).open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {"passed": report["all_checks_passed"], "checks": len(rows), "source_sha256": report["source_sha256"]}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    synthetic(parser.parse_args().synthetic)
