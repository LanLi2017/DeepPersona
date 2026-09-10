#!/usr/bin/env python3
"""Required-PASSED SWE endpoint with explicit process integrity and strict sensitivity."""
import argparse
import ast
from enum import Enum
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'runs/swe-diversity-selection/swe-execution-bridge/harness/swebench/harness'
PINNED = {
    'grading.py': 'a095273010175001a6779e365b3b77abb8657c7ac2e4f80b4bc360faa6f09483',
    'constants/__init__.py': '8895ab5313d874349c4b07bd223bb9eb00acaee377b432e67cc7ad840341e79b',
    'log_parsers/python.py': 'cd56156414f8327221e525665ace9b184f7d73e83b272d9eb3f545fb17c2d9bc',
}
RULE = ('Primary requires completed normal exit0/1, no timeout, nonempty FAIL_TO_PASS, '
        'every named FAIL_TO_PASS/PASS_TO_PASS exactly PASSED, and for exit1 at least '
        'one FAILED/ERROR in the full actual parser map. Strict exit0 sensitivity is '
        'reported separately. Exact official skip/XFAIL semantics are diagnostic only.')


@lru_cache(maxsize=1)
def official():
    for name, digest in PINNED.items():
        assert hashlib.sha256((HARNESS / name).read_bytes()).hexdigest() == digest
    names = {'FAIL_TO_PASS', 'PASS_TO_PASS', 'FAIL_TO_FAIL', 'PASS_TO_FAIL'}
    constants = HARNESS / 'constants/__init__.py'
    nodes = [n for n in ast.parse(constants.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in {'EvalType', 'TestStatus', 'ResolvedStatus'}
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = dict(Enum=Enum, re=re, Any=Any)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(constants), 'exec'), ns)
    funcs = {'_resolve_case', 'test_passed', 'test_maintained', 'test_failed',
             'get_eval_tests_report', 'compute_fail_to_pass', 'compute_pass_to_pass', 'get_resolution_status'}
    grading = HARNESS / 'grading.py'
    nodes = [n for n in ast.parse(grading.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in funcs]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(grading), 'exec'), ns)
    parser = HARNESS / 'log_parsers/python.py'
    nodes = [n for n in ast.parse(parser.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'parse_log_sympy']
    ns['TestSpec'] = Any
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(parser), 'exec'), ns)
    return ns


def parse_log(text):
    return official()['parse_log_sympy'](text, None)


def grade(tests, parsed, exit_code, *, completed=True, timed_out=False):
    assert set(tests) == {'FAIL_TO_PASS', 'PASS_TO_PASS'}
    assert isinstance(parsed, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items())
    assert all(isinstance(v, list) and len(v) == len(set(v)) and all(isinstance(x, str) for x in v) for v in tests.values())
    nonempty = bool(tests['FAIL_TO_PASS'])
    normal_exit = type(exit_code) is int and exit_code in (0, 1)
    exit_consistent = exit_code == 0 or any(v in {'FAILED', 'ERROR'} for v in parsed.values())
    integrity = bool(completed is True and timed_out is False and normal_exit and exit_consistent)
    statuses = {suite: {test: parsed.get(test, 'MISSING') for test in names} for suite, names in tests.items()}
    required_passed = all(status == 'PASSED' for suite in statuses.values() for status in suite.values())
    primary = bool(integrity and nonempty and required_passed)
    baseline = bool(integrity and nonempty and
                    all(v in {'FAILED', 'ERROR'} for v in statuses['FAIL_TO_PASS'].values()) and
                    all(v == 'PASSED' for v in statuses['PASS_TO_PASS'].values()))
    ns = official()
    report = ns['get_eval_tests_report'](parsed, tests, eval_type=ns['EvalType'].PASS_AND_FAIL)
    resolution = ns['get_resolution_status'](report)
    return dict(primary_resolved=primary, strict_exit0_resolved=bool(primary and exit_code == 0),
                execution_integrity_valid=integrity, completed=completed, timed_out=timed_out,
                normal_exit=normal_exit, exit_status_consistent=bool(exit_consistent), exit_code=exit_code,
                nonempty_fail_to_pass=nonempty, required_statuses=statuses, all_required_passed=required_passed,
                baseline_required_pattern=baseline, official_resolution_status=resolution,
                official_resolved=bool(integrity and nonempty and resolution == 'RESOLVED_FULL'),
                official_test_report=report)


def synthetic():
    tests = {'FAIL_TO_PASS': ['target'], 'PASS_TO_PASS': ['regression']}
    good = {'target': 'PASSED', 'regression': 'PASSED'}
    cases = [
        ('normal_pass', good, 0, {}, True, True),
        ('unlisted_existing_error', dict(good, unrelated='ERROR'), 1, {}, True, False),
        ('unlisted_existing_failure', dict(good, unrelated='FAILED'), 1, {}, True, False),
        ('exit1_without_any_failure', good, 1, {}, False, False),
        ('signal', dict(good, unrelated='ERROR'), -9, {}, False, False),
        ('abnormal_exit', dict(good, unrelated='ERROR'), 2, {}, False, False),
        ('timeout_despite_printed_pass', good, 0, {'timed_out': True}, False, False),
        ('incomplete_despite_printed_pass', good, 0, {'completed': False}, False, False),
        ('missing_target', {'regression': 'PASSED'}, 0, {}, False, False),
        ('missing_regression', {'target': 'PASSED'}, 0, {}, False, False),
        ('regression_failure', dict(good, regression='FAILED'), 1, {}, False, False),
        ('target_failure', dict(good, target='FAILED'), 1, {}, False, False),
        ('xfail_is_diagnostic_only', dict(good, target='XFAIL'), 0, {}, False, False),
        ('skipped_regression_is_diagnostic_only', dict(good, regression='SKIPPED'), 0, {}, False, False),
        ('boolean_not_exitcode', good, False, {}, False, False),
    ]
    rows = []
    for name, parsed, code, kwargs, primary, strict in cases:
        result = grade(tests, parsed, code, **kwargs)
        assert result['primary_resolved'] is primary and result['strict_exit0_resolved'] is strict, name
        if name in {'xfail_is_diagnostic_only', 'skipped_regression_is_diagnostic_only'}:
            assert result['official_resolved']
        rows.append(dict(name=name, passed=True))
    assert not grade({'FAIL_TO_PASS': [], 'PASS_TO_PASS': ['regression']}, good, 0)['primary_resolved']
    rows.append(dict(name='empty_targets_rejected', passed=True))
    result = grade(tests, dict(target='ERROR', regression='PASSED'), 1)
    assert result['baseline_required_pattern'] and not result['primary_resolved']
    rows.append(dict(name='baseline_required_pattern', passed=True))
    return dict(all_checks_passed=True, checks=rows, count=len(rows), rule=RULE,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), official_source_sha256=PINNED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(synthetic(), indent=2))
