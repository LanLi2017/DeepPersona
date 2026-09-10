#!/usr/bin/env python3
"""Separate required-test control; preserve132's original strict gate and dose."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'runs/swe-diversity-selection/swe-localize-repair-support'
OUT = SUPPORT / 'full_training_required_tests'
DOC = ROOT / 'docs/swe_required_test_control_plan.md'

spec = importlib.util.spec_from_file_location('required_test_train132', ROOT / 'scripts/132_swe_localize_full_train.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.OUT = OUT
q.EVALUATION = SUPPORT / 'evaluation_required_tests'
q.INFERENCE = SUPPORT / 'autonomous_inference_required_tests'
# q.INITIALS intentionally remains the original134 public-initial barrier.
CONTROL = 'required_test_control_v1'


def checks():
    # Inherited report correctly hashes132 and the unchanged learner helper.
    q.checks()


def freeze():
    assert not (OUT / 'protocol.json').exists(), 'One prospective required-test control only'
    assert not (OUT / 'run').exists() and not (OUT / 'outer_dispatch.json').exists()
    required = [Path(__file__), ROOT / 'scripts/145_swe_required_test_train_launcher.py',
                ROOT / 'scripts/141_swe_required_test_grading.py',
                ROOT / 'scripts/142_swe_required_test_readiness.py', DOC]
    for path in required:
        assert path.is_file(), str(path)
    original_save = q.h.save

    def save_with_control_binding(path, value):
        if Path(path) == OUT / 'protocol.json':
            assert value['evaluation_readiness_attested'] and not value['evaluation_private_values_read']
            assert value['expected_optimizer_steps'] == 18 and value['expected_causal_row_exposures'] == 255
            assert value['expected_task_exposures'] == 69 and value['epochs'] == 3 and value['task_batch_size'] == 4
            assert value['initial_adapter_path'] == str(q.SMOKE / 'run/initial_adapter.pt')
            value['inputs_sha256'].update({str(p): q.h.sha(p) for p in required})
            value.update(
                control_namespace=CONTROL,
                scope='Separate prospective ordinary-supervision control using official required-test scoring; no diversity claim',
                benchmark_scoring_correction=True,
                correction_reason='Original strict20-task readiness failed18/20 because two gold processes returned exit1 from unlisted pre-existing errors; all20 official required-test pairs pass. Preserve the failed strict control and test the full original cohort under explicitly frozen required-test semantics.',
                original_strict_control_preserved=True,
                original_strict_training_namespace=str(SUPPORT / 'full_training'),
                original_strict_readiness_namespace=str(SUPPORT / 'evaluation_readiness_versions'),
                evaluation_scoring='Official required FAIL_TO_PASS and PASS_TO_PASS statuses; unlisted failures and process exits retained as diagnostics. No changes to20tasks,23sources,training dose or inference limits.',
                control_plan_sha256=q.h.sha(DOC),
                wrapper_sha256=q.h.sha(Path(__file__)),
                active_launcher_sha256=q.h.sha(ROOT / 'scripts/145_swe_required_test_train_launcher.py'),
            )
        return original_save(path, value)

    q.h.save = save_with_control_binding
    try:
        q.freeze()
    finally:
        q.h.save = original_save


def run():
    protocol = q.h.read(OUT / 'protocol.json')
    assert protocol['control_namespace'] == CONTROL and protocol['benchmark_scoring_correction']
    assert protocol['original_strict_control_preserved']
    assert protocol['wrapper_sha256'] == q.h.sha(Path(__file__))
    assert protocol['active_launcher_sha256'] == q.h.sha(ROOT / 'scripts/145_swe_required_test_train_launcher.py')
    assert protocol['control_plan_sha256'] == q.h.sha(DOC)
    q.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['checks', 'freeze', 'run'])
    globals()[parser.parse_args().stage]()
