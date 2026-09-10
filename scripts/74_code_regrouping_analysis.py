#!/usr/bin/env python3
"""Frozen descriptive analysis of the 22-task identical-multiset intervention."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.stats import rankdata

OUT = Path('runs/swe-diversity-selection/code-regrouping-interaction-v2')
NOTE = Path('docs/swe_regrouping_analysis_protocol.md')
SEED = 2026090974
RESAMPLES = 20000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {'percentile95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            'valid_resamples': len(finite), 'undefined_resamples': len(values) - len(finite)}


def mean_report(values, indices):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'minimum': float(values.min()), 'maximum': float(values.max()),
            'positive': int((values > 0).sum()), 'negative': int((values < 0).sum()),
            'zero': int((values == 0).sum()),
            'bootstrap_mean': interval(values[indices].mean(axis=1))}


def correlation(x, y):
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((x * x).sum(axis=-1) * (y * y).sum(axis=-1))
    return np.divide((x * y).sum(axis=-1), denominator,
                     out=np.full(np.shape(denominator), np.nan), where=denominator > 0)


def associations(x, y, seed):
    x, y = np.asarray(x, float), np.asarray(y, float)
    support = np.isfinite(x) & np.isfinite(y)
    x, y = x[support], y[support]
    result = {'tasks': len(x), 'undefined_feature_tasks': int((~support).sum())}
    if len(x) < 2:
        return {**result, 'pearson': None, 'spearman': None}
    indices = np.random.default_rng(seed).integers(len(x), size=(RESAMPLES, len(x)))
    for name, xx, yy in [('pearson', x, y),
                         ('spearman', rankdata(x), rankdata(y))]:
        point = float(correlation(xx, yy))
        bx, by = x[indices], y[indices]
        if name == 'spearman':
            bx, by = rankdata(bx, axis=1), rankdata(by, axis=1)
        result[name] = {'estimate': point if np.isfinite(point) else None,
                        'bootstrap': interval(correlation(bx, by))}
    return result


def contrast(losses, tasks=21):
    assert set(losses) == {'mixed', 'uv', 'vu'}
    assert all(len(v) == tasks for v in losses.values())
    a = {k: np.asarray(v, float) for k, v in losses.items()}
    assert all(np.isfinite(v).all() for v in a.values())
    return float(np.mean((a['uv'] + a['vu']) / 2 - a['mixed']))


def freeze(out):
    target = out / 'analysis_protocol.json'
    assert not target.exists(), 'Analysis already frozen'
    # Check existence only; never inspect smoke or partial finite-path outcomes.
    assert not (out / 'per_task.jsonl').exists(), 'Freeze must precede full outcomes'
    protocol = read(out / 'protocol.json')
    assert len(protocol['task_ids']) == 22 and protocol['etas'] == [1.0, 10.0]
    sources = [Path(__file__), NOTE, out / 'protocol.json', out / 'groups.json',
               out / 'selected_raw.jsonl', out / 'reference_tokens.jsonl',
               Path('scripts/72_code_regrouping_interaction_v2.py')]
    save(target, {'utc': datetime.now(timezone.utc).isoformat(),
                  'frozen_before_full_outcomes': True,
                  'finite_path_smoke_outcomes_read': False,
                  'task_ids': protocol['task_ids'], 'etas': [1.0, 10.0],
                  'bootstrap_resamples': RESAMPLES, 'bootstrap_seed': SEED,
                  'inference': 'Descriptive paired source-task bootstrap only; fixed shared21 measurement tasks; no p-values, outcome selection, fitted coefficients, or confirmatory claim.',
                  'source_sha256': {str(p): sha(p) for p in sources}})
    print(json.dumps({'frozen': str(target), 'sha256': sha(target)}))


def analyze(out):
    frozen = read(out / 'analysis_protocol.json')
    for name, expected in frozen['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    completed = read(out / 'completed_manifest.json')
    for name, expected in completed['sha256'].items():
        assert sha(out / name) == expected, name
    summary, manifest = read(out / 'summary.json'), read(out / 'manifest.json')
    assert summary['technical_checks_passed'] and summary['tasks'] == 22
    assert summary['calibration_tasks'] == summary['measurement_tasks'] == 21
    assert manifest['full_root_review_flag'] and not summary['root_review_required_before_full']
    rows = [json.loads(line) for line in (out / 'per_task.jsonl').read_text().splitlines()]
    assert [r['task_id'] for r in rows] == frozen['task_ids'] == summary['task_ids']
    assert len({r['task_id'] for r in rows}) == 22
    indices = np.random.default_rng(SEED).integers(22, size=(RESAMPLES, 22))
    b = np.array([r['score_B'] for r in rows], float)
    diversity = np.array([r['diversity']['contrast'] if r['diversity'] else np.nan for r in rows])
    assert np.isfinite(b).all()
    assert np.all(diversity[np.isfinite(diversity)] >= -1e-10)
    for r in rows:
        assert len(set(r['occurrence_tokens_per_path'].values())) == 1
        if r['diversity']:
            assert abs(r['diversity']['contrast'] - r['diversity']['identity']) < 1e-10
        assert abs(contrast(r['results']['0.0']['measurement_losses'])) <= 1e-7
    report = {'scope': frozen['inference'], 'source_tasks': 22, 'fixed_measurement_tasks': 21,
              'all_tasks_retained': True, 'both_eta_settings_reported': True,
              'duplicate_content_tasks': sum(bool(r['same_token_content_pairs']) for r in rows),
              'undefined_diversity_task_ids': [r['task_id'] for r in rows if r['diversity'] is None],
              'B_positive': int((b > 0).sum()), 'B_negative': int((b < 0).sum()),
              'B_zero_randomized_ties': int((b == 0).sum()), 'by_eta': {}, 'per_task': []}
    for eta in frozen['etas']:
        key = str(eta)
        delta = np.array([contrast(r['results'][key]['measurement_losses']) for r in rows])
        null = np.array([contrast(r['frozen_gradient_null'][key]['measurement_losses']) for r in rows])
        prediction = eta ** 2 * b
        lift = .5 * np.sign(b) * delta
        residual = delta - prediction
        for i, r in enumerate(rows):
            assert np.isclose(delta[i], r['results'][key]['mixed_advantage'], rtol=0, atol=1e-14)
            assert np.isclose(null[i], r['frozen_gradient_null'][key]['mixed_advantage'], rtol=0, atol=1e-14)
            assert np.isclose(prediction[i], r['results'][key]['prediction'], rtol=1e-12, atol=1e-16)
            assert np.isclose(residual[i], r['results'][key]['residual'], rtol=1e-12, atol=1e-14)
            report['per_task'].append({'task_id': r['task_id'], 'eta': eta, 'B': float(b[i]),
                'diversity_contrast': float(diversity[i]) if np.isfinite(diversity[i]) else None,
                'delta': float(delta[i]), 'B_sign_selector_lift': float(lift[i]),
                'prediction': float(prediction[i]), 'calibration_transfer_prediction_residual': float(residual[i]),
                'frozen_gradient_null_delta': float(null[i]), 'delta_minus_null': float(delta[i] - null[i])})
        report['by_eta'][key] = {
            'always_mixed_advantage': mean_report(delta, indices),
            'B_sign_selector_lift_vs_fifty_fifty': mean_report(lift, indices),
            'B_vs_delta': associations(b, delta, SEED + 1),
            'raw_diversity_contrast_vs_delta': associations(diversity, delta, SEED + 2),
            'calibration_transfer_prediction_residual': {
                'mean': float(residual.mean()), 'median_absolute': float(np.median(abs(residual))),
                'max_absolute': float(max(abs(residual))),
                'interpretation': 'Includes calibration-to-measurement mismatch, higher-order effects and numerical error; not a pure Taylor remainder.'},
            'frozen_gradient_null_scale': {
                'mean': float(null.mean()), 'median_absolute': float(np.median(abs(null))),
                'max_absolute': float(max(abs(null))),
                'max_endpoint_absolute_gap': max(r['frozen_gradient_null'][key]['max_endpoint_absolute_gap'] for r in rows),
                'actual_median_absolute': float(np.median(abs(delta))),
                'actual_max_absolute': float(max(abs(delta))),
                'tasks_abs_actual_le_abs_own_null': int((abs(delta) <= abs(null)).sum()),
                'interpretation': 'Descriptive measured numerical scale, not a universal error bound, significance cutoff, or exclusion rule.'},
            'delta_minus_null_diagnostic': mean_report(delta - null, indices),
            'prediction_sign_agreement_tasks': int((np.sign(prediction) == np.sign(delta)).sum()),
            'zero_step_max_absolute_contrast': max(abs(contrast(r['results']['0.0']['measurement_losses'])) for r in rows)}
    report['input_sha256'] = {name: sha(out / name) for name in
                              ['analysis_protocol.json', 'completed_manifest.json', 'per_task.jsonl']}
    save(out / 'analysis.json', report)
    print(json.dumps({'written': str(out / 'analysis.json'), 'source_tasks': 22, 'outcomes': len(report['per_task'])}))


def self_test():
    assert contrast({'uv': [4, 6], 'vu': [2, 8], 'mixed': [2, 4]}, tasks=2) == 2
    b, delta = np.array([-2., 0., 2.]), np.array([-4., 7., 6.])
    assert np.array_equal(.5 * np.sign(b) * delta, [2, 0, 3])
    assert float(correlation(np.array([1., 2., 3.]), np.array([3., 2., 1.]))) == -1
    assert np.isnan(correlation(np.ones(3), np.arange(3.)))
    result = associations([1, 1, 2, np.nan], [1, 1, 3, 4], SEED)
    assert result['tasks'] == 3 and result['spearman']['estimate'] == 1
    assert result['spearman']['bootstrap']['undefined_resamples'] > 0
    print('Synthetic sign, tie, average-loss, constant-feature and missing-support checks passed; no experiment outcomes read.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'analyze', 'self-test'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.out)
    elif args.stage == 'analyze':
        analyze(args.out)
    else:
        self_test()
