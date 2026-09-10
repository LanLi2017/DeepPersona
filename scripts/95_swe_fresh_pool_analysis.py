#!/usr/bin/env python3
"""Frozen neutral-pool support summary; descriptive, no selector tuning."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/swe-diversity-selection/swe-fresh-development'
EVAL = OUT / 'private/candidate-evaluation'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


class DropDocstrings(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node
    visit_ClassDef = visit_Module
    visit_FunctionDef = visit_Module
    visit_AsyncFunctionDef = visit_Module


def main():
    assert not (OUT / 'pool_summary.json').exists()
    assert read(EVAL / 'completed.json')['slots'] == 48
    for name, digest in read(EVAL / 'completed_manifest.json').items():
        assert sha(Path(name)) == digest, name
    plan = read(OUT / 'pool_analysis_plan.json')
    for name, digest in plan['review_sha256'].items():
        assert sha(ROOT / name) == digest
    rows = [json.loads(line) for line in (EVAL / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['key'] for r in rows}) == 48
    patches = {r['key']: r for r in read(OUT / 'generation/patch_manifest.json')}
    mapping = read(OUT / 'blinded_reviews/key_mapping.json')
    reviews = {label: {r['candidate_id']: r for r in read(OUT / f'blinded_reviews/review_{label}.json')['annotations']} for label in ['a', 'b']}
    observations = []
    for row in rows:
        statuses = [v for group in row['test_statuses'].values() for v in group.values()]
        observed = bool(statuses) and all(v in ['PASSED', 'FAILED', 'ERROR'] for v in statuses)
        expected = observed and row['execution_status'] == 'tested' and all(v == 'PASSED' for v in statuses)
        assert row['resolved'] == expected
        assert row['correctness_observed'] == observed or row['execution_status'] != 'tested'
        for suite, group in row['test_statuses'].items():
            for test, value in group.items():
                assert row['reward_vector'][suite][test] == (1 if value == 'PASSED' else 0 if value in ['FAILED', 'ERROR'] else None)
        patch = patches[row['key']]
        structures = []
        for name in patch.get('files', []):
            path = EVAL / 'candidates' / row['key'] / 'repo' / name
            node = DropDocstrings().visit(ast.parse(path.read_bytes().decode('utf-8')))
            structures.append((name, ast.dump(node, include_attributes=False)))
        signature = hashlib.sha256(json.dumps(structures).encode()).hexdigest() if structures else None
        cid = mapping.get(row['key'])
        observations.append({'key':row['key'], 'instance_id':row['instance_id'], 'sample':row['sample'],
            'execution_status':row['execution_status'], 'correctness_observed':row['correctness_observed'],
            'resolved':row['resolved'], 'exit_code':row.get('exit_code'), 'patch_sha256':patch.get('patch_sha256'),
            'normalized_ast_sha256':signature, 'candidate_id':cid,
            'mechanism':{label:reviews[label][cid]['mechanism_cluster'] for label in reviews} if cid else {},
            'quality_prediction':{label:reviews[label][cid]['quality_score'] for label in reviews} if cid else {}})
    tasks = []
    for task in [r['instance_id'] for r in read(OUT / 'selection.json')['tasks']]:
        selected = [r for r in observations if r['instance_id'] == task]
        correct = [r for r in selected if r['resolved']]
        tasks.append({'instance_id':task, 'slots':len(selected), 'observed':sum(r['correctness_observed'] for r in selected),
            'resolved':len(correct), 'unique_patch_texts':len({r['patch_sha256'] for r in selected}),
            'unique_normalized_asts':len({r['normalized_ast_sha256'] for r in selected}),
            'successful_unique_patch_texts':len({r['patch_sha256'] for r in correct}),
            'successful_unique_normalized_asts':len({r['normalized_ast_sha256'] for r in correct}),
            'all_mechanism_counts':{label:len({r['mechanism'][label] for r in selected if r['mechanism']}) for label in reviews},
            'successful_mechanism_counts':{label:len({r['mechanism'][label] for r in correct}) for label in reviews}})
    consistency = defaultdict(set)
    for row in observations:
        consistency[row['instance_id'], row['patch_sha256']].add((row['correctness_observed'], row['resolved']))
    assert all(len(v) == 1 for v in consistency.values()), 'Identical-patch grade inconsistency'
    multiple = [r['instance_id'] for r in tasks if min(r['successful_mechanism_counts'].values()) >= 2]
    assert plan['pre_outcome_gate_unattainable']
    result = {'slots':48, 'tasks':tasks, 'resolved':sum(r['resolved'] for r in observations),
        'execution_status_counts':dict(Counter(r['execution_status'] for r in observations)),
        'observed_correctness_slots':sum(r['correctness_observed'] for r in observations),
        'identical_patch_outcomes_consistent':True, 'successful_multi_mechanism_tasks':multiple,
        'three_task_support_gate_passed':False, 'length_matching_not_needed_for_failed_upper_bound':True,
        'scope':'Descriptive one-shot repair proposal development pool; host required-test success, not official container evaluation, full trajectory diversity, a selection effect, or downstream learning gain.',
        'normalized_ast_definition':'Changed production files, comments and leading module/class/function docstrings removed; identifiers/constants retained. Syntactic identity, not semantic proof.',
        'reviews':'Two blinded automated code reviews; same/different partitions agree42/42 pairs, not human ground truth.',
        'nonzero_exit_resolved_slots':[r['key'] for r in observations if r['resolved'] and r['exit_code'] != 0],
        'analysis_plan_sha256':sha(OUT / 'pool_analysis_plan.json'), 'script_sha256':sha(Path(__file__)),
        'observations':observations}
    (OUT / 'pool_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'observations'}, indent=2))


if __name__ == '__main__':
    main()
