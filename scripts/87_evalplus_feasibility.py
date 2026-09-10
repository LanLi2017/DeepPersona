#!/usr/bin/env python3
"""Pin MBPP+ opaque release; inspect IDs and four development schemas only."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import urllib.request

OUT = Path('runs/swe-diversity-selection/code-evalplus-feasibility')
ROOT = Path('runs/swe-diversity-selection')
API = 'https://api.github.com/repos/'
SOURCE_FILES = ['LICENSE', 'README.md', 'pyproject.toml', 'setup.cfg', 'requirements.txt',
                'evalplus/data/mbpp.py', 'evalplus/data/utils.py', 'evalplus/data/__init__.py',
                'evalplus/evaluate.py', 'evalplus/eval/__init__.py', 'evalplus/eval/utils.py',
                'evalplus/eval/_special_oracle.py', 'evalplus/gen/util/__init__.py',
                'evalplus/config.py', 'docs/evalperf.md']


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'DeepPersona-bounded-feasibility'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def remote_json(url):
    return json.loads(fetch(url))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def signatures(code):
    try:
        return {n.name: ast.unparse(n.args) for n in ast.parse(code).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    except SyntaxError:
        return {}


def main():
    assert not OUT.exists(), 'Preserve previous feasibility outputs'
    OUT.mkdir()
    release = remote_json(API + 'evalplus/mbppplus_release/releases/tags/v0.2.0')
    source = remote_json(API + 'evalplus/evalplus/commits/master')
    repository = remote_json(API + 'evalplus/mbppplus_release')
    revision = source['sha']
    assets = [{k: a.get(k) for k in ['id', 'name', 'size', 'created_at', 'updated_at', 'browser_download_url', 'digest']} for a in release['assets']]
    candidates = [a for a in assets if a['name'].lower() == 'mbppplus.jsonl.gz']
    assert len(candidates) == 1, [a['name'] for a in assets]
    asset = candidates[0]
    save(OUT / 'download_protocol.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'runner_revision': revision,
        'release_tag': release['tag_name'], 'release_id': release['id'], 'release_assets': assets,
        'dataset_repository_license_metadata': repository.get('license'),
        'scope': 'Opaque compressed release pinned by SHA256; IDs extracted as bytes only. Full JSON decoding only for2train+2cal records selected by ascending supportedID. No final/measurement test fields decoded, no execution or generation.',
        'source_script_sha256': digest(Path(__file__).read_bytes())})
    blob = fetch(asset['browser_download_url'])
    assert len(blob) == asset['size']
    (OUT / 'MbppPlus-v0.2.0.opaque.jsonl.gz').write_bytes(blob)
    task_lines = {}
    # Non-selected records remain opaque byte strings: do not decode their JSON fields.
    for line in gzip.decompress(blob).splitlines():
        match = re.search(rb'"task_id"\s*:\s*"Mbpp/(\d+)"', line)
        assert match is not None
        task = int(match.group(1))
        assert task not in task_lines
        task_lines[task] = line
    assert len(task_lines) == 378
    original = json.loads((ROOT / 'code-learning-pilot/protocol.json').read_text())
    expansion = json.loads((ROOT / 'code-learning-expansion/protocol.json').read_text())
    train = sorted(set(original['train_task_ids']) | set(expansion['train_task_ids']))
    split_ids = {'train': train, 'calibration': original['calibration_task_ids'],
                 'measurement': original['measurement_task_ids'], 'final_locked': original['later_test_task_ids']}
    assert [len(split_ids[k]) for k in split_ids] == [120, 21, 21, 107]
    mapping = {k: {'original_count': len(v), 'supported_ids': sorted(set(v) & task_lines.keys()),
                   'unsupported_ids': sorted(set(v) - task_lines.keys())} for k, v in split_ids.items()}
    save(OUT / 'membership.json', {'dataset_ids': sorted(task_lines), 'splits': mapping,
                                 'final_fields_accessed': ['task_id'], 'tests_or_outcomes_executed': False})
    selected = {k: mapping[k]['supported_ids'][:2] for k in ['train', 'calibration']}
    inspect_ids = {v for values in selected.values() for v in values}
    assert not inspect_ids & set(split_ids['final_locked'] + split_ids['measurement'])
    save(OUT / 'schema_allowlist.json', {'policy': 'First2numerically sorted supportedIDs in each development split', 'selected': selected})
    ours = {}
    for pool in ['code-learning-pilot', 'code-learning-expansion']:
        for r in rows(ROOT / pool / 'inputs.jsonl'):
            if r['task_id'] in inspect_ids:
                ours[r['task_id']] = r
    inspected = []
    for task in sorted(inspect_ids):
        r = json.loads(task_lines[task])
        sig = signatures(r['prompt'] + r['canonical_solution'])
        inspected.append({'task_id': task, 'evalplus_task_id': r['task_id'],
            'keys_and_types': {k: type(v).__name__ for k, v in r.items()},
            'entry_point': r['entry_point'], 'canonical_interface': sig.get(r['entry_point']),
            'evalplus_prompt': r['prompt'], 'existing_prompt': ours[task]['prompt'],
            'existing_interfaces': ours[task].get('interfaces'),
            'prompt_exactly_equal': r['prompt'] == ours[task]['prompt'],
            'entrypoint_in_existing_public_interfaces': r['entry_point'] in re.findall(r'def\s+(\w+)\s*\(', ours[task]['prompt']),
            'base_input_count': len(r['base_input']), 'plus_input_count': len(r['plus_input']),
            'base_input_first_two': r['base_input'][:2], 'plus_input_first_two': r['plus_input'][:2],
            'atol': r.get('atol'), 'contract_type': type(r.get('contract')).__name__})
    save(OUT / 'development_schema.json', inspected)
    source_hashes, missing = {}, {}
    for name in SOURCE_FILES:
        url = f'https://raw.githubusercontent.com/evalplus/evalplus/{revision}/{name}'
        try:
            content = fetch(url)
        except urllib.error.HTTPError as error:
            missing[name] = error.code
            continue
        path = OUT / 'runner_source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        source_hashes[name] = digest(content)
    versions = {}
    for package in ['evalplus', 'numpy', 'multipledispatch', 'psutil', 'wget', 'appdirs', 'tempdir', 'termcolor', 'fire', 'tree-sitter', 'tree-sitter-python']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary = {'dataset_version': 'v0.2.0', 'dataset_tasks': 378,
               'dataset_asset': asset, 'compressed_sha256': digest(blob), 'runner_revision': revision,
               'runner_commit_date': source['commit']['committer']['date'],
               'runner_source_sha256': source_hashes, 'source_paths_not_found': missing,
               'membership_counts': {k: len(v['supported_ids']) for k, v in mapping.items()},
               'development_schema_task_ids': sorted(inspect_ids), 'installed_packages': versions,
               'opaque_archive_contains_full_release': True, 'nonallowlisted_test_fields_decoded': False,
               'canonical_code_executed': False, 'grading_run': False, 'gpu_used': False, 'api_spend_usd': 0}
    save(OUT / 'summary.json', summary)
    save(OUT / 'completed_manifest.json', {'sha256': {str(p.relative_to(OUT)): digest(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ['dataset_tasks', 'compressed_sha256', 'runner_revision', 'membership_counts', 'development_schema_task_ids', 'source_paths_not_found']}))


if __name__ == '__main__':
    main()
