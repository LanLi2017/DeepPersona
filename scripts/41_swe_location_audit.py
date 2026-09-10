#!/usr/bin/env python3
"""Prepare an outcome-hidden development repair-pair audit and verify edit locations."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from unidiff import PatchSet

sp = importlib.util.spec_from_file_location('semantic', Path(__file__).with_name('40_swe_semantic.py'))
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
s = m.s
OUT = s.ROOT / 'location-audit'


def auxiliary(name):
    return bool(re.search(r'(^|/)(tests?|docs?|\.eggs|\.venv|\.tox|__pycache__|node_modules)(/|$)|(^|/)(test_|reproduc|debug)|\.(md|rst|log)$|\.egg-info/', name))


def inputs():
    plan, pools = s.load_pools('dev')
    raw = {r['id']: r for p in sorted((s.OUT/'patches').glob('*.jsonl')) for r in s.read_jsonl(p)}
    return plan, pools, raw


def prepare():
    if (OUT/'pairs.json').exists():
        raise FileExistsError('Pair sample already frozen')
    OUT.mkdir(exist_ok=True)
    plan, pools, raw = inputs()
    candidates = {}
    for iid, rows in pools.items():
        for r in rows:
            if not r['valid'] or r['patch_chars'] > 16000:
                continue
            blocks = [b for b in PatchSet(raw[r['id']]['generated_patch']) if not auxiliary(b.path)]
            if not blocks or any(not b.path.endswith('.py') for b in blocks):
                continue
            edits = '\n'.join(line.value for b in blocks for h in b for line in h if line.is_added or line.is_removed)
            if sum(1 for b in blocks for h in b for line in h if line.is_added or line.is_removed) > 80:
                continue
            candidates[r['id']] = {'instance_id': iid, 'runtime_signature': s.digest(''.join(str(b) for b in blocks)),
                'files': sorted({b.path for b in blocks}), 'hints': sorted({(b.path, h.section_header.strip()) for b in blocks for h in b}),
                'tokens': set(re.findall(r'[A-Za-z_]\w*|\d+|[^\w\s]', edits)), 'raw_hash': r['patch_hash']}
    buckets = {'same_runtime_auxiliary_candidate': [], 'different_location_candidate': [], 'different_edit_candidate': []}
    for iid, rows in pools.items():
        eligible = [r['id'] for r in rows if r['id'] in candidates]
        for a, b in combinations(eligible, 2):
            x, y = candidates[a], candidates[b]
            if x['raw_hash'] == y['raw_hash']:
                continue
            overlap = len(x['tokens'] & y['tokens']) / max(1, len(x['tokens'] | y['tokens']))
            kind = None
            if x['runtime_signature'] == y['runtime_signature']:
                kind = 'same_runtime_auxiliary_candidate'
            elif x['files'] == y['files'] and x['hints'] != y['hints'] and overlap >= .5:
                kind = 'different_location_candidate'
            elif overlap <= .45:
                kind = 'different_edit_candidate'
            if kind:
                buckets[kind].append({'instance_id': iid, 'a': a, 'b': b, 'sampling_stratum': kind,
                                      'sampling_token_overlap': overlap})
    pairs, seen = [], {'dfm__emcee-295', 'scrapinghub__price-parser-43'}
    for kind, rows in buckets.items():
        rows.sort(key=lambda r: s.digest(f"{s.SEED}:location-pair:{r['a']}:{r['b']}"))
        count = 0
        for r in rows:
            if r['instance_id'] not in seen:
                pairs.append(r)
                seen.add(r['instance_id'])
                count += 1
            if count == 6:
                break
    for iid, a, b in [('dfm__emcee-295', 'data/train-00000-of-00012.parquet:5354', 'data/train-00000-of-00012.parquet:5367'),
                      ('scrapinghub__price-parser-43', 'data/train-00003-of-00012.parquet:2661', 'data/train-00003-of-00012.parquet:2657')]:
        pairs.append({'instance_id': iid, 'a': a, 'b': b, 'sampling_stratum': 'outcome_guided_diagnostic'})
    pairs.sort(key=lambda r: s.digest('packet-order:'+r['a']+r['b']))
    for j, r in enumerate(pairs):
        r['pair_id'] = f'P{j+1:02d}'
        if int(s.digest('orientation:'+r['a'])[:8], 16) % 2:
            r['a'], r['b'] = r['b'], r['a']
        assert plan['issues'][r['instance_id']]['split'] == 'dev'
    s.write_json(OUT/'pairs.json', pairs)
    s.write_json(OUT/'protocol.json', {'utc': datetime.now(timezone.utc).isoformat(), 'seed': s.SEED,
        'script_sha256': s.digest(Path(__file__).read_text()), 'api_spend_usd': 0,
        'sampling': 'Six pairs per heuristic stratum, distinct issues, then two explicitly outcome-guided diagnostics; no outcome labels loaded by sampler.',
        'eligibility': 'Valid Python runtime diffs, <=80 runtime changed lines, <=16000 raw patch characters.',
        'strata_are_not_gold': True, 'main_pairs': len(pairs)-2, 'diagnostic_pairs': 2,
        'available_pairs': {k: len(v) for k, v in buckets.items()}, 'pairs_sha256': s.digest((OUT/'pairs.json').read_text()),
        'scope': 'Construct-validity audit only; no new selector tuning, efficacy evaluation, or reserve access.',
        'annotation': 'Hide outcomes, metric scores, sampling strata and extracted claims; annotate full patch and exact base-code context first. Assistant annotations are provisional, not independent human gold.',
        'location_rule': 'Apply hunks in memory only after exact old-text match; resolve innermost qualified AST function/class on before and after sides; abstain on missing source, patch mismatch or syntax failure.',
        'report': 'Exact claim-set collapses, location mention omissions, auxiliary-edit invariance; no tuned metric weights or AUC on heuristic strata.'})
    print(json.dumps({'pairs': len(pairs), 'strata': {k: sum(r['sampling_stratum']==k for r in pairs) for k in [*buckets, 'outcome_guided_diagnostic']}}))


def fetch(smoke):
    plan, _, raw = inputs()
    pairs = json.loads((OUT/'pairs.json').read_text())
    if smoke:
        pairs = [r for r in pairs if r['sampling_stratum'] == 'outcome_guided_diagnostic']
    jobs = {}
    for r in pairs:
        issue = plan['issues'][r['instance_id']]
        for cid in [r['a'], r['b']]:
            for b in PatchSet(raw[cid]['generated_patch']):
                if not auxiliary(b.path) and b.path.endswith('.py') and not b.is_added_file:
                    url = f"https://raw.githubusercontent.com/{issue['repo']}/{issue['base_commit']}/{urllib.parse.quote(b.source_file.removeprefix('a/'))}"
                    jobs[url] = {'url': url, 'repo': issue['repo'], 'base_commit': issue['base_commit'], 'path': b.source_file.removeprefix('a/')}
    dest = OUT/'sources'
    dest.mkdir(exist_ok=True)

    def download(job):
        path = dest/(s.digest(job['url'])+'.json')
        if path.exists():
            return json.loads(path.read_text())['status']
        try:
            with urllib.request.urlopen(job['url'], timeout=30) as response:
                data = response.read()
            record = {**job, 'status': 'ok', 'sha256': hashlib.sha256(data).hexdigest(), 'text': data.decode('utf-8')}
        except (urllib.error.URLError, UnicodeDecodeError) as e:
            record = {**job, 'status': 'unavailable', 'error_type': type(e).__name__}
        s.write_json(path, record)
        return record['status']
    with ThreadPoolExecutor(max_workers=4) as pool:
        status = list(pool.map(download, jobs.values()))
    print(json.dumps({'requested_sources': len(jobs), 'ok': status.count('ok'), 'unavailable': status.count('unavailable')}))


def scopes(text):
    tree = ast.parse(text)
    result = []

    def walk(node, parents):
        named = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        path = parents + [node.name] if named else parents
        if named:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            result.append((start, node.end_lineno, '.'.join(path)))
        for child in ast.iter_child_nodes(node):
            walk(child, path)
    walk(tree, [])
    return result


def locate(text, lines):
    ranges = scopes(text)
    return sorted({min(((end-start, name) for start, end, name in ranges if start <= line <= end),
                       default=(0, '<module>'))[1] for line in lines})


def patched(before, block):
    source = before.splitlines(keepends=True)
    after, cursor, oldlines, newlines = [], 0, [], []
    for hunk in block:
        start = hunk.source_start-1 if hunk.source_length else hunk.source_start
        if start < cursor:
            raise ValueError('Overlapping hunks')
        after.extend(source[cursor:start])
        cursor = start
        for line in hunk:
            if line.line_type == '\\':
                continue
            if line.is_context or line.is_removed:
                if cursor >= len(source) or source[cursor].rstrip('\n') != line.value.rstrip('\n'):
                    raise ValueError('Base content mismatch')
                if line.is_removed and line.value.strip() and not line.value.lstrip().startswith('#'):
                    oldlines.append(cursor+1)
                cursor += 1
            if line.is_added or line.is_context:
                if line.is_added and line.value.strip() and not line.value.lstrip().startswith('#'):
                    newlines.append(len(after)+1)
                after.append(line.value)
    after.extend(source[cursor:])
    return ''.join(after), oldlines, newlines


def analyze(smoke):
    plan, pools, raw = inputs()
    pairs = json.loads((OUT/'pairs.json').read_text())
    if smoke:
        pairs = [r for r in pairs if r['sampling_stratum']=='outcome_guided_diagnostic']
    mapping = json.loads((m.OUT/'mapping.json').read_text())
    _, reviews = m.responses()
    files, contexts = {}, {}
    for pair in pairs:
        issue = plan['issues'][pair['instance_id']]
        for cid in [pair['a'], pair['b']]:
            files[cid], contexts[cid] = [], []
            for block in PatchSet(raw[cid]['generated_patch']):
                if auxiliary(block.path) or not block.path.endswith('.py'):
                    continue
                url = f"https://raw.githubusercontent.com/{issue['repo']}/{issue['base_commit']}/{urllib.parse.quote(block.source_file.removeprefix('a/'))}"
                record = {'file': block.path, 'hunk_hints': [h.section_header for h in block], 'base_url': None if block.is_added_file else url}
                try:
                    before = '' if block.is_added_file else json.loads((OUT/'sources'/(s.digest(url)+'.json')).read_text())['text']
                    after, oldlines, newlines = patched(before, block)
                    record.update(status='verified', before_symbols=locate(before, oldlines), after_symbols=locate(after, newlines),
                                  before_lines=oldlines, after_lines=newlines)
                    spans = set()
                    base = before.splitlines()
                    for h in block:
                        spans.update(range(max(0, h.source_start-7), min(len(base), h.source_start+h.source_length+4)))
                    contexts[cid].append({'file': block.path, 'url': None if block.is_added_file else url, 'text': '\n'.join(f'{i+1}: {base[i]}' for i in sorted(spans))})
                except (KeyError, FileNotFoundError, ValueError, SyntaxError) as e:
                    record.update(status='unresolved', reason=type(e).__name__)
                files[cid].append(record)
    feature = np.load(m.OUT/'selection_features.npz')
    ix = {i: j for j, i in enumerate(feature['issue_ids'].tolist())}
    reports = []
    for r in pairs:
        iid = r['instance_id']
        positions = {row['id']: j for j, row in enumerate(pools[iid])}
        a, b = r['a'], r['b']
        claims = {cid: reviews[mapping[cid]['key']]['repair_steps'] if mapping[cid]['key'] else [] for cid in [a,b]}
        symbols = {cid: sorted({f"{f['file']}::{name}" for f in files[cid] if f['status']=='verified' for name in f['before_symbols']+f['after_symbols']}) for cid in [a,b]}
        reports.append({**r, 'semantic_distance': float(feature['semantic'][ix[iid], positions[a], positions[b]]),
            'filtered_token_distance': float(feature['filtered_tokens'][ix[iid], positions[a], positions[b]]),
            'claims_a': claims[a], 'claims_b': claims[b], 'identical_claim_sets': set(claims[a])==set(claims[b]),
            'locations_a': symbols[a], 'locations_b': symbols[b], 'different_verified_locations': bool(symbols[a] and symbols[b] and symbols[a]!=symbols[b]),
            'location_records_a': files[a], 'location_records_b': files[b]})
    prefix = 'smoke_' if smoke else ''
    s.write_json(OUT/f'{prefix}analysis.json', reports)
    s.write_json(OUT/f'{prefix}contexts.json', contexts)
    summary = {'pairs':len(pairs), 'runtime_files': sum(map(len, files.values())),
               'unresolved_files': sum(f['status']!='verified' for v in files.values() for f in v),
               'identical_claim_pairs': [r['pair_id'] for r in reports if r['identical_claim_sets']],
               'identical_claims_different_locations': [r['pair_id'] for r in reports if r['identical_claim_sets'] and r['different_verified_locations']]}
    s.write_json(OUT/f'{prefix}summary.json', summary)
    print(json.dumps(summary, indent=2))


def packet():
    _, _, raw = inputs()
    pairs = json.loads((OUT/'pairs.json').read_text())
    contexts = json.loads((OUT/'contexts.json').read_text())
    issues = {r['instance_id']:r for r in s.read_jsonl(s.OUT/'issue_metadata.jsonl')}
    dest = OUT/'annotation'
    dest.mkdir(exist_ok=True)
    template = []
    for r in pairs:
        text = f"# {r['pair_id']} — repair comparison\n\n{issues[r['instance_id']]['problem_statement']}\n"
        for side in ['a','b']:
            cid = r[side]
            text += f"\n## Patch {side.upper()}\n\n````diff\n{raw[cid]['generated_patch']}\n````\n"
            text += '\n### Base-code context (read only)\n'
            for ctx in contexts[cid]:
                label = f"[{ctx['file']}]({ctx['url']})" if ctx['url'] else f"{ctx['file']} (new file; no base content)"
                text += f"\n{label}\n\n````text\n{ctx['text']}\n````\n"
        (dest/f"{r['pair_id']}.md").write_text(text)
        template.append({'pair_id':r['pair_id'],'relation':None,'locations_a':[],'locations_b':[],
                         'operation_a':None,'operation_b':None,'condition_or_dataflow_difference':None,
                         'auxiliary_only_difference':None,'confidence':None,'evidence':None})
    path = dest/'annotations_blank.jsonl'
    if not path.exists():
        path.write_text(''.join(json.dumps(r)+'\n' for r in template))
    index = '# Repair-pair annotation packet\n\nStart with the [rubric](../../../../docs/swe_repair_location_audit.md). '
    index += 'Annotate actual edits before viewing any metric outputs. Published correctness labels and model reviews are omitted.\n\n'
    index += '\n'.join(f"- [{r['pair_id']}]({r['pair_id']}.md)" for r in pairs)+'\n'
    (dest/'README.md').write_text(index)
    print(json.dumps({'packet':str(dest/'README.md'),'pairs':len(pairs)}))


def report():
    from collections import Counter
    records = json.loads((OUT/'analysis.json').read_text())
    notes = {r['pair_id']: r for r in s.read_jsonl(OUT/'assistant_provisional_annotations.jsonl')}
    controls = []
    for r in records:
        def distance(a, b):
            a, b = set(a), set(b)
            return 1-len(a & b)/len(a | b) if a and b else None
        controls.append({'pair_id': r['pair_id'], 'provisional_relation': notes[r['pair_id']]['relation'],
            'sampling_stratum': r['sampling_stratum'], 'semantic_distance': r['semantic_distance'],
            'file_jaccard': distance([x.split('::')[0] for x in r['locations_a']], [x.split('::')[0] for x in r['locations_b']]),
            'scope_jaccard': distance(r['locations_a'], r['locations_b']),
            'scope_is_only_structural_control': True})
    auxiliary_pairs = [r for r in records if r['sampling_stratum']=='same_runtime_auxiliary_candidate']
    main = [r for r in records if r['sampling_stratum']!='outcome_guided_diagnostic']
    summary = {'api_spend_usd': 0, 'provisional_only': True,
        'main_relation_counts': dict(Counter(notes[r['pair_id']]['relation'] for r in main)),
        'diagnostic_relation_counts': dict(Counter(notes[r['pair_id']]['relation'] for r in records if r not in main)),
        'identical_runtime_pairs': len(auxiliary_pairs),
        'identical_runtime_nonzero_semantic_distance': [{'pair_id':r['pair_id'],'distance':r['semantic_distance']} for r in auxiliary_pairs if r['semantic_distance']>1e-6],
        'location_signature_invariance': all(c['scope_jaccard']==0 for c in controls if c['sampling_stratum']=='same_runtime_auxiliary_candidate'),
        'pure_location_contrasts_in_main_sample': sum(notes[r['pair_id']]['relation']=='same_operation_different_location' for r in main),
        'decision': 'Do not promote claims or a location-only signature to a selection reward. Obtain independent annotations and fresh outcome-independent location contrasts; retain source-grounded operation and control-flow context.'}
    s.write_json(OUT/'structural_controls.json', controls)
    s.write_json(OUT/'findings.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare','fetch','analyze','packet','report'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.stage in ['prepare','packet','report']:
        {'prepare':prepare,'packet':packet,'report':report}[args.stage]()
    else:
        {'fetch':fetch,'analyze':analyze}[args.stage](args.smoke)
