#!/usr/bin/env python3
"""Deterministic gold-blind SWE source packets; reads only a public task export."""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'runs/swe-diversity-selection/swe-fresh-development'
EXTENSIONS = {'.py', '.pyi', '.pyx', '.pxd', '.c', '.h', '.cpp', '.rst', '.md',
              '.txt', '.toml', '.cfg', '.ini', '.yaml', '.yml', '.json', '.html', '.js', '.css'}
WORD = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')
MAX_FILE_BYTES = 1_000_000
MAX_CHUNK_LINES = 140
OVERLAP_LINES = 12
MAX_CANDIDATES = 600
MAX_SELECTED_FILES = 16
SOFT_FILE_TOKENS = 6500


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def terms(text):
    """Keep whole identifiers as well as snake/camel components."""
    result = []
    for word in WORD.findall(text):
        lower = word.lower()
        if len(lower) > 1:
            result.append(lower)
        pieces = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', word).lower().split('_')
        if len(pieces) > 1:
            result.extend(piece for piece in pieces if len(piece) > 1 and piece != lower)
    return result


@dataclass
class Chunk:
    path: str
    start: int
    end: int
    symbol: str
    kind: str
    tf: Counter
    length: int
    score: float = 0.0
    bm25: float = 0.0
    path_boost: float = 0.0
    symbol_boost: float = 0.0
    direct_symbol_anchor: bool = False


def spans(path, text):
    lines = text.splitlines(keepends=True)
    n = len(lines)
    if not n:
        return []
    raw = []
    if Path(path).suffix in {'.py', '.pyi'}:
        try:
            tree = ast.parse(text)
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    if isinstance(node, ast.ClassDef) and node.end_lineno - start > MAX_CHUNK_LINES:
                        raw.append((start, min(node.end_lineno, start + 25), node.name, 'class_header'))
                        for child in node.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                begin = min([child.lineno] + [d.lineno for d in child.decorator_list])
                                raw.append((begin, child.end_lineno, node.name + '.' + child.name, 'method'))
                    else:
                        raw.append((start, node.end_lineno, node.name, type(node).__name__))
        except (SyntaxError, ValueError, RecursionError):
            pass
    # Fixed windows ensure coverage of module-level code and large class bodies.
    raw.extend((start, min(n, start + MAX_CHUNK_LINES - 1), '', 'window')
               for start in range(1, n + 1, MAX_CHUNK_LINES - OVERLAP_LINES))
    out = []
    seen = set()
    for begin, end, symbol, kind in raw:
        for start in range(begin, end + 1, MAX_CHUNK_LINES - OVERLAP_LINES):
            stop = min(end, start + MAX_CHUNK_LINES - 1)
            if (start, stop) in seen:
                continue
            seen.add((start, stop))
            words = terms(path + '\n' + ''.join(lines[start - 1:stop]))
            out.append(Chunk(path, start, stop, symbol, kind, Counter(words), len(words)))
            if stop == end:
                break
    return out


def merge(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


def render(task_id, metadata, issue, chosen, files):
    blocks = [f'# Public repair context: {task_id}\n',
              'This packet contains the issue statement and excerpts from the original base source.\n',
              'Repository: ' + str(metadata.get('repo', metadata.get('repository', 'unspecified'))) + '\n',
              'Base commit: ' + str(metadata.get('base_commit', 'unspecified')) + '\n',
              '\n## Issue statement\n\n' + issue.rstrip() + '\n',
              '\n## Original source excerpts\n',
              'Line ranges are inclusive. Gaps between excerpts are omitted source, not edits.\n']
    for path, intervals in chosen.items():
        file = files[path]
        blocks.append(f'\n### {path}\n\nOriginal file SHA256: `{file["sha256"]}`\n')
        for start, end in merge(intervals):
            code = ''.join(file['lines'][start - 1:end])
            fence = '`' * max(3, 1 + max([len(m) for m in re.findall(r'`+', code)] or [0]))
            language = 'python' if Path(path).suffix in {'.py', '.pyi'} else ''
            blocks.append(f'\nLines {start}–{end}:\n\n{fence}{language}\n' + code
                          + ('' if code.endswith('\n') else '\n') + fence + '\n')
    return ''.join(blocks)


def build_task(task_dir, encoding, budget):
    public = task_dir.resolve()
    for name in ['task.json', 'problem_statement.md', 'source']:
        assert (public / name).exists(), (public.name, name)
        assert (public / name).resolve().is_relative_to(public), 'No external symlink reads'
    meta_blob = (public / 'task.json').read_bytes()
    issue_blob = (public / 'problem_statement.md').read_bytes()
    metadata = json.loads(meta_blob)
    issue = issue_blob.decode('utf-8')
    files, skipped, chunks = {}, [], []
    source = public / 'source'
    for file in sorted(source.rglob('*')):
        if not file.is_file():
            continue
        relative = file.relative_to(source).as_posix()
        if file.is_symlink() or not file.resolve().is_relative_to(source.resolve()):
            skipped.append({'path': relative, 'reason': 'symlink'})
            continue
        if file.suffix.lower() not in EXTENSIONS and file.name not in {'Makefile', 'Dockerfile', 'setup.py'}:
            skipped.append({'path': relative, 'reason': 'extension'})
            continue
        if file.stat().st_size > MAX_FILE_BYTES:
            skipped.append({'path': relative, 'reason': 'size'})
            continue
        blob = file.read_bytes()
        if b'\x00' in blob:
            skipped.append({'path': relative, 'reason': 'binary'})
            continue
        try:
            text = blob.decode('utf-8')
        except UnicodeDecodeError:
            skipped.append({'path': relative, 'reason': 'encoding'})
            continue
        files[relative] = {'sha256': digest(blob), 'bytes': len(blob),
                           'lines': text.splitlines(keepends=True)}
        chunks.extend(spans(relative, text))
    assert chunks, 'No source chunks available'
    query = Counter(terms(issue))
    df = Counter(term for chunk in chunks for term in chunk.tf if term in query)
    avg = sum(chunk.length for chunk in chunks) / len(chunks)
    issue_lower = issue.lower()
    mentioned = set(WORD.findall(issue))
    calls = set(re.findall(r'\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(', issue))
    for chunk in chunks:
        total = 0.0
        for term in sorted(query.keys() & chunk.tf.keys()):
            tf = chunk.tf[term]
            idf = math.log(1 + (len(chunks) - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * chunk.length / avg)) * (1 + math.log(query[term]))
        chunk.bm25 = total
        path = chunk.path.lower()
        leaf = Path(path).name
        if path in issue_lower or path.replace('/', '.').removesuffix('.py') in issue_lower:
            chunk.path_boost = 24.0
        elif leaf in issue_lower and len(leaf) > 5:
            chunk.path_boost = 12.0
        if chunk.symbol and any(s in mentioned for s in chunk.symbol.split('.')):
            chunk.symbol_boost = 8.0
        owner, _, member = chunk.symbol.rpartition('.')
        chunk.direct_symbol_anchor = bool(chunk.symbol and (chunk.symbol in calls
            or (owner in calls and member in {'__new__', '__init__'})))
        chunk.score = total + chunk.path_boost + chunk.symbol_boost
    ranked = sorted(chunks, key=lambda c: (not c.direct_symbol_anchor, -c.score, c.path, c.start, c.end, c.kind))
    chosen, selected = {}, []
    def count(text):
        return len(encoding.encode(text, disallowed_special=()))
    assert count(render(public.name, metadata, issue, {}, files)) < budget, 'Issue alone exceeds packet budget'
    # First pass limits any single file from monopolizing the packet; the second
    # permits additional relevant material when there is remaining room.
    for limited in [True, False]:
        for chunk in ranked[:MAX_CANDIDATES]:
            if chunk.score <= 0:
                continue
            if chunk.path not in chosen and len(chosen) >= MAX_SELECTED_FILES:
                continue
            old = chosen.get(chunk.path, [])
            proposed = merge(old + [(1, min(60, len(files[chunk.path]['lines']))), (chunk.start, chunk.end)])
            if merge(old) == proposed:
                continue
            if limited:
                file_text = ''.join(''.join(files[chunk.path]['lines'][a - 1:b]) for a, b in proposed)
                if count(file_text) > SOFT_FILE_TOKENS:
                    continue
            candidate = {**chosen, chunk.path: proposed}
            packet = render(public.name, metadata, issue, candidate, files)
            if count(packet) > budget:
                continue
            chosen = candidate
            selected.append({'path': chunk.path, 'lines': [chunk.start, chunk.end],
                'symbol': chunk.symbol, 'kind': chunk.kind, 'score': chunk.score,
                'bm25': chunk.bm25, 'path_boost': chunk.path_boost,
                'symbol_boost': chunk.symbol_boost, 'direct_symbol_anchor': chunk.direct_symbol_anchor, 'selection_pass': 1 if limited else 2})
    assert chosen, 'No source span fits budget'
    packet = render(public.name, metadata, issue, chosen, files)
    token_count = count(packet)
    assert token_count <= budget
    source_spans = []
    for path, intervals in chosen.items():
        for start, end in merge(intervals):
            assert 1 <= start <= end <= len(files[path]['lines'])
            exact = ''.join(files[path]['lines'][start - 1:end])
            assert exact.rstrip('\n') in packet
            source_spans.append({'path': path, 'start_line': start, 'end_line': end,
                'file_sha256': files[path]['sha256'], 'span_utf8_sha256': digest(exact.encode())})
    manifest = {'task_id': public.name, 'packet_sha256': digest(packet.encode()),
        'packet_tokens': token_count, 'tokenizer': 'tiktoken/o200k_base',
        'token_count_is_model_specific_guarantee': False, 'budget': budget,
        'issue_sha256': digest(issue_blob), 'public_task_json_sha256': digest(meta_blob),
        'public_source_files_read': {path: {'sha256': value['sha256'], 'bytes': value['bytes'],
            'lines': len(value['lines'])} for path, value in files.items()},
        'skipped_files': skipped, 'candidate_chunks': len(chunks),
        'source_spans': source_spans, 'selection_events': selected,
        'top_retrieval_candidates': [{'path': c.path, 'lines': [c.start, c.end],
            'symbol': c.symbol, 'score': c.score, 'direct_symbol_anchor': c.direct_symbol_anchor} for c in ranked[:40]],
        'private_or_evaluator_inputs_read': False}
    return packet, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-root', type=Path, default=DEFAULT / 'public')
    parser.add_argument('--output-root', type=Path, default=DEFAULT / 'generation_context')
    parser.add_argument('--budget', type=int, default=24000)
    parser.add_argument('--verify', action='store_true', help='Rebuild from public inputs and compare existing artifacts')
    args = parser.parse_args()
    public = args.public_root.resolve()
    output = args.output_root.resolve()
    assert public.is_dir() and not output.is_relative_to(public), 'Separate public input and output roots required'
    tasks = sorted(p for p in public.iterdir() if p.is_dir() and not p.is_symlink())
    assert len(tasks) == 6, 'Expected exactly six freshly frozen public tasks'
    encoding = tiktoken.get_encoding('o200k_base')
    algorithm = {'source_script_sha256': digest(Path(__file__).read_bytes()),
        'scope': 'Only public/<id>/task.json,problem_statement.md,source/** are corpus inputs',
        'ranking': 'BM25(k1=1.5,b=.75), full-identifier+snake/camel terms; explicit issue path boosts24/12, symbol boost8',
        'chunk_max_lines': MAX_CHUNK_LINES, 'window_overlap_lines': OVERLAP_LINES,
        'candidate_limit': MAX_CANDIDATES, 'file_limit': MAX_SELECTED_FILES,
        'soft_file_tokens_first_pass': SOFT_FILE_TOKENS,
        'budget': args.budget, 'tokenizer': 'o200k_base', 'tiktoken_version': tiktoken.__version__,
        'tie_break': 'Exact issue call-symbol definitions and class constructors first; score descending, path,start,end,kind ascending',
        'auto_context': 'First60 original lines of each selected file; overlapping intervals merged',
        'api_calls': 0, 'gpu_used': False, 'tasks': []}
    if not args.verify:
        assert not (output / 'manifest.json').exists() and not list(output.glob('*.txt')), 'Preserve existing context packets'
        output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        packet, manifest = build_task(task, encoding, args.budget)
        packet_path = output / (task.name + '.txt')
        retrieval_path = output / (task.name + '.retrieval.json')
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
        if args.verify:
            assert packet_path.read_bytes() == packet.encode('utf-8'), task.name
            assert retrieval_path.read_text() == manifest_text, task.name
        else:
            packet_path.write_text(packet)
            retrieval_path.write_text(manifest_text)
        algorithm['tasks'].append({'task_id': task.name, 'file': packet_path.name,
            'retrieval_file': retrieval_path.name, 'packet_bytes': len(packet.encode()), 'packet_tokens': manifest['packet_tokens'],
            'packet_sha256': manifest['packet_sha256'], 'selected_files': len({s['path'] for s in manifest['source_spans']}),
            'retrieval_manifest_sha256': digest(manifest_text.encode())})
        print(json.dumps(algorithm['tasks'][-1]), flush=True)
    manifest_text = json.dumps(algorithm, indent=2, sort_keys=True) + '\n'
    if args.verify:
        assert (output / 'manifest.json').read_text() == manifest_text
        print('Deterministic rebuild verified for all six packets.', flush=True)
    else:
        (output / 'manifest.json').write_text(manifest_text)


if __name__ == '__main__':
    main()
