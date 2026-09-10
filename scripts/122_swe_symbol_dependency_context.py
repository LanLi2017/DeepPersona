#!/usr/bin/env python3
"""One prospective source-only issue-symbol / syntactic one-hop retrieval policy."""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import keyword
import math
from pathlib import Path
import re
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "runs/swe-diversity-selection/swe-sympy-patch-sft"
PRIOR = ORIGINAL.parent / "swe-sympy-retrieval-policies"
OUT = ORIGINAL.parent / "swe-sympy-symbol-context"
POLICY = "symbol_dependency"
BUDGET = 24576
MAX_ATTEMPTS = 600
MAX_WINDOWS = 12
WORKERS = 8
UNKNOWN = "!unknown"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(p, value):
    with p.open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expression(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = expression(node.value)
        return base + "." + node.attr if base else None
    return None


def scope_walk(body):
    stack = list(reversed(body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def bound_names(node):
    """Bindings whose payload is sometimes a string rather than a Store node."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return {node.id}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {item.asname or item.name.split(".")[0] for item in node.names if item.name != "*"}
    if isinstance(node, ast.ExceptHandler):
        return {node.name} if node.name else set()
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return {node.name} if node.name else set()
    if isinstance(node, ast.MatchMapping):
        return {node.rest} if node.rest else set()
    return set()


def wildcard_import(node):
    return isinstance(node, ast.ImportFrom) and any(item.name == "*" for item in node.names)


def module_name(path):
    name = path[:-3].replace("/", ".")
    return name.removesuffix(".__init__")


def import_bindings(node, module, is_package):
    found = []
    if isinstance(node, ast.Import):
        for item in node.names:
            found.append(
                (item.asname or item.name.split(".")[0], item.name if item.asname else item.name.split(".")[0])
            )
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            package = module.split(".") if is_package else module.split(".")[:-1]
            if node.level > len(package):
                return []
            prefix = ".".join(package[: len(package) - node.level + 1])
            target = ".".join(x for x in [prefix, node.module] if x)
        else:
            target = node.module or ""
        for item in node.names:
            if item.name != "*":
                found.append((item.asname or item.name, target + "." + item.name))
    return found


class DefinitionGraph:
    def __init__(self, files):
        self.nodes = {}
        self.bindings = defaultdict(lambda: defaultdict(set))
        self.methods = defaultdict(lambda: defaultdict(set))
        self.wildcard_modules = set()
        self.modules = set()
        self.leaves = defaultdict(set)
        self.parse_errors = []
        self.edges = []
        self.unresolved_calls = Counter()
        trees = {}
        for path, text in sorted(files.items()):
            module = module_name(path)
            self.modules.add(module)
            try:
                trees[path] = ast.parse(text)
            except (SyntaxError, ValueError, RecursionError) as error:
                self.parse_errors.append({"path": path, "error": type(error).__name__})
        for path, tree in trees.items():
            module = module_name(path)
            if any(wildcard_import(n) for n in scope_walk(tree.body)):
                self.wildcard_modules.add(module)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    identifier = self.add(path, module, node, node.name, kind, None)
                    self.bindings[module][node.name].add("@" + identifier)
                    if kind == "class":
                        for child in node.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                child_id = self.add(
                                    path, module, child, node.name + "." + child.name, "method", identifier
                                )
                                self.methods[identifier][child.name].add("@" + child_id)
                            else:
                                for item in scope_walk([child]):
                                    for name in bound_names(item):
                                        self.methods[identifier][name].add(UNKNOWN)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name, value in import_bindings(node, module, path.endswith("/__init__.py")):
                        self.bindings[module][name].add(value)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = expression(node.value) if node.value is not None else None
                    for target in targets:
                        for stored in ast.walk(target):
                            if isinstance(stored, ast.Name):
                                self.bindings[module][stored.id].add(
                                    module + "." + value if value and isinstance(target, ast.Name) else UNKNOWN
                                )
                else:
                    for item in scope_walk([node]):
                        for name in bound_names(item):
                            self.bindings[module][name].add(UNKNOWN)
        for identifier, node in self.nodes.items():
            tree = node["ast"]
            locals_ = defaultdict(set)
            if node["kind"] != "class":
                arguments = tree.args.posonlyargs + tree.args.args + tree.args.kwonlyargs
                arguments += [x for x in [tree.args.vararg, tree.args.kwarg] if x is not None]
                for arg in arguments:
                    locals_[arg.arg].add(UNKNOWN)
            for item in scope_walk(tree.body):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    locals_[item.name].add(UNKNOWN)
                elif isinstance(item, (ast.Import, ast.ImportFrom)):
                    for name, value in import_bindings(item, node["module"], node["path"].endswith("/__init__.py")):
                        locals_[name].add(value)
                else:
                    for name in bound_names(item):
                        locals_[name].add(UNKNOWN)
            local_wildcard = any(wildcard_import(x) for x in scope_walk(tree.body))
            decorators = {expression(d) for d in getattr(tree, "decorator_list", [])}
            positional = [] if node["kind"] == "class" else tree.args.posonlyargs + tree.args.args
            receiver = None
            if node["kind"] == "method" and positional:
                first = positional[0].arg
                # Any unknown decorator may change descriptor binding; do not infer its receiver.
                if first == "self" and not decorators:
                    receiver = "self"
                if (
                    first == "cls"
                    and decorators == {"classmethod"}
                    and "classmethod" not in self.bindings[node["module"]]
                    and node["module"] not in self.wildcard_modules
                ):
                    receiver = "cls"
                # Reassignment of the receiver makes even same-class lookup uncertain.
                if any(receiver in bound_names(x) for x in scope_walk(tree.body)):
                    receiver = None
            for call in (x for x in scope_walk(tree.body) if isinstance(x, ast.Call)):
                text = expression(call.func)
                if not text:
                    self.unresolved_calls["dynamic_call_expression"] += 1
                    continue
                if receiver and text.startswith(receiver + "."):
                    parts = text.split(".")
                    target, reason = self.resolve_refs(
                        self.methods[node["owner"]].get(parts[1], set()), parts[2:], 0, set()
                    )
                elif local_wildcard:
                    target, reason = None, "wildcard_scope_uncertainty"
                else:
                    target, reason = self.resolve_in_module(text, node["module"], locals_)
                if target is not None:
                    self.edges.append({"source": identifier, "target": target, "line": call.lineno, "reference": text})
                else:
                    self.unresolved_calls[reason] += 1
        self.edges.sort(key=lambda e: (e["source"], e["target"], e["line"], e["reference"]))

    def add(self, path, module, node, qualname, kind, owner):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        identifier = module + ":" + qualname + "@" + str(start)
        self.nodes[identifier] = {
            "path": path,
            "module": module,
            "qualname": qualname,
            "start": start,
            "end": node.end_lineno,
            "kind": kind,
            "owner": owner,
            "ast": node,
        }
        self.leaves[node.name].add(identifier)
        return identifier

    def resolve_refs(self, refs, attrs, depth, seen):
        if not refs:
            return None, "not_found"
        if len(refs) != 1:
            return None, "ambiguous_binding"
        ref = next(iter(refs))
        if ref == UNKNOWN:
            return None, "shadowed_or_dynamic_binding"
        if ref.startswith("@"):
            identifier = ref[1:]
            if not attrs:
                return identifier, "resolved"
            if self.nodes[identifier]["kind"] != "class":
                return None, "dynamic_receiver_attribute"
            return self.resolve_refs(self.methods[identifier].get(attrs[0], set()), attrs[1:], depth, seen)
        if depth >= 4:
            return None, "alias_hop_limit"
        key = (ref, tuple(attrs))
        if key in seen:
            return None, "alias_cycle"
        seen = seen | {key}
        parts = (ref + "." + ".".join(attrs)).rstrip(".").split(".")
        # Longest explicit module prefix; never global-name lookup in source calls.
        for stop in range(len(parts), 0, -1):
            module = ".".join(parts[:stop])
            if module not in self.modules:
                continue
            if stop == len(parts):
                return None, "module_not_definition"
            if module in self.wildcard_modules:
                return None, "wildcard_scope_uncertainty"
            return self.resolve_refs(self.bindings[module].get(parts[stop], set()), parts[stop + 1 :], depth + 1, seen)
        return None, "external_or_unresolved_module"

    def resolve_in_module(self, text, module, locals_=None):
        parts = text.split(".")
        local = locals_ or {}
        if parts[0] not in local and module in self.wildcard_modules:
            return None, "wildcard_scope_uncertainty"
        refs = local[parts[0]] if parts[0] in local else self.bindings[module].get(parts[0], set())
        return self.resolve_refs(refs, parts[1:], 0, set())

    def issue_seeds(self, issue):
        bindings = defaultdict(set)
        snippet_errors = []
        issue_wildcard = False
        issue_star_modules = set()
        for snippet in re.findall(r"```(?:python3|python|py)?[ \t]*\n(.*?)```", issue, re.S):
            try:
                tree = ast.parse(snippet)
            except (SyntaxError, ValueError, RecursionError) as error:
                snippet_errors.append(type(error).__name__)
                continue
            for item in scope_walk(tree.body):
                if wildcard_import(item):
                    if item.level == 0 and item.module in self.modules:
                        issue_star_modules.add(item.module)
                    else:
                        issue_wildcard = True
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    resolved = dict(import_bindings(node, "", False))
                    for name in bound_names(node):
                        bindings[name].add(resolved.get(name, UNKNOWN))
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        for item in ast.walk(target):
                            if isinstance(item, ast.Name):
                                bindings[item.id].add(UNKNOWN)
                else:
                    for item in scope_walk([node]):
                        for name in bound_names(item):
                            bindings[name].add(UNKNOWN)
        references = sorted(set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", issue)))
        rows = []
        seeds = set()
        for reference in references:
            if keyword.iskeyword(reference):
                continue
            parts = reference.split(".")
            kind = "call" if re.search(r"(?<![\w.])" + re.escape(reference) + r"\s*\(", issue) else "identifier"
            if issue_wildcard:
                identifier, reason, route = None, "wildcard_scope_uncertainty", "issue_wildcard_import"
            elif parts[0] in bindings:
                identifier, reason = self.resolve_refs(bindings[parts[0]], parts[1:], 0, set())
                route = "issue_import_or_shadow"
            elif issue_star_modules and len(parts) == 1:
                candidates = []
                uncertain = False
                for module in sorted(issue_star_modules):
                    refs = self.bindings[module].get(parts[0], set())
                    if refs:
                        candidate, status = self.resolve_refs(refs, [], 0, set())
                        if candidate is None:
                            uncertain = True
                        else:
                            candidates.append(candidate)
                unique = set(candidates)
                route = "lexical_indexed_star_export"
                if len(unique) == 1 and not uncertain:
                    identifier, reason = next(iter(unique)), "resolved"
                else:
                    identifier, reason = None, "ambiguous_star_export" if unique or uncertain else "not_found"
            elif parts[0] == "sympy":
                identifier, reason = self.resolve_refs({"sympy"}, parts[1:], 0, set())
                route = "explicit_sympy"
            else:
                identifier, reason = self.resolve_in_module(reference, "sympy")
                route = "public_sympy_export"
                if identifier is None and reason == "not_found":
                    candidates = self.leaves.get(parts[0], set())
                    if len(candidates) == 1:
                        identifier, reason = self.resolve_refs({"@" + next(iter(candidates))}, parts[1:], 0, set())
                        route = "globally_unique_exact_name"
                    elif len(candidates) > 1:
                        reason = "ambiguous_global_name"
                        route = "global_exact_name"
            if identifier is not None:
                seeds.add(identifier)
            rows.append({"reference": reference, "kind": kind, "route": route, "status": reason, "node": identifier})
        neighbors = set()
        edges = []
        for edge in self.edges:
            if edge["source"] in seeds or edge["target"] in seeds:
                neighbors.update([edge["source"], edge["target"]])
                edges.append(edge)
        return (
            seeds,
            neighbors - seeds,
            {"issue_references": rows, "issue_snippet_parse_errors": snippet_errors, "one_hop_edges": edges},
        )


def pack_symbols(task_id, metadata, issue, files, ranked, query, df, average, encoding, budget):
    m = RANKING
    graph = DefinitionGraph(
        {name: "".join(data["lines"]) for name, data in files.items() if TARGETS.path_allowed(name)}
    )
    seeds, neighbors, provenance = graph.issue_seeds(issue)
    words = set(m.WORD.findall(issue))
    calls = set(re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", issue))
    issue_lower = issue.lower()

    def relevance(name, start, end, symbol):
        terms = Counter(m.terms(name + "\n" + "".join(files[name]["lines"][start - 1 : end])))
        length = sum(terms.values())
        score = 0.0
        for term in sorted(query.keys() & terms.keys()):
            tf = terms[term]
            idf = math.log(1 + (len(ranked) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average)) * (1 + math.log(query[term]))
        path = name.lower()
        leaf = Path(path).name
        score += (
            24.0
            if path in issue_lower or path.replace("/", ".").removesuffix(".py") in issue_lower
            else (12.0 if leaf in issue_lower and len(leaf) > 5 else 0.0)
        )
        if symbol and any(s in words for s in symbol.split(".")):
            score += 8.0
        owner, _, member = symbol.rpartition(".")
        direct = bool(symbol and (symbol in calls or (owner in calls and member in {"__new__", "__init__"})))
        return (not direct, -score, name, start, end, symbol)

    ranks = {
        identifier: relevance(n["path"], n["start"], n["end"], n["qualname"])
        for identifier, n in graph.nodes.items()
        if identifier in seeds | neighbors
    }
    depths = [sorted(seeds, key=lambda x: ranks[x]), sorted(neighbors, key=lambda x: ranks[x])]
    windows = {}
    for identifier in seeds | neighbors:
        node = graph.nodes[identifier]
        candidates = []
        for start in range(node["start"], node["end"] + 1, m.MAX_CHUNK_LINES - m.OVERLAP_LINES):
            end = min(node["end"], start + m.MAX_CHUNK_LINES - 1)
            candidates.append((relevance(node["path"], start, end, node["qualname"]), identifier, start, end))
            if end == node["end"]:
                break
        windows[identifier] = sorted(candidates)[:MAX_WINDOWS]
    chosen = {}
    selected = []
    attempts = []
    seen = set()
    pass_index = 0

    def count(text):
        return len(encoding.encode(text, disallowed_special=()))

    assert count(m.render(task_id, metadata, issue, {}, files)) < budget, "Issue alone exceeds packet budget"

    def try_pack(name, intervals, event):
        if name not in chosen and len(chosen) >= 16:
            return "file_limit"
        old = chosen.get(name, [])
        proposed = m.merge(old + [(1, min(60, len(files[name]["lines"])))] + intervals)
        if m.merge(old) == proposed:
            return "already_present"
        key = (name, tuple(proposed))
        if key in seen:
            return "duplicate_proposal"
        if len(attempts) >= MAX_ATTEMPTS:
            return "attempt_limit"
        seen.add(key)
        record = dict(path=name, proposed_intervals=proposed, pass_number=pass_index + 1, **event)
        if pass_index == 0:
            file_text = "".join("".join(files[name]["lines"][a - 1 : b]) for a, b in proposed)
            record["native_file_tokens"] = count(file_text)
            if record["native_file_tokens"] > 6500:
                record.update(accepted=False, reason="soft_file_budget")
                attempts.append(record)
                return "soft_file_budget"
        candidate = {**chosen, name: proposed}
        record["native_prompt_tokens"] = count(m.render(task_id, metadata, issue, candidate, files))
        if record["native_prompt_tokens"] > budget:
            record.update(accepted=False, reason="hard_prompt_budget")
            attempts.append(record)
            return "hard_prompt_budget"
        chosen[name] = proposed
        record.update(accepted=True, reason="accepted")
        attempts.append(record)
        selected.append(
            dict(
                path=name,
                lines=event["lines"],
                kind=event["kind"],
                depth=event.get("depth"),
                node=event.get("node"),
                selection_pass=pass_index + 1,
            )
        )
        return "accepted"

    def node_intervals(identifier, start, end, is_window):
        node = graph.nodes[identifier]
        intervals = [(start, end)]
        if is_window:
            intervals.append((node["start"], min(node["end"], node["start"] + 25)))
        if node["owner"]:
            owner = graph.nodes[node["owner"]]
            intervals.append((owner["start"], min(owner["end"], owner["start"] + 25)))
        return intervals

    for pass_index in range(2):
        seen = set()
        for depth, identifiers in enumerate(depths):
            pending = []
            for identifier in identifiers:
                if len(attempts) >= MAX_ATTEMPTS:
                    break
                node = graph.nodes[identifier]
                result = try_pack(
                    node["path"],
                    node_intervals(identifier, node["start"], node["end"], False),
                    dict(kind="symbol_block", depth=depth, node=identifier, lines=[node["start"], node["end"]]),
                )
                if result not in {"accepted", "already_present", "file_limit"}:
                    pending.extend(windows[identifier])
            for _, identifier, start, end in sorted(pending):
                if len(attempts) >= MAX_ATTEMPTS:
                    break
                node = graph.nodes[identifier]
                try_pack(
                    node["path"],
                    node_intervals(identifier, start, end, True),
                    dict(kind="symbol_window", depth=depth, node=identifier, lines=[start, end]),
                )
        for chunk in ranked[:600]:
            if len(attempts) >= MAX_ATTEMPTS:
                break
            if chunk.score <= 0:
                continue
            try_pack(
                chunk.path,
                [(chunk.start, chunk.end)],
                dict(kind="bm25_fallback", depth=None, node=None, lines=[chunk.start, chunk.end]),
            )
    nodes = [
        dict(id=identifier, **{k: v for k, v in node.items() if k != "ast"})
        for identifier, node in sorted(graph.nodes.items())
    ]
    detail = dict(
        definitions=len(nodes),
        resolved_syntactic_edges=len(graph.edges),
        unresolved_source_calls=dict(graph.unresolved_calls),
        parse_errors=graph.parse_errors,
        wildcard_modules=sorted(graph.wildcard_modules),
        definition_index_sha256=hashlib.sha256(json.dumps(nodes, sort_keys=True).encode()).hexdigest(),
        call_edges_sha256=hashlib.sha256(json.dumps(graph.edges, sort_keys=True).encode()).hexdigest(),
        seeds=[dict(node=x, relevance=list(ranks[x])) for x in depths[0]],
        one_hop_neighbors=[dict(node=x, relevance=list(ranks[x])) for x in depths[1]],
        **provenance,
        packing_attempts=attempts,
        attempt_limit_reached=len(attempts) == MAX_ATTEMPTS,
        node_window_limit=MAX_WINDOWS,
        containment_edges=False,
    )
    return chosen, selected, detail


def build_packet(task_dir, encoding, budget, policy):
    m = RANKING
    EXTENSIONS, MAX_FILE_BYTES = m.EXTENSIONS, m.MAX_FILE_BYTES
    WORD = m.WORD
    spans, terms, merge, render, digest = m.spans, m.terms, m.merge, m.render, m.digest
    public = task_dir.resolve()
    for name in ["task.json", "problem_statement.md", "source"]:
        assert (public / name).exists(), (public.name, name)
        assert (public / name).resolve().is_relative_to(public), "No external symlink reads"
    meta_blob = (public / "task.json").read_bytes()
    issue_blob = (public / "problem_statement.md").read_bytes()
    metadata = json.loads(meta_blob)
    issue = issue_blob.decode("utf-8")
    files, skipped, chunks = {}, [], []
    source = public / "source"
    for file in sorted(source.rglob("*")):
        if not file.is_file():
            continue
        relative = file.relative_to(source).as_posix()
        if file.is_symlink() or not file.resolve().is_relative_to(source.resolve()):
            skipped.append({"path": relative, "reason": "symlink"})
            continue
        if file.suffix.lower() not in EXTENSIONS and file.name not in {"Makefile", "Dockerfile", "setup.py"}:
            skipped.append({"path": relative, "reason": "extension"})
            continue
        if file.stat().st_size > MAX_FILE_BYTES:
            skipped.append({"path": relative, "reason": "size"})
            continue
        blob = file.read_bytes()
        if b"\x00" in blob:
            skipped.append({"path": relative, "reason": "binary"})
            continue
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append({"path": relative, "reason": "encoding"})
            continue
        files[relative] = {"sha256": digest(blob), "bytes": len(blob), "lines": text.splitlines(keepends=True)}
        if TARGETS.path_allowed(relative):
            chunks.extend(spans(relative, text))
    assert chunks, "No source chunks available"
    query = Counter(terms(issue))
    df = Counter(term for chunk in chunks for term in chunk.tf if term in query)
    avg = sum(chunk.length for chunk in chunks) / len(chunks)
    issue_lower = issue.lower()
    mentioned = set(WORD.findall(issue))
    calls = set(re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", issue))
    for chunk in chunks:
        total = 0.0
        for term in sorted(query.keys() & chunk.tf.keys()):
            tf = chunk.tf[term]
            idf = math.log(1 + (len(chunks) - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * chunk.length / avg)) * (1 + math.log(query[term]))
        chunk.bm25 = total
        path = chunk.path.lower()
        leaf = Path(path).name
        if path in issue_lower or path.replace("/", ".").removesuffix(".py") in issue_lower:
            chunk.path_boost = 24.0
        elif leaf in issue_lower and len(leaf) > 5:
            chunk.path_boost = 12.0
        if chunk.symbol and any(s in mentioned for s in chunk.symbol.split(".")):
            chunk.symbol_boost = 8.0
        owner, _, member = chunk.symbol.rpartition(".")
        chunk.direct_symbol_anchor = bool(
            chunk.symbol and (chunk.symbol in calls or (owner in calls and member in {"__new__", "__init__"}))
        )
        chunk.score = total + chunk.path_boost + chunk.symbol_boost
    ranked = sorted(chunks, key=lambda c: (not c.direct_symbol_anchor, -c.score, c.path, c.start, c.end, c.kind))
    chosen, selected, symbol_graph = pack_symbols(
        public.name, metadata, issue, files, ranked, query, df, avg, encoding, budget
    )

    def count(text):
        return len(encoding.encode(text, disallowed_special=()))

    assert chosen, "No source span fits budget"
    packet = render(public.name, metadata, issue, chosen, files)
    token_count = count(packet)
    assert token_count <= budget
    source_spans = []
    for path, intervals in chosen.items():
        for start, end in merge(intervals):
            assert 1 <= start <= end <= len(files[path]["lines"])
            exact = "".join(files[path]["lines"][start - 1 : end])
            assert exact.rstrip("\n") in packet
            source_spans.append(
                {
                    "path": path,
                    "start_line": start,
                    "end_line": end,
                    "file_sha256": files[path]["sha256"],
                    "span_utf8_sha256": digest(exact.encode()),
                }
            )
    manifest = {
        "task_id": public.name,
        "packet_sha256": digest(packet.encode()),
        "packet_tokens": token_count,
        "tokenizer": "tiktoken/o200k_base",
        "token_count_is_model_specific_guarantee": False,
        "budget": budget,
        "issue_sha256": digest(issue_blob),
        "public_task_json_sha256": digest(meta_blob),
        "public_source_files_read": {
            path: {"sha256": value["sha256"], "bytes": value["bytes"], "lines": len(value["lines"])}
            for path, value in files.items()
        },
        "skipped_files": skipped,
        "candidate_chunks": len(chunks),
        "source_spans": source_spans,
        "selection_events": selected,
        "top_retrieval_candidates": [
            {
                "path": c.path,
                "lines": [c.start, c.end],
                "symbol": c.symbol,
                "score": c.score,
                "direct_symbol_anchor": c.direct_symbol_anchor,
            }
            for c in ranked[:40]
        ],
        "private_or_evaluator_inputs_read": False,
        "policy": policy,
        "symbol_graph": symbol_graph,
    }
    return packet, manifest


def synthetic(output):
    """Exercise scope and exact native packing using fabricated source only."""
    global RANKING, TARGETS, MAX_ATTEMPTS
    helper = load("symbol_native109", ROOT / "scripts/109_swe_patch_sft_context.py")
    RANKING = helper.load_ranking()
    TARGETS = load("symbol_targets110", ROOT / "scripts/110_swe_patch_sft_targets.py")
    enc = helper.NativeEncoding(helper.tokenizer())
    checks = []

    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})
        assert value, name

    source = """def helper(): return 1
if flag:
    from elsewhere import helper
class Plain:
    def helper(self): return 1
    def good(self): return self.helper()
    @descriptor_alias
    def uncertain(self): return self.helper()
def exception():
    try: pass
    except ValueError as helper: helper()
def pattern(value):
    match value:
        case {'callback': helper}: helper()
def deleted():
    del helper
    helper()
"""
    graph = DefinitionGraph(
        {
            "sympy/__init__.py": "from .core import exposed\n",
            "sympy/core.py": "def exposed(): return 2\n",
            "sympy/scope.py": source,
        }
    )
    edges = {(graph.nodes[e["source"]]["qualname"], graph.nodes[e["target"]]["qualname"]) for e in graph.edges}
    check("conditional_import_shadow", graph.resolve_in_module("helper", "sympy.scope")[0] is None)
    check("exception_pattern_del_and_decorator_shadow", edges == {("Plain.good", "Plain.helper")})
    seeds, _, provenance = graph.issue_seeds("```python3\nimport sympy as sp\nsp.exposed()\n```")
    check(
        "python3_issue_alias",
        any(r["reference"] == "sp.exposed" and r["node"] in seeds for r in provenance["issue_references"]),
    )
    wildcard = DefinitionGraph(
        {"sympy/wild.py": "def helper(): return 1\nfrom other import *\ndef caller(): return helper()\n"}
    )
    check("wildcard_uncertain", not wildcard.edges)
    conditional_method = DefinitionGraph(
        {
            "sympy/conditional.py": "class Example:\n    def helper(self): return 1\n    if flag:\n        def helper(self): return 2\n    def caller(self): return self.helper()\n"
        }
    )
    check("conditional_method_shadow", not conditional_method.edges)
    star_seeds, _, star_provenance = graph.issue_seeds("```python\nfrom sympy import *\nexposed()\n```")
    check(
        "known_issue_star_lexical_export",
        any(r["reference"] == "exposed" and r["node"] in star_seeds for r in star_provenance["issue_references"]),
    )
    unknown_seeds, _, _ = graph.issue_seeds("```python\nfrom unknown import *\nexposed()\n```")
    check("unknown_issue_star_uncertain", not unknown_seeds)
    issue = "repair alpha() and beta(); preserve trailing bytes.  \n\n"
    files = {
        "sympy/__init__.py": "from .alpha import alpha\nfrom .beta import beta\n",
        "sympy/alpha.py": "def alpha():\n    return 1\n",
        "sympy/beta.py": "def beta():\n    return 2\n",
        "sympy/fallback.py": "# repair alpha beta fallback relevant context\nvalue = 3\n",
        "sympy/tests/test_alpha.py": 'def alpha(): raise RuntimeError("excluded")\n',
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="symbol122-synthetic-") as temporary:
        task = Path(temporary) / "synthetic"
        (task / "source").mkdir(parents=True)
        (task / "task.json").write_text(json.dumps({"repo": "sympy/sympy", "base_commit": "synthetic"}))
        (task / "problem_statement.md").write_text(issue)

        def write_files():
            for path, text in files.items():
                dest = task / "source" / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text)

        write_files()
        packet, record = build_packet(task, enc, BUDGET, POLICY)
        packet2, record2 = build_packet(task, enc, BUDGET, POLICY)
        check("deterministic_packet_and_provenance", (packet, record) == (packet2, record2))
        ids = enc.encode(packet)
        direct = enc.tok.apply_chat_template(
            enc.messages(packet), tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(direct, list):
            direct = direct["input_ids"]
        check("exact_native_ids_and_cap", ids == direct and len(ids) <= BUDGET)
        check(
            "complete_issue_and_generation_boundary",
            issue in packet and enc.render(packet).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        )
        check("production_only", all(TARGETS.path_allowed(r["path"]) for r in record["source_spans"]))
        check(
            "seeds_before_bm25_fill",
            record["selection_events"][0]["kind"] == "symbol_block"
            and any(e["kind"] == "bm25_fallback" for e in record["selection_events"]),
        )
        boundary_packet, boundary = build_packet(task, enc, len(ids), POLICY)
        check("inclusive_exact_hard_budget", boundary_packet == packet and boundary["packet_tokens"] == len(ids))
        smaller_packet, smaller = build_packet(task, enc, len(ids) - 1, POLICY)
        check(
            "one_token_tighter_rejects_overflow", smaller["packet_tokens"] <= len(ids) - 1 and smaller_packet != packet
        )
        # A large first seed must defer windows while a later seed gets its full-block attempt.
        files["sympy/alpha.py"] = (
            "def alpha():\n"
            + "".join(f"    value_{i} = {i}  # alpha exact public line\n" for i in range(1700))
            + "    return 1\n"
        )
        write_files()
        _, large = build_packet(task, enc, BUDGET, POLICY)
        attempts = large["symbol_graph"]["packing_attempts"]
        check(
            "large_block_soft_rejection",
            any(e["kind"] == "symbol_block" and e["reason"] == "soft_file_budget" for e in attempts),
        )
        first_window = next(i for i, e in enumerate(attempts) if e["kind"] == "symbol_window")
        check(
            "all_seed_blocks_before_windows",
            sum(e["kind"] == "symbol_block" and e.get("depth") == 0 for e in attempts[:first_window]) == 2,
        )
        check(
            "common_top12_window_bound",
            all(
                len({tuple(e["lines"]) for e in attempts if e.get("node") == n and e["kind"] == "symbol_window"})
                <= MAX_WINDOWS
                for n in {e.get("node") for e in attempts}
            ),
        )
        check(
            "soft_rejections_eligible_second_pass",
            any(
                e["pass_number"] == 2 and e["kind"] == "symbol_block" and "native_file_tokens" not in e
                for e in attempts
            ),
        )
        check(
            "bounded_attempts_and_files", len(attempts) <= 600 and len({e["path"] for e in large["source_spans"]}) <= 16
        )
        check(
            "dedup_per_pass",
            all(
                len([(e["path"], str(e["proposed_intervals"])) for e in attempts if e["pass_number"] == p])
                == len({(e["path"], str(e["proposed_intervals"])) for e in attempts if e["pass_number"] == p})
                for p in [1, 2]
            ),
        )
        original_limit = MAX_ATTEMPTS
        try:
            MAX_ATTEMPTS = 3
            _, limited = build_packet(task, enc, BUDGET, POLICY)
            check(
                "attempt_counter_stops_exactly",
                len(limited["symbol_graph"]["packing_attempts"]) == 3
                and limited["symbol_graph"]["attempt_limit_reached"],
            )
        finally:
            MAX_ATTEMPTS = original_limit
    report = {
        "all_checks_passed": all(x["passed"] for x in checks),
        "kernel_sha256": sha(__file__),
        "checks": checks,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Fabricated AST and public-source packets only; exact cached native tokenizer, no real tasks/targets/models/GPU/API.",
        "native_small_prompt_tokens": len(ids),
        "large_prompt_tokens": large["packet_tokens"],
    }
    save(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, required=True)
    synthetic(parser.parse_args().synthetic)
