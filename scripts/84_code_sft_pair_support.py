#!/usr/bin/env python3
"""CPU-only correct-pair support audit; no grading, fitting, generation or endpoint freeze."""

import ast
from collections import Counter
from itertools import combinations
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize
from transformers import AutoTokenizer

ROOT = Path("runs/swe-diversity-selection")
OUT = ROOT / "code-sft-selection-support"
MODEL = Path(
    "/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.open()]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


class NoDocs(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
        return node


def features(code):
    tree = NoDocs().visit(ast.parse(code))
    canonical = ast.dump(tree, include_attributes=False)
    # Unparse is for diagnostic features only; original verified code remains the target.
    surface = ast.unparse(tree)
    lexical = [
        (token.type, token.string)
        for token in tokenize.generate_tokens(io.StringIO(surface).readline)
        if token.type
        not in {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
        }
    ]
    return (
        sha(canonical),
        sha(json.dumps(lexical)),
        set(re.findall(r"[A-Za-z_]\w*|\d+|[^\w\s]", surface)),
        {type(node).__name__ for node in ast.walk(tree)},
    )


def jaccard(a, b):
    return 1 - len(a & b) / len(a | b) if a | b else 0


def pair_stats(rows):
    pairs = []
    for a, b in combinations(rows, 2):
        if a["ast_sha256"] == b["ast_sha256"]:
            continue
        pairs.append(
            dict(
                record_ids=[a["record_id"], b["record_id"]],
                tokens=a["target_tokens"] + b["target_tokens"],
                ast_jaccard=jaccard(a["_nodes"], b["_nodes"]),
                token_jaccard=jaccard(a["_tokens"], b["_tokens"]),
            )
        )
    matched = []
    for i, j in combinations(range(len(pairs)), 2):
        a, b = pairs[i], pairs[j]
        gap = abs(a["tokens"] - b["tokens"]) / min(a["tokens"], b["tokens"])
        if gap <= 0.05:
            matched.append(
                dict(
                    pair_indices=[i, j],
                    relative_token_gap=gap,
                    disjoint=set(a["record_ids"]).isdisjoint(b["record_ids"]),
                    ast_score_differs=abs(a["ast_jaccard"] - b["ast_jaccard"]) > 1e-12,
                    token_score_differs=abs(a["token_jaccard"] - b["token_jaccard"]) > 1e-12,
                )
            )
    return dict(
        pairs=pairs,
        matched_edges=matched,
        pair_count=len(pairs),
        matched_edge_count=len(matched),
        ast_varying_edges=sum(x["ast_score_differs"] for x in matched),
        token_varying_edges=sum(x["token_score_differs"] for x in matched),
        disjoint_edges=sum(x["disjoint"] for x in matched),
        disjoint_ast_varying_edges=sum(x["disjoint"] and x["ast_score_differs"] for x in matched),
    )


assert not OUT.exists()
OUT.mkdir()
audit = ROOT / "code-extraction-audit"
interface = ROOT / "code-extraction-interface-audit"
sources = {r["record_id"]: r for r in lines(audit / "inputs.jsonl") if r["split"] == "train"}
outcomes = {r["record_id"]: r for r in lines(audit / "outcomes.jsonl") if r["split"] == "train"}
cases = {f"train:{r['record_id']}": r for r in read(interface / "cases.json")}
case_results = {f"train:{r['record_id']}": r for r in read(interface / "results.json")}
assert len(sources) == len(outcomes) == 960 and len(cases) == 3
paths = [
    Path(__file__),
    audit / "inputs.jsonl",
    audit / "outcomes.jsonl",
    interface / "cases.json",
    interface / "results.json",
    interface / "protocol.json",
    Path("scripts/79_code_interface_extraction_audit.py"),
]
for name, digest in read(interface / "completed_manifest.json")["sha256"].items():
    assert hashlib.sha256((interface / name).read_bytes()).hexdigest() == digest
prompts = {}
for pool in ["code-learning-pilot", "code-learning-expansion"]:
    path = ROOT / pool / "inputs.jsonl"
    paths.append(path)
    prompts.update({r["task_id"]: r for r in lines(path) if r["split"] == "train"})
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
records = []
for rid, source in sources.items():
    case = cases.get(rid)
    correct = case_results[rid]["grade"]["correct"] if case else outcomes[rid]["first_correct"]
    if not correct:
        continue
    code = case["selected_code"] if case else source["first_code"]
    required = set(re.findall(r"def\s+(\w+)\s*\(", "\n".join(prompts[source["task_id"]]["interfaces"])))
    defined = {
        n.name for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert required <= defined
    target = "```python\n" + code.strip() + "\n```"
    ast_hash, lexical_hash, tokens, nodes = features(code)
    token_ids = tok.encode(target, add_special_tokens=False) + [tok.eos_token_id]
    records.append(
        dict(
            record_id=rid,
            task_id=source["task_id"],
            sample=source["sample"],
            code=code,
            target=target,
            target_tokens=len(token_ids),
            target_token_ids=token_ids,
            target_token_sha256=sha(json.dumps(token_ids)),
            ast_sha256=ast_hash,
            normalized_lexical_sha256=lexical_hash,
            _tokens=tokens,
            _nodes=nodes,
        )
    )
assert len(records) == 626
all_tasks = []
pair_details = []
for task in sorted(prompts):
    rows = [r for r in records if r["task_id"] == task]
    reps = {}
    for r in rows:
        previous = reps.get(r["ast_sha256"])
        if previous is None or sha("sft-support-representative:" + r["record_id"]) < sha(
            "sft-support-representative:" + previous["record_id"]
        ):
            reps[r["ast_sha256"]] = r
    raw = pair_stats(rows)
    dedup = pair_stats(list(reps.values()))
    stats = dict(
        task_id=task,
        successful_records=len(rows),
        original_first_mixed=0
        < sum(outcomes[rid]["first_correct"] for rid in sources if sources[rid]["task_id"] == task)
        < 8,
        interface_mixed=0 < len(rows) < 8,
        unique_target_token_sequences=len({r["target_token_sha256"] for r in rows}),
        unique_ast=len(reps),
        unique_normalized_lexical_sequences=len({r["normalized_lexical_sha256"] for r in rows}),
        canonical_representative_ids=[r["record_id"] for r in reps.values()],
        target_token_lengths=[r["target_tokens"] for r in rows],
        raw={k: v for k, v in raw.items() if k not in ["pairs", "matched_edges"]},
        deduplicated={k: v for k, v in dedup.items() if k not in ["pairs", "matched_edges"]},
    )
    all_tasks.append(stats)
    pair_details.append(dict(task_id=task, raw=raw, deduplicated=dedup))


def aggregate(tasks):
    result = dict(
        tasks=len(tasks),
        successful_records=sum(r["successful_records"] for r in tasks),
        tasks_any_success=sum(r["successful_records"] > 0 for r in tasks),
        tasks_at_least2_success=sum(r["successful_records"] >= 2 for r in tasks),
        tasks_at_least3_success=sum(r["successful_records"] >= 3 for r in tasks),
        tasks_at_least2_unique_ast=sum(r["unique_ast"] >= 2 for r in tasks),
        tasks_at_least3_unique_ast=sum(r["unique_ast"] >= 3 for r in tasks),
        unique_ast_total=sum(r["unique_ast"] for r in tasks),
        unique_target_sequences_total=sum(r["unique_target_token_sequences"] for r in tasks),
        unique_ast_count_histogram=dict(sorted(Counter(r["unique_ast"] for r in tasks).items())),
    )
    for view in ["raw", "deduplicated"]:
        result[view] = {
            key: dict(tasks=sum(r[view][key] > 0 for r in tasks), total=sum(r[view][key] for r in tasks))
            for key in [
                "pair_count",
                "matched_edge_count",
                "ast_varying_edges",
                "token_varying_edges",
                "disjoint_edges",
                "disjoint_ast_varying_edges",
            ]
        }
        result[view]["tasks_multiple_pair_choices"] = sum(r[view]["pair_count"] >= 2 for r in tasks)
    return result


summary = dict(
    populations={
        "all120_training_tasks": aggregate(all_tasks),
        "all94_tasks_with_any_correct": aggregate([r for r in all_tasks if r["successful_records"] > 0]),
        "original_FIRST_mixed30": aggregate([r for r in all_tasks if r["original_first_mixed"]]),
        "interface_mixed": aggregate([r for r in all_tasks if r["interface_mixed"]]),
        "all8_interface_correct": aggregate([r for r in all_tasks if r["successful_records"] == 8]),
    },
    eligible_deduplicated_ast_varying_task_ids=[
        r["task_id"] for r in all_tasks if r["deduplicated"]["ast_varying_edges"]
    ],
    eligible_deduplicated_disjoint_ast_varying_task_ids=[
        r["task_id"] for r in all_tasks if r["deduplicated"]["disjoint_ast_varying_edges"]
    ],
    max_target_tokens=max(r["target_tokens"] for r in records),
    target_tokens_over768=[r["record_id"] for r in records if r["target_tokens"] > 768],
    new_grades=0,
    api_spend_usd=0,
    gpu_jobs=0,
)
save(
    "protocol.json",
    dict(
        scope="CPU support/design audit only; no endpoint or selection protocol frozen",
        extraction="First fenced block containing all required public top-level definitions; reuse79 labels for3 exceptions, otherwise75 FIRST labels; no verifier-based block choice",
        ast_identity="ast.dump include_attributes=False after recursively removing module/function/class docstrings; identifiers and constants retained; comments ignored by parser",
        token_identity="Exact model token sequence of original interface block stripped at boundaries and Python-fenced plus EOS",
        normalized_lexical_identity="Python token type/string sequence from docstring-free AST unparse, ignoring whitespace/comment token types; diagnostic only",
        deduplication="Choose one original verified surface per unique AST by minimum SHA256(sft-support-representative:record_id); support sensitivity, not final training selection",
        group="K2 distinct-AST correct programs from same task; compare alternative groups with abs(tokenSumA-tokenSumB)/min<=.05, no relaxation",
        scores="Diagnostic set-Jaccard distances on docstring-free lexical tokens and AST node types; no utility gradients computed or scores fit",
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        model_snapshot=str(MODEL),
        seed=None,
        no_measurement_or_final_scoring=True,
    ),
)
save("task_support.json", all_tasks)
save("pair_support.json", pair_details)
(OUT / "successful_records.jsonl").write_text(
    "".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n" for r in records)
)
save("summary.json", summary)
save(
    "completed_manifest.json",
    dict(sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}),
)
print(json.dumps(summary))
