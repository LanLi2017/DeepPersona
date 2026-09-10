#!/usr/bin/env python3
"""Exact maintainer-diff targets. CLI is synthetic-only; no dataset access."""

import argparse
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "runs/swe-diversity-selection/swe-sympy-patch-sft/target_converter_checks"
HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?$")


class TargetError(ValueError):
    """A retained, explicitly unsupported/non-exact target disposition."""

    def __init__(self, reason):
        self.reason = self.code = reason
        super().__init__(reason)


TargetRejection = TargetError


def require(condition, reason):
    if not condition:
        raise TargetError(reason)


def path_allowed(name):
    p = PurePosixPath(name)
    return (
        isinstance(name, str)
        and re.fullmatch(r"[A-Za-z0-9_./-]+", name)
        and str(p) == name
        and not p.is_absolute()
        and p.parts[0] == "sympy"
        and ".." not in p.parts
        and p.suffix == ".py"
        and not any(x in {"test", "tests", "testing", "config", "configs"} or x.startswith("test_") for x in p.parts)
        and not p.name.endswith("_test.py")
        and p.name not in {"conftest.py", "setup.py", "config.py"}
    )


def byte_lines(data):
    """Split on LF only: preserve CRLF, bare CR and missing terminal newline."""
    pieces = data.split(b"\n")
    return [x + b"\n" for x in pieces[:-1]] + ([pieces[-1]] if pieces[-1] else [])


def boundaries(data):
    result = [0]
    for line in byte_lines(data):
        result.append(result[-1] + len(line))
    return result


def _parse(base, patch):
    """Strict git unified diff; source coordinates and bytes are authoritative."""
    require(isinstance(patch, bytes) and bool(patch), "empty_or_nonbyte_patch")
    lines = byte_lines(patch)
    files = {}
    i = 0
    while i < len(lines):
        header = re.fullmatch(rb"diff --git a/([^\n ]+) b/([^\n ]+)\n?", lines[i])
        require(header is not None, "unsupported_diff_header")
        try:
            name = header[1].decode("utf-8")
            other = header[2].decode("utf-8")
        except UnicodeError as error:
            raise TargetError("non_utf8_path") from error
        require(name == other and path_allowed(name), "unsupported_path_or_rename")
        require(name in base and isinstance(base[name], bytes), "not_existing_byte_file")
        require(name not in files, "duplicate_file_section")
        try:
            base[name].decode("utf-8")
        except UnicodeError as error:
            raise TargetError("non_utf8_source") from error
        i += 1
        # Only an ordinary unchanged-mode index header is accepted.
        if i < len(lines) and lines[i].startswith(b"index "):
            require(
                re.fullmatch(rb"index [0-9a-f]+\.\.[0-9a-f]+(?: 100644| 100755)?\n?", lines[i]),
                "unsupported_index_or_file_mode",
            )
            i += 1
        require(
            i + 1 < len(lines)
            and lines[i] == b"--- a/" + name.encode() + b"\n"
            and lines[i + 1] == b"+++ b/" + name.encode() + b"\n",
            "new_deleted_binary_mode_or_unsupported_file_headers",
        )
        i += 2
        source = base[name]
        offsets = boundaries(source)
        hunks = []
        delta = 0
        previous_end = 0
        while i < len(lines) and not lines[i].startswith(b"diff --git "):
            match = HUNK.fullmatch(lines[i])
            require(match is not None, "unsupported_hunk_header")
            old_start, old_count, new_start, new_count = (
                int(match[1]),
                int(match[2]) if match[2] is not None else 1,
                int(match[3]),
                int(match[4]) if match[4] is not None else 1,
            )
            old_index = old_start - 1 if old_count else old_start
            new_index = new_start - 1 if new_count else new_start
            require(0 <= old_index <= old_index + old_count < len(offsets), "hunk_out_of_bounds")
            require(
                old_index >= previous_end and new_index == old_index + delta, "overlap_or_inconsistent_hunk_coordinates"
            )
            i += 1
            records = []
            while i < len(lines) and not lines[i].startswith((b"@@ ", b"diff --git ")):
                line = lines[i]
                if line == b"\\ No newline at end of file\n" or line == b"\\ No newline at end of file":
                    require(records and records[-1][1].endswith(b"\n"), "invalid_no_newline_marker")
                    records[-1] = (records[-1][0], records[-1][1][:-1])
                else:
                    require(line[:1] in (b" ", b"+", b"-"), "unsupported_hunk_record")
                    records.append((line[:1], line[1:]))
                i += 1
            require(
                sum(t != b"+" for t, _ in records) == old_count and sum(t != b"-" for t, _ in records) == new_count,
                "hunk_line_count_mismatch",
            )
            old = b"".join(v for t, v in records if t != b"+")
            new = b"".join(v for t, v in records if t != b"-")
            start, end = offsets[old_index], offsets[old_index + old_count]
            require(source[start:end] == old, "hunk_original_bytes_mismatch")
            require(old != new, "unchanged_hunk")
            hunks.append((start, end, new))
            delta += new_count - old_count
            previous_end = old_index + old_count
        require(hunks, "no_text_hunks")
        files[name] = hunks
    return files


def _apply_coordinates(source, hunks):
    result = source
    for start, end, new in reversed(hunks):
        result = result[:start] + new + result[end:]
    return result


def _merge(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _expand(span, offsets):
    # Prefer one unchanged line on the left; when unavailable, extend right.
    start, end = span
    left = bisect_left(offsets, start) - 1
    if left >= 0:
        return offsets[left], end
    right = bisect_right(offsets, end)
    require(right < len(offsets), "no_nonempty_unique_anchor")
    return start, offsets[right]


def _replacement(source, span, hunks):
    start, end = span
    selected = [(a - start, b - start, new) for a, b, new in hunks if start <= a and b <= end]
    return _apply_coordinates(source[start:end], selected)


def _file_edits(name, source, hunks):
    require(source, "empty_source_cannot_have_nonempty_old_anchor")
    offsets = boundaries(source)
    spans = _merge([(a, b) for a, b, _ in hunks])
    # Every expansion consumes original source extent; merging preserves all hunks.
    while True:
        failed = None
        current = source
        edits = []
        for j in range(len(spans) - 1, -1, -1):
            start, end = spans[j]
            old = source[start:end]
            if (
                not old
                or source.find(old) != source.rfind(old)
                or current.find(old) < 0
                or current.find(old) != current.rfind(old)
            ):
                failed = j
                break
            new = _replacement(source, spans[j], hunks)
            try:
                edit = dict(path=name, old=old.decode("utf-8"), new=new.decode("utf-8"))
            except UnicodeError as error:
                raise TargetError("non_utf8_target") from error
            require(old != new, "unchanged_merged_edit")
            edits.append(edit)
            current = current.replace(old, new, 1)
        if failed is None:
            require(current == _apply_coordinates(source, hunks), "converter_roundtrip_mismatch")
            return edits
        spans[failed] = _expand(spans[failed], offsets)
        spans = _merge(spans)


def convert(base: Mapping[str, bytes], patch: bytes) -> list[dict[str, str]]:
    """Canonical path order / descending original offsets, <=20 exact edits.

    Raises TargetError for unsupported or non-exact targets. Reads no files.
    Empty source insertions cannot satisfy the nonempty-old contract and reject.
    """
    require(isinstance(base, Mapping), "base_must_be_mapping")
    parsed = _parse(base, patch)
    edits = []
    for name in sorted(parsed):
        edits += _file_edits(name, base[name], parsed[name])
    require(0 < len(edits) <= 20, "edit_count_outside_1_to_20")
    return edits


def apply_edits(base, edits):
    """Independent sequential contract check, without hunk coordinates."""
    result = dict(base)
    require(0 < len(edits) <= 20, "edit_count_outside_1_to_20")
    for edit in edits:
        require(set(edit) == {"path", "old", "new"}, "invalid_edit_schema")
        name, old, new = (edit[k] for k in ("path", "old", "new"))
        require(path_allowed(name) and name in result, "unsupported_edit_path")
        require(isinstance(old, str) and isinstance(new, str) and old and old != new, "invalid_edit_strings")
        old, new = old.encode("utf-8"), new.encode("utf-8")
        require(result[name].count(old) == 1, "nonunique_sequential_anchor")
        result[name] = result[name].replace(old, new, 1)
    return result


def canonical_target(edits):
    return json.dumps(
        dict(rationale="", edits=[dict(path=e["path"], old=e["old"], new=e["new"]) for e in edits]),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def check_target(base, patch, edits, visible_intervals, tokenizer, max_tokens=2048):
    """Check immutable context and completion budget; never expand context.

    visible_intervals: path -> [(start_line, end_line), ...], 1-based inclusive.
    Each whole anchor must fit ONE interval; separated excerpts cannot form an anchor.
    tokenizer must expose encode(..., add_special_tokens=False) and eos_token_id.
    """
    require(edits == convert(base, patch), "target_not_canonical_converter_output")
    parsed = _parse(base, patch)
    gold = dict(base)
    for name, hunks in parsed.items():
        gold[name] = _apply_coordinates(base[name], hunks)
    require(apply_edits(base, edits) == gold, "target_roundtrip_mismatch")
    anchors = validate_visibility(base, edits, visible_intervals)
    serialized = serialize_target(edits, tokenizer)
    require(serialized["target_tokens_including_eos"] <= max_tokens, "completion_token_budget_exceeded")
    return dict(
        **serialized,
        anchors=anchors,
        patched_sha256={name: hashlib.sha256(gold[name]).hexdigest() for name in sorted(parsed)},
    )


def validate_visibility(base, edits, visible_intervals):
    """Return original anchor coordinates, or reject; never alter retrieval."""
    anchors = []
    for edit in edits:
        name = edit["path"]
        data = base[name]
        old = edit["old"].encode()
        require(old and data.find(old) >= 0 and data.find(old) == data.rfind(old), "nonunique_original_anchor")
        start = data.index(old)
        end = start + len(old)
        first = data[:start].count(b"\n") + 1
        last = data[: end - 1].count(b"\n") + 1
        intervals = visible_intervals.get(name, [])
        require(
            all(type(a) is int and type(b) is int and 1 <= a <= b for a, b in intervals), "invalid_visibility_intervals"
        )
        require(any(a <= first <= last <= b for a, b in intervals), "anchor_not_wholly_visible")
        anchors.append(dict(path=name, start_byte=start, end_byte=end, start_line=first, end_line=last))
    return anchors


def serialize_target(edits, tokenizer):
    """Serialize/count without filtering, so token fit is independently reported."""
    target = canonical_target(edits)
    require(type(tokenizer.eos_token_id) is int, "missing_single_eos_token")
    ids = list(tokenizer.encode(target, add_special_tokens=False)) + [tokenizer.eos_token_id]
    return dict(
        target=target, target_token_ids=ids, target_tokens_including_eos=len(ids), eos_token_id=tokenizer.eos_token_id
    )


def synthetic_checks(output):
    """Only generated fixtures. git diff / git apply is the independent oracle."""
    require(not output.exists(), "checks_output_already_exists")
    output.mkdir(parents=True)
    results = []

    class ByteTokenizer:
        eos_token_id = 256

        def encode(self, text, add_special_tokens=False):
            require(not add_special_tokens, "unexpected_special_tokens")
            return list(text.encode())

    cases = [
        ("replace", b"a = 1\nb = 2\n", b"a = 1\nb = 3\n"),
        ("crlf", b"a = 1\r\n\r\nb = 2\r\n", b"a = 1\r\n\r\nb = 3\r\n"),
        ("insert_empty_line", b"a = 1\nb = 2\n", b"a = 1\n\nb = 2\n"),
        ("delete_empty_line", b"a = 1\n\nb = 2\n", b"a = 1\nb = 2\n"),
        ("insert_begin", b"a = 1\n", b"b = 2\na = 1\n"),
        ("insert_end", b"a = 1\n", b"a = 1\nb = 2\n"),
        ("delete_all_content", b"a = 1\n", b""),
        ("no_final_newline", b"a = 1\nb = 2", b"a = 1\nb = 3"),
        ("add_final_newline", b"a = 1", b"a = 1\n"),
        ("remove_final_newline", b"a = 1\n", b"a = 1"),
        ("crlf_no_final_newline", b"a = 1\r\nb = 2", b"a = 1\r\nb = 3"),
        ("utf8", 'label = "α"\n'.encode(), 'label = "β"\n'.encode()),
        ("repeated_anchor", b"x = 1\nx = 1\nx = 1\n", b"x = 1\nx = 2\nx = 1\n"),
        ("overlap_after_extension", b"x = 1\nx = 1\nx = 1\nx = 1\n", b"x = 2\nx = 1\nx = 3\nx = 1\n"),
        ("sequential_new_collision", b"x = 1\nunique = 0\ny = 2\n", b"x = 3\nunique = 0\nx = 1\n"),
    ]
    cases += [(name + "_context3", before, after) for name, before, after in list(cases)]

    def git(repo, *args):
        proc = subprocess.run(
            ["git", "-c", "core.autocrlf=false", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout

    for name, before, after in cases:
        with tempfile.TemporaryDirectory(prefix="swe110_") as tmp:
            repo = Path(tmp)
            path = repo / "sympy/core/example.py"
            path.parent.mkdir(parents=True)
            git(repo, "init", "-q")
            path.write_bytes(before)
            git(repo, "add", ".")
            path.write_bytes(after)
            patch = git(
                repo,
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--unified=" + ("3" if name.endswith("_context3") else "0"),
                "--",
                "sympy/core/example.py",
            )
            path.write_bytes(before)
            patchfile = repo / "fixture.patch"
            patchfile.write_bytes(patch)
            git(repo, "apply", "--unidiff-zero", "--whitespace=nowarn", str(patchfile))
            oracle = path.read_bytes()
            require(oracle == after, "synthetic_git_oracle_mismatch")
            base = {"sympy/core/example.py": before}
            edits = convert(base, patch)
            require(apply_edits(base, edits)["sympy/core/example.py"] == oracle, "synthetic_roundtrip_failed")
            require(convert(base, patch) == edits, "nondeterministic_conversion")
            checked = check_target(
                base, patch, edits, {"sympy/core/example.py": [(1, len(byte_lines(before)))]}, ByteTokenizer()
            )
            # Context visibility and completion constraints must reject, not adapt.
            for intervals, cap, expected in [
                ({}, 2048, "anchor_not_wholly_visible"),
                ({"sympy/core/example.py": [(1, len(byte_lines(before)))]}, 1, "completion_token_budget_exceeded"),
            ]:
                try:
                    check_target(base, patch, edits, intervals, ByteTokenizer(), cap)
                except TargetError as error:
                    require(str(error) == expected, "unexpected_constraint_failure")
                else:
                    raise TargetError("missing_constraint_rejection")
            fixture = output / name
            fixture.mkdir()
            (fixture / "before.bin").write_bytes(before)
            (fixture / "after.bin").write_bytes(oracle)
            (fixture / "patch.diff").write_bytes(patch)
            (fixture / "target.json").write_text(checked["target"], encoding="utf-8")
            results.append(
                dict(
                    case=name,
                    passed=True,
                    edits=len(edits),
                    target_tokens_with_synthetic_byte_tokenizer=checked["target_tokens_including_eos"],
                )
            )
    # Python bytes.count misses overlapping matches: two identical lines in three.
    overlap_base = {"sympy/a.py": b"x = 1\nx = 1\nx = 1\n"}
    overlap_patch = b"diff --git a/sympy/a.py b/sympy/a.py\n--- a/sympy/a.py\n+++ b/sympy/a.py\n@@ -2,2 +2,2 @@\n x = 1\n-x = 1\n+x = 2\n"
    overlap_edits = convert(overlap_base, overlap_patch)
    with tempfile.TemporaryDirectory(prefix="swe110_overlap_") as tmp:
        repo = Path(tmp)
        path = repo / "sympy/a.py"
        path.parent.mkdir(parents=True)
        git(repo, "init", "-q")
        path.write_bytes(overlap_base["sympy/a.py"])
        patchfile = repo / "fixture.patch"
        patchfile.write_bytes(overlap_patch)
        git(repo, "apply", "--unidiff-zero", "--whitespace=nowarn", str(patchfile))
        require(
            apply_edits(overlap_base, overlap_edits)["sympy/a.py"] == path.read_bytes(),
            "overlapping_occurrence_roundtrip",
        )
    results.append(dict(case="overlapping_duplicate_occurrences", passed=True))
    # Explicit unsupported file operations and malformed source coordinates.
    good = b"diff --git a/sympy/a.py b/sympy/a.py\n--- a/sympy/a.py\n+++ b/sympy/a.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    rejected = {
        "test_path": (good.replace(b"sympy/a.py", b"sympy/tests/a.py"), {"sympy/tests/a.py": b"a = 1\n"}),
        "config_path": (good.replace(b"sympy/a.py", b"sympy/config.py"), {"sympy/config.py": b"a = 1\n"}),
        "outside_path": (good.replace(b"sympy/a.py", b"other/a.py"), {"other/a.py": b"a = 1\n"}),
        "missing_file": (good, {}),
        "coordinate_mismatch": (good.replace(b"@@ -1 +1 @@", b"@@ -2 +2 @@"), {"sympy/a.py": b"a = 1\n"}),
        "content_mismatch": (good, {"sympy/a.py": b"a = 3\n"}),
        "new_file": (good.replace(b"--- a/sympy/a.py", b"--- /dev/null"), {"sympy/a.py": b"a = 1\n"}),
        "deleted_file": (good.replace(b"+++ b/sympy/a.py", b"+++ /dev/null"), {"sympy/a.py": b"a = 1\n"}),
        "rename": (good.replace(b"b/sympy/a.py", b"b/sympy/b.py"), {"sympy/a.py": b"a = 1\n"}),
        "binary": (b"diff --git a/sympy/a.py b/sympy/a.py\nGIT binary patch\n", {"sympy/a.py": b"a = 1\n"}),
        "mode_change": (
            good.replace(b"--- a/", b"old mode 100644\nnew mode 100755\n--- a/"),
            {"sympy/a.py": b"a = 1\n"},
        ),
    }
    rejected["empty_existing_source"] = (
        b"diff --git a/sympy/a.py b/sympy/a.py\n--- a/sympy/a.py\n+++ b/sympy/a.py\n@@ -0,0 +1 @@\n+a = 1\n",
        {"sympy/a.py": b""},
    )
    many = [f"sympy/file{i:02d}.py" for i in range(21)]
    rejected["over_20_edits"] = (
        b"".join(good.replace(b"sympy/a.py", n.encode()) for n in many),
        {n: b"a = 1\n" for n in many},
    )
    # Two files in reverse input order must produce sorted canonical path order.
    two_base = {"sympy/z.py": b"a = 1\n", "sympy/a.py": b"a = 1\n"}
    two_patch = good.replace(b"sympy/a.py", b"sympy/z.py") + good
    two_edits = convert(two_base, two_patch)
    require([e["path"] for e in two_edits] == ["sympy/a.py", "sympy/z.py"], "incorrect_path_order")
    results.append(dict(case="canonical_multifile_order", passed=True))
    # A multi-line anchor may not bridge separately supplied visibility intervals.
    span_base = {"sympy/a.py": b"a = 1\nb = 2\n"}
    span_patch = good.replace(b"@@ -1 +1 @@", b"@@ -1,2 +1,2 @@").replace(b"+a = 2\n", b"+a = 2\n b = 2\n")
    span_edits = convert(span_base, span_patch)
    try:
        validate_visibility(span_base, span_edits, {"sympy/a.py": [(1, 1), (2, 2)]})
    except TargetError as error:
        require(error.code == "anchor_not_wholly_visible", "incorrect_visibility_failure")
    else:
        raise TargetError("visibility_intervals_were_combined")
    results.append(dict(case="disjoint_visibility_rejected", passed=True))
    for name, (patch, base) in rejected.items():
        try:
            convert(base, patch)
        except TargetError as error:
            results.append(dict(case=name, passed=True, rejection=str(error)))
        else:
            raise TargetError("unsupported_fixture_accepted:" + name)
    summary = dict(
        passed=True,
        utc=datetime.now(timezone.utc).isoformat(),
        cases=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle="Independent git apply --unidiff-zero on synthetic fixtures",
        tokenizer="Synthetic byte tokenizer plus one EOS; no claim of real Qwen budget admission",
        real_data_accessed=False,
        api_calls=0,
        gpu_used=False,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(passed=True, cases=len(results), output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["synthetic-checks"])
    parser.add_argument("--output", type=Path, default=CHECKS)
    args = parser.parse_args()
    synthetic_checks(args.output)
