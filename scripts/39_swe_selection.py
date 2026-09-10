#!/usr/bin/env python3
"""Fixed-pool SWE repair selection: freeze, fetch patches, develop, then evaluate."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path("runs/swe-diversity-selection")
OUT = ROOT / "s1"
REV = "68195a1450865274106246d0d0296a1d6807b88e"
SEED = 20260909
SUBSETS = np.array(list(combinations(range(16), 4)))
PAIRS = np.array(list(combinations(range(4), 2)))
LAMBDAS = [0, 0.05, 0.1, 0.25, 0.5, 1, 2]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def provenance():
    return {"utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
            "script_sha256": digest(Path(__file__).read_text()), "api_spend_usd": 0}


def prepare():
    path = OUT / "split.json"
    if path.exists():
        raise FileExistsError("Split is already frozen")
    metadata = read_jsonl(ROOT / "public-audit/metadata.jsonl")
    issues = {r["instance_id"]: r for r in read_jsonl(OUT / "issue_metadata.jsonl")}
    pools = defaultdict(list)
    for r in metadata:
        if r["model_name"] == "swe-agent-llama-70b":
            pools[r["instance_id"]].append(r)
    eligible = sorted(i for i, rows in pools.items() if len(rows) >= 16)
    missing = sorted(set(eligible) - set(issues))
    if missing:
        write_json(OUT / "missing_issue_metadata.json", missing)
        raise ValueError(f"Missing issue metadata for {len(missing)} eligible issues")
    parent = {i: i for i in eligible}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i in eligible:
        r = issues[i]
        keys = [("commit", r["repo"], r["base_commit"]),
                ("text", digest(" ".join(r["problem_statement"].split())))]
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i in eligible:
        groups[root(i)].append(i)
    groups = sorted(groups.values(), key=lambda g: digest(f"{SEED}:split:{min(g)}"))
    inspected = "AnalogJ__lexicon-336"
    groups.sort(key=lambda g: inspected not in g)
    split, counts = {}, Counter()
    for group in groups:
        name = "dev" if counts["dev"] < 100 else "test" if counts["test"] < 300 else "reserve"
        for i in group:
            split[i] = {"split": name, "cluster": min(group), "repo": issues[i]["repo"],
                        "base_commit": issues[i]["base_commit"]}
            counts[name] += 1
    candidates, labels = [], {}
    for i, spec in split.items():
        if spec["split"] == "reserve":
            continue
        ordered = sorted(pools[i], key=lambda r: digest(f"{SEED}:pool:{r['source_file']}:{r['source_row']}"))
        for r in ordered[:16]:
            cid = f"{r['source_file']}:{r['source_row']}"
            candidates.append({"id": cid, "instance_id": i, "source_file": r["source_file"],
                               "source_row": r["source_row"]})
            labels[cid] = r["target"]
    plan = {"provenance": provenance(), "dataset_revision": REV, "counts": dict(counts),
            "issue_families": len(groups), "issues": split, "candidates": candidates,
            "protocol": {"n": 16, "k": 4, "lambda_grid": LAMBDAS,
                         "primary": "development-chosen quality+cheap diversity versus quality-only",
                         "quality_recipes": ["patch", "submission", "combined", "short"],
                         "metrics": ["tokens", "files"], "bootstrap": "10000 issue-family resamples",
                         "overlap": "exclude test issues sharing a valid normalized patch with dev in the same repo",
                         "duplicates": "retain within-issue duplicates; they represent pool redundancy",
                         "tie_break": "candidate SHA256 order, then lexicographic subset order"}}
    write_json(path, plan)
    write_json(OUT / "labels.json", labels)
    print(json.dumps({"counts": dict(counts), "issue_families": len(groups), "candidates": len(candidates)}))


def fetch(smoke):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    plan = json.loads((OUT / "split.json").read_text())
    by_file = defaultdict(list)
    dev = sorted(i for i, v in plan["issues"].items() if v["split"] == "dev")[:3]
    for r in plan["candidates"]:
        if not smoke or r["instance_id"] in dev:
            by_file[r["source_file"]].append(r)
    dest = OUT / ("smoke-patches" if smoke else "patches")
    dest.mkdir(exist_ok=True)

    def shard(item):
        source, selected = item
        output = dest / (Path(source).stem + ".jsonl")
        if output.exists():
            return f"cached {output.name}"
        fs = HfFileSystem(token=False)
        with fs.open(f"datasets/nebius/SWE-agent-trajectories@{REV}/{source}", "rb", block_size=65536) as stream:
            table = pq.ParquetFile(stream).read(columns=["instance_id", "model_name", "exit_status", "generated_patch"])
            rows = table.take([r["source_row"] for r in selected]).to_pylist()
        with output.open("w") as f:
            for locator, row in zip(selected, rows):
                assert locator["instance_id"] == row["instance_id"]
                assert row["model_name"] == "swe-agent-llama-70b"
                f.write(json.dumps({**row, "id": locator["id"]}) + "\n")
        return f"{output.name}: {len(rows)} patches"

    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(shard, sorted(by_file.items())):
            print(result, flush=True)


def patch_features(row):
    patch = row["generated_patch"] or ""
    result = subprocess.run(["git", "apply", "--numstat", "-z"], input=patch,
                            text=True, capture_output=True)
    valid = bool(patch.strip()) and result.returncode == 0 and bool(result.stdout.strip())
    files = sorted(set(re.findall(r"^\+\+\+ b/(.+)$|^--- a/(.+)$", patch, re.M)))
    files = sorted({name for pair in files for name in pair if name})
    changes = [line for line in patch.splitlines()
               if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    tokens = sorted(set(re.findall(r"[A-Za-z_]\w*|\d+|[^\w\s]", "\n".join(changes)))) if valid else []
    normalized = "\n".join(files) + "\n" + "\n".join(" ".join(line.split()) for line in changes)
    return {"id": row["id"], "instance_id": row["instance_id"], "valid": valid,
            "submitted": row["exit_status"] == "submitted", "exit_status": row["exit_status"],
            "files": files if valid else [], "tokens": tokens, "changed_lines": len(changes),
            "patch_chars": len(patch), "patch_hash": digest(patch),
            "normalized_hash": digest(normalized) if valid else None}


def features(smoke):
    src = OUT / ("smoke-patches" if smoke else "patches")
    rows = [r for p in sorted(src.glob("*.jsonl")) for r in read_jsonl(p)]
    assert rows
    plan = json.loads((OUT / "split.json").read_text())
    if not smoke:
        assert {r["id"] for r in rows} == {r["id"] for r in plan["candidates"]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs = list(pool.map(patch_features, rows))
    name = "smoke-features.jsonl" if smoke else "features.jsonl"
    with (OUT / name).open("w") as f:
        for row in fs:
            f.write(json.dumps(row) + "\n")
    hashes, by_issue = defaultdict(set), defaultdict(list)
    for row in fs:
        by_issue[row["instance_id"]].append(row)
        if row["normalized_hash"]:
            repo = plan["issues"][row["instance_id"]]["repo"]
            hashes[(repo, row["normalized_hash"])].add(row["instance_id"])
    overlaps = [sorted(ids) for ids in hashes.values()
                if any(plan["issues"][i]["split"] == "dev" for i in ids)
                and any(plan["issues"][i]["split"] == "test" for i in ids)]
    exclude = sorted({i for ids in overlaps for i in ids if plan["issues"][i]["split"] == "test"})
    audit = {"provenance": provenance(), "rows": len(fs), "issues": len(by_issue),
             "valid_patches": sum(r["valid"] for r in fs), "cross_split_patch_groups": overlaps,
             "exclude_test_issues": exclude,
             "mean_unique_valid_patches": float(np.mean([
                 len({r["normalized_hash"] for r in rs if r["valid"]}) for rs in by_issue.values()]))}
    write_json(OUT / ("smoke-feature-audit.json" if smoke else "feature-audit.json"), audit)
    print(json.dumps(audit, indent=2))


def scores(rows, recipe):
    valid = np.array([r["valid"] for r in rows], dtype=float)
    submitted = np.array([r["submitted"] for r in rows], dtype=float)
    combined = (2 * valid + submitted) / 3
    return {"patch": valid, "submission": submitted, "combined": combined,
            "short": 0.9 * combined + 0.1 * valid / (1 + np.log1p([r["changed_lines"] for r in rows]))}[recipe]


def distances(rows, metric):
    sets = [set(r[metric]) for r in rows]
    d = np.zeros((16, 16))
    for i, j in combinations(range(16), 2):
        if sets[i] and sets[j]:
            d[i, j] = d[j, i] = 1 - len(sets[i] & sets[j]) / len(sets[i] | sets[j])
    return d


def choose(rows, recipe, metric=None, lam=0, dedup=False):
    q = scores(rows, recipe)
    if dedup:
        order = sorted(range(16), key=lambda i: -q[i])
        selected, seen = [], set()
        for i in order:
            key = rows[i]["normalized_hash"]
            if key and key not in seen:
                selected.append(i)
                seen.add(key)
            if len(selected) == 4:
                break
        selected += [i for i in order if i not in selected][:4-len(selected)]
        return np.array(selected)
    objective = q[SUBSETS].mean(axis=1) if lam != "only" else np.zeros(len(SUBSETS))
    if metric and lam != 0:
        d = distances(rows, metric)
        diversity = d[SUBSETS[:, PAIRS[:, 0]], SUBSETS[:, PAIRS[:, 1]]].mean(axis=1)
        objective += (1 if lam == "only" else lam) * diversity
    return SUBSETS[np.argmax(objective)]


def load_pools(split):
    plan = json.loads((OUT / "split.json").read_text())
    audit = json.loads((OUT / "feature-audit.json").read_text())
    all_rows = {r["id"]: r for r in read_jsonl(OUT / "features.jsonl")}
    pools = defaultdict(list)
    for row in plan["candidates"]:
        i = row["instance_id"]
        if plan["issues"][i]["split"] == split and i not in audit["exclude_test_issues"]:
            pools[i].append(all_rows[row["id"]])
    assert all(len(rs) == 16 for rs in pools.values())
    return plan, pools


def outcomes(pools, specs, labels):
    results = {}
    for i, rows in pools.items():
        y = np.array([labels[r["id"]] for r in rows], dtype=float)
        c = int(y.sum())
        record = {"c": c, "oracle": float(c > 0),
                  "random": {"coverage": 1-comb(16-c, 4)/comb(16, 4), "accuracy": c/16}}
        for name, spec in specs.items():
            selected = choose(rows, **spec)
            record[name] = {"coverage": float(y[selected].max()), "accuracy": float(y[selected].mean()),
                            "selected": [rows[j]["id"] for j in selected],
                            "mean_changed_lines": float(np.mean([rows[j]["changed_lines"] for j in selected])),
                            "mean_changed_files": float(np.mean([len(rows[j]["files"]) for j in selected])),
                            "valid_rate": float(np.mean([rows[j]["valid"] for j in selected]))}
        results[i] = record
    return results


def develop():
    if (OUT / "frozen_selectors.json").exists():
        raise FileExistsError("Selectors already frozen")
    _, pools = load_pools("dev")
    labels = json.loads((OUT / "labels.json").read_text())
    qspecs = {recipe: {"recipe": recipe} for recipe in ["patch", "submission", "combined", "short"]}
    qout = outcomes(pools, qspecs, labels)

    def means(data, names):
        return {name: float(np.mean([r[name]["coverage"] for r in data.values()])) for name in names}

    qs = means(qout, qspecs)
    recipe = max(qs, key=qs.get)
    specs = {"quality": {"recipe": recipe}, "dedup": {"recipe": recipe, "dedup": True}}
    grid = {f"{m}:{lam}": {"recipe": recipe, "metric": m, "lam": lam}
            for m in ["tokens", "files"] for lam in LAMBDAS}
    grid.update({f"only:{m}": {"recipe": recipe, "metric": m, "lam": "only"} for m in ["tokens", "files"]})
    gs = means(outcomes(pools, grid, labels), grid)
    best = max((name for name in grid if not name.startswith("only:")), key=gs.get)
    only = max((name for name in grid if name.startswith("only:")), key=gs.get)
    nonzero = max((name for name in grid if grid[name]["lam"] not in [0, "only"]), key=gs.get)
    specs.update(quality_diversity=grid[best], diversity_only=grid[only],
                 quality_diversity_nonzero=grid[nonzero])
    write_json(OUT / "development.json", {"provenance": provenance(), "issues": len(pools),
               "quality_coverage": qs, "grid_coverage": gs})
    frozen = {"provenance": provenance(), "specs": specs, "quality_winner": recipe,
              "grid_winner": best, "diversity_winner": only, "nonzero_winner": nonzero,
              "split_sha256": digest((OUT / "split.json").read_text()),
              "feature_audit_sha256": digest((OUT / "feature-audit.json").read_text())}
    write_json(OUT / "frozen_selectors.json", frozen)
    print(json.dumps({"quality_coverage": qs, "grid_coverage": gs, "frozen": frozen}, indent=2))


def evaluate():
    frozen = json.loads((OUT / "frozen_selectors.json").read_text())
    assert digest((OUT / "split.json").read_text()) == frozen["split_sha256"]
    assert digest((OUT / "feature-audit.json").read_text()) == frozen["feature_audit_sha256"]
    plan, pools = load_pools("test")
    result = outcomes(pools, frozen["specs"], json.loads((OUT / "labels.json").read_text()))
    write_json(OUT / "evaluation_per_issue.json", result)
    ids = list(result)
    groups = defaultdict(list)
    for j, i in enumerate(ids):
        groups[plan["issues"][i]["cluster"]].append(j)
    groups = list(groups.values())
    draws = np.random.default_rng(SEED).integers(len(groups), size=(10000, len(groups)))
    weights = np.array([len(g) for g in groups])

    def estimate(values):
        values = np.array(values)
        totals = np.array([values[g].sum() for g in groups])
        boot = totals[draws].sum(axis=1)/weights[draws].sum(axis=1)
        return {"mean": float(values.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist()}

    names = ["random", *frozen["specs"]]
    aggregate = {name: {field: estimate([result[i][name][field] for i in ids])
                       for field in ["coverage", "accuracy"]} for name in names}
    contrasts = {}
    for a, b in [("quality", "random"), ("quality_diversity", "quality"),
                 ("quality_diversity", "dedup"), ("dedup", "quality"), ("diversity_only", "random"),
                 ("quality_diversity_nonzero", "quality")]:
        delta = [result[i][a]["coverage"]-result[i][b]["coverage"] for i in ids]
        contrasts[f"{a}-minus-{b}"] = {**estimate(delta), "wins": sum(x>0 for x in delta),
                                     "losses": sum(x<0 for x in delta)}
    labels = json.loads((OUT / "labels.json").read_text())
    conflicts = []
    for i, rows in pools.items():
        by_hash = defaultdict(set)
        for row in rows:
            if row["valid"]:
                by_hash[row["patch_hash"]].add(labels[row["id"]])
        if any(len(values) > 1 for values in by_hash.values()):
            conflicts.append(i)
    nuisance = {name: {field: float(np.mean([result[i][name][field] for i in ids]))
                       for field in ["mean_changed_lines", "mean_changed_files", "valid_rate"]}
                for name in frozen["specs"]}
    report = {"provenance": provenance(), "issues": len(ids), "issue_families": len(groups),
              "identical_valid_patch_label_conflict_issues": conflicts, "selected_patch_diagnostics": nuisance,
              "pool_correct_count_histogram": dict(Counter(result[i]["c"] for i in ids)),
              "oracle_coverage": estimate([result[i]["oracle"] for i in ids]),
              "oracle_gain_over_quality": estimate([result[i]["oracle"]-result[i]["quality"]["coverage"] for i in ids]),
              "aggregate": aggregate, "contrasts": contrasts, "frozen_selectors": frozen}
    write_json(OUT / "evaluation.json", report)
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "fetch", "features", "develop", "evaluate"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ["fetch", "features"]:
        {"fetch": fetch, "features": features}[args.stage](args.smoke)
    else:
        {"prepare": prepare, "develop": develop, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
