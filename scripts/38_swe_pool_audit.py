#!/usr/bin/env python3
"""Audit public SWE repair pool support using narrow Parquet columns, without model calls."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
import subprocess
import urllib.request

DATASET = "nebius/SWE-agent-trajectories"
REVISION = "68195a1450865274106246d0d0296a1d6807b88e"
COLUMNS = ["instance_id", "model_name", "target", "exit_status"]


def coverage(n, c, k):
    return 1 - comb(n - c, k) / comb(n, k)


def summarize(rows):
    by_issue, by_model = defaultdict(list), defaultdict(list)
    for r in rows:
        if not isinstance(r["target"], bool):
            raise ValueError("Missing or nonbinary success label")
        by_issue[r["instance_id"]].append(r)
        by_model[(r["instance_id"], r["model_name"])].append(r)
    result = {
        "rows": len(rows), "issues": len(by_issue), "issue_model_groups": len(by_model),
        "successes": sum(r["target"] for r in rows),
        "models": dict(Counter(r["model_name"] for r in rows)),
        "exit_status": dict(Counter(r["exit_status"] for r in rows)),
        "exit_success_counts": [
            {"exit_status": status, "success": success, "count": count}
            for (status, success), count in sorted(Counter(
                (str(r["exit_status"]), r["target"]) for r in rows).items())],
        "eligibility": [],
    }
    groupings = [("issue", by_issue), ("issue_model", by_model)]
    groupings += [(f"model:{model}", {key: rs for key, rs in by_model.items() if key[1] == model})
                  for model in sorted(result["models"])]
    for grouping, pools in groupings:
        for n in [8, 16]:
            eligible = [rs for rs in pools.values() if len(rs) >= n]
            counts = [(len(rs), sum(r["target"] for r in rs)) for rs in eligible]
            if not counts:
                continue
            random4 = sum(coverage(m, c, 4) for m, c in counts) / len(counts)
            pool_n = sum(coverage(m, c, n) for m, c in counts) / len(counts)
            result["eligibility"].append({
                "grouping": grouping, "candidate_n": n, "selected_k": 4,
                "groups": len(counts),
                "unique_issues": len({rs[0]["instance_id"] for rs in eligible}),
                "all_fail_full_pools": sum(c == 0 for m, c in counts),
                "all_success_full_pools": sum(c == m for m, c in counts),
                "mixed_full_pools": sum(0 < c < m for m, c in counts),
                "mean_full_pool_accuracy": sum(c / m for m, c in counts) / len(counts),
                "expected_random_coverage4": random4,
                "expected_random_pool_coverage_n": pool_n,
                "expected_oracle_selection_headroom": pool_n - random4,
            })
    result["interpretation"] = (
        "Support audit, not a selector result. Expectations are exact for uniform sampling "
        "without replacement from each published full pool: select N, then k=4. "
        "Issue-model rows weight groups, not unique issues; mixed status describes the full "
        "published pool, not the N-subsample. Model names do not establish matched settings "
        "or independent sampling. Counts precede trajectory-duplicate/provenance audits."
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/swe-diversity-selection/public-audit"))
    args = ap.parse_args()
    out = args.out / "smoke" if args.smoke else args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "metadata.jsonl"
    if args.fetch:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        url = f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"
        with urllib.request.urlopen(url, timeout=30) as response:
            meta = json.load(response)
        if meta["sha"] != REVISION:
            raise ValueError("Dataset revision mismatch")
        paths = sorted(f["rfilename"] for f in meta["siblings"] if f["rfilename"].endswith(".parquet"))
        if args.smoke:
            paths = paths[:1]

        def read(path):
            fs = HfFileSystem(token=False)
            remote = f"datasets/{DATASET}@{REVISION}/{path}"
            with fs.open(remote, "rb", block_size=65536) as stream:
                pf = pq.ParquetFile(stream)
                table = pf.read(columns=COLUMNS)
                rows = table.to_pylist()
            for index, row in enumerate(rows):
                row.update(source_file=path, source_row=index)
            print(f"{path}: {len(rows)} metadata rows", flush=True)
            return rows

        with ThreadPoolExecutor(max_workers=3) as pool:
            rows = [row for shard in pool.map(read, paths) for row in shard]
        with cache.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        manifest = {
            "dataset": DATASET, "revision": REVISION, "columns": COLUMNS,
            "files": paths, "smoke": args.smoke, "model_api_spend_usd": 0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "script": str(Path(__file__).relative_to(Path.cwd())),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        rows = [json.loads(line) for line in cache.open()]
    result = summarize(rows)
    result["analysis"] = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
