#!/usr/bin/env python3
"""Paired comparison of F2 inference-diversity arms (same problem set, same K).

Usage: analyze_infdiv.py <run_dir_neutral> <run_dir_static> [<run_dir_adaptive> ...]
First dir is the baseline; every other arm is compared to it with a paired bootstrap
over problems (coverage@K = any-correct, pass@1 = mean per-sample accuracy).
"""
import json, random, sys
from pathlib import Path


def load(d):
    by_q = {}
    for line in (Path(d) / "raw.jsonl").open():
        r = json.loads(line)
        by_q.setdefault(r["qidx"], []).append(r)
    arm = json.loads((Path(d) / "metrics.json").read_text())["arm"]
    return arm, by_q


def stats(by_q):
    cov = {q: int(any(r["correct"] for r in v)) for q, v in by_q.items()}
    p1 = {q: sum(r["correct"] for r in v) / len(v) for q, v in by_q.items()}
    da = {q: len(set(r["extracted"] for r in v if r["extracted"])) for q, v in by_q.items()}
    return cov, p1, da


def boot_ci(diffs, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    ms = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def main():
    runs = [load(d) for d in sys.argv[1:]]
    base_arm, base_q = runs[0]
    bcov, bp1, _ = stats(base_q)
    qs = sorted(base_q)
    print(f"paired over {len(qs)} problems, baseline={base_arm}")
    print(f"{'arm':<10}{'pass@1':>8}{'cov@k':>8}{'dANS':>6}   {'d_pass1 [95% CI]':<24}{'d_cov [95% CI]':<24}")
    for arm, by_q in runs:
        cov, p1, da = stats(by_q)
        p1m = sum(p1.values()) / len(p1)
        covm = sum(cov.values()) / len(cov)
        dam = sum(da.values()) / len(da)
        if arm == base_arm:
            print(f"{arm:<10}{p1m:>8.3f}{covm:>8.3f}{dam:>6.2f}")
            continue
        dp1 = [p1[q] - bp1[q] for q in qs]
        dcv = [cov[q] - bcov[q] for q in qs]
        lo1, hi1 = boot_ci(dp1)
        loc, hic = boot_ci(dcv)
        print(f"{arm:<10}{p1m:>8.3f}{covm:>8.3f}{dam:>6.2f}   "
              f"{sum(dp1)/len(dp1):+.3f} [{lo1:+.3f},{hi1:+.3f}]  "
              f"{sum(dcv)/len(dcv):+.3f} [{loc:+.3f},{hic:+.3f}]")


if __name__ == "__main__":
    main()
