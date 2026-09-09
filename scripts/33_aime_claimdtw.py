"""
"Reduced" Metric (b): claim:dtw only, run entirely locally via Ollama -- no GPU, no OpenAI key.
The other two ensemble components (l15/l4head trained heads) need raw access to Qwen3-8B's
internal hidden states, which Ollama's API does not expose regardless of hardware, plus trained
weight files that don't exist in this checkout -- see docs/aime_passk_diversity_design.md §4.4.
This is a weaker stand-in for the validated 3-way ensemble, not a replacement for it.

Pipeline per unique pool trajectory:
  1. extract "claims" (logical skeleton, wording stripped) via qwen2.5:1.5b (same default parser
     model as scripts/09_traj_diversity.py's graph method)
  2. embed each claim via nomic-embed-text (same embedding model as scripts/09/31)
Then per group: claim:dtw distance (scripts/23_logdist_seqot.py's d_dtw, vectorized) between
every pair of members' claim-embedding matrices, averaged -> claim_dtw_div.

Usage:
  python scripts/33_aime_claimdtw.py <run_dir>
"""
import argparse
import importlib.util
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import ollama

EXTRACT_SYS = """You extract the logical skeleton of a mathematical solution as a list of atomic claims.

Rules:
- One claim per list item: a short, self-contained mathematical statement the solution asserts and uses (definitions it introduces, case splits, derived equations/bounds, computed intermediate values, the final answer).
- Express claims in terms of the problem's original quantities. If the solution introduces auxiliary variables, describe them by their defining property, not their letter name (e.g. "the number of even-position beads" not "k").
- Canonical form: plain digits (no thousands separators), simplified expressions, no LaTeX decoration beyond what is needed.
- EXCLUDE: prose, motivation, restatements of the problem, verification chatter, dead ends that the solution abandons.
- Preserve the solution's actual logic -- do not correct errors or add missing steps.
Output JSON: {"claims": ["...", "..."]}"""

PARSER_MODEL = "qwen2.5:1.5b"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_claims(text):
    try:
        resp = ollama.chat(
            model=PARSER_MODEL,
            messages=[{"role": "system", "content": EXTRACT_SYS},
                      {"role": "user", "content": text[:3000]}],
            options={"temperature": 0, "num_predict": 500},
        )
        raw = resp["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        claims = json.loads(m.group()).get("claims", []) if m else []
        claims = [str(c) for c in claims]
        return claims if claims else [text[:150]]
    except Exception:
        return [text[:150]]


def d_dtw(X, Y):
    if not len(X) or not len(Y):
        return 1.0
    C = 1 - X @ Y.T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    s09 = _load(Path(__file__).parent / "09_traj_diversity.py", "s09")

    pool = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.run_dir) / "raw" / "pool.jsonl")}
    groups = [json.loads(l) for l in open(Path(args.run_dir) / "raw" / "groups_scored.jsonl")]

    print(f"pool trajectories: {len(pool)}  groups: {len(groups)}")
    print("extracting claims + embedding (one-time cost per unique trajectory)...")
    claim_embs = {}
    for i, (tid, rec) in enumerate(pool.items()):
        claims = extract_claims(rec["generation"])
        vecs = s09.embed_texts(claims)
        claim_embs[tid] = np.array(vecs)
        claim_embs[tid] /= np.linalg.norm(claim_embs[tid], axis=1, keepdims=True) + 1e-9
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pool)}", flush=True)

    pair_cache = {}

    def dtw_pair(a, b):
        key = tuple(sorted((a, b)))
        if key not in pair_cache:
            pair_cache[key] = d_dtw(claim_embs[a], claim_embs[b])
        return pair_cache[key]

    print("scoring groups...")
    out_path = Path(args.run_dir) / "raw" / "groups_scored_claimdtw.jsonl"
    with open(out_path, "x") as f:
        for i, g in enumerate(groups):
            pairs = list(combinations(g["member_ids"], 2))
            scores = [dtw_pair(a, b) for a, b in pairs]
            g_out = dict(g)
            g_out["claim_dtw_div"] = sum(scores) / len(scores)
            f.write(json.dumps(g_out) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(groups)}", flush=True)

    print(f"\nSaved: {out_path}")
    print(f"unique pairs scored: {len(pair_cache)}")


if __name__ == "__main__":
    main()
