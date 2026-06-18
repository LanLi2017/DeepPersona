"""Throwaway probe: map Qwen3-8B-thinking pass@1 across E-SPL sets to find the
moderate-accuracy regime (headroom for prompt evolution). One batched generate over
all sets. Delete after use."""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("evo", "scripts/09_espl_axis_evo.py")
evo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo)

from deeppersona import espl_axis as ax
from deeppersona.data_espl import load_raw

MAXTOK = 24000
NP = 8          # problems per set
NS = 2          # samples per problem (coarse pass@1)
SETS = ["amc", "aime", "BeyondAIME", "hmmt_nov_2025"]

sc, renderer, tok = evo.make_sampler("Qwen/Qwen3-8B", "qwen3")
reqs = []
for name in SETS:
    for i, d in enumerate(load_raw(name)[:NP]):
        reqs.append({"set": name, "item_idx": i, "system": ax.render_system_prompt({}),
                     "problem": d["problem"], "gold": d["groundtruth"], "n_samples": NS, "temperature": 0.7})
evo.generate(sc, renderer, tok, reqs, 0.95, MAXTOK, 0, 64)
evo.grade_reqs(reqs)

for name in SETS:
    rs = [r for r in reqs if r["set"] == name]
    c = sum(sum(r["correct"]) for r in rs)
    n = sum(len(r["gens"]) for r in rs)
    lens = [gl for r in rs for gl in r["gen_lens"]]
    trunc = sum(gl >= MAXTOK for gl in lens)
    solved = sum(1 for r in rs if sum(r["correct"]) > 0)  # pass@2 per problem
    print(f"[{name}] pass@1~{c}/{n}={c/n:.2f}  any-correct {solved}/{len(rs)}  "
          f"med_len={int(np.median(lens))} max_len={max(lens)} trunc={trunc}/{n}", flush=True)
