"""Difficulty/token probe for Qwen3.6-35B-A3B (MoE, ~3B active) before committing to a
stronger-model evolution run. Checks headroom on BeyondAIME (hard test) + AIME/AIME25
(train / same-dist held-out). Delete after use."""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("evo", "scripts/09_espl_axis_evo.py")
evo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo)

from tinker_cookbook import model_info
from deeppersona import espl_axis as ax
from deeppersona.data_espl import load_raw

MODEL = "Qwen/Qwen3.6-35B-A3B"
RENDERER = model_info.get_recommended_renderer_name(MODEL)
print(f"[probe] model={MODEL} recommended_renderer={RENDERER}", flush=True)

MAXTOK, NP, NS = 30000, 8, 2
SETS = ["aime", "AIME25", "BeyondAIME"]
sc, renderer, tok = evo.make_sampler(MODEL, RENDERER)
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
    print(f"[{name}] pass@1~{c}/{n}={c/n:.2f}  any-correct {sum(1 for r in rs if sum(r['correct'])>0)}/{len(rs)}  "
          f"med_len={int(np.median(lens))} max_len={max(lens)} trunc={trunc}/{n}", flush=True)
