"""Throwaway probe: real thinking-token usage + untruncated accuracy for Qwen3-8B
on aime_and_amc, root (empty) prompt. Sets the full-run max_new_tokens budget and
confirms there is fitness signal (floor-effect check). Delete after use."""
import importlib.util
import sys

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("evo", "scripts/09_espl_axis_evo.py")
evo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo)

from deeppersona import espl_axis as ax
from deeppersona.data_espl import load_espl_items

MAXTOK = 30000
tr, _ = load_espl_items("aime_and_amc", n_test=40, seed=0)
items = tr[:6]
sc, renderer, tok = evo.make_sampler("Qwen/Qwen3-8B", "qwen3")
reqs = [{"item_idx": it["idx"], "system": ax.render_system_prompt({}), "problem": it["problem"],
         "gold": it["gold_answer"], "n_samples": 2, "temperature": 0.7} for it in items]
evo.generate(sc, renderer, tok, reqs, 0.95, MAXTOK, 0, 64)
evo.grade_reqs(reqs)

mx = 0
for r in reqs:
    mx = max(mx, max(r["gen_lens"]))
    trunc = [gl >= MAXTOK for gl in r["gen_lens"]]
    print(f"item {r['item_idx']}: lens={r['gen_lens']} trunc={trunc} correct={r['correct']} "
          f"pred={r['preds']} gold={r['gold']}", flush=True)
allc = sum(sum(r["correct"]) for r in reqs)
alln = sum(len(r["gens"]) for r in reqs)
print(f"\n[probe] untruncated-ish accuracy: {allc}/{alln}  max_gen_len={mx} (cap={MAXTOK})", flush=True)
