"""Disambiguation: do the scaled-run champions beat root on a HELD-OUT same-difficulty
AIME set (AIME25, not in training)? Distinguishes real in-distribution generalization
from TrueSkill selection-overfitting to the 90 training problems. Reuses the champions;
no new evolution. Delete after use."""
import importlib.util
import json
import sys

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("evo", "scripts/09_espl_axis_evo.py")
evo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo)

from deeppersona import espl_axis as ax
from deeppersona.data_espl import load_raw

RUN = "runs/espl-axis-20260617-212056Z-aime"
SET, GROUP, MAXTOK = "AIME25", 10, 28000

ax.configure_axes("persona5")
data = load_raw(SET)
items = [{"idx": i, "problem": d["problem"], "gold_answer": d["groundtruth"]} for i, d in enumerate(data)]
fe = json.load(open(f"{RUN}/final_eval.json"))
champs = {c["program_id"]: c["principles"] for c in fe["champions"]}


class Args:
    top_p, max_new_tokens, seed, concurrency, temp, test_group_size = 0.95, MAXTOK, 0, 64, 0.7, GROUP


args = Args()
sc, renderer, tok = evo.make_sampler("Qwen/Qwen3-8B", "qwen3")
print(f"[heldout] {SET}: {len(items)} problems x group {GROUP}", flush=True)

root_eval, _ = evo.eval_system(sc, renderer, tok, ax.render_system_prompt({}), items, GROUP, args, "heldout-root")
print(f"[heldout] ROOT pass@1={root_eval['pass1']:.3f} pass@{GROUP}={root_eval['passk'][GROUP]:.3f}", flush=True)
for pid, pr in champs.items():
    ev, _ = evo.eval_system(sc, renderer, tok, ax.render_system_prompt(pr), items, GROUP, args, f"heldout-id{pid}")
    ci, sig = evo.passk_ci_vs_root(ev, root_eval, GROUP, 0)
    print(f"[heldout] id{pid} ({len(pr)}p) pass@1={ev['pass1']:.3f} pass@{GROUP}={ev['passk'][GROUP]:.3f} "
          f"| vs root: pass@1 d={ev['pass1']-root_eval['pass1']:+.3f} pass@{GROUP} d={ev['passk'][GROUP]-root_eval['passk'][GROUP]:+.3f} "
          f"| first_sig_k={sig} | CI k1={ci[1]} k5={ci[5]} k{GROUP}={ci[GROUP]}", flush=True)
