"""Cheap stronger-model probe: do the 8B-evolved champion prompts (general competition-math
strategy) help Qwen3.6-35B-A3B vs root on held-out sets? Leading indicator before any full
35B evolution. AIME25 = clean low-truncation held-out; BeyondAIME = hard (truncation-limited
at 30k for this model). Paired-bootstrap CIs vs root. Delete after use."""
import importlib.util
import json
import sys

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("evo", "scripts/09_espl_axis_evo.py")
evo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo)

from deeppersona import espl_axis as ax
from deeppersona.data_espl import load_raw

MODEL, RENDERER = "Qwen/Qwen3.6-35B-A3B", "qwen3_5"
RUN = "runs/espl-axis-20260617-212056Z-aime"
# (set, n_items, group, max_tokens)
PLAN = [("AIME25", 30, 8, 30000), ("BeyondAIME", 30, 6, 30000)]

ax.configure_axes("persona5")
fe = json.load(open(f"{RUN}/final_eval.json"))
champs = {c["program_id"]: c["principles"] for c in fe["champions"]}
sc, renderer, tok = evo.make_sampler(MODEL, RENDERER)
print(f"[transfer] model={MODEL} renderer={RENDERER} champions={list(champs)}", flush=True)


class Args:
    top_p, seed, concurrency, temp = 0.95, 0, 64, 0.7


for SET, NI, GROUP, MAXTOK in PLAN:
    data = load_raw(SET)[:NI]
    items = [{"idx": i, "problem": d["problem"], "gold_answer": d["groundtruth"]} for i, d in enumerate(data)]
    args = Args()
    args.max_new_tokens, args.test_group_size = MAXTOK, GROUP
    print(f"\n========== {SET}: {len(items)} problems x group {GROUP}, max_tokens {MAXTOK} ==========", flush=True)
    root_eval, _ = evo.eval_system(sc, renderer, tok, ax.render_system_prompt({}), items, GROUP, args, f"{SET}-root")
    print(f"[{SET}] ROOT pass@1={root_eval['pass1']:.3f} pass@{GROUP}={root_eval['passk'][GROUP]:.3f}", flush=True)
    for pid, pr in champs.items():
        ev, _ = evo.eval_system(sc, renderer, tok, ax.render_system_prompt(pr), items, GROUP, args, f"{SET}-id{pid}")
        ci, sig = evo.passk_ci_vs_root(ev, root_eval, GROUP, 0)
        print(f"[{SET}] id{pid} ({len(pr)}p) pass@1={ev['pass1']:.3f} pass@{GROUP}={ev['passk'][GROUP]:.3f} "
              f"| d pass@1={ev['pass1']-root_eval['pass1']:+.3f} d pass@{GROUP}={ev['passk'][GROUP]-root_eval['passk'][GROUP]:+.3f} "
              f"| first_sig_k={sig} | CI k1={ci[1]} k{GROUP}={ci[GROUP]}", flush=True)
