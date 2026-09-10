#!/usr/bin/env python3
"""N4 — ENSinreg logical-diversity as a dense GRPO reward, in the N-regime target regime
(Qwen3-8B non-thinking, AIME+AMC, 2048 tok). Forks scripts/10 (Tinker GRPO core); the
lever moves from rollout INPUTS (F1) to the REWARD.

Per-rollout reward, gated by correctness (N1-dry/N-regime: ungated diversity anticorrelates
with correctness — an ungated bonus would pay for being lost):

  r_i = correct_i + lam * correct_i * LOO_i,  LOO_i = V(C) - V(C \\ i)

where C = the question's CORRECT rollouts, V = Vendi over pairwise ENSinreg distances
(l15_inreg + l4_inreg heads from N3b, scaled by judge-diff-pair sd, no mean-centering;
tau likewise from N3b). "Solve it, and solve it a different way." lam=0 -> vanilla GRPO.

Rollouts are encoded with the BASE model locally (heads were trained on base features;
LoRA drift at lr 1e-5 / 20 steps is assumed small — flagged approximation).

  set -a; . ./.tinker_env; set +a
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/30_n4_divgrpo.py --lam 0.5 --smoke
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/30_n4_divgrpo.py --lam 0    # baseline arm
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/30_n4_divgrpo.py --lam 0.5  # diversity arm
"""
from __future__ import annotations

import argparse, importlib.util, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deeppersona.data_espl import load_espl_items
from deeppersona.diversity_metrics import answer_entropy, distinct_n
from deeppersona.manifest import write_manifest
from deeppersona.config import RunConfig
from deeppersona.passk import pass_at_k, pass_at_k_for_range


def _load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s6 = _load("scripts/23_logdist_seqot.py", "s6")
s10 = _load("scripts/10_diversity_grpo.py", "s10")
s9 = _load("scripts/29_n3b_ingame.py", "s9")

RP = Path("runs/regime-probe")
CHUNK = 128
NEUTRAL = s10.NEUTRAL


# ── ENSinreg encoder (base model, local GPU) ────────────────────────────────
class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).cuda().eval()
        assert next(self.model.parameters()).is_cuda
        self.W15 = torch.nn.Linear(4096, 256, bias=False)
        self.W15.load_state_dict(torch.load(RP / "head_l15_inreg.pt", map_location="cpu"))
        self.W4 = torch.nn.Linear(4096, 256, bias=False)
        self.W4.load_state_dict(torch.load(RP / "head_l4_inreg.pt", map_location="cpu"))

    def embed(self, texts, bs=8):
        # -> per text: (l15 vec [256], l4 chunk matrix [n,256])
        torch, out = self.torch, []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=4096,
                             return_tensors="pt").to("cuda")
                hs = self.model(**b, output_hidden_states=True).hidden_states[18]
                for r in range(hs.shape[0]):
                    v = hs[r][b["attention_mask"][r].bool()]
                    tr = torch.nn.functional.normalize(v.mean(0), dim=-1).float().cpu()
                    cs = torch.stack([v[j:j + CHUNK].mean(0) for j in range(0, v.shape[0], CHUNK)])
                    cs = torch.nn.functional.normalize(cs, dim=-1).float().cpu()
                    f15 = torch.nn.functional.normalize(self.W15(tr), dim=-1).numpy()
                    fc = torch.nn.functional.normalize(self.W4(cs), dim=-1).numpy()
                    out.append((f15, fc))
        return out


def n3b_scales():
    # sd of each component + tau of the ensemble over N3b judge diff-method pairs
    ids, _, _ = s9.ids_texts()
    idx = {s: i for i, s in enumerate(ids)}
    F15 = np.load(RP / "emb_l15_inreg.npz")["l15"]
    z = np.load(RP / "chunk_l4_inreg_head.npz")
    FC, coff = z["E"], {k: tuple(v) for k, v in json.loads(str(z["offsets"])).items()}
    diff, _ = s9.judge_pairs()
    d15 = [float(1 - F15[idx[a]] @ F15[idx[b]]) for a, b in diff]
    d4 = [s6.d_chamfer(FC[slice(*coff[a])], FC[slice(*coff[b])]) for a, b in diff]
    sd15, sd4 = float(np.std(d15)), float(np.std(d4))
    tau = float(np.mean([0.5 * (a / sd15 + b / sd4) for a, b in zip(d15, d4)]))
    return sd15, sd4, tau


def ens_D(embs, sd15, sd4):
    n = len(embs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d15 = float(1 - embs[i][0] @ embs[j][0])
            d4 = s6.d_chamfer(embs[i][1], embs[j][1])
            D[i, j] = D[j, i] = 0.5 * (d15 / sd15 + d4 / sd4)
    return D


def loo_credits(embs, correct, sd15, sd4, tau):
    # gated LOO Vendi among correct rollouts; wrong rollouts (and <2 correct) get 0
    n = len(embs)
    cred = [0.0] * n
    ci = [i for i in range(n) if correct[i]]
    if len(ci) < 2:
        return cred, 1.0
    D = ens_D([embs[i] for i in ci], sd15, sd4)
    v_full = s6.vendi(D, tau)
    for pos, i in enumerate(ci):
        keep = [p for p in range(len(ci)) if p != pos]
        v_wo = s6.vendi(D[np.ix_(keep, keep)], tau) if len(keep) >= 2 else 1.0
        cred[i] = max(0.0, v_full - v_wo)
    return cred, v_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default="qwen3_disable_thinking")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--group-total", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--rl-loss", default="importance_sampling")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--batch-questions", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--test-group-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    G = args.group_total
    if args.smoke:
        args.n_steps, args.batch_questions, args.n_test = 1, 2, 4
        G = args.group_total = 4
        args.max_tokens = 512
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    enc = Encoder()
    sd15, sd4, tau = n3b_scales()
    print(f"[setup] lam={args.lam} G={G} sd15={sd15:.4f} sd4={sd4:.4f} tau={tau:.3f}", flush=True)

    tok = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tok)
    svc = tinker.ServiceClient()
    training_client = svc.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
    adam = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temp,
                               top_p=args.top_p, stop=renderer.get_stop_sequences())

    train_items, test_items = load_espl_items("aime_and_amc", n_test=40, seed=0)
    for it in train_items + test_items:
        it["gold"] = it.pop("gold_answer")
    if args.smoke:
        train_items, test_items = train_items[:4], test_items[:args.n_test]
    else:
        test_items = test_items[:args.n_test]
    print(f"[data] aime_and_amc train={len(train_items)} test={len(test_items)}", flush=True)

    cond = f"lam{args.lam}" + (f"-{args.tag}" if args.tag else "")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = REPO_ROOT / "runs" / (f"n4grpo-{ts}-{cond}-s{args.seed}" + ("-smoke" if args.smoke else ""))
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, RunConfig(model_id=args.model, model_revision="tinker", task="math",
                                      condition=cond, seed=args.seed, split="train",
                                      n_items=len(train_items), run_name="n4grpo", smoke=args.smoke),
                   REPO_ROOT)
    (run_dir / "args.json").write_text(json.dumps({**vars(args), "sd15": sd15, "sd4": sd4,
                                                   "tau": tau}, indent=2))
    metrics_f = (run_dir / "metrics.jsonl").open("w")
    print(f"[run] {run_dir}", flush=True)

    def eval_neutral(sampling_client, step):
        cells = [(0, q, NEUTRAL, args.test_group_size, it) for q, it in enumerate(test_items)]
        data = s10.sample_cells(sampling_client, renderer, sp, cells)
        passk, vfull_all, vfull_corr, raw = [], [], [], []
        for q, it in enumerate(test_items):
            rec = data[(0, q)]
            c = int(sum(rec["rewards"]))
            passk.append(pass_at_k_for_range(args.test_group_size, c))
            embs = enc.embed(rec["comps"])
            D = ens_D(embs, sd15, sd4)
            vfull_all.append(s6.vendi(D, tau))
            _, vc = loo_credits(embs, rec["rewards"], sd15, sd4, tau)
            vfull_corr.append(vc)
            for slot, (g, pred, ok) in enumerate(zip(rec["comps"], rec["preds"], rec["rewards"])):
                raw.append({"step": step, "item_idx": it["idx"], "slot": slot, "gold": it["gold"],
                            "pred": pred, "correct": int(ok), "generation": g,
                            "problem": it["problem"]})
        with (run_dir / "raw" / f"eval_step{step:04d}.jsonl").open("w") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")
        pk = np.mean(passk, axis=0).tolist()
        return {"pass@1": pk[0], "pass@k": pk, "V_ens": float(np.mean(vfull_all)),
                "V_ens_correct": float(np.mean(vfull_corr))}

    order = list(range(len(train_items)))
    random.Random(args.seed).shuffle(order)
    B = args.batch_questions
    for step in range(args.n_steps):
        t0 = time.time()
        spath = training_client.save_weights_for_sampler(
            name=f"{step:06d}", ttl_seconds=7 * 24 * 3600).result().path
        sampling_client = svc.create_sampling_client(model_path=spath)

        eval_stats = None
        if step == 0 or step % args.eval_every == 0 or step == args.n_steps - 1:
            eval_stats = eval_neutral(sampling_client, step)
            print(f"[eval s{step}] pass@1={eval_stats['pass@1']:.3f} "
                  f"pass@{args.test_group_size}={eval_stats['pass@k'][-1]:.3f} "
                  f"V_ens={eval_stats['V_ens']:.3f} V_corr={eval_stats['V_ens_correct']:.3f}",
                  flush=True)

        start = (step * B) % len(order)
        batch = [train_items[order[(start + i) % len(order)]] for i in range(B)]
        cells = [(0, q, NEUTRAL, G, it) for q, it in enumerate(batch)]
        data = s10.sample_cells(sampling_client, renderer, sp, cells)

        # composite reward: correctness + gated LOO diversity credit
        rewards, v_packs, credits_log = [], [], []
        for q in range(B):
            rec = data[(0, q)]
            embs = enc.embed(rec["comps"])
            cred, v_full = loo_credits(embs, rec["rewards"], sd15, sd4, tau)
            r = [c + args.lam * c * d for c, d in zip(rec["rewards"], cred)]
            rewards.append(r)
            v_packs.append(v_full)
            credits_log.append(cred)
        rew = torch.tensor(rewards).unsqueeze(0)  # [1, B, G]
        adv = (rew - rew.mean(dim=-1, keepdim=True)).squeeze(0).tolist()

        datums, skipped = [], 0
        for q in range(B):
            a = adv[q]
            if all(abs(x) < 1e-9 for x in a):
                skipped += 1
                continue
            rec = data[(0, q)]
            for tokens, ob, lp, av in zip(rec["tokens"], rec["ob_lens"], rec["logprobs"], a):
                inp = [int(t) for t in tokens[:-1]]
                tgt = tokens[1:]
                all_lp = [0.0] * ob + lp
                all_av = [0.0] * ob + [av] * (len(inp) - ob)
                assert len(inp) == len(tgt) == len(all_lp) == len(all_av)
                datums.append(types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=inp),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(tgt)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_lp)),
                        "advantages": TensorData.from_torch(torch.tensor(all_av)),
                    }))

        corr = [data[(0, q)]["rewards"] for q in range(B)]
        train_stats = {
            "avg_reward": float(rew.mean().item()),
            "pass@1": float(np.mean([np.mean(c) for c in corr])),
            f"pass@{G}_union": float(np.mean([pass_at_k(G, int(sum(c)), G) for c in corr])),
            "V_ens_correct": float(np.mean(v_packs)),
            "mean_credit": float(np.mean([c for cl in credits_log for c in cl])),
            "answer_entropy": float(np.mean([answer_entropy(data[(0, q)]["preds"]) for q in range(B)])),
            "distinct4": float(np.mean([distinct_n(data[(0, q)]["comps"], 4) for q in range(B)])),
            "n_skipped": skipped, "n_datums": len(datums),
        }
        with (run_dir / "raw" / f"train_step{step:04d}.jsonl").open("w") as f:
            for q in range(B):
                rec = data[(0, q)]
                for slot, (g, pred, ok) in enumerate(zip(rec["comps"], rec["preds"], rec["rewards"])):
                    f.write(json.dumps({"step": step, "item_idx": batch[q]["idx"], "slot": slot,
                                        "gold": batch[q]["gold"], "pred": pred, "correct": int(ok),
                                        "credit": credits_log[q][slot], "adv": adv[q][slot],
                                        "generation": g}) + "\n")

        did_update = False
        if datums:
            fbf = training_client.forward_backward(datums, loss_fn=args.rl_loss)
            osf = training_client.optim_step(adam)
            fbf.result(); osf.result()
            did_update = True

        rec_line = {"step": step, "lam": args.lam, "secs": round(time.time() - t0, 1),
                    "updated": did_update, "train": train_stats}
        if eval_stats is not None:
            rec_line["eval"] = eval_stats
        metrics_f.write(json.dumps(rec_line) + "\n"); metrics_f.flush()
        print(f"[step {step}] r={train_stats['avg_reward']:.3f} p@1={train_stats['pass@1']:.3f} "
              f"p@{G}={train_stats[f'pass@{G}_union']:.3f} V_corr={train_stats['V_ens_correct']:.3f} "
              f"credit={train_stats['mean_credit']:.3f} ansH={train_stats['answer_entropy']:.3f} "
              f"datums={len(datums)} skip={skipped} {rec_line['secs']}s", flush=True)

    metrics_f.close()
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
