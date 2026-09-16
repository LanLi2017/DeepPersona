"""
AIME 2024 multi-model rollout generation via AWS Bedrock, with oversample-until-mixed
sampling: draw rollouts one at a time per (model, item) until both >=1 correct and >=1
incorrect are seen, capped at --n-max. Items still degenerate at the cap are dropped.

See docs/aime_passk_diversity_design.md for the full design (Step 1 of 5).

Output JSONL uses `condition` = model key so it's directly compatible with
scripts/09_traj_diversity.py's per-(condition, item_idx) grouping.

Usage:
  python scripts/28_aime_multimodel_rollout.py --smoke                  # 3 items x 1 model, n_max=4
  python scripts/28_aime_multimodel_rollout.py --models qwen32b --yes   # single model, no prompt
  python scripts/28_aime_multimodel_rollout.py --yes                   # all 5 models, full run
"""
import argparse
import datetime
import json
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

REGION = "us-east-2"

MODELS = {
    "qwen32b": "qwen.qwen3-32b-v1:0",                     # ON_DEMAND -- bare model ID works
    "gpt56sol": "us.openai.gpt-5.6-sol",                  # INFERENCE_PROFILE-only -- needs the us. prefix
    "gpt56terra": "us.openai.gpt-5.6-terra",              # unconfirmed profile id, same pattern assumed
    "claudeopus5": "us.anthropic.claude-opus-5",          # INFERENCE_PROFILE-only
    "claudesonnet5": "us.anthropic.claude-sonnet-5",      # INFERENCE_PROFILE-only
    "llama70b": "meta.llama3-3-70b-instruct-v1:0",  # access-check fallback, not in the design's model roster
    "mistrallarge3": "mistral.mistral-large-3-675b-instruct",  # ON_DEMAND, 64000 max_tokens OK, confirmed 2026-09-02
    "kimik2thinking": "moonshot.kimi-k2-thinking",  # ON_DEMAND, 64000 max_tokens OK; also in us-west-2 (scripts/22 legacy)
    "haiku45": "us.anthropic.claude-haiku-4-5-20251001-v1:0",  # INFERENCE_PROFILE, 64000 OK, no verified pricing (Anthropic absent from Pricing API here)
}
# NOTE (2026-09-02): gpt56sol/gpt56terra/claudeopus5/claudesonnet5 all still return
# AccessDeniedException via EITHER calling convention (bare ON_DEMAND id or the us.* inference
# profile) -- confirmed the model IDs are valid (get_foundation_model recognizes them) and that the
# profile-based calling convention is correct, but this AWS account has no model-access grant for
# any of the four. Needs to be requested in the Bedrock console (Model access) before these work.

# Per-model output-token ceilings confirmed empirically 2026-09-02 (ValidationException probing) --
# these are well below the 64k the design doc assumed from the Kimi K2 precedent. Requesting above
# these fails the whole call with ValidationException before any generation/cost happens.
MODEL_MAX_TOKENS = {
    "qwen32b": 16384,       # 32768 advertised but errors; 16384 confirmed OK
    "llama70b": 8192,
}

PROMPT_TMPL = """{persona}Solve the following AIME problem. The answer is a non-negative integer at most 999.
Show your complete reasoning, then state your final answer as \\boxed{{answer}}.

Problem: {problem}"""

# Verbatim from scripts/22_aime_bedrock.py -- Step 1 sampling-strategy comparison (2026-09-08):
# does varying the GENERATION prompt (not just temperature) produce more genuinely mixed groups,
# vs the plain single-prompt-repeated-at-temp-1.0 approach used so far?
AXIS_PERSONAS = [
    "You are a university algebra professor who solves problems through symbolic manipulation and equation systems.",
    "You are a competition geometry coach who visualizes problems spatially and applies geometric constructions.",
    "You are a combinatorics expert who systematically counts using bijections and generating functions.",
    "You are a number theory specialist who identifies modular arithmetic patterns and divisibility structures.",
    "You are a Putnam competition winner who tries multiple approaches and seeks elegant shortcuts.",
    "You are a math olympiad coach who first estimates the answer intuitively, then verifies rigorously.",
]

MATCHED_PERSONA = (
    "You are an expert AIME competition coach who has trained hundreds of high school students. "
    "You systematically identify the key mathematical concepts, carefully set up equations or inequalities, "
    "and solve step by step showing all work clearly."
)

PERSONA_CONDITIONS = ["vanilla", "axis", "matched"]


def make_prompt(problem, persona_cond, slot):
    if persona_cond == "vanilla":
        persona = ""
    elif persona_cond == "axis":
        persona = AXIS_PERSONAS[slot % len(AXIS_PERSONAS)] + "\n\n"
    else:
        persona = MATCHED_PERSONA + "\n\n"
    return PROMPT_TMPL.format(persona=persona, problem=problem)


# Sampling-method comparison (2026-09-11): persona-prompting pre-imposes an identity on the model
# before it even sees the problem, which may just constrain it rather than diversify it (esp. an
# axis persona mismatched to the problem's actual domain). These three alternatives try to induce
# genuine method diversity without fixing the model into an external frame up front.
SAMPLING_METHODS = ["plain", "randtemp", "selfdiverse", "branch"]

SELFDIVERSE_INSTRUCTION = (
    "Solve the problem again using a genuinely different method from the one above -- different "
    "key idea, not just different wording. Do not reuse the same formulas, case structure, or "
    "overall approach. Show your complete reasoning, then state your final answer as \\boxed{answer}."
)


def stitch_continuation(prefix, response_text):
    """Bedrock Converse's assistant-prefill continuation behavior is provider-specific: Anthropic
    models continue purely from the cut point (response excludes the prefix); Qwen echoes the
    given prefix back verbatim before continuing. Detect which happened rather than hardcoding
    per-model, so this doesn't silently break if the vendor changes behavior."""
    check_len = min(40, len(prefix))
    if check_len and prefix[-check_len:] in response_text[:check_len + 40]:
        return response_text  # echoed the prefix -- response is already the full text
    return prefix + response_text


def branch_point(text):
    """Cut a prior solution at a random point in its middle third, snapped to a sentence
    boundary, so the resampled continuation diverges partway through rather than at the very
    start (~persona-like) or the very end (~no real divergence)."""
    lo, hi = int(len(text) * 0.3), int(len(text) * 0.7)
    if hi <= lo:
        return text
    cut = random.randint(lo, hi)
    boundary = text.rfind(". ", lo, cut + 1)
    return text[:boundary + 1] if boundary != -1 else text[:cut]


def build_messages(persona_cond, slot, sampling_method, prob, prior_recs):
    """Returns the Converse `messages` list for this draw."""
    base_prompt = make_prompt(prob["problem"], persona_cond, slot)
    if sampling_method in ("plain", "randtemp") or slot == 0 or not prior_recs:
        return [{"role": "user", "content": [{"text": base_prompt}]}]
    if sampling_method == "selfdiverse":
        return [
            {"role": "user", "content": [{"text": base_prompt}]},
            {"role": "assistant", "content": [{"text": prior_recs[-1]["generation"]}]},
            {"role": "user", "content": [{"text": SELFDIVERSE_INSTRUCTION}]},
        ]
    if sampling_method == "branch":
        prefix = branch_point(prior_recs[-1]["generation"])
        return [
            {"role": "user", "content": [{"text": base_prompt}]},
            {"role": "assistant", "content": [{"text": prefix}]},
        ]
    raise ValueError(sampling_method)


def load_aime2024():
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    rows = [r for r in ds if "2024" in r["url"]]
    return [{"item_idx": i, "problem": r["problem"], "answer": int(r["answer"])}
            for i, r in enumerate(rows)]


def extract_answer(text):
    m = re.search(r"\\boxed\{(\d+)\}", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:answer is|answer:|final answer[:\s]+)\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b(\d{1,3})\b", text)
    return int(nums[-1]) if nums else None


def call_bedrock(client, model_id, messages, max_tokens, temperature=1.0, retries=4):
    for attempt in range(retries):
        try:
            resp = client.converse_stream(
                modelId=model_id,
                messages=messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            break
        except (client.exceptions.ThrottlingException,
                client.exceptions.InternalServerException,
                client.exceptions.ModelTimeoutException,
                client.exceptions.ValidationException) as e:
            # ValidationException is usually permanent (bad request), but observed 2026-09-11 to
            # be intermittent for Claude assistant-prefill continuations (the "branch" sampling
            # method) -- retrying the identical request succeeded on manual reproduction.
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)
    text_buf, think_buf = "", ""
    in_tok = out_tok = 0
    for event in resp["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "reasoningContent" in delta:
                think_buf += delta["reasoningContent"].get("text", "")
            elif "text" in delta:
                text_buf += delta["text"]
        elif "metadata" in event:
            u = event["metadata"].get("usage", {})
            in_tok = u.get("inputTokens", 0)
            out_tok = u.get("outputTokens", 0)
    parts = []
    if think_buf:
        parts.append(f"<think>\n{think_buf}\n</think>")
    if text_buf:
        parts.append(text_buf)
    return "\n\n".join(parts), in_tok, out_tok


def run_item(client, model_key, model_id, persona_cond, temperature, sampling_method,
            prob, n_max, max_tokens, f, write_lock):
    """Draw rollouts one at a time until both a correct and incorrect are seen, capped at n_max.

    Each draw is written to disk immediately (not buffered until the item finishes) -- a paid-for
    API call that gets killed mid-item (background timeout, process kill) must not also lose its
    already-received result.
    """
    recs = []
    seen_correct = seen_incorrect = False
    full_condition = f"{model_key}_{persona_cond}_t{temperature}_{sampling_method}"
    while len(recs) < n_max and not (seen_correct and seen_incorrect):
        slot = len(recs)
        messages = build_messages(persona_cond, slot, sampling_method, prob, recs)
        draw_temp = random.uniform(0.6, 1.0) if sampling_method == "randtemp" else temperature
        try:
            response_text, in_tok, out_tok = call_bedrock(client, model_id, messages, max_tokens, draw_temp)
        except Exception as e:
            return recs, False, str(e)
        if sampling_method == "branch" and slot > 0:
            prefix = messages[-1]["content"][0]["text"]
            generation = stitch_continuation(prefix, response_text)
        else:
            generation = response_text
        pred = extract_answer(generation)
        correct = pred is not None and pred == prob["answer"]
        seen_correct |= correct
        seen_incorrect |= not correct
        rec = {
            "item_idx": prob["item_idx"], "condition": full_condition, "model": model_key,
            "persona_cond": persona_cond, "sample_slot": slot, "sampling_method": sampling_method,
            "temperature": draw_temp, "gold": prob["answer"], "pred": pred, "correct": correct,
            "gen_len": out_tok, "in_tok": in_tok, "generation": generation,
        }
        recs.append(rec)
        with write_lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
    mixed = seen_correct and seen_incorrect
    return recs, mixed, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--personas", nargs="+", default=["vanilla"], choices=PERSONA_CONDITIONS)
    ap.add_argument("--temperatures", nargs="+", type=float, default=[1.0],
                    help="Bedrock Converse caps this at [0,1] for all providers tested so far")
    ap.add_argument("--sampling-methods", nargs="+", default=["plain"], choices=SAMPLING_METHODS)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.n_items, args.n_max, args.models = 3, 4, args.models[:1]

    problems = load_aime2024()[:args.n_items]
    max_calls = (len(problems) * len(args.models) * len(args.personas)
                * len(args.temperatures) * len(args.sampling_methods) * args.n_max)
    eff_tokens = {mk: min(args.max_tokens, MODEL_MAX_TOKENS.get(mk, args.max_tokens)) for mk in args.models}
    print(f"items={len(problems)}  models={args.models}  personas={args.personas}  "
          f"temperatures={args.temperatures}  sampling_methods={args.sampling_methods}  n_max={args.n_max}")
    print(f"worst-case calls={max_calls} (stops early per-item once pass+fail both seen)")
    print(f"effective max_tokens per model (clamped to Bedrock's per-model ceiling): {eff_tokens}")
    print(f"UNKNOWN per-model $/call for gpt-5.6-*/claude-opus-5 -- "
          f"run --smoke first and read actual token usage before a full run.")

    if not args.yes and not args.smoke:
        ans = input("Proceed? [y/N] ")
        if ans.lower() != "y":
            return

    tag = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%SZ") + f"-pid{os.getpid()}"
    outdir = Path(args.out) if args.out else \
        Path("runs") / f"aime24-multimodel-rollout-{tag}"
    # exist_ok=False: two concurrent runs racing on the same second-granularity tag must fail
    # loudly, not silently truncate each other's output (this destroyed real API spend once already).
    (outdir / "raw").mkdir(parents=True, exist_ok=False)

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    manifest = {
        "models": {k: MODELS[k] for k in args.models}, "personas": args.personas,
        "temperatures": args.temperatures, "sampling_methods": args.sampling_methods, "region": REGION,
        "n_items": len(problems), "n_max": args.n_max, "max_tokens": args.max_tokens,
        "git_sha": git_sha, "timestamp": tag,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(read_timeout=300, connect_timeout=10))
    out_path = outdir / "raw" / "samples.jsonl"
    dropped = []
    errors = []
    per_cond = {f"{mk}_{pc}_t{t}_{sm}": {"calls": 0, "in": 0, "out": 0}
                for mk in args.models for pc in args.personas for t in args.temperatures
                for sm in args.sampling_methods}
    total_in = total_out = total_calls = 0

    jobs = [(mk, pc, t, sm, prob) for mk in args.models for pc in args.personas
            for t in args.temperatures for sm in args.sampling_methods for prob in problems]
    write_lock = threading.Lock()
    with open(out_path, "x") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_item, client, mk, MODELS[mk], pc, t, sm, prob, args.n_max,
                            min(args.max_tokens, MODEL_MAX_TOKENS.get(mk, args.max_tokens)),
                            f, write_lock): (mk, pc, t, sm, prob)
                for mk, pc, t, sm, prob in jobs}
        for fut in as_completed(futs):
            mk, pc, t, sm, prob = futs[fut]
            cond = f"{mk}_{pc}_t{t}_{sm}"
            recs, mixed, err = fut.result()
            total_calls += len(recs)
            total_in += sum(r["in_tok"] for r in recs)
            total_out += sum(r["gen_len"] for r in recs)
            per_cond[cond]["calls"] += len(recs)
            per_cond[cond]["in"] += sum(r["in_tok"] for r in recs)
            per_cond[cond]["out"] += sum(r["gen_len"] for r in recs)
            if err:
                errors.append((cond, prob["item_idx"], err))
                print(f"  [{cond}] item={prob['item_idx']:2d} ERROR after {len(recs)} draws: {err[:120]}")
            elif not mixed:
                dropped.append((cond, prob["item_idx"]))
                print(f"  [{cond}] item={prob['item_idx']:2d} DROPPED (degenerate after {len(recs)} draws)")
            else:
                n_corr = sum(r["correct"] for r in recs)
                print(f"  [{cond}] item={prob['item_idx']:2d} mixed after {len(recs)} draws "
                      f"({n_corr} correct)")

    print(f"\nSaved: {out_path}")
    print(f"Dropped (degenerate): {len(dropped)}/{len(jobs)} -- {dropped}")
    if errors:
        print(f"Errors: {len(errors)}/{len(jobs)} (model likely not enabled on this account) -- "
              f"{sorted(set(cond for cond, _, _ in errors))}")
    print(f"Calls: {total_calls}  Tokens: in={total_in:,} out={total_out:,}")
    print("\nPer-condition token usage:")
    for cond, d in per_cond.items():
        print(f"  {cond:<25} calls={d['calls']:3d}  in={d['in']:>8,}  out={d['out']:>8,}")
    print(f"\nNOTE: no verified $/token rate for these models -- check AWS Bedrock console/billing "
          f"for actual cost before scaling up n_items or n_max.")


if __name__ == "__main__":
    main()
