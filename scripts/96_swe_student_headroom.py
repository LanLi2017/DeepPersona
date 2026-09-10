#!/usr/bin/env python3
"""Six fixed public-packet greedy FP32 attempts; no APIs or evaluator inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/swe-diversity-selection/swe-student-headroom'
MODEL = Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
SEED = 20260909
spec = importlib.util.spec_from_file_location('generation92', ROOT / 'scripts/92_swe_fresh_generation.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
COHORT = g.COHORT


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def utc(): return datetime.now(timezone.utc).isoformat()
def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prepare():
    from transformers import AutoTokenizer
    assert not OUT.exists()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    addition = ('Return a JSON object conforming exactly to this JSON Schema. Do not wrap it in Markdown fences. '
                'Escape newlines inside JSON strings.\n' + json.dumps(g.SCHEMA, sort_keys=True) + '\n\n')
    OUT.mkdir()
    (OUT / 'responses').mkdir()
    (OUT / 'prompts').mkdir()
    paths = [Path(__file__).resolve(), ROOT / 'scripts/92_swe_fresh_generation.py',
             COHORT / 'selection.json', COHORT / 'generation_context/manifest.json']
    jobs, inputs = [], []
    for row in read(COHORT / 'selection.json')['tasks']:
        task = row['instance_id']
        packet = COHORT / 'generation_context' / (task + '.txt')
        meta = COHORT / 'public' / task / 'task.json'
        paths += [packet, meta]
        for name, digest in read(meta)['source_sha256'].items():
            assert sha(meta.parent / 'source' / name) == digest
        messages = [{'role': 'system', 'content': g.SYSTEM}, {'role': 'user', 'content': addition + packet.read_bytes().decode('utf-8')}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        target = OUT / 'prompts' / (task + '.txt')
        target.write_text(prompt)
        paths.append(target)
        job = dict(key=task + '__0', instance_id=task, sample=0, packet=str(packet.relative_to(ROOT)), packet_sha256=sha(packet), input_tokens=len(ids))
        jobs.append(job)
        inputs.append(dict(**job, messages=messages, prompt=prompt, prompt_token_ids=ids))
    assert len(jobs) == 6
    save(OUT / 'jobs.json', jobs)
    save(OUT / 'inputs.json', inputs)
    paths += [OUT / 'jobs.json', OUT / 'inputs.json']
    config = read(MODEL / 'config.json')
    maximum = max(j['input_tokens'] for j in jobs) + 4096
    assert maximum <= config['max_position_embeddings']
    kv_bytes = 2 * config['num_hidden_layers'] * config['num_key_value_heads'] * config['head_dim'] * 4 * maximum
    model_files = [p for p in MODEL.iterdir() if p.is_file()]
    save(OUT / 'protocol.json', dict(utc=utc(), scope='Gold-exposed SWE development cohort; student headroom only, no learning or selection claim.',
        attempts=6, attempts_per_task=1, order=[j['instance_id'] for j in jobs], smoke_task=jobs[0]['instance_id'],
        model=str(MODEL), model_sha256={p.name: sha(p) for p in model_files}, seed=SEED,
        system=g.SYSTEM, explicit_schema_instruction=addition,
        format_difference='Same92SYSTEM and exact frozen packets; schema supplied in text, no API/grammar schema enforcement.',
        tokenizer_chat_template=tokenizer.chat_template, enable_thinking=False,
        engine=dict(max_model_len=maximum, max_num_seqs=1, max_num_batched_tokens=1024,
                    enable_chunked_prefill=True, gpu_memory_utilization=.94, enforce_eager=True,
                    enable_prefix_caching=False, dtype='float32', kv_cache_dtype='auto'),
        generation=dict(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=4096, n=1, logprobs=1, seed=SEED),
        precision='FP32 model and KV; TF32 off including NVIDIA_TF32_OVERRIDE=0 in workers; no LoRA, quantization or lower precision fallback.',
        native_max_position_embeddings=config['max_position_embeddings'], kv_cache_max_bytes=kv_bytes,
        memory_estimate=dict(weights_gib_from_prior_load=30.62, fp32_kv_gib=kv_bytes / 2**30, excludes_runtime_workspace=True),
        smoke_gate='Exactly one output, exact prompt IDs, finite chosen logprobs, no runtime error. Syntax/JSON/patch invalidity and truncation are retained outcomes, not reasons to retry.',
        no_retry_rule='First selected task is in-study smoke, never regenerated; five remaining slots once after technical gate. Preserve runtime failures without fallback.',
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        versions={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'vllm']},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), api_spend_usd=0))
    print(json.dumps(dict(protocol_sha256=sha(OUT / 'protocol.json'), input_tokens=[j['input_tokens'] for j in jobs], max_model_len=maximum, fp32_kv_gib=kv_bytes / 2**30)), flush=True)


def validate():
    p = read(OUT / 'protocol.json')
    for name, digest in p['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in p['model_sha256'].items():
        assert sha(MODEL / name) == digest, name
    return p


def run(stage):
    import torch
    from vllm import LLM, SamplingParams
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    assert os.environ.get('NVIDIA_TF32_OVERRIDE') == '0'
    p = validate()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    startfile = OUT / (stage + '_started.json')
    assert not startfile.exists(), 'No implicit retry'
    rows = read(OUT / 'inputs.json')
    if stage == 'smoke':
        rows = rows[:1]
    else:
        assert read(OUT / 'smoke_summary.json')['technical_gate_passed']
        rows = rows[1:]
    save(startfile, dict(utc=utc(), pid=os.getpid(), protocol_sha256=sha(OUT / 'protocol.json'),
         device=torch.cuda.get_device_name(0), versions={k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'vllm']},
         tf32_override=os.environ['NVIDIA_TF32_OVERRIDE'], physical_gpu=3))
    started = time.monotonic()
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), seed=SEED, disable_log_stats=True, **p['engine'])
    config = llm.llm_engine.vllm_config
    assert config.model_config.dtype == torch.float32
    assert config.cache_config.cache_dtype == 'auto'
    completed = []
    for row in rows:
        path = OUT / 'responses' / (row['key'] + '.json')
        assert not path.exists()
        tick = time.monotonic()
        save(OUT / (row['key'] + '_dispatched.json'), dict(utc=utc(), key=row['key']))
        reply = llm.generate([{'prompt_token_ids': row['prompt_token_ids']}], SamplingParams(**p['generation']), use_tqdm=False)[0]
        assert len(reply.outputs) == 1
        seq = reply.outputs[0]
        ids = list(seq.token_ids)
        lp = [float(values[token].logprob) for token, values in zip(ids, seq.logprobs)]
        record = dict(key=row['key'], instance_id=row['instance_id'], sample=0, output_text=seq.text,
                      response_status='completed' if seq.finish_reason == 'stop' else 'incomplete',
                      prompt_token_ids=reply.prompt_token_ids, generated_token_ids=ids, chosen_token_logprobs=lp,
                      finish_reason=seq.finish_reason, stop_reason=seq.stop_reason, elapsed_seconds=time.monotonic()-tick,
                      input_tokens=len(reply.prompt_token_ids), output_tokens=len(ids), truncated=seq.finish_reason == 'length')
        save(path, record)
        assert reply.prompt_token_ids == row['prompt_token_ids']
        assert len(lp) == len(ids) and ids and torch.isfinite(torch.tensor(lp)).all()
        completed.append(dict(key=row['key'], output_tokens=len(ids), finish_reason=seq.finish_reason, response_sha256=sha(path)))
        print(json.dumps(completed[-1]), flush=True)
    save(OUT / (stage + '_summary.json'), dict(technical_gate_passed=True, completed=completed,
         elapsed_seconds=time.monotonic()-started, protocol_sha256=sha(OUT / 'protocol.json'), api_spend_usd=0))


def patches():
    validate()
    assert all((OUT / 'responses' / (j['key'] + '.json')).exists() for j in read(OUT / 'jobs.json'))
    # Reuse the frozen92validator unchanged. Its patch-only call graph contains no API or ledger calls.
    g.OUT = OUT
    g.validate_sources = validate
    g.build_patches()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'smoke', 'remaining', 'patches'])
    args = parser.parse_args()
    try:
        if args.stage == 'prepare': prepare()
        elif args.stage == 'patches': patches()
        else: run(args.stage)
    except Exception as error:
        if OUT.exists():
            save(OUT / (args.stage + '_failure.json'), dict(utc=utc(), error_type=type(error).__name__, error=str(error), fallback_used=False))
        raise
