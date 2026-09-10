#!/usr/bin/env python3
"""Persistent local inference queue; never executes generated actions or tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/swe-diversity-selection/swe-student-tools'
QUEUE = OUT / 'queue'
MODEL = Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
SEED = 20260909
MAX_REQUESTS = 48
MAX_INPUT_TOKENS = 36864
ENGINE = dict(dtype='float32', kv_cache_dtype='auto', tensor_parallel_size=2,
              max_model_len=40960, max_num_seqs=1, max_num_batched_tokens=1024,
              enable_chunked_prefill=True, enforce_eager=True,
              enable_prefix_caching=False, disable_custom_all_reduce=True,
              gpu_memory_utilization=.90)
GENERATION = dict(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=4096,
                  n=1, logprobs=1, seed=SEED)
NUMERICAL_SCOPE = ('FP32 model and KV storage. PyTorch TF32 flags and NVIDIA_TF32_OVERRIDE=0 are set, '
                   'but the installed Triton attention dot defaults may use TF32. No IEEE FP32 '
                   'arithmetic claim; TP2 can also change discrete outputs versus the earlier TP1 baseline.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def exclusive(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def run(protocol_path):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2,3', 'Only physical GPUs2,3 are allowed'
    assert os.environ.get('NVIDIA_TF32_OVERRIDE') == '0'
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    protocol = read(protocol_path)
    hashes = protocol['source_sha256']
    own_path = Path(__file__).resolve()
    assert any((ROOT / name).resolve() == own_path and digest == sha(own_path) for name, digest in hashes.items()), 'Root must freeze102 source before worker launch'
    for name, digest in hashes.items():
        assert sha(ROOT / name) == digest, name
    assert not (QUEUE / 'stop').exists(), 'Stop already requested'
    for name in ['requests', 'dispatched', 'responses']:
        (QUEUE / name).mkdir(parents=True, exist_ok=True)
    assert not list((QUEUE / 'dispatched').glob('*.json')), 'Existing dispatches require explicit root review, not worker restart'
    assert not list((QUEUE / 'responses').glob('*.json'))
    versions = {p: importlib.metadata.version(p) for p in ['vllm', 'torch', 'transformers', 'triton']}
    manifest = dict(utc=utc(), pid=os.getpid(), protocol_sha256=sha(protocol_path), worker_source_sha256=sha(own_path),
                    versions=versions, model=str(MODEL), physical_gpus=[2, 3], engine=ENGINE,
                    generation=GENERATION, seed=SEED, max_requests=MAX_REQUESTS,
                    numerical_scope=NUMERICAL_SCOPE, api_spend_usd=0)
    exclusive(QUEUE / 'worker_started.json', manifest)
    model_hashes = {p.name: sha(p) for p in MODEL.iterdir() if p.is_file()}
    for name, digest in protocol.get('model_sha256', {}).items():
        assert model_hashes[name] == digest, name
    save(QUEUE / 'model_manifest.json', model_hashes)
    import torch
    from vllm import LLM, SamplingParams
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    tick = time.monotonic()
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), seed=SEED, disable_log_stats=True, **ENGINE)
    config = llm.llm_engine.vllm_config
    assert config.model_config.dtype == torch.float32 and config.cache_config.cache_dtype == 'auto'
    assert config.parallel_config.tensor_parallel_size == 2
    save(QUEUE / 'ready.json', dict(**manifest, model_manifest_sha256=sha(QUEUE / 'model_manifest.json'),
                                  initialization_seconds=time.monotonic()-tick))
    print(json.dumps(dict(event='ready', pid=os.getpid())), flush=True)
    dispatched = 0
    while dispatched < MAX_REQUESTS and not (QUEUE / 'stop').exists():
        pending = [p for p in sorted((QUEUE / 'requests').glob('*.json')) if not (QUEUE / 'dispatched' / p.name).exists()]
        if not pending:
            time.sleep(.25)
            continue
        request_path = pending[0]
        request_bytes = request_path.read_bytes()
        request = json.loads(request_bytes)
        jobid = request['jobid']
        assert isinstance(jobid, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,160}', jobid) and request_path.name == jobid + '.json'
        tokens = request['prompt_token_ids']
        assert isinstance(tokens, list) and 0 < len(tokens) <= MAX_INPUT_TOKENS
        assert all(type(token) is int and 0 <= token < config.model_config.get_vocab_size() for token in tokens)
        assert type(request['max_tokens']) is int and request['max_tokens'] == GENERATION['max_tokens']
        response_path = QUEUE / 'responses' / request_path.name
        assert not response_path.exists()
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        exclusive(QUEUE / 'dispatched' / request_path.name,
                  dict(jobid=jobid, utc=utc(), request_sha256=request_sha256, pid=os.getpid(),
                       protocol_sha256=manifest['protocol_sha256'], input_tokens=len(tokens), max_tokens=4096))
        dispatched += 1
        started = time.monotonic()
        reply = llm.generate([{'prompt_token_ids': tokens}], SamplingParams(**GENERATION), use_tqdm=False)[0]
        assert len(reply.outputs) == 1
        seq = reply.outputs[0]
        ids = list(seq.token_ids)
        logprobs = [float(values[token].logprob) for token, values in zip(ids, seq.logprobs)]
        checks = dict(exact_prompt_tokens=list(reply.prompt_token_ids) == tokens,
                      chosen_logprobs_complete=len(logprobs) == len(ids),
                      chosen_logprobs_finite=all(math.isfinite(v) for v in logprobs),
                      token_budget=0 < len(ids) <= 4096,
                      request_unchanged=sha(request_path) == request_sha256)
        response = dict(jobid=jobid, utc=utc(), request_sha256=request_sha256,
                        protocol_sha256=manifest['protocol_sha256'], prompt_token_ids=list(reply.prompt_token_ids),
                        generated_token_ids=ids, chosen_token_logprobs=logprobs, text=seq.text, output_text=seq.text,
                        finish_reason=seq.finish_reason, stop_reason=seq.stop_reason,
                        response_status='completed' if seq.finish_reason == 'stop' else 'incomplete',
                        input_tokens=len(tokens), output_tokens=len(ids), truncated=seq.finish_reason == 'length',
                        elapsed_seconds=time.monotonic()-started, versions=versions,
                        technical_checks=checks, technical_checks_passed=all(checks.values()), numerical_scope=NUMERICAL_SCOPE)
        save(response_path, response)
        assert all(checks.values()), jobid
        print(json.dumps(dict(event='response', jobid=jobid, output_tokens=len(ids), finish_reason=seq.finish_reason)), flush=True)
    save(QUEUE / 'worker_stopped.json', dict(utc=utc(), dispatched=dispatched,
         reason='request_limit' if dispatched == MAX_REQUESTS else 'stop_file', protocol_sha256=manifest['protocol_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=OUT / 'protocol.json')
    args = parser.parse_args()
    try:
        run(args.protocol.resolve())
    except Exception as error:
        if QUEUE.exists() and not (QUEUE / 'worker_failure.json').exists():
            save(QUEUE / 'worker_failure.json', dict(utc=utc(), error_type=type(error).__name__, error=str(error),
                 policy='No automatic retry; any dispatch without a response remains unknown for root inspection.'))
        raise
