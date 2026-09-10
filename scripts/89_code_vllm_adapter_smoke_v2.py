#!/usr/bin/env python3
"""Two previously used training probes; FP32 vLLM adapter-path smoke only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('runs/swe-diversity-selection')
OUT = ROOT / 'code-vllm-adapter-smoke-v2'
CONTROL = ROOT / 'code-corrected-sft-control'
MODEL = Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
SEED = 20260909


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def readlines(p):
    return [json.loads(line) for line in p.open()]


def prepare():
    import torch
    from safetensors.torch import load_file, save_file
    assert not OUT.exists()
    OUT.mkdir()
    source = CONTROL / 'headroom/base_raw.jsonl'
    rows = [r for r in readlines(source) if r['task_id'] in [784, 750]]
    assert [r['task_id'] for r in rows] == [784, 750]
    save(OUT / 'inputs.json', rows)
    initial = torch.load(CONTROL / 'smoke/initial_adapter.pt', map_location='cpu', weights_only=True)
    state = {k.replace('.default.', '.'): v.contiguous() for k, v in initial.items()}
    assert state.keys() == load_file(str(CONTROL / 'full/adapter/adapter_model.safetensors')).keys()
    assert all(torch.count_nonzero(v) == 0 for k, v in state.items() if '.lora_B.' in k)
    zero = OUT / 'zero_adapter'
    zero.mkdir()
    shutil.copyfile(CONTROL / 'full/adapter/adapter_config.json', zero / 'adapter_config.json')
    save_file(state, str(zero / 'adapter_model.safetensors'))
    sources = [Path(__file__), source, OUT / 'inputs.json', CONTROL / 'smoke/initial_adapter.pt']
    sources += list(zero.iterdir())
    sources += [CONTROL / 'full/adapter' / n for n in ['adapter_config.json', 'adapter_model.safetensors']]
    save(OUT / 'protocol.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), model_snapshot=str(MODEL), seed=SEED,
        amendment='Original86 stopped before model loading: installed LoRAConfig accepts torch.float32 dtype object but not float32 string. Only argument type changes; same numerical precision and gates.',
        tasks=[784, 750], scope='Technical smoke on already evaluated held-in training probes; no calibration/measurement/final evaluation',
        precision='FP32 base and LoRA; TF32off; eager; prefix caching disabled',
        jobs=['base', 'zero_adapter', 'trained_adapter', 'base_repeat'],
        generation=dict(temperature=0, top_p=1, max_tokens=768, logprobs=1),
        engine=dict(max_model_len=2048, max_num_seqs=2, gpu_memory_utilization=.85,
                    enable_lora=True, max_lora_rank=8, max_loras=1, lora_dtype='float32'),
        gates='All finite aligned chosen-token logprobs; exact base/zero/base_repeat greedy tokens; nonzero trained checkpoint and changed outputs or aligned logprobs',
        input_sha256={str(p): sha(p) for p in sources}, physical_gpu=3, api_spend_usd=0))
    print('Frozen two-task FP32 adapter-engine smoke.')


def run():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    p = json.loads((OUT / 'protocol.json').read_text())
    for name, digest in p['input_sha256'].items():
        assert sha(Path(name)) == digest
    assert not (OUT / 'raw.jsonl').exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    save(OUT / 'manifest.json', dict(protocol_sha256=sha(OUT / 'protocol.json'), pid=os.getpid(),
         versions={k: importlib.metadata.version(k) for k in ['vllm', 'torch', 'transformers']},
         utc=datetime.now(timezone.utc).isoformat(), api_spend_usd=0))
    rows = json.loads((OUT / 'inputs.json').read_text())
    started = time.monotonic()
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), dtype='float32', seed=SEED,
              enforce_eager=True, disable_log_stats=True, enable_prefix_caching=False,
              **{**p['engine'], 'lora_dtype': torch.float32})
    jobs = [('base', None), ('zero_adapter', LoRARequest('zero', 1, str(OUT / 'zero_adapter'))),
            ('trained_adapter', LoRARequest('trained', 2, str(CONTROL / 'full/adapter'))), ('base_repeat', None)]
    records = []
    with (OUT / 'raw.jsonl').open('w') as handle:
        for name, request in jobs:
            replies = llm.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in rows],
                                   SamplingParams(**p['generation'], seed=SEED), lora_request=request, use_tqdm=False)
            for row, reply in zip(rows, replies):
                seq = reply.outputs[0]
                assert reply.prompt_token_ids == row['prompt_token_ids']
                lp = [float(values[t].logprob) for t, values in zip(seq.token_ids, seq.logprobs)]
                assert len(lp) == len(seq.token_ids) and torch.isfinite(torch.tensor(lp)).all()
                record = dict(job=name, task_id=row['task_id'], prompt_token_ids=reply.prompt_token_ids,
                              generated_token_ids=list(seq.token_ids), chosen_token_logprobs=lp,
                              generation=seq.text, finish_reason=seq.finish_reason, stop_reason=seq.stop_reason)
                records.append(record)
                handle.write(json.dumps(record) + '\n')
                handle.flush()
            print(json.dumps(dict(job=name, elapsed_seconds=time.monotonic()-started)), flush=True)
    by = {(r['job'], r['task_id']): r for r in records}
    for task in p['tasks']:
        base = by['base', task]['generated_token_ids']
        assert base == by['zero_adapter', task]['generated_token_ids'] == by['base_repeat', task]['generated_token_ids']
    changed = [task for task in p['tasks'] if by['base', task]['generated_token_ids'] != by['trained_adapter', task]['generated_token_ids']]
    logprob_changed = any(by['base', t]['chosen_token_logprobs'] != by['trained_adapter', t]['chosen_token_logprobs'] for t in p['tasks'])
    assert changed or logprob_changed
    save(OUT / 'summary.json', dict(technical_checks_passed=True, tasks=2, generations=8,
         exact_no_update_and_base_repeat=True, trained_output_changed_tasks=changed,
         hf_base_output_changed_tasks=[r['task_id'] for r in rows if r['generated_token_ids'] != by['base', r['task_id']]['generated_token_ids']],
         interpretation='Engine validation only; HF comparison descriptive, no selection or functional-performance claim',
         elapsed_seconds=time.monotonic()-started, api_spend_usd=0))
    save(OUT / 'completed_manifest.json', dict(sha256={f.name: sha(f) for f in OUT.iterdir() if f.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        prepare() if args.stage == 'prepare' else run()
    except Exception as error:
        if OUT.exists():
            save(OUT / 'failure.json', dict(error_type=type(error).__name__, message=str(error), fallback_used=False))
        raise
