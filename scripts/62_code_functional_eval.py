#!/usr/bin/env python3
"""FP32 greedy functional evaluation on frozen MBPP calibration tasks only."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

POOL = Path('runs/swe-diversity-selection/code-learning-pilot')
MODEL = Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
SEED = 20260909
HELPER = Path(__file__).with_name('52_code_learning_pool.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_inputs(task_ids):
    protocol = json.loads((POOL / 'protocol.json').read_text())
    assert sha(POOL / 'inputs.jsonl') == protocol['frozen_inputs_sha256']
    assert sha(POOL / 'evaluator_targets.jsonl') == protocol['targets_sha256']
    ids = task_ids if task_ids is not None else protocol['calibration_task_ids'][:2]
    assert ids and len(ids) == len(set(ids))
    assert set(ids) <= set(protocol['calibration_task_ids']), 'Calibration task IDs only; measurement/final test forbidden'
    rows = {r['task_id']: r for line in (POOL / 'inputs.jsonl').open() if (r := json.loads(line))['split'] == 'calibration'}
    targets = {r['task_id']: r for line in (POOL / 'evaluator_targets.jsonl').open() if (r := json.loads(line))['split'] == 'calibration'}
    expected = json.loads((POOL / 'generation_manifest.json').read_text())['generation_script_sha256']
    assert sha(HELPER) == expected, 'Frozen grader helper changed'
    spec = importlib.util.spec_from_file_location('functional_pool52', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return [rows[i] for i in ids], targets, helper


def greedy(model, tokenizer, rows, batch_size, out, stem):
    import torch
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.temperature = None
    config.top_p = None
    config.top_k = None
    config.max_new_tokens = 768
    config.num_beams = 1
    config.num_return_sequences = 1
    config.return_dict_in_generate = True
    config.output_scores = False
    config.pad_token_id = tokenizer.pad_token_id
    eos = config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    results = []
    started = time.time()
    with (out / f'{stem}_raw.jsonl').open('w') as f:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False) for r in batch]
            inputs = tokenizer(texts, padding=True, return_tensors='pt')
            assert tokenizer.padding_side == 'left'
            width = inputs['input_ids'].shape[1]
            with torch.inference_mode():
                reply = model.generate(**inputs.to('cuda'), generation_config=config)
            suffixes = reply.sequences[:, width:].cpu().tolist()
            for j, (row, text, padded_suffix) in enumerate(zip(batch, texts, suffixes)):
                stop = next((k + 1 for k, token in enumerate(padded_suffix) if token in eos_ids), None)
                tokens = padded_suffix[:stop] if stop is not None else padded_suffix
                prompt_tokens = inputs['input_ids'][j][inputs['attention_mask'][j].bool()].cpu().tolist()
                assert prompt_tokens == tokenizer.encode(text)
                record = {'task_id': row['task_id'], 'user_prompt': row['prompt'], 'rendered_prompt': text,
                          'prompt_token_ids': prompt_tokens,
                          'padded_prompt_token_ids': inputs['input_ids'][j].cpu().tolist(),
                          'attention_mask': inputs['attention_mask'][j].cpu().tolist(),
                          'generated_token_ids': tokens, 'generated_suffix_with_batch_padding': padded_suffix,
                          'generation': tokenizer.decode(tokens, skip_special_tokens=True),
                          'finish_reason': 'eos' if stop is not None else 'length' if len(tokens) == 768 else 'other',
                          'batch_index': offset // batch_size, 'batch_task_ids': [r['task_id'] for r in batch]}
                f.write(json.dumps(record) + '\n'); f.flush()
                results.append(record)
            print(json.dumps({'mode': stem, 'tasks_completed': len(results), 'total': len(rows),
                              'seconds': round(time.time() - started, 2)}), flush=True)
    return results


def no_update_check(model, tokenizer, rows, batch_size, out):
    from peft import LoraConfig, PeftModel, get_peft_model
    baseline = greedy(model, tokenizer, rows, batch_size, out, 'base')
    adapter = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=0,
                                             target_modules=['q_proj', 'v_proj'],
                                             layers_to_transform=[model.config.num_hidden_layers - 1]))
    checkpoint = out / 'no_update_adapter'
    adapter.save_pretrained(checkpoint)
    model = adapter.unload()
    reloaded = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False).eval()
    result = greedy(reloaded, tokenizer, rows, batch_size, out, 'reloaded_no_update')
    assert all(a['generated_token_ids'] == b['generated_token_ids'] for a, b in zip(baseline, result))
    save(out / 'no_update_equivalence.json', {'exact_generated_token_equivalence': True,
         'task_ids': [r['task_id'] for r in rows], 'batch_size': batch_size,
         'adapter_sha256': {str(p.relative_to(out)): sha(p) for p in checkpoint.iterdir() if p.is_file()},
         'scope': 'Fresh zero-effect last-layer rank8 q/v LoRA, saved/reloaded, same precision and batch boundaries'})
    return result


def run(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'PhysicalGPU2 only'
    assert not args.out.exists(), 'Use a fresh output directory; completed outputs are immutable'
    rows, targets, helper = load_inputs(args.task_ids)
    if args.adapter:
        assert (args.adapter / 'adapter_config.json').is_file()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.out.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'seed': SEED, 'physical_gpu': 2, 'model_snapshot': str(MODEL),
        'dtype': 'float32', 'attention': 'sdpa', 'tf32': False,
        'padding_side': 'left', 'attention_mask': True, 'batch_size': args.batch_size,
        'batch_task_ids': [[r['task_id'] for r in rows[i:i + args.batch_size]] for i in range(0, len(rows), args.batch_size)],
        'task_ids': [r['task_id'] for r in rows], 'split': 'calibration_only',
        'mode': 'no_update_equivalence' if args.check_no_update else 'adapter' if args.adapter else 'base',
        'adapter': str(args.adapter) if args.adapter else None,
        'adapter_sha256': {p.name: sha(p) for p in args.adapter.iterdir() if p.is_file()} if args.adapter else {},
        'generation': {'do_sample': False, 'max_new_tokens': 768, 'enable_thinking': False},
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'input_sha256': {str(p): sha(p) for p in [POOL/'protocol.json', POOL/'inputs.jsonl', POOL/'evaluator_targets.jsonl', HELPER]},
        'runner_sha256': sha(Path(__file__)),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft']},
        'api_spend_usd': 0, 'pid': os.getpid(),
        'limitation': 'MBPP calibration smoke; provided-test execution is not SWE repair evaluation. Grader has no filesystem/network isolation.'}
    save(args.out / 'manifest.json', manifest)
    (args.out / 'runner_snapshot.py').write_text(Path(__file__).read_text())
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True,
                                                attn_implementation='sdpa').cuda().eval()
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point())
    model.config.use_cache = True
    if args.check_no_update:
        records = no_update_check(model, tokenizer, rows, args.batch_size, args.out)
    else:
        if args.adapter:
            model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False).eval()
            assert all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point())
        records = greedy(model, tokenizer, rows, args.batch_size, args.out, 'eval')
    grades = []
    with (args.out / 'grades.jsonl').open('w') as f:
        for r in records:
            result = {'task_id': r['task_id'], **helper.grade(r['generation'], targets[r['task_id']])}
            f.write(json.dumps(result) + '\n'); f.flush(); grades.append(result)
    summary = {'tasks': len(grades), 'correct': sum(r['correct'] for r in grades),
               'pass_at_1': sum(r['correct'] for r in grades) / len(grades),
               'valid_python': sum(r['valid_python'] for r in grades),
               'truncated': sum(r['finish_reason'] == 'length' for r in records),
               'generated_tokens': sum(len(r['generated_token_ids']) for r in records),
               'measurement_or_final_test_scored': False, 'api_spend_usd': 0}
    save(args.out / 'summary.json', summary)
    save(args.out / 'completed_manifest.json', {'status': 'complete', 'sha256': {p.name: sha(p) for p in args.out.iterdir() if p.is_file()}})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--adapter', type=Path)
    modes.add_argument('--base', action='store_true')
    modes.add_argument('--check-no-update', action='store_true')
    parser.add_argument('--task-ids', nargs='+', type=int)
    parser.add_argument('--batch-size', choices=[2, 4], type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
