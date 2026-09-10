#!/usr/bin/env python3
"""One frozen patch-SFT smoke/full run; no inference or evaluation-private reads."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/swe-diversity-selection/swe-sympy-patch-sft'
DEST = OUT / 'learner'
MODEL = Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
LIMIT_SECONDS = 3600


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
def save(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2); f.write('\n')
def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

def gpu_identity():
    raw = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=30)
    index, uuid, name, memory = [x.strip() for x in raw.strip().split(',')]
    assert index == '2'
    return dict(index=2, uuid=uuid, name=name, total_mib=int(memory))


def batches(records):
    for offset in range(0, len(records), 4): yield records[offset:offset + 4]


def smoke_records(records):
    longest = min(records, key=lambda r: (-(len(r['prompt_token_ids']) + len(r['target_token_ids'])), r['instance_id']))
    completion = min(records, key=lambda r: (-len(r['target_token_ids']), r['instance_id']))
    return [longest] if longest['instance_id'] == completion['instance_id'] else [longest, completion]


def freeze():
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as qwen
    import transformers.integrations.sdpa_attention as sdpa
    import peft.tuners.lora.layer as lora
    import peft.utils.save_and_load as peft_io
    import torch.optim.adamw as adamw
    assert not DEST.exists(), 'No overwrite or alternative recipe in this experiment'
    quality = read(OUT / 'runtime/summary.json')
    selection = read(OUT / 'selection.json')
    source_ids = [r['instance_id'] for r in selection['tasks'] if r['role'] == 'source']
    eval_ids = {r['instance_id'] for r in selection['tasks'] if r['role'] == 'evaluation'}
    admitted = quality['admitted_source_ids']
    assert len(admitted) == len(set(admitted)) >= 16 and set(admitted) <= set(source_ids)
    assert len(eval_ids) == 20 and len(quality['evaluation_runnable_ids']) == 20
    assert set(quality['evaluation_runnable_ids']) == eval_ids
    assert not set(admitted) & eval_ids
    # Only hash private runtime evidence; do not deserialize its tests/patches.
    quality_manifest = read(OUT / 'runtime/completed_manifest.json')
    assert isinstance(quality_manifest, dict)
    assert str(OUT / 'runtime/summary.json') in {str(resolve(p)) for p in quality_manifest}
    for name, digest in quality_manifest.items(): assert sha(resolve(name)) == digest, name
    target_summary = read(OUT / 'source_targets/summary.json')
    assert target_summary['prequalification_gate_passed'] and target_summary['technical_failures'] == 0
    paths = [Path(__file__), OUT / 'runtime/summary.json', OUT / 'runtime/completed_manifest.json',
             OUT / 'selection.json', OUT / 'selection_protocol.json', OUT / 'context/protocol.json',
             OUT / 'context/manifest.json', OUT / 'source_targets/protocol.json', OUT / 'source_targets/summary.json',
             OUT / 'source_targets/completed_manifest.json']
    for protocol_path, field in [(OUT/'selection_protocol.json', 'inputs_sha256'),
                                  (OUT/'context/protocol.json', 'source_sha256'),
                                  (OUT/'context/parallel_execution_amendment.json', 'source_sha256'),
                                  (OUT/'source_targets/protocol.json', 'inputs_sha256')]:
        paths.append(protocol_path)
        for name, digest in read(protocol_path)[field].items():
            assert sha(resolve(name)) == digest, name
            paths.append(resolve(name))
    # These manifest-bound records contain exactly the source targets already admitted by112.
    target_manifest = read(OUT / 'source_targets/completed_manifest.json')
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    records = []
    for iid in source_ids:
        if iid not in admitted: continue
        cp = OUT/'context'/f'{iid}.json'; tp = OUT/'source_targets'/iid/'record.json'
        expected = {str(resolve(k)): v for k, v in target_manifest.items()}
        assert str(tp) in expected and sha(tp) == expected[str(tp)]
        c, t = read(cp), read(tp)
        assert c['instance_id'] == t['instance_id'] == iid and c['role'] == t['role'] == 'source'
        assert c['status'] == 'ready' and t['prequalified'] and t['independent_git_reconstruction']
        assert 'technical_failure' not in t
        prompt = tok.apply_chat_template(c['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert prompt == c['prompt_token_ids'] and len(prompt) == c['prompt_tokens'] <= 6144
        target = tok.encode(t['target'], add_special_tokens=False) + [tok.eos_token_id]
        assert target == t['target_token_ids'] and 0 < len(target) <= 2048 and len(prompt) + len(target) <= 8192
        assert target[-1] == t['eos_token_id'] == tok.eos_token_id
        assert sha(OUT/'context'/f'{iid}.prompt.txt') == c['prompt_sha256']
        paths += [cp, tp, OUT/'context'/f'{iid}.prompt.txt', OUT/'context'/f'{iid}.packet.txt']
        records.append(dict(instance_id=iid, prompt_token_ids=prompt, target_token_ids=target,
                            context_record_sha256=sha(cp), target_record_sha256=sha(tp)))
    assert len(records) == len(admitted)
    model_files = sorted(p for p in MODEL.iterdir() if p.is_file() and p.suffix in {'.json', '.safetensors', '.jinja', '.txt'})
    index = read(MODEL/'model.safetensors.index.json')
    assert all(MODEL/n in model_files for n in set(index['weight_map'].values()))
    assert read(MODEL/'config.json')['num_hidden_layers'] == 36
    paths += [Path(m.__file__) for m in [qwen, sdpa, lora, peft_io, adamw]] + model_files
    inputs = {str(p): sha(p) for p in sorted(set(paths))}
    software = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft', 'safetensors', 'accelerate', 'tokenizers']}
    hardware = gpu_identity()
    DEST.mkdir()
    with (DEST/'training_records.jsonl').open('x') as f:
        for r in records: f.write(json.dumps(r) + '\n')
    inputs[str(DEST/'training_records.jsonl')] = sha(DEST/'training_records.jsonl')
    save(DEST/'protocol.json', dict(utc=utc(), inputs_sha256=inputs, software=software, hardware=hardware,
        training_ids=[r['instance_id'] for r in records], evaluation_ids=sorted(eval_ids),
        smoke_ids=[r['instance_id'] for r in smoke_records(records)], model=str(MODEL), seed=0,
        epochs=3, accumulation=4, microbatch=1, task_order='Frozen source-slot order repeated each epoch; no shuffle',
        expected_updates=3*math.ceil(len(records)/4), expected_exposures=3*len(records),
        completion_token_exposures=3*sum(len(r['target_token_ids']) for r in records),
        objective='Per-example mean completion-token NLL including EOS; actual accumulation-batch mean; prompt loss masked',
        logits_to_keep='CUDA arange(prompt_length-1,total_length-1); project completion prediction positions only',
        optimizer=dict(name='AdamW',lr=1e-4,betas=[0.9,0.999],eps=1e-8,weight_decay=0),
        adapter=dict(targets=['q_proj','v_proj'],layers=36,r=8,alpha=16,dropout=0,parameters=3833856),
        precision=dict(base='BF16',adapter_parameters='FP32',optimizer_state='FP32',autocast=False,
            adapter_compute='PEFT casts activation to FP32 for adapter matmuls and output back to base BF16; not strictFP32',
            cross_entropy='FP32 logits',tf32=False,attention='SDPA, FLASH_ATTENTION only; no fallback',
            checkpointing='nonreentrant',use_cache=False),
        limits=dict(gpu=2,total_seconds=LIMIT_SECONDS,includes='Validation, load, all smoke checks, reset, fulltraining and checkpoint save',retry=False),
        extrapolation='1.25 * smoke_update_seconds / smoke_examples * total_full_exposures +120seconds reserve <= time remaining',
        smoke='Actual longest serialized sequence and longest completion (unique1or2examples), one accumulated update; no-update/base equality, finite gradients, changed adapter, exact standardPEFT reload and exact reset',
        full='Reset exact pre-smoke tensors and fresh optimizer; one recipe, no sweep',
        inference_or_private_evaluation=False,api_spend_usd=0))
    print(json.dumps(dict(frozen=True,training_tasks=len(records),smoke_ids=[r['instance_id'] for r in smoke_records(records)])),flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'Launch only with CUDA_VISIBLE_DEVICES=2'
    work = DEST/'run'; work.mkdir()  # Permanent single-use marker, including failed attempts.
    started = time.monotonic()
    def deadline(signum, frame): raise TimeoutError('Cumulative 60-minute run limit; no retry')
    previous = signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, LIMIT_SECONDS)
    torch = None
    try:
        p = read(DEST/'protocol.json')
        save(work/'started.json', dict(utc=utc(),protocol_sha256=sha(DEST/'protocol.json'),limit_seconds=LIMIT_SECONDS))
        for name,digest in p['inputs_sha256'].items(): assert sha(name)==digest,name
        assert {n:importlib.metadata.version(n) for n in p['software']} == p['software']
        assert gpu_identity() == p['hardware']
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from peft.tuners.lora.layer import LoraLayer
        from safetensors.torch import load_file
        from torch.nn.attention import sdpa_kernel, SDPBackend
        assert torch.cuda.device_count()==1
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
        records=[json.loads(l) for l in (DEST/'training_records.jsonl').read_text().splitlines()]
        smoke=smoke_records(records);assert [r['instance_id'] for r in smoke]==p['smoke_ids']
        model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,dtype=torch.bfloat16,
                                                 attn_implementation='sdpa',device_map={'':'cuda:0'})
        model.config.use_cache=False;model.eval()
        def forward(r, probe=False):
            start=len(r['prompt_token_ids']);ids=r['prompt_token_ids']+r['target_token_ids']
            tokens=torch.tensor([ids],device='cuda',dtype=torch.long)
            keep=torch.arange(start-1,start if probe else len(ids)-1,device='cuda')
            logits=model(input_ids=tokens,use_cache=False,logits_to_keep=keep).logits
            assert logits.shape[1] == (1 if probe else len(r['target_token_ids']))
            if probe: return logits.detach().cpu()
            loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),tokens[:,start:].reshape(-1))
            return loss, str(logits.dtype)
        def probe(r):
            model.eval()
            with torch.no_grad(): return forward(r,True)
        def tensor_digest(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            original_probe=probe(smoke[0])
            model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0,
                                               target_modules=['q_proj','v_proj'],bias='none'))
            params={n:v for n,v in model.named_parameters() if v.requires_grad}
            assert sum(v.numel() for v in params.values())==3833856
            assert len([m for m in model.modules() if isinstance(m,LoraLayer)])==72
            for v in params.values(): v.data=v.data.float()
            assert all(v.dtype==torch.float32 for v in params.values())
            base={n:(v.data_ptr(),v._version,str(v.dtype)) for n,v in model.named_parameters() if not v.requires_grad}
            assert all(dtype=='torch.bfloat16' for _,_,dtype in base.values())
            initial={n:v.detach().cpu().clone() for n,v in params.items()}
            assert all(torch.count_nonzero(v)==0 for n,v in initial.items() if '.lora_B.' in n)
            torch.save(initial,work/'initial_adapter.pt')
            assert torch.equal(original_probe,probe(smoke[0])), 'Zero-update adapter changed base logits'
            dtype_trace={}
            def dtype_hook(module, args, output):
                dtype_trace.update(input=str(args[0].dtype),output=str(output.dtype),
                    base_weight=str(module.base_layer.weight.dtype),A=str(module.lora_A['default'].weight.dtype),B=str(module.lora_B['default'].weight.dtype))
            first=next(m for m in model.modules() if isinstance(m,LoraLayer));hook=first.register_forward_hook(dtype_hook)
            def check_base():
                assert base=={n:(v.data_ptr(),v._version,str(v.dtype)) for n,v in model.named_parameters() if not v.requires_grad}
            def optimizer(): return torch.optim.AdamW(params.values(),lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0)
            def update(batch,opt,epoch,step,handle):
                model.train();opt.zero_grad(set_to_none=True);losses=[]
                for r in batch:
                    loss,logit_dtype=forward(r);assert torch.isfinite(loss)
                    (loss/len(batch)).backward();value=float(loss.detach());losses.append(value)
                    handle.write(json.dumps(dict(instance_id=r['instance_id'],epoch=epoch,step=step,nll=value,
                        prompt_tokens=len(r['prompt_token_ids']),completion_tokens=len(r['target_token_ids']),
                        accumulation_divisor=len(batch),logit_dtype=logit_dtype,loss_dtype=str(loss.dtype)))+'\n');handle.flush()
                    del loss
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
                norm=float(torch.sqrt(sum(v.grad.square().sum() for v in params.values())))
                assert norm>0
                opt.step();assert all(torch.isfinite(v).all() for v in params.values());check_base()
                assert all(s.dtype==torch.float32 for state in opt.state.values() for s in state.values() if torch.is_tensor(s))
                return dict(epoch=epoch,step=step,ids=[r['instance_id'] for r in batch],gradient_norm=norm,mean_nll=sum(losses)/len(losses))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            opt=optimizer();torch.cuda.synchronize();smoke_started=time.monotonic()
            with (work/'smoke_items.jsonl').open('x') as handle: smoke_step=update(smoke,opt,0,1,handle)
            torch.cuda.synchronize();smoke_seconds=time.monotonic()-smoke_started
            changed={n:v.detach().cpu().clone() for n,v in params.items()}
            delta=float(torch.sqrt(sum((changed[n]-initial[n]).square().sum() for n in params)))
            assert delta>0
            updated_probe=probe(smoke[0]);model.save_pretrained(work/'smoke_adapter',safe_serialization=True)
            saved=load_file(str(work/'smoke_adapter/adapter_model.safetensors'))
            state=get_peft_model_state_dict(model)
            assert saved.keys()==state.keys() and all(torch.equal(saved[n],v.detach().cpu()) for n,v in state.items())
            del state
            with torch.no_grad():
                for v in params.values():v.zero_()
            set_peft_model_state_dict(model,saved,adapter_name='default')
            assert all(torch.equal(v.detach().cpu(),changed[n]) for n,v in params.items())
            assert torch.equal(probe(smoke[0]),updated_probe), 'Saved/reloaded smoke logits differ'
            with torch.no_grad():
                for n,v in params.items():v.copy_(initial[n].to(v.device))
            assert all(torch.equal(v.detach().cpu(),initial[n]) for n,v in params.items())
            assert torch.equal(probe(smoke[0]),original_probe), 'Exact reset failed'
            check_base();hook.remove();del opt,changed,saved
            for v in params.values():v.grad=None
            estimate=1.25*smoke_seconds/len(smoke)*p['expected_exposures']+120
            remaining=LIMIT_SECONDS-(time.monotonic()-started)
            save(work/'smoke_summary.json',dict(technical_checks_passed=True,smoke_step=smoke_step,
                smoke_update_seconds=smoke_seconds,adapter_delta_l2=delta,actual_adapter_dtypes=dtype_trace,
                base_probe_sha256=tensor_digest(original_probe),updated_probe_sha256=tensor_digest(updated_probe),
                probe_max_abs_change=float((updated_probe.float()-original_probe.float()).abs().max()),
                zero_update_exact=True,saved_reload_exact=True,reset_exact=True,base_storage_versions_unchanged=True,
                estimated_full_seconds_with_reserve=estimate,remaining_seconds=remaining,full_fits_budget=estimate<=remaining,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started))
            assert estimate<=remaining, 'Smoke estimate cannot fit fixed remaining budget; no fallback'
            opt=optimizer();assert len(opt.state)==0
            step=0;online=[]
            with (work/'training_items.jsonl').open('x') as items,(work/'training_steps.jsonl').open('x') as steps:
                for epoch in range(1,4):
                    for batch in batches(records):
                        step+=1;row=update(batch,opt,epoch,step,items);online.append(row['mean_nll'])
                        row['elapsed_seconds']=time.monotonic()-started;steps.write(json.dumps(row)+'\n');steps.flush()
                        print(json.dumps(row),flush=True)
            assert step==p['expected_updates'];check_base()
            final_delta=float(torch.sqrt(sum((v.detach().cpu()-initial[n]).square().sum() for n,v in params.items())))
            assert final_delta>0
            model.eval();model.gradient_checkpointing_disable();model.save_pretrained(work/'adapter',safe_serialization=True)
            saved=load_file(str(work/'adapter/adapter_model.safetensors'));state=get_peft_model_state_dict(model)
            assert saved.keys()==state.keys() and all(torch.equal(saved[n],v.detach().cpu()) for n,v in state.items())
            torch.cuda.synchronize()
            save(work/'summary.json',dict(completed=True,trained=True,inference_run=False,optimizer_steps=step,
                item_exposures=p['expected_exposures'],completion_token_exposures=p['completion_token_exposures'],
                elapsed_seconds=time.monotonic()-started,limit_seconds=LIMIT_SECONDS,adapter_delta_l2=final_delta,
                base_storage_versions_unchanged=True,standard_peft_checkpoint_exact=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_runtime=torch.version.cuda,device_name=torch.cuda.get_device_name(0),api_spend_usd=0))
            files=[f for f in work.rglob('*') if f.is_file()]
            save(work/'completed_manifest.json',{str(f):sha(f) for f in files})
    except BaseException as error:
        save(work/'failure.json',dict(utc=utc(),type=type(error).__name__,message=str(error),
            elapsed_seconds=time.monotonic()-started,limit_seconds=LIMIT_SECONDS,retry=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_initialized() else None))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:run()
