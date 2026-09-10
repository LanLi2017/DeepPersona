#!/usr/bin/env python3
"""Technical-only actual-source long-context smoke. Never quality-grade/full-train."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'runs/swe-diversity-selection/swe-sympy-patch-sft'
CAPACITY = ROOT/'runs/swe-diversity-selection/swe-sympy-context-capacity'
OUT = CAPACITY/'learner_smoke'
HELPER = ROOT/'scripts/113_swe_patch_sft_train.py'
spec = importlib.util.spec_from_file_location('helpers113',HELPER)
h = importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
MODEL = h.MODEL
LIMIT = 3600


def native_prompt_ids(tok, messages):
    result=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    ids=result if isinstance(result,list) else result['input_ids']
    assert isinstance(ids,list) and ids and all(type(i) is int for i in ids)
    return ids


def freeze():
    """114 handoff is checked here; no runtime quality labels are required/read."""
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as qwen
    import transformers.integrations.sdpa_attention as sdpa
    import peft.tuners.lora.layer as lora
    import peft.utils.save_and_load as peft_io
    import torch.optim.adamw as adamw
    assert not OUT.exists(), 'Single prospective technical smoke; no overwrite'
    summary=h.read(CAPACITY/'score_summary.json')
    cap=summary['smallest_cap_meeting_source_gate']
    assert cap in [12288,24576] and summary['technical_failures']==0 and summary['minimum_required']==16
    assert cap==min(r['cap'] for r in summary['caps'] if r['prequalified']>=16)
    scored=next(r for r in summary['caps'] if r['cap']==cap)
    ids=[r['instance_id'] for r in scored['tasks'] if r['prequalified']]
    assert len(ids)==len(set(ids))==scored['prequalified'] and len(ids)>=16
    assert summary['evaluation_new_contexts']==0 and summary['evaluation_private_values_read'] is False
    selection=h.read(OLD/'selection.json');source=[r['instance_id'] for r in selection['tasks'] if r['role']=='source']
    assert set(ids)<=set(source)
    manifest=h.read(CAPACITY/'completed_manifest.json')
    for name,digest in manifest.items():assert h.sha(h.resolve(name))==digest,name
    paths=[Path(__file__),HELPER,CAPACITY/'score_summary.json',CAPACITY/'protocol.json',CAPACITY/'completed_manifest.json',
           OLD/'selection.json',OLD/'source_targets/completed_manifest.json',OLD/'source_targets/protocol.json']
    paths += [h.resolve(name) for name in manifest]
    build=h.read(CAPACITY/'build_manifest.json')
    assert build['assigned_slots']==len(build['tasks'])==64 and build['private_inputs_read'] is False
    assert {(r['instance_id'],r['cap']) for r in build['tasks']}=={(iid,k) for iid in source for k in [12288,24576]}
    assert build['protocol_sha256']==h.sha(CAPACITY/'protocol.json')
    for name,digest in build['artifact_sha256'].items():assert h.sha(h.resolve(name))==digest,name;paths.append(h.resolve(name))
    for pp,field in [(CAPACITY/'protocol.json','source_sha256'),(CAPACITY/'score_protocol.json','inputs_sha256'),(OLD/'context/protocol.json','source_sha256'),
                     (OLD/'source_targets/protocol.json','inputs_sha256')]:
        paths.append(pp)
        for name,digest in h.read(pp)[field].items():assert h.sha(h.resolve(name))==digest,name;paths.append(h.resolve(name))
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    target_manifest={str(h.resolve(k)):v for k,v in h.read(OLD/'source_targets/completed_manifest.json').items()}
    records=[]
    for iid in source:
        if iid not in ids:continue
        cp=CAPACITY/str(cap)/f'{iid}.json'
        c=h.read(cp);tp=OLD/'source_targets'/iid/'record.json';t=h.read(tp)
        assert h.sha(tp)==target_manifest[str(tp)]
        assert c['instance_id']==t['instance_id']==iid and c['status']=='ready' and c['role']=='source' and c['cap']==cap
        assert c['protocol_sha256']==h.sha(CAPACITY/'protocol.json')
        assert h.sha(CAPACITY/str(cap)/f'{iid}.prompt.txt')==c['prompt_sha256']
        assert h.sha(CAPACITY/str(cap)/f'{iid}.packet.txt')==c['packet_sha256']
        assert t['conversion_supported'] and t['independent_git_reconstruction'] and t['completion_fits']
        assert 'technical_failure' not in t
        prompt=native_prompt_ids(tok,c['messages'])
        target=tok.encode(t['target'],add_special_tokens=False)+[tok.eos_token_id]
        assert prompt==c['prompt_token_ids'] and 0<len(prompt)<=cap
        assert target==t['target_token_ids'] and 0<len(target)<=2048
        assert target[-1]==t['eos_token_id']==tok.eos_token_id
        paths += [cp,tp]
        records.append(dict(instance_id=iid,prompt_token_ids=prompt,target_token_ids=target,
                            context_record_sha256=h.sha(cp),target_record_sha256=h.sha(tp),quality_verified=False))
    assert len(records)==len(ids)
    chosen=h.smoke_records(records)
    config=h.read(MODEL/'config.json');assert config['num_hidden_layers']==36
    assert max(len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in records)<=config['max_position_embeddings']
    model_files=sorted(p for p in MODEL.iterdir() if p.is_file() and p.suffix in {'.json','.safetensors','.jinja','.txt'})
    index=h.read(MODEL/'model.safetensors.index.json')
    assert all(MODEL/n in model_files for n in set(index['weight_map'].values()))
    paths += model_files+[Path(m.__file__) for m in [qwen,sdpa,lora,peft_io,adamw]]
    inputs={str(p):h.sha(p) for p in sorted(set(paths))}
    software={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','safetensors','accelerate','tokenizers']}
    hardware=h.gpu_identity()
    OUT.mkdir()
    h.save(OUT/'smoke_records.json',chosen)
    inputs[str(OUT/'smoke_records.json')]=h.sha(OUT/'smoke_records.json')
    h.save(OUT/'protocol.json',dict(utc=h.utc(),scope='Training-interface development on already exposed source gold; technical-only, quality unverified',
        inputs_sha256=inputs,software=software,hardware=hardware,model=str(MODEL),seed=0,
        cuda_device_order='PCI_BUS_ID',torch_device_uuid_must_match_frozen_physical_gpu2=True,
        chosen_prompt_cap=cap,max_completion_tokens=2048,max_sequence_tokens=cap+2048,
        prequalified_source_ids=[r['instance_id'] for r in records],smoke_ids=[r['instance_id'] for r in chosen],
        selection='Longest actual serialized sequence, then longest actual completion if different; ties by instanceID; unchanged113helper',
        objective='Mean per-example mean-completion-token NLL including EOS; one actual1or2example accumulation update',
        steady_state_probe='Repeat same examples forward/backward with initialized Adam states retained, no second optimizer step; clear gradients before reset',
        optimizer=dict(name='AdamW',lr=1e-4,betas=[0.9,0.999],eps=1e-8,weight_decay=0),
        adapter=dict(layers=36,targets=['q_proj','v_proj'],r=8,alpha=16,dropout=0,parameters=3833856),
        precision=dict(base='BF16',adapter_parameters='FP32',adapter_compute='FP32 A/B matmuls, result cast to BF16',optimizer_state='FP32',
                       loss='FP32 crossentropy',autocast=False,tf32=False,attention='SDPA FLASH_ATTENTION only; no fallback'),
        checkpointing='nonreentrant',use_cache=False,logits='Only completion prediction positions via tensor logits_to_keep',
        validation=['Unchanged base versus zero-adapter probe logits exact','Finite gradients and nonzero adapter delta',
                    'Actual LoRA layer and A input/output dtype hooks','Standard PEFT checkpoint saved tensors and reloaded tensors/logits exact',
                    'Reset original adapter tensors and original probe exact','Frozen base parameter byte hash plus storage/version/dtype checks'],
        total_shared_gpu_budget_seconds=LIMIT,prior_gpu_seconds=0,
        external_timeout=dict(required=True,term_seconds=3595,kill_grace_seconds=5,hard_ceiling_seconds=3600,
            launcher='timeout --signal=TERM --kill-after=5s 3595s env SWE_SMOKE_EXTERNAL_TIMEOUT=3595+5 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/115_swe_sympy_long_context_smoke.py run'),
        accounting='This single-use run charges validation/load/smoke/hash/save time. Future full run must subtract max(internal elapsed, root-measured external terminal duration); no retry reset. Terminal duration must be recorded externally, especially if killed.',
        future_estimate='1.20*max(first_update_seconds,steady_state_probe_seconds)/smoke_examples*(3*prequalified_sources)+120seconds load/save reserve; descriptive conservative dose, not quality approval',
        full_training=False,inference=False,quality_verification=False,evaluation_private_access=False,api_spend_usd=0))
    print(json.dumps(dict(frozen=True,cap=cap,smoke_ids=[r['instance_id'] for r in chosen])),flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    assert os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
    assert os.environ.get('SWE_SMOKE_EXTERNAL_TIMEOUT')=='3595+5', 'Use frozen external process-group timeout and set acknowledgment'
    work=OUT/'run';work.mkdir()  # Failed or killed attempts remain permanently consumed.
    started=time.monotonic();torch=None
    def interrupted(signum,frame):raise TimeoutError('Shared60minute smoke/full budget or external timeout; no fallback/retry')
    old_alarm=signal.signal(signal.SIGALRM,interrupted);old_term=signal.signal(signal.SIGTERM,interrupted)
    signal.setitimer(signal.ITIMER_REAL,3595)
    try:
        p=h.read(OUT/'protocol.json');h.save(work/'started.json',dict(utc=h.utc(),protocol_sha256=h.sha(OUT/'protocol.json'),external_timeout=p['external_timeout']))
        for name,digest in p['inputs_sha256'].items():assert h.sha(name)==digest,name
        assert {n:importlib.metadata.version(n) for n in p['software']}==p['software']
        assert h.gpu_identity()==p['hardware']
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
        from peft.tuners.lora.layer import LoraLayer
        from safetensors.torch import load_file
        from torch.nn.attention import sdpa_kernel,SDPBackend
        assert torch.cuda.device_count()==1
        torch_uuid=str(torch.cuda.get_device_properties(0).uuid)
        normalize_uuid=lambda value:str(value).lower().removeprefix('gpu-')
        assert normalize_uuid(torch_uuid)==normalize_uuid(p['hardware']['uuid']), 'CUDA logical0 is not frozen physical GPU2'
        h.save(work/'device_identity.json',dict(cuda_device_order=os.environ['CUDA_DEVICE_ORDER'],
            visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch_logical0_uuid=torch_uuid,physical_gpu2_uuid=p['hardware']['uuid'],matched=True))
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
        records=h.read(OUT/'smoke_records.json');assert [r['instance_id'] for r in records]==p['smoke_ids']
        model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,local_files_only=True,
            attn_implementation='sdpa',device_map={'':'cuda:0'}).eval();model.config.use_cache=False
        def forward(r,probe=False):
            start=len(r['prompt_token_ids']);ids=r['prompt_token_ids']+r['target_token_ids']
            tokens=torch.tensor([ids],device='cuda',dtype=torch.long)
            keep=torch.arange(start-1,start if probe else len(ids)-1,device='cuda')
            logits=model(input_ids=tokens,use_cache=False,logits_to_keep=keep).logits
            assert logits.shape[1]==(1 if probe else len(r['target_token_ids']))
            if probe:return logits.detach().cpu()
            return torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),tokens[:,start:].reshape(-1)),str(logits.dtype)
        def probe():
            model.eval()
            with torch.no_grad():return forward(records[0],True)
        def tensor_hash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        def frozen_hash():
            digest=hashlib.sha256()
            for name,v in model.named_parameters():
                if v.requires_grad:continue
                digest.update(json.dumps([name,list(v.shape),str(v.dtype)]).encode())
                for chunk in v.detach().reshape(-1).split(8*1024*1024):
                    digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            original=probe()
            model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],bias='none'))
            params={n:v for n,v in model.named_parameters() if v.requires_grad}
            assert sum(v.numel() for v in params.values())==3833856
            modules=[m for m in model.modules() if isinstance(m,LoraLayer)];assert len(modules)==72
            for v in params.values():v.data=v.data.float()
            assert all(v.dtype==torch.float32 for v in params.values())
            base={n:(v.data_ptr(),v._version,str(v.dtype)) for n,v in model.named_parameters() if not v.requires_grad}
            assert all(dtype=='torch.bfloat16' for _,_,dtype in base.values())
            initial={n:v.detach().cpu().clone() for n,v in params.items()}
            assert all(torch.count_nonzero(v)==0 for n,v in initial.items() if '.lora_B.' in n)
            torch.save(initial,work/'initial_adapter.pt')
            assert torch.equal(original,probe()),'Zero adapter changes base probe'
            before_hash=frozen_hash();trace={}
            def layer_hook(module,args,result):trace['layer']=dict(input=str(args[0].dtype),output=str(result.dtype),base_weight=str(module.base_layer.weight.dtype))
            def a_hook(module,args,result):trace['A']=dict(input=str(args[0].dtype),output=str(result.dtype),weight=str(module.weight.dtype))
            def b_hook(module,args,result):trace['B']=dict(input=str(args[0].dtype),output=str(result.dtype),weight=str(module.weight.dtype))
            hooks=[modules[0].register_forward_hook(layer_hook),modules[0].lora_A['default'].register_forward_hook(a_hook),modules[0].lora_B['default'].register_forward_hook(b_hook)]
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
            opt=torch.optim.AdamW(params.values(),lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0)
            assert len(opt.state)==0;opt.zero_grad(set_to_none=True);torch.cuda.synchronize();update_start=time.monotonic()
            with (work/'smoke_items.jsonl').open('x') as f:
                for r in records:
                    loss,logit_dtype=forward(r);assert torch.isfinite(loss)
                    (loss/len(records)).backward()
                    f.write(json.dumps(dict(instance_id=r['instance_id'],nll=float(loss.detach()),
                        prompt_tokens=len(r['prompt_token_ids']),completion_tokens=len(r['target_token_ids']),
                        accumulation_divisor=len(records),logit_dtype=logit_dtype,loss_dtype=str(loss.dtype),quality_verified=False))+'\n');f.flush();del loss
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
            gradnorm=float(torch.sqrt(sum(v.grad.square().sum() for v in params.values())));assert gradnorm>0
            opt.step();assert all(torch.isfinite(v).all() for v in params.values())
            assert all(s.dtype==torch.float32 for state in opt.state.values() for s in state.values() if torch.is_tensor(s))
            torch.cuda.synchronize();update_seconds=time.monotonic()-update_start
            # Second backward checks peak memory with Adam state resident, without a second update.
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();steady_start=time.monotonic()
            with (work/'steady_state_items.jsonl').open('x') as f:
                for r in records:
                    loss,logit_dtype=forward(r);assert torch.isfinite(loss)
                    (loss/len(records)).backward()
                    f.write(json.dumps(dict(instance_id=r['instance_id'],nll=float(loss.detach()),
                        prompt_tokens=len(r['prompt_token_ids']),completion_tokens=len(r['target_token_ids']),
                        accumulation_divisor=len(records),optimizer_states_resident=True,optimizer_step_performed=False,
                        logit_dtype=logit_dtype,loss_dtype=str(loss.dtype),quality_verified=False))+'\n');f.flush();del loss
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
            torch.cuda.synchronize();steady_seconds=time.monotonic()-steady_start
            opt.zero_grad(set_to_none=True)
            changed={n:v.detach().cpu().clone() for n,v in params.items()}
            delta=float(torch.sqrt(sum((changed[n]-initial[n]).square().sum() for n in params)));assert delta>0
            updated=probe()
            assert trace['A']==trace['B']==dict(input='torch.float32',output='torch.float32',weight='torch.float32')
            assert trace['layer']==dict(input='torch.bfloat16',output='torch.bfloat16',base_weight='torch.bfloat16')
            model.save_pretrained(work/'updated_adapter',safe_serialization=True)
            saved=load_file(str(work/'updated_adapter/adapter_model.safetensors'));state=get_peft_model_state_dict(model)
            assert saved.keys()==state.keys() and all(torch.equal(saved[n],v.detach().cpu()) for n,v in state.items())
            del state
            with torch.no_grad():
                for v in params.values():v.zero_()
            set_peft_model_state_dict(model,saved,adapter_name='default')
            assert all(torch.equal(v.detach().cpu(),changed[n]) for n,v in params.items())
            assert torch.equal(probe(),updated),'Reload probe differs'
            with torch.no_grad():
                for n,v in params.items():v.copy_(initial[n].to(v.device))
            assert all(torch.equal(v.detach().cpu(),initial[n]) for n,v in params.items())
            assert torch.equal(probe(),original),'Initial reset probe differs'
            assert base=={n:(v.data_ptr(),v._version,str(v.dtype)) for n,v in model.named_parameters() if not v.requires_grad}
            after_hash=frozen_hash();assert before_hash==after_hash
            for hook in hooks:hook.remove()
            del opt
            for v in params.values():v.grad=None
            torch.cuda.synchronize();elapsed=time.monotonic()-started
            estimate=1.20*max(update_seconds,steady_seconds)/len(records)*(3*len(p['prequalified_source_ids']))+120
            h.save(work/'summary.json',dict(utc=h.utc(),technical_checks_passed=True,quality_verified=False,full_training_run=False,
                inference_run=False,evaluation_private_access=False,smoke_ids=p['smoke_ids'],selected_prompt_cap=p['chosen_prompt_cap'],
                accumulation_examples=len(records),gradient_norm=gradnorm,adapter_delta_l2=delta,actual_dtypes=trace,
                base_probe_sha256=tensor_hash(original),updated_probe_sha256=tensor_hash(updated),
                probe_max_abs_change=float((updated.float()-original.float()).abs().max()),zero_update_exact=True,
                standard_peft_saved_reloaded_exact=True,initial_reset_exact=True,
                frozen_base_parameter_sha256_before=before_hash,frozen_base_parameter_sha256_after=after_hash,
                base_storage_version_dtype_unchanged=True,smoke_update_seconds=update_seconds,
                steady_state_probe_seconds=steady_seconds,steady_state_optimizer_states_resident=True,
                optimizer_steps=1,gradient_example_exposures=2*len(records),steady_state_extra_optimizer_steps=0,
                gradient_completion_token_exposures=2*sum(len(r['target_token_ids']) for r in records),
                future_compute_reserve_factor=1.20,future_load_save_reserve_seconds=120,
                total_elapsed_seconds=elapsed,shared_budget_seconds=LIMIT,remaining_seconds_before_external_charge=LIMIT-elapsed,
                future_max_prequalified_dose_estimate_seconds=estimate,future_max_dose_fits_internal_budget=estimate<=LIMIT-elapsed,
                external_terminal_duration_required_before_future_full=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_runtime=torch.version.cuda,device=torch.cuda.get_device_name(0),api_spend_usd=0,
                interpretation='Technical memory/numerics/save-reset check only; targets are not execution-quality verified. Future full training requires source-quality dose and all20evaluation runtimes plus remaining shared budget.'))
            h.save(work/'completed_manifest.json',{str(f):h.sha(f) for f in work.rglob('*') if f.is_file()})
    except BaseException as error:
        h.save(work/'failure.json',dict(utc=h.utc(),type=type(error).__name__,message=str(error),
            total_elapsed_seconds=time.monotonic()-started,shared_budget_seconds=LIMIT,retry=False,fallback=False,
            external_terminal_duration_required=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_initialized() else None))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);a=parser.parse_args()
    if a.stage=='freeze':freeze()
    else:run()
