#!/usr/bin/env python3
"""Verify a saved functional adapter changes calibration logits and unload restores base."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path('runs/swe-diversity-selection/code-functional-development')
OUT = ROOT / 'checkpoint_effect_check'
ADAPTER = ROOT / 'checkpoints/model_typicality_seed2_lr1_epochs3/adapter'
HELPER = Path('scripts/62_code_functional_eval.py')
spec = importlib.util.spec_from_file_location('effect_eval62', HELPER)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def prepare():
    assert not OUT.exists(), 'Checkpoint effect protocol already frozen'
    rows, targets, _ = e.load_inputs([586])
    assert rows[0]['task_id'] == 586 and targets[586]['split'] == 'calibration'
    OUT.mkdir()
    paths = [Path(__file__),HELPER,e.HELPER,e.POOL/'inputs.jsonl',e.POOL/'evaluator_targets.jsonl',ROOT/'functional_protocol.json']
    paths += [p for p in ADAPTER.iterdir() if p.is_file()]
    e.save(OUT/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),task_id=586,adapter=str(ADAPTER),
        model=str(e.MODEL),gpu=2,precision='FullFP32,TF32off',
        purpose='Applied-adapter sanity only: saved tensor identity, enabled unmerged adapters, nonzeroB, changed teacher-forced logits, exact unload restoration',
        reference='Frozen calibration586 code fenced exactly as prior referenceNLL study; no new accuracy endpoint',
        input_sha256={str(p):e.sha(p) for p in paths},api_spend_usd=0))
    print('Checkpoint effect check frozen; GPU run not launched')


def run():
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from peft.tuners.lora.layer import LoraLayer
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
    assert not (OUT/'results.json').exists(), 'Completed check immutable'
    p = json.loads((OUT/'protocol.json').read_text())
    for name,sha in p['input_sha256'].items():
        assert e.sha(Path(name)) == sha
    rows,targets,_ = e.load_inputs([586])
    torch.manual_seed(e.SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tok=AutoTokenizer.from_pretrained(e.MODEL,local_files_only=True)
    prompt=tok.apply_chat_template([{'role':'user','content':rows[0]['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=tok.encode(prompt,add_special_tokens=False)
    completion=tok.encode('```python\n'+targets[586]['code'].strip()+'\n```',add_special_tokens=False)+[tok.eos_token_id]
    tokens=torch.tensor([prefix+completion],device='cuda')
    e.save(OUT/'input.json',dict(task_id=586,prompt=prompt,prompt_token_ids=prefix,completion_token_ids=completion))
    model=AutoModelForCausalLM.from_pretrained(e.MODEL,dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    assert all(v.dtype==torch.float32 for v in model.parameters() if v.is_floating_point())
    def forward(m):
        with torch.inference_mode():
            logits=m(input_ids=tokens,use_cache=False).logits[:,len(prefix)-1:-1,:].float().cpu()
        nll=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,len(prefix):].cpu().reshape(-1),reduction='none')
        return logits,nll
    base,base_nll=forward(model)
    active=PeftModel.from_pretrained(model,str(ADAPTER),is_trainable=False).eval()
    saved=load_file(str(ADAPTER/'adapter_model.safetensors'))
    actual=get_peft_model_state_dict(active)
    assert actual.keys()==saved.keys() and all(torch.equal(v.detach().cpu(),saved[k]) for k,v in actual.items())
    bnorm=float(torch.sqrt(sum(v.square().sum() for k,v in saved.items() if '.lora_B.' in k)))
    assert bnorm>0
    layers=[m for m in active.modules() if isinstance(m,LoraLayer)]
    status=active.get_model_status()
    assert status.enabled is True and status.active_adapters==['default']
    assert len(layers)==2 and all(not m.disable_adapters and m.active_adapters==['default'] and not m.merged for m in layers)
    adapted,adapted_nll=forward(active)
    delta=adapted-base
    assert torch.isfinite(delta).all() and float(delta.abs().max())>0
    restored=model=active.unload()
    unloaded,unloaded_nll=forward(restored)
    assert torch.equal(base,unloaded) and torch.equal(base_nll,unloaded_nll)
    e.save(OUT/'per_token_nll.json',dict(base=base_nll.tolist(),adapter=adapted_nll.tolist(),unloaded=unloaded_nll.tolist()))
    e.save(OUT/'results.json',dict(task_id=586,completion_tokens=len(completion),saved_tensor_identity=True,
        lora_layers=len(layers),adapters_enabled=status.enabled,active_adapters=status.active_adapters,lora_B_norm=bnorm,
        max_logit_difference=float(delta.abs().max()),rms_logit_difference=float(delta.square().mean().sqrt()),
        base_mean_nll=float(base_nll.mean()),adapter_mean_nll=float(adapted_nll.mean()),
        mean_nll_decrease=float((base_nll-adapted_nll).mean()),unload_restores_exact_base_logits=True,
        purpose=p['purpose'],functional_accuracy_endpoint=False,measurement_or_final_used=False,api_spend_usd=0))
    e.save(OUT/'completed_manifest.json',dict(sha256={f.name:e.sha(f) for f in OUT.iterdir() if f.is_file()}))
    print((OUT/'results.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else run()
