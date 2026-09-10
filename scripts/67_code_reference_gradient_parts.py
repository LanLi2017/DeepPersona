#!/usr/bin/env python3
"""Exploratory calibration-reference gradient decomposition, body vs literal scaffold."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

POOL=Path('runs/swe-diversity-selection/code-learning-pilot')
SOURCE=Path('runs/swe-diversity-selection/code-update-transfer-fp32')
INITIAL=Path('runs/swe-diversity-selection/code-update-transfer/initial_adapter.pt')
OUT=Path('runs/swe-diversity-selection/code-reference-gradient-parts')
MODEL=Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
SEED=20260909


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def calibration_rows(path):
    return [json.loads(line) for line in path.open() if line.startswith('{"split": "calibration",')]


def source_calibration(ids):
    with zipfile.ZipFile(SOURCE/'gradients.npz') as z:
        with z.open('ids.npy') as f: keys=np.load(f,allow_pickle=False).tolist()
        assert all(key.startswith('calibration:') for key in ids)
        with z.open('gradients.npy') as f:
            assert np.lib.format.read_magic(f)==(1,0)
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            assert not fortran and len(shape)==2
            offset=f.tell();result=[]
            for key in ids:
                f.seek(offset+keys.index(key)*shape[1]*dtype.itemsize)
                result.append(np.frombuffer(f.read(shape[1]*dtype.itemsize),dtype=dtype).copy())
    return np.stack(result)


def prepare():
    from transformers import AutoTokenizer
    assert not (OUT/'protocol.json').exists(),'Already frozen'
    OUT.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    inputs={r['task_id']:r for r in calibration_rows(POOL/'inputs.jsonl')}
    targets=sorted(calibration_rows(POOL/'evaluator_targets.jsonl'),key=lambda r:r['task_id'])
    assert len(targets)==len(inputs)==21
    records=[]
    for row in targets:
        body=row['code'].strip();prefix='```python\n';text=prefix+body+'\n```'
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        completion=encoded['input_ids']+[tokenizer.eos_token_id]
        offsets=[list(x) for x in encoded['offset_mapping']]+[[0,0]]
        start,end=len(prefix),len(prefix)+len(body)
        body_mask=[int(b>start and a<end) for a,b in offsets[:-1]]+[0]
        assert 0<sum(body_mask)<len(completion)
        prompt=tokenizer.apply_chat_template([{'role':'user','content':inputs[row['task_id']]['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompt_tokens=tokenizer.encode(prompt,add_special_tokens=False)
        assert len(prompt_tokens)+len(completion)<=2048
        records.append({'id':f'calibration:{row["task_id"]}','task_id':row['task_id'],
                        'prompt_token_ids':prompt_tokens,'completion_token_ids':completion,
                        'completion_text':text,'token_offsets':offsets,'body_character_interval':[start,end],
                        'body_mask':body_mask,'scaffold_mask':[1-x for x in body_mask]})
    (OUT/'token_parts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    original=source_calibration([r['id'] for r in records])
    np.savez(OUT/'original_calibration_gradients.npz',ids=np.array([r['id'] for r in records]),gradients=original)
    save(OUT/'protocol.json',{'seed':SEED,'model_snapshot':str(MODEL),'gpu':3,'dtype':'float32','tf32':False,
         'adapter':{'last_layer_only':True,'modules':['q_proj','v_proj'],'rank':8,'alpha':16,'dropout':0,'initial':str(INITIAL)},
         'diagnostic':'Exploratory after an observed approximately.98 calibration-measurement mean-gradient cosine; not a preregistered confirmation, no metric tuning or functional outcomes.',
         'token_rule':'Any tokenizer offset overlapping reference code.strip() is body, including boundary-spanning tokens. Prefix```python/newline, suffix newline/fence, and EOS otherwise scaffold. Body is not asserted semantic.',
         'loss_rule':'Each component is SUM selected tokenNLL / TOTAL completion tokens; body+scaffold must recover original whole meanNLL gradient.',
         'task_ids':[r['task_id'] for r in records],'smoke_task_ids':[r['task_id'] for r in records[:2]],
         'numeric_gate':{'max_relative_l2_error':1e-5,'max_absolute_error':1e-6},
         'source_access':'Calibration JSON rows only decoded; only calibration numeric rows selectively read from source npz. Measurement IDs/header may be seen; no measurement-reference or gradient values used.',
         'input_sha256':{str(p):sha(p) for p in [INITIAL,OUT/'token_parts.jsonl',OUT/'original_calibration_gradients.npz',SOURCE/'gradients.npz']},
         'script_sha256':sha(Path(__file__)),'api_spend_usd':0,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    (OUT/'runner_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({'frozen_tasks':21,'smoke':[r['task_id'] for r in records[:2]],'calibration_only':True}),flush=True)


def run(smoke):
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForCausalLM
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==protocol['script_sha256']
    for path,digest in protocol['input_sha256'].items():assert sha(Path(path))==digest
    if not smoke:assert json.loads((OUT/'smoke_summary.json').read_text())['technical_checks_passed']
    prefix='smoke_' if smoke else ''
    assert not (OUT/f'{prefix}summary.json').exists(),'Run already complete'
    records=[json.loads(l) for l in (OUT/'token_parts.jsonl').open()][:2 if smoke else 21]
    source=np.load(OUT/'original_calibration_gradients.npz');original={k:g for k,g in zip(source['ids'].tolist(),source['gradients'])}
    torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    save(OUT/f'{prefix}manifest.json',{'protocol_sha256':sha(OUT/'protocol.json'),'pid':os.getpid(),'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},'task_ids':[r['task_id'] for r in records],'api_spend_usd':0})
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').cuda()
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],layers_to_transform=[model.config.num_hidden_layers-1])).eval()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    fixed=torch.load(INITIAL,map_location='cpu',weights_only=True)
    assert set(fixed)=={n for n,p in params}
    with torch.no_grad():
        for name,p in params:p.copy_(fixed[name])
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point())
    parts={k:[] for k in ['whole','body','scaffold']};rows=[];started=time.time()
    for index,r in enumerate(records):
        tokens=torch.tensor([r['prompt_token_ids']+r['completion_token_ids']],device='cuda')
        start=len(r['prompt_token_ids']);count=len(r['completion_token_ids'])
        logits=model(input_ids=tokens,use_cache=False).logits[:,start-1:-1,:].float()
        nll=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,start:].reshape(-1),reduction='none')
        values={}
        for component in ['whole','body','scaffold']:
            model.zero_grad(set_to_none=True)
            mask=torch.ones(count,device='cuda') if component=='whole' else torch.tensor(r[component+'_mask'],device='cuda')
            loss=(nll*mask).sum()/count
            loss.backward(retain_graph=component!='scaffold')
            gradient=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten() for _,p in params]).cpu().numpy().copy()
            assert np.isfinite(gradient).all()
            parts[component].append(gradient);values[component]=float(loss.detach())
        whole,body,scaffold=[parts[k][-1] for k in ['whole','body','scaffold']]
        delta=body+scaffold-whole;source_delta=whole-original[r['id']]
        checks={'sum_relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(whole),1e-30)),
                'sum_max_absolute':float(np.max(np.abs(delta))),
                'original_relative_l2':float(np.linalg.norm(source_delta)/max(np.linalg.norm(original[r['id']]),1e-30)),
                'original_max_absolute':float(np.max(np.abs(source_delta)))}
        row={'id':r['id'],'tokens':count,'body_tokens':sum(r['body_mask']),'loss_parts':values,'norms':{k:float(np.linalg.norm(parts[k][-1])) for k in parts},**checks}
        rows.append(row)
        with (OUT/f'{prefix}per_task.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        limit=protocol['numeric_gate']
        assert max(checks['sum_relative_l2'],checks['original_relative_l2'])<limit['max_relative_l2_error']
        assert max(checks['sum_max_absolute'],checks['original_max_absolute'])<limit['max_absolute_error']
        print(json.dumps({'tasks':index+1,'total':len(records),'seconds':round(time.time()-started,2),'sum_rel':checks['sum_relative_l2'],'original_rel':checks['original_relative_l2']}),flush=True)
    arrays={k:np.stack(v) for k,v in parts.items()}
    np.savez(OUT/f'{prefix}component_gradients.npz',ids=np.array([r['id'] for r in records]),parameter_names=np.array([n for n,p in params]),**arrays)
    means={k:v.mean(axis=0,dtype=np.float64) for k,v in arrays.items()}
    def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
    mean_whole=means['whole'];denom=float(mean_whole@mean_whole)
    summary={'technical_checks_passed':True,'tasks':len(records),'mean_gradient_norms':{k:float(np.linalg.norm(v)) for k,v in means.items()},
             'mean_gradient_cosines':{'body_whole':cosine(means['body'],mean_whole),'scaffold_whole':cosine(means['scaffold'],mean_whole),'body_scaffold':cosine(means['body'],means['scaffold'])},
             'projection_contribution_to_mean_whole':{k:float(means[k]@mean_whole/denom) for k in ['body','scaffold']},
             'total_tokens':sum(r['tokens'] for r in rows),'body_tokens':sum(r['body_tokens'] for r in rows),
             'max_additive_relative_error':max(r['sum_relative_l2'] for r in rows),'max_original_relative_error':max(r['original_relative_l2'] for r in rows),
             'interpretation':'Projection shares sum to1 up to numerical error and can be negative; they are not causal importance. Body tokens are not established semantics. Calibration-only exploratory diagnostic, not a calibration-measurement decomposition.',
             'measurement_reference_or_gradient_values_used':False,'functional_outcomes_used':False,'api_spend_usd':0}
    save(OUT/f'{prefix}summary.json',summary);print(json.dumps(summary),flush=True)
    save(OUT/f'{prefix}completed_manifest.json',{'sha256':{p.name:sha(p) for p in [OUT/'protocol.json',OUT/'token_parts.jsonl',OUT/f'{prefix}manifest.json',OUT/f'{prefix}component_gradients.npz',OUT/f'{prefix}per_task.jsonl',OUT/f'{prefix}summary.json']},'api_spend_usd':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    prepare() if args.stage=='prepare' else run(args.smoke)
