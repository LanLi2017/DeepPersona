#!/usr/bin/env python3
"""Separately frozen FIRST-block reward regrouping, fixed960original rollouts."""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path('runs/swe-diversity-selection')
POOL=ROOT/'code-learning-pilot'
EXPANSION=ROOT/'code-learning-expansion'
OUT=ROOT/'code-regrouping-first-block'
AUDIT=ROOT/'code-extraction-audit'
INITIAL=ROOT/'code-update-transfer/initial_adapter.pt'
MODEL=Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
SEED=20260909
PATHS={'mixed':['M','M'],'uv':['U','V'],'vu':['V','U']}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def read(p):return [json.loads(l) for l in p.open()]


def prepare():
    from transformers import AutoTokenizer
    assert not (OUT/'protocol.json').exists(),'Already frozen'
    raw={f'{r["task_id"]}:{r["sample"]}':r for d in [POOL,EXPANSION] for r in read(d/'raw.jsonl')}
    labels={r['record_id'].removeprefix('train:'):r['first_correct'] for line in (AUDIT/'outcomes.jsonl').open() if '"split": "train"' in line and (r:=json.loads(line))['split']=='train'}
    assert len(labels)==960 and set(labels)==set(raw)
    by=defaultdict(list)
    for key,r in raw.items():by[r['task_id']].append(key)
    tasks=[]
    for task,keys in by.items():
        correct=sorted([k for k in keys if labels[k]],key=lambda k:digest(f'{SEED}:regroup-choice:{task}:{k}'))
        failed=sorted([k for k in keys if not labels[k]],key=lambda k:digest(f'{SEED}:regroup-choice:{task}:{k}'))
        if min(len(correct),len(failed))<2:continue
        c1,c2=correct[:2];f1,f2=failed[:2]
        groups={'U':[c1,c1,f1,f1],'V':[c2,c2,f2,f2],'M':[c1,c2,f1,f2]}
        multisets={name:Counter(k for group in path for k in groups[group]) for name,path in PATHS.items()}
        assert multisets['mixed']==multisets['uv']==multisets['vu']==Counter({k:2 for k in [c1,c2,f1,f2]})
        assert all(sum(labels[k] for k in group)==2 for group in groups.values())
        tokens={name:sum(len(raw[k]['generated_token_ids']) for group in path for k in groups[group]) for name,path in PATHS.items()}
        assert len(set(tokens.values()))==1
        unique=[c1,c2,f1,f2]
        tasks.append({'task_id':task,'record_ids':unique,'groups':groups,'rewards':{k:int(labels[k]) for k in unique},
                      'occurrence_tokens_per_path':tokens,'same_token_content_pairs':[[a,b] for i,a in enumerate(unique) for b in unique[i+1:] if raw[a]['generated_token_ids']==raw[b]['generated_token_ids']]})
    tasks.sort(key=lambda r:digest(f'{SEED}:regroup-task:{r["task_id"]}'))
    assert len(tasks)==17
    selected=sorted({k for t in tasks for k in t['record_ids']})
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    inputs={r['task_id']:r for r in read(POOL/'inputs.jsonl')}
    refs=[]
    for row in sorted(read(POOL/'evaluator_targets.jsonl'),key=lambda r:r['task_id']):
        if row['split'] not in ['calibration','measurement']:continue
        prompt=tokenizer.apply_chat_template([{'role':'user','content':inputs[row['task_id']]['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        completion=tokenizer.encode('```python\n'+row['code'].strip()+'\n```',add_special_tokens=False)+[tokenizer.eos_token_id]
        refs.append({'id':f'{row["split"]}:{row["task_id"]}','task_id':row['task_id'],'split':row['split'],
                     'prompt_token_ids':tokenizer.encode(prompt,add_special_tokens=False),'generated_token_ids':completion})
    assert Counter(r['split'] for r in refs)=={'calibration':21,'measurement':21}
    assert not {t['task_id'] for t in tasks}&{r['task_id'] for r in refs}
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'groups.json',tasks)
    for file,records in [('selected_raw.jsonl',[{'id':k,**raw[k]} for k in selected]),('reference_tokens.jsonl',refs)]:
        (OUT/file).write_text(''.join(json.dumps(r)+'\n' for r in records))
    sources=[INITIAL,AUDIT/'outcomes.jsonl',OUT/'groups.json',OUT/'selected_raw.jsonl',OUT/'reference_tokens.jsonl']+[d/f for d in [POOL,EXPANSION] for f in ['protocol.json','raw.jsonl','grades.jsonl']]
    protocol={'seed':SEED,'scope':'Exploratory exact-multiset regrouping after prior method-development outcomes; not confirmation, functional gain, or novelty claim.',
              'model_snapshot':str(MODEL),'gpu':3,'dtype':'float32','tf32':False,'attention':'SDPA MATH forced for every forward and Hessian product',
              'adapter':{'last_layer_only':True,'modules':['q_proj','v_proj'],'rank':8,'alpha':16,'dropout':0,'initial':str(INITIAL)},
              'task_ids':[r['task_id'] for r in tasks],'selection':'All17FIRST-block-eligible tasks; four distinct recordIDs chosen by SHA(seed:regroup-choice:task:id), independent of metrics. Duplicate contents retained.',
              'paths':PATHS,'etas':[1.0,10.0],'objective':'J_group=mean_occurrences((reward-.5)*mean_completion_token_NLL); plainSGD, second gradient recomputed, no clipping/momentum/decay.',
              'score':'u=gradJU,v=gradJV,d=(u-v)/2; Hd=Hessian((JU-JV)/2); B=-mean_calibration_gradient dot(Hd*d); predicted mixed loss-decrease advantage eta^2*B.',
              'diversity':'Mean cosine distance over6unordered occurrencepairs; contrast D(M)-(D(U)+D(V))/2=||(unitgc1+unitgf1)-(unitgc2+unitgf2)||^2/12. Exact zero-norm gives nulldiversity, retains outcomes.',
              'target':'Mean actual measurement21 referenceNLL; compare mean(UVloss,VUloss) minus Mloss, never loss ofmean parameters. No measurement gradients asfeatures; no functionalgrid outcomes read.',
              'smoke':{'task_ids':[tasks[0]['task_id']],'calibration_ids':[r['task_id'] for r in refs if r['split']=='calibration'][:2],'measurement_ids':[r['task_id'] for r in refs if r['split']=='measurement'][:2]},
              'finite_difference':{'epsilons':[.1,.3],'relative_gate':.05,'absolute_gate':1e-6,'probe':'Hd*d plus firstcorrect-trajectory Hessian times fixedseed Gaussian direction scaledL2 .01; probe alwayscomputed, not afeature.'},
              'nulls':'Exact reset repeatedbaseline; eta0 allpaths; frozen initialgradients physically applied intwo steps, record FP32 endpoint/loss discrepancies.',
              'numeric_gates':{'reset_max_absolute':0.0,'zero_step_loss_absolute':1e-7,'frozen_gradient_endpoint_absolute':1e-6,'diversity_identity_absolute':1e-10},
              'full_requires_root_smoke_review':True,'source_sha256':{str(p):sha(p) for p in sources},'runner_sha256':sha(Path(__file__)),'api_spend_usd':0,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    protocol['numerical_amendment']={'old_fd_epsilons':[.01,.003],'new_fd_epsilons':[.1,.3],
        'unchanged_relative_gate':.05,'unchanged_absolute_gate':1e-6,
        'rationale':'Root reviewed diagnostic71 showing FP32 finite-difference resolution failure at original auxiliary steps; only FD steps/logging/output path amended, before any finite-path outcomes.',
        'original_gate_remains_failed':True,
        'evidence_sha256':{str(p):sha(p) for p in [Path('scripts/69_code_regrouping_interaction.py'),ROOT/'code-regrouping-interaction/protocol.json',ROOT/'code-regrouping-interaction/smoke_failure.json',Path('scripts/71_code_hvp_fd_diagnostic.py'),ROOT/'code-regrouping-interaction/fd-diagnostic/protocol.json',ROOT/'code-regrouping-interaction/fd-diagnostic/summary.json']}}
    protocol['reward_policy_amendment']={'primary_reward':'FIRST fenced-block execution as operationally audited by75; fallback/extraction details frozen in75 code and audit inputs.',
        'reason':'75 found23of88selected oldrecords change fail-to-pass underFIRST, breaking balance in13of22oldselected groups. Oldreward remains recorded asLAST-block execution, not relabeled.',
        'prior22_study_status':'Technicalsmokeonly, nofull22taskrun; original69failureand72technicalsmoke preserved.',
        'fixed_population':'Same960rawrollouts on120train tasks, no resampling. All17eligible underFIRST, four IDs SHAselected afresh withsameformula.',
        'evidence_sha256':{str(p):sha(p) for p in [Path('scripts/72_code_regrouping_interaction_v2.py'),ROOT/'code-regrouping-interaction-v2/protocol.json',ROOT/'code-regrouping-interaction-v2/reward_extraction_impact.json',AUDIT/'outcomes.jsonl',AUDIT/'inputs.jsonl']}}
    save(OUT/'protocol.json',protocol);(OUT/'runner_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps({'tasks':17,'smoke_task':tasks[0]['task_id'],'selected_records':len(selected),'duplicate_content_tasks':sum(bool(t['same_token_content_pairs']) for t in tasks)}),flush=True)


def run(smoke,approved):
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForCausalLM
    from torch.nn.attention import sdpa_kernel,SDPBackend
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==protocol['runner_sha256']
    for p,h in protocol['source_sha256'].items():assert sha(Path(p))==h
    if not smoke:
        assert approved,'Root smoke review required beforefull'
        assert json.loads((OUT/'smoke_summary.json').read_text())['technical_checks_passed']
    prefix='smoke_' if smoke else ''
    assert not (OUT/f'{prefix}summary.json').exists()
    groups=json.loads((OUT/'groups.json').read_text())[:1 if smoke else 17]
    raw={r['id']:r for r in read(OUT/'selected_raw.jsonl')}
    refs=read(OUT/'reference_tokens.jsonl')
    cal=[r for r in refs if r['split']=='calibration'][:2 if smoke else 21]
    measurement=[r for r in refs if r['split']=='measurement'][:2 if smoke else 21]
    torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    save(OUT/f'{prefix}manifest.json',{'pid':os.getpid(),'protocol_sha256':sha(OUT/'protocol.json'),'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},'tasks':[g['task_id'] for g in groups],'calibration_ids':[r['id'] for r in cal],'measurement_ids':[r['id'] for r in measurement],'full_root_review_flag':approved,'api_spend_usd':0})
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').cuda();model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],layers_to_transform=[model.config.num_hidden_layers-1])).eval()
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for n,p in named]
    initial_dict=torch.load(INITIAL,map_location='cpu',weights_only=True)
    assert set(initial_dict)=={n for n,p in named}
    with torch.no_grad():
        for n,p in named:p.copy_(initial_dict[n])
    initial=torch.cat([p.detach().flatten() for p in params]).clone()
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point())
    records={**raw,**{r['id']:r for r in cal+measurement}}
    def assign(vector):
        offset=0
        with torch.no_grad():
            for p in params:p.copy_(vector[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
    def current():return torch.cat([p.detach().flatten() for p in params]).clone()
    def nll(key):
        r=records[key];n=len(r['prompt_token_ids']);tokens=torch.tensor([r['prompt_token_ids']+r['generated_token_ids']],device='cuda')
        assert tokens.shape[1]<=2048
        with sdpa_kernel(SDPBackend.MATH):logits=model(input_ids=tokens,use_cache=False).logits[:,n-1:-1,:]
        return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,n:].reshape(-1),reduction='mean')
    def flat(values):return torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for p,v in zip(params,values)])
    def grad(key):return flat(torch.autograd.grad(nll(key),params,allow_unused=True)).detach()
    def weighted_gradient(weights):return sum(weight*grad(key) for key,weight in weights.items())
    def hvp(weights,direction):
        result=torch.zeros_like(direction)
        for key,weight in weights.items():
            first=torch.autograd.grad(nll(key),params,create_graph=True,allow_unused=True)
            dot=(flat(first)*direction.detach()).sum()
            second=flat(torch.autograd.grad(dot,params,allow_unused=True)).detach()
            result+=weight*second
        return result
    def measurement_losses():
        with torch.no_grad():return [float(nll(r['id'])) for r in measurement]
    def finite_difference(weights,direction,exact):
        checks=[]
        for eps in protocol['finite_difference']['epsilons']:
            assign(initial+eps*direction);plus=weighted_gradient(weights)
            assign(initial-eps*direction);minus=weighted_gradient(weights)
            approximate=(plus-minus)/(2*eps);error=torch.linalg.vector_norm(approximate-exact).item();scale=torch.linalg.vector_norm(exact).item()
            checks.append({'epsilon':eps,'absolute_l2':error,'relative_l2':error/max(scale,1e-30),'exact_hvp_norm':scale})
            with (OUT/f'{prefix}finite_difference.jsonl').open('a') as f:
                f.write(json.dumps({'weights':weights,'direction_norm':float(torch.linalg.vector_norm(direction)),**checks[-1]})+'\n');f.flush()
        assign(initial)
        gate=protocol['finite_difference']
        assert any(c['relative_l2']<gate['relative_gate'] or c['absolute_l2']<gate['absolute_gate'] for c in checks),'HVP finite difference failed'
        return checks
    started=time.time();assign(initial)
    cal_grads=[grad(r['id']) for r in cal];cal_mean=torch.stack(cal_grads).mean(0)
    np.savez(OUT/f'{prefix}calibration_gradients.npz',ids=np.array([r['id'] for r in cal]),gradients=torch.stack(cal_grads).cpu().numpy())
    baseline=measurement_losses();assert baseline==measurement_losses()
    rows=[];vectors={}
    for index,g in enumerate(groups):
        assign(initial);assert torch.equal(current(),initial)
        keys=g['record_ids'];c1,c2,f1,f2=keys
        raw_grads={k:grad(k) for k in keys}
        weights={name:{k:count*(g['rewards'][k]-.5)/4 for k,count in Counter(occurrences).items()} for name,occurrences in g['groups'].items()}
        fixed={name:sum(w*raw_grads[k] for k,w in coeffs.items()) for name,coeffs in weights.items()}
        d=(fixed['U']-fixed['V'])/2
        dweights={c1:.125,c2:-.125,f1:-.125,f2:.125}
        hessian_d=hvp(dweights,d)
        b=-float(cal_mean.double()@hessian_d.double())
        fd=finite_difference(dweights,d,hessian_d) if smoke else None
        if smoke:
            rng=np.random.default_rng(SEED);probe=torch.tensor(rng.normal(size=initial.numel()),device='cuda',dtype=torch.float32);probe=.01*probe/torch.linalg.vector_norm(probe)
            probe_hvp=hvp({c1:1.0},probe);probe_fd=finite_difference({c1:1.0},probe,probe_hvp)
        else:probe_fd=None
        grads={k:v.double().cpu().numpy() for k,v in raw_grads.items()};norms={k:float(np.linalg.norm(v)) for k,v in grads.items()}
        diversity=None
        if all(n>0 for n in norms.values()):
            units={k:v/norms[k] for k,v in grads.items()}
            div={name:float(np.mean([1-units[a]@units[b] for i,a in enumerate(occurrences) for b in occurrences[i+1:]])) for name,occurrences in g['groups'].items()}
            contrast=div['M']-(div['U']+div['V'])/2
            formula=float(np.sum((units[c1]+units[f1]-units[c2]-units[f2])**2)/12)
            assert abs(contrast-formula)<protocol['numeric_gates']['diversity_identity_absolute']
            diversity={'by_group':div,'contrast':contrast,'identity':formula}
        results={};frozen={}
        for eta in [0.0]+protocol['etas']:
            losses={};ends={};step_norms={}
            for name,path in PATHS.items():
                assign(initial);updates=[]
                for group in path:
                    gradient=weighted_gradient(weights[group]);updates.append(float(torch.linalg.vector_norm(eta*gradient)));assign(current()-eta*gradient)
                endpoint=current();assert torch.isfinite(endpoint).all()
                ends[name]=endpoint;step_norms[name]=updates;losses[name]=measurement_losses();assert np.isfinite(losses[name]).all()
            contrast=float(np.mean((np.array(losses['uv'])+np.array(losses['vu']))/2-np.array(losses['mixed'])))
            results[str(eta)]={'measurement_losses':losses,'mixed_advantage':contrast,'prediction':eta*eta*b,'residual':contrast-eta*eta*b,'step_update_norms':step_norms,
                               'endpoint_mixed_minus_ordermean_norm':float(torch.linalg.vector_norm(ends['mixed']-(ends['uv']+ends['vu'])/2))}
            if eta==0:
                assert max(abs(x-y) for values in losses.values() for x,y in zip(values,baseline))<=protocol['numeric_gates']['zero_step_loss_absolute']
                continue
            fends={};floss={}
            for name,path in PATHS.items():
                assign(initial)
                for group in path:assign(current()-eta*fixed[group])
                fends[name]=current();floss[name]=measurement_losses()
            gap=max(float(torch.max(torch.abs(fends['mixed']-fends[k]))) for k in ['uv','vu'])
            assert gap<protocol['numeric_gates']['frozen_gradient_endpoint_absolute']
            frozen[str(eta)]={'max_endpoint_absolute_gap':gap,'measurement_losses':floss,'mixed_advantage':float(np.mean((np.array(floss['uv'])+np.array(floss['vu']))/2-np.array(floss['mixed'])))}
        assign(initial);assert torch.equal(current(),initial) and measurement_losses()==baseline
        row={'task_id':g['task_id'],'score_B':b,'d_norm':float(torch.linalg.vector_norm(d)),'Hd_d_norm':float(torch.linalg.vector_norm(hessian_d)),
             'raw_gradient_norms':norms,'diversity':diversity,'finite_difference':fd,'independent_probe_fd':probe_fd,'results':results,'frozen_gradient_null':frozen,
             'same_token_content_pairs':g['same_token_content_pairs'],'occurrence_tokens_per_path':g['occurrence_tokens_per_path']}
        rows.append(row)
        with (OUT/f'{prefix}per_task.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        for key,value in {**raw_grads,'d':d,'Hd_d':hessian_d}.items():vectors[f'{g["task_id"]}:{key}']=value.cpu().numpy()
        np.savez(OUT/f'{prefix}vectors.npz',ids=np.array(list(vectors)),vectors=np.stack(list(vectors.values())))
        print(json.dumps({'tasks':index+1,'total':len(groups),'seconds':round(time.time()-started,2),'task_id':g['task_id'],'B':b,'advantages':{eta:r['mixed_advantage'] for eta,r in results.items()}}),flush=True)
    summary={'technical_checks_passed':True,'tasks':len(rows),'calibration_tasks':len(cal),'measurement_tasks':len(measurement),'measurement_baseline':baseline,
             'task_ids':[r['task_id'] for r in rows],'etas':protocol['etas'],'root_review_required_before_full':smoke,'api_spend_usd':0,
             'interpretation':'Exploratory two-step fixed-pool SGD reference-NLL mechanism; identical data multiset, not functional accuracy, semantic diversity validity, or novelty.'}
    save(OUT/f'{prefix}summary.json',summary)
    save(OUT/f'{prefix}completed_manifest.json',{'sha256':{p.name:sha(p) for p in [OUT/'protocol.json',OUT/f'{prefix}manifest.json',OUT/f'{prefix}calibration_gradients.npz',OUT/f'{prefix}vectors.npz',OUT/f'{prefix}per_task.jsonl',OUT/f'{prefix}summary.json']}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--smoke',action='store_true');parser.add_argument('--full-approved',action='store_true');args=parser.parse_args()
    try:prepare() if args.stage=='prepare' else run(args.smoke,args.full_approved)
    except Exception as error:
        if OUT.exists():save(OUT/('smoke_failure.json' if args.smoke else 'failure.json'),{'error_type':type(error).__name__,'message':str(error),'fallback_used':False})
        raise
