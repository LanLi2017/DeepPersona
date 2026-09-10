#!/usr/bin/env python3
"""Numerical diagnostic after69 HVP probe failure; no regrouping outcomes or relaxed gates."""
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path('runs/swe-diversity-selection/code-regrouping-interaction')
OUT=ROOT/'fd-diagnostic'
INITIAL=Path('runs/swe-diversity-selection/code-update-transfer/initial_adapter.pt')
MODEL=Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
SEED=20260909
EPS=[.001,.003,.01,.03,.1,.3,1.0]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def run():
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForCausalLM
    from torch.nn.attention import sdpa_kernel,SDPBackend
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    assert not OUT.exists(),'Diagnostic outputs alreadyexist'
    OUT.mkdir()
    group=json.loads((ROOT/'groups.json').read_text())[0];keys=group['record_ids'];c1,c2,f1,f2=keys
    records={r['id']:r for line in (ROOT/'selected_raw.jsonl').open() if (r:=json.loads(line))['id'] in keys}
    sourcepaths=[Path('scripts/69_code_regrouping_interaction.py'),ROOT/'runner_snapshot.py',ROOT/'protocol.json',ROOT/'smoke_failure.json',ROOT/'groups.json',ROOT/'selected_raw.jsonl',INITIAL]
    protocol={'scope':'Diagnostic-only logging rerun following auxiliary HVP finite-difference failure; original gates unchanged, no finite regrouping or functional outcomes.',
              'task_id':group['task_id'],'record_ids':keys,'epsilon_sweep':EPS,'original_epsilons':[.01,.003],
              'original_gate':{'relative_l2':.05,'absolute_l2':1e-6},'precision':'FP32,TF32off,SDPA MATH',
              'checks':'OriginalHd*d; originalGaussian raw-Hessian probe; exact repeatedbasegradient; Hessian bilinear symmetry; FD actualparameterdirection error; firstandsecondscalar finite differences.',
              'source_sha256':{str(p):sha(p) for p in sourcepaths},'runner_sha256':sha(Path(__file__)),'api_spend_usd':0}
    save(OUT/'protocol.json',protocol);(OUT/'runner_snapshot.py').write_text(Path(__file__).read_text())
    save(OUT/'manifest.json',{'pid':os.getpid(),'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},'gpu':3,'api_spend_usd':0})
    torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').cuda();model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],layers_to_transform=[model.config.num_hidden_layers-1])).eval()
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for n,p in named]
    saved=torch.load(INITIAL,map_location='cpu',weights_only=True)
    with torch.no_grad():
        for n,p in named:p.copy_(saved[n])
    initial=torch.cat([p.detach().flatten() for p in params]).clone()
    def assign(vector):
        offset=0
        with torch.no_grad():
            for p in params:p.copy_(vector[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
    def nll(key):
        r=records[key];n=len(r['prompt_token_ids']);tokens=torch.tensor([r['prompt_token_ids']+r['generated_token_ids']],device='cuda')
        with sdpa_kernel(SDPBackend.MATH):logits=model(input_ids=tokens,use_cache=False).logits[:,n-1:-1,:]
        return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens[:,n:].reshape(-1),reduction='mean')
    def flat(values):return torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for p,v in zip(params,values)])
    def grad(key):return flat(torch.autograd.grad(nll(key),params,allow_unused=True)).detach()
    def weighted(weights):return sum(weight*grad(key) for key,weight in weights.items())
    def hvp(weights,direction):
        result=torch.zeros_like(direction)
        for key,weight in weights.items():
            first=torch.autograd.grad(nll(key),params,create_graph=True,allow_unused=True)
            dot=(flat(first)*direction.detach()).sum()
            result+=weight*flat(torch.autograd.grad(dot,params,allow_unused=True)).detach()
        return result
    def scalar(weights):
        with torch.no_grad():return sum(weight*float(nll(key)) for key,weight in weights.items())
    def norm(x):return float(torch.linalg.vector_norm(x.double()))
    def dot(a,b):return float(a.double()@b.double())
    initial_grads={k:grad(k) for k in keys}
    repeated=grad(c1)
    repeated_error=float(torch.max(torch.abs(repeated-initial_grads[c1])))
    weights={name:{k:count*(group['rewards'][k]-.5)/4 for k,count in Counter(occurrences).items()} for name,occurrences in group['groups'].items()}
    fixed={name:sum(w*initial_grads[k] for k,w in terms.items()) for name,terms in weights.items()}
    d=(fixed['U']-fixed['V'])/2;dweights={c1:.125,c2:-.125,f1:-.125,f2:.125}
    rng=np.random.default_rng(SEED)
    probe=torch.tensor(rng.normal(size=initial.numel()),device='cuda',dtype=torch.float32);probe=.01*probe/torch.linalg.vector_norm(probe)
    symmetry_vector=torch.tensor(rng.normal(size=initial.numel()),device='cuda',dtype=torch.float32);symmetry_vector=.01*symmetry_vector/torch.linalg.vector_norm(symmetry_vector)
    vectors={'initial':initial,'d':d,'probe':probe,**initial_grads};checks=[];started=time.time()
    for name,terms,direction in [('Hd_d',dweights,d),('raw_probe',{c1:1.0},probe)]:
        assign(initial);exact=hvp(terms,direction);basegrad=weighted(terms);baseloss=scalar(terms)
        vectors[name+'_exact']=exact
        for eps in EPS:
            plus_params=initial+eps*direction;minus_params=initial-eps*direction
            assign(plus_params);plus=weighted(terms);lossplus=scalar(terms)
            assign(minus_params);minus=weighted(terms);lossminus=scalar(terms)
            approximate=(plus-minus)/(2*eps)
            effective_direction=((plus_params.double()-minus_params.double())/(2*eps)).float()
            assign(initial);effective_hvp=hvp(terms,effective_direction)
            error=norm(approximate-exact);relative=error/max(norm(exact),1e-30)
            row={'component':name,'epsilon':eps,'direction_norm':norm(direction),'exact_hvp_norm':norm(exact),'fd_hvp_norm':norm(approximate),
                 'absolute_l2_error':error,'relative_l2_error':relative,'fd_exact_cosine':dot(approximate,exact)/max(norm(approximate)*norm(exact),1e-30),
                 'existing_gate_would_pass':relative<.05 or error<1e-6,
                 'parameter_direction_relative_error':norm(effective_direction-direction)/max(norm(direction),1e-30),
                 'zero_parameter_perturbation_fraction':float((plus_params==minus_params).float().mean()),
                 'effective_direction_hvp_relative_error':norm(approximate-effective_hvp)/max(norm(effective_hvp),1e-30),
                 'gradient_difference_norm':norm(plus-minus),'basegradient_norm':norm(basegrad),
                 'scalar_first_fd':(lossplus-lossminus)/(2*eps),'scalar_first_autograd':dot(basegrad,direction),
                 'scalar_second_fd':(lossplus-2*baseloss+lossminus)/(eps*eps),'scalar_second_autograd':dot(direction,exact)}
            checks.append(row)
            with (OUT/'checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            vectors[f'{name}:fd:{eps}']=approximate
            print(json.dumps({k:row[k] for k in ['component','epsilon','exact_hvp_norm','absolute_l2_error','relative_l2_error','parameter_direction_relative_error','existing_gate_would_pass']}),flush=True)
    assign(initial)
    hp=hvp({c1:1.0},probe);hs=hvp({c1:1.0},symmetry_vector)
    left,right=dot(symmetry_vector,hp),dot(probe,hs)
    symmetry={'left':left,'right':right,'absolute_error':abs(left-right),'relative_error':abs(left-right)/max(abs(left),abs(right),1e-30)}
    summary={'diagnostic_complete':True,'original_gate_unchanged':True,'repeated_basegradient_max_absolute_error':repeated_error,'hessian_symmetry':symmetry,
             'original_epsilon_checks':[r for r in checks if r['epsilon'] in [.01,.003]],
             'all_epsilon_error_summary':[{k:r[k] for k in ['component','epsilon','relative_l2_error','absolute_l2_error','existing_gate_would_pass']} for r in checks],
             'seconds':time.time()-started,'full_regrouping_outcomes_run':False,'api_spend_usd':0}
    np.savez(OUT/'vectors.npz',ids=np.array(list(vectors)),vectors=torch.stack(list(vectors.values())).cpu().numpy())
    save(OUT/'summary.json',summary)
    assert all(sha(Path(p))==h for p,h in protocol['source_sha256'].items()),'Original sources changed'
    save(OUT/'completed_manifest.json',{'original_sources_unchanged':True,'sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'done':True,'repeated_gradient_error':repeated_error,'symmetry':symmetry,'seconds':summary['seconds']}),flush=True)


if __name__=='__main__':run()
