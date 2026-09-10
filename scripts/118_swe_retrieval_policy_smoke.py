#!/usr/bin/env python3
"""117-selected source policy → unchanged115technical smoke; no fabricated114files."""
import argparse
import importlib.metadata
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'runs/swe-diversity-selection/swe-sympy-retrieval-policies'
OLD=ROOT/'runs/swe-diversity-selection/swe-sympy-patch-sft'
OUT=STUDY/'learner_smoke'
KERNEL=ROOT/'scripts/115_swe_sympy_long_context_smoke.py'
spec=importlib.util.spec_from_file_location('kernel115',KERNEL)
k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
h=k.h
POLICIES=['production_chunks','production_files']
CAP=24576


def freeze():
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as qwen
    import transformers.integrations.sdpa_attention as sdpa
    import peft.tuners.lora.layer as lora
    import peft.utils.save_and_load as peft_io
    import torch.optim.adamw as adamw
    assert not OUT.exists(), 'Single prospective smoke; no overwrite'
    summary=h.read(STUDY/'score_summary.json')
    assert summary['technical_failures']==0 and summary['minimum_required']==16 and summary['prompt_cap']==CAP
    assert summary['evaluation_new_contexts']==0 and summary['evaluation_private_values_read'] is False
    scores={r['policy']:r for r in summary['policies']}
    assert set(scores)==set(POLICIES)
    winner=min(POLICIES,key=lambda name:(-scores[name]['prequalified'],POLICIES.index(name)))
    assert scores[winner]['prequalified']>=16 and summary['chosen_policy']==winner
    ids=[r['instance_id'] for r in scores[winner]['tasks'] if r['prequalified']]
    assert len(ids)==len(set(ids))==scores[winner]['prequalified']
    selection=h.read(OLD/'selection.json')
    sources=[r['instance_id'] for r in selection['tasks'] if r['role']=='source']
    assert len(sources)==32 and set(ids)<=set(sources)
    protocol=h.read(STUDY/'protocol.json')
    assert protocol['prompt_cap']==CAP and protocol['completion_cap_including_eos']==2048
    assert protocol['policies']==POLICIES
    assert {r['instance_id'] for r in protocol['source_tasks']}==set(sources)
    paths=[Path(__file__),KERNEL,k.HELPER,STUDY/'protocol.json',STUDY/'score_summary.json',
           STUDY/'completed_manifest.json',OLD/'selection.json',OLD/'source_targets/completed_manifest.json']
    for name,digest in h.read(STUDY/'completed_manifest.json').items():
        assert h.sha(h.resolve(name))==digest,name;paths.append(h.resolve(name))
    build=h.read(STUDY/'build_manifest.json')
    assert len(build['tasks'])==build['assigned_slots']==64 and build['private_inputs_read'] is False
    assert {(r['instance_id'],r['policy']) for r in build['tasks']}=={(i,p) for i in sources for p in POLICIES}
    assert build['protocol_sha256']==h.sha(STUDY/'protocol.json')
    for name,digest in build['artifact_sha256'].items():
        assert h.sha(h.resolve(name))==digest,name;paths.append(h.resolve(name))
    for pp,field in [(STUDY/'protocol.json','source_sha256'),(STUDY/'score_protocol.json','inputs_sha256'),
                     (OLD/'context/protocol.json','source_sha256'),(OLD/'source_targets/protocol.json','inputs_sha256')]:
        paths.append(pp)
        for name,digest in h.read(pp)[field].items():
            assert h.sha(h.resolve(name))==digest,name;paths.append(h.resolve(name))
    tok=AutoTokenizer.from_pretrained(str(k.MODEL),local_files_only=True)
    old_targets={str(h.resolve(n)):v for n,v in h.read(OLD/'source_targets/completed_manifest.json').items()}
    records=[]
    for iid in sources:
        if iid not in ids:continue
        cp=STUDY/winner/f'{iid}.json';tp=OLD/'source_targets'/iid/'record.json'
        c,t=h.read(cp),h.read(tp);assert h.sha(tp)==old_targets[str(tp)]
        assert c['instance_id']==t['instance_id']==iid and c['role']=='source' and c['status']=='ready'
        assert c['policy']==winner and c['protocol_sha256']==h.sha(STUDY/'protocol.json')
        assert h.sha(STUDY/winner/f'{iid}.prompt.txt')==c['prompt_sha256']
        assert h.sha(STUDY/winner/f'{iid}.packet.txt')==c['packet_sha256']
        assert t['conversion_supported'] and t['independent_git_reconstruction'] and t['completion_fits']
        assert 'technical_failure' not in t
        prompt=k.native_prompt_ids(tok,c['messages'])
        target=tok.encode(t['target'],add_special_tokens=False)+[tok.eos_token_id]
        assert prompt==c['prompt_token_ids'] and 0<len(prompt)<=CAP
        assert target==t['target_token_ids'] and 0<len(target)<=2048
        assert target[-1]==t['eos_token_id']==tok.eos_token_id
        paths += [cp,tp]
        records.append(dict(instance_id=iid,prompt_token_ids=prompt,target_token_ids=target,
                            context_record_sha256=h.sha(cp),target_record_sha256=h.sha(tp),quality_verified=False))
    assert len(records)==len(ids)
    chosen=h.smoke_records(records)
    config=h.read(k.MODEL/'config.json');assert config['num_hidden_layers']==36
    assert max(len(r['prompt_token_ids'])+len(r['target_token_ids']) for r in records)<=config['max_position_embeddings']
    files=sorted(p for p in k.MODEL.iterdir() if p.is_file() and p.suffix in {'.json','.safetensors','.jinja','.txt'})
    assert all(k.MODEL/n in files for n in set(h.read(k.MODEL/'model.safetensors.index.json')['weight_map'].values()))
    paths+=files+[Path(m.__file__) for m in [qwen,sdpa,lora,peft_io,adamw]]
    inputs={str(p):h.sha(p) for p in sorted(set(paths))}
    software={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','safetensors','accelerate','tokenizers']}
    hardware=h.gpu_identity();OUT.mkdir()
    h.save(OUT/'smoke_records.json',chosen);inputs[str(OUT/'smoke_records.json')]=h.sha(OUT/'smoke_records.json')
    h.save(OUT/'protocol.json',dict(utc=h.utc(),scope='Source-only retrieval-interface development; technical smoke on quality-unverified targets',
        inputs_sha256=inputs,software=software,hardware=hardware,model=str(k.MODEL),seed=0,
        chosen_policy=winner,chosen_prompt_cap=CAP,max_completion_tokens=2048,max_sequence_tokens=CAP+2048,
        policy_choice='Highest prequalified count>=16; production_chunks wins ties; no evaluation contexts or outcomes used',
        prequalified_source_ids=[r['instance_id'] for r in records],smoke_ids=[r['instance_id'] for r in chosen],
        smoke_choice='Unchanged113longest actual serialized sequence then longest completion if different; ID ties',
        implementation='Invoke unchanged115.run after assigning115.OUT to this learner_smoke directory; never invoke115.freeze or write114lookalike files',
        kernel_sha256=h.sha(KERNEL),cuda_device_order='PCI_BUS_ID',torch_device_uuid_must_match_frozen_physical_gpu2=True,
        objective='Mean per-example completion-token NLL including EOS; one1or2example accumulated update',
        steady_state_probe='Sameexamples backward with Adam states resident, no second optimizer step, gradients cleared before reset',
        optimizer=dict(name='AdamW',lr=1e-4,betas=[0.9,0.999],eps=1e-8,weight_decay=0),
        adapter=dict(layers=36,targets=['q_proj','v_proj'],r=8,alpha=16,dropout=0,parameters=3833856),
        precision=dict(base='BF16',adapter_parameters='FP32',adapter_compute='FP32 A/B matmuls, result castBF16',optimizer_state='FP32',
            loss='FP32 crossentropy',autocast=False,tf32=False,attention='SDPA FLASH_ATTENTION only; no fallback'),
        checkpointing='nonreentrant',use_cache=False,logits='Tensor-indexed completion prediction positions only',
        validation='Unchanged115zero-adapter/base probe, finite gradients/delta, A/B/layer dtype hooks, exact standardPEFT save/reload/reset, frozen-base bytehash and storage/version/dtype checks',
        total_shared_gpu_budget_seconds=3600,prior_gpu_seconds=0,
        external_timeout=dict(required=True,term_seconds=3595,kill_grace_seconds=5,hard_ceiling_seconds=3600,
            launcher='timeout --signal=TERM --kill-after=5s 3595s env SWE_SMOKE_EXTERNAL_TIMEOUT=3595+5 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/118_swe_retrieval_policy_smoke.py run'),
        accounting='Single-use; count validation/load/hash/smoke/save; futurefull subtracts max(internalelapsed,rootexternalterminalduration); no retryreset',
        future_estimate='1.20*max(firstupdate,steadypass)/examples*(3*prequalifiedN)+120seconds load/save reserve',
        quality_verification=False,full_training=False,inference=False,evaluation_private_access=False,api_spend_usd=0))
    print({'frozen':True,'chosen_policy':winner,'smoke_ids':[r['instance_id'] for r in chosen]},flush=True)


def run():
    k.OUT=OUT
    k.run()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);a=parser.parse_args()
    if a.stage=='freeze':freeze()
    else:run()
