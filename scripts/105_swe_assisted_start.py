#!/usr/bin/env python3
"""Two fixed tasks × three evidence arms; seven new native-tool responses each."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/swe-diversity-selection/swe-assisted-start'

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

s=module('student103','103_swe_student_tools.py');w=module('worker102','102_swe_local_tool_worker.py')
m=s.m;sha,read,save=s.sha,s.read,s.save
BASELINE=s.OUT;TEACHER=OUT.parent/'swe-debugging-trajectories-v2';CONTROL=OUT.parent/'swe-tool-competence'
NOTICE='An externally supplied diagnostic step follows. Its code was selected by another agent, not by you. Any observation shown is a recorded result from the original unmodified source. The result may instead be explicitly withheld. This supplied step occupies one of eight interaction steps; seven model responses remain.'
CUE='Continue repairing the issue using the available tools. Seven model responses remain. Submit your final repair through edit_source.'
TASKS=['sympy__sympy-24562','sympy__sympy-23824']
ARMS=['W','T','F']
EXPECTED=[('a4cf4e35eae7c41753c61c5d0553eb0fc0d6126e57a2a1bcc43ce738ca6b2a4a','d5828821914225ed7b78f63acbdab0dc5e36a21e609caa67caa446b6320a5dd4'),('f5bb91f99b1abe71fc6ea781031bfa9dae178bd073f1c0b3294715652d78ce4b','a86d7e980a30454d3128866aa7428bafac96ece9b5cba069a4398beec11cc01b')]

def validate():
    p=read(OUT/'protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return p

s.OUT=OUT;m.OUT=OUT;m.validate=validate;s.validate=validate
w.OUT=OUT;w.QUEUE=OUT/'queue';w.MAX_REQUESTS=42

def prepare():
    assert not OUT.exists();OUT.mkdir();tok=s.tokenizer()
    controls=read(CONTROL/'summary.json');assert controls['slots']==controls['compliant']==controls['runtime_ok']==6
    assert read(CONTROL/'independent_result_audit.json')['all_checks_passed']
    base={r['instance_id']:r for r in read(BASELINE/'summary.json')['rows']}
    assert all(not base[i]['resolved'] and base[i]['execution_status']=='invalid_generation' for i in TASKS)
    assert [r['instance_id'] for r in read(s.COHORT/'selection.json')['tasks'][:2]]==TASKS
    for name in ['requests','dispatched','responses']:(OUT/'queue'/name).mkdir(parents=True)
    sources=[Path(__file__).resolve(),ROOT/'scripts/103_swe_student_tools.py',ROOT/'scripts/102_swe_local_tool_worker.py',ROOT/'scripts/100_swe_debugging_trajectories_v2.py',ROOT/'scripts/92_swe_fresh_generation.py',ROOT/'scripts/93_swe_candidate_evaluation.py',ROOT/'scripts/85_swe_execution_bridge.py',ROOT/'docs/swe_assisted_start_protocol.md',CONTROL/'summary.json',CONTROL/'independent_result_audit.json',BASELINE/'summary.json',BASELINE/'protocol.json',s.COHORT/'selection.json',s.COHORT/'generation_context/manifest.json']
    jobs=[];matches=[]
    for iid,digests in zip(TASKS,EXPECTED):
        public=s.COHORT/'public'/iid;packet=s.COHORT/'generation_context'/(iid+'.txt');teacher=TEACHER/'episodes'/(iid+'__0')
        request=teacher/'actions/01.request.json';result=teacher/'actions/01.result.json'
        assert (sha(request),sha(result))==digests
        action=read(request);assert action['name']=='python_probe'
        observation={k:read(result)[k] for k in ['exit_code','timed_out','output','output_truncated']}
        observation['output']=observation['output'].replace(str(teacher/'repo'),'<REPO>')
        external=dict(name=action['name'],arguments=action['arguments'],observation=observation)
        meta=read(public/'task.json');sources.extend([public/'task.json',packet,request,result])
        for name,digest in meta['source_sha256'].items():assert sha(public/'source'/name)==digest,name
        sizes={}
        for sample,arm in enumerate(ARMS):
            key=f'{iid}__{sample}';folder=OUT/'episodes'/key;folder.mkdir(parents=True)
            shutil.copytree(public/'source',folder/'repo')
            user=packet.read_bytes().decode()+'\n\n'+NOTICE
            history=[dict(role='system',content=s.STUDENT_SYSTEM),dict(role='user',content=user)]
            if arm in ['W','T']:
                history.append(dict(type='function_call',name=action['name'],arguments=json.dumps(action['arguments']),call_id=key+':external'))
                record=observation if arm=='T' else dict(observation_withheld=True,notice='The result of the externally supplied probe is withheld. No execution status or output is provided.')
                history.append(dict(type='function_call_output',call_id=key+':external',output=json.dumps(record)))
            else:
                history[-1]['content']+='\n\n<external_evidence>\n'+json.dumps(external)+'\n</external_evidence>'
            history.append(dict(role='user',content=CUE))
            state=dict(key=key,instance_id=iid,arm=arm,turns=1,status='ready',changed=[],history=history)
            chat,prompt,ids=s.render(state,tok)
            assert len(ids)<=36864 and len(json.dumps(dict(history=history,tools=m.TOOLS),ensure_ascii=False).encode())<=260000
            save(folder/'initial_input.json',dict(messages=chat,prompt=prompt,prompt_token_ids=ids));sources.append(folder/'initial_input.json')
            save(folder/'state.json',state);save(folder/'external_evidence.json',external);sources.append(folder/'external_evidence.json')
            sizes[arm]=len(ids);jobs.append(dict(key=key,instance_id=iid,sample=sample,arm=arm,initial_input_tokens=len(ids)))
        ratio=abs(sizes['T']-sizes['F'])/min(sizes['T'],sizes['F'])
        matches.append(dict(instance_id=iid,input_tokens=sizes,TF_relative_difference=ratio,within_five_percent=ratio<=.05))
    assert len(jobs)==6;save(OUT/'jobs.json',jobs);sources.append(OUT/'jobs.json')
    old=read(BASELINE/'protocol.json')
    protocol=dict(utc=w.utc(),scope='Fixed two-task assisted-start evidence feasibility, not learning/diversity/generalization.',
        model=str(w.MODEL),model_revision=w.MODEL.name,model_sha256=old['model_sha256'],seed=w.SEED,engine=w.ENGINE,generation=w.GENERATION,numerical_scope=w.NUMERICAL_SCOPE,physical_gpus=[2,3],
        system=s.STUDENT_SYSTEM,native_chat_tools=s.CHAT_TOOLS,tokenizer_chat_template=tok.chat_template,enable_thinking=False,
        tasks=TASKS,arms=ARMS,notice=NOTICE,continuation_cue=CUE,equal_evidence_token_check=matches,
        external_step='Exact episode0 action01 before any edit. Normalize original absolute repository path to <REPO>; omit seconds/raw_log_sha256 only. No teacher commentary, patch or later actions.',
        max_new_model_responses_per_episode=7,initial_budget_steps=1,max_worker_requests=42,max_input_tokens=36864,max_output_tokens=4096,max_serialized_history_bytes=260000,
        global_worker_wall_seconds=5400,termination_grace_seconds=30,runtime_cap='Launch worker under timeout --signal=TERM --kill-after=30s 5400s; close any interrupted pending dispatch as unknown after confirmed termination, no retry. Also cease new submit after5400s from worker start.',
        no_retry=True,api_spend_usd=0,smoke='First fixed W episode is technical smoke before other five; success does not gate continuation.',
        primary='Nonempty terminal patch passes required target and all regressions; grade all six only after generation ends. Retain patches at budget/context/protocol limits.',
        gate='T resolves both24562and23824 while W resolves neither; no loss relative to baseline. Two paired development tasks, no significance or population claim.',
        stopping='Exactly six slots; no second teacher step, added hint, replacement task or seed. Gate failure closes this branch for current learner/cohort.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),initial_inputs=matches)),flush=True)

def submit(key):
    started=read(OUT/'queue/worker_started.json')['utc']
    if (datetime.now(timezone.utc)-datetime.fromisoformat(started)).total_seconds()>=5400:
        folder=OUT/'episodes'/key;state=read(folder/'state.json');assert state['status']=='ready'
        state['status']='global_runtime_limit';save(folder/'state.json',state);return
    s.submit(key)

def finalize():
    with m.lock():
        validate();assert not (OUT/'patch_manifest.json').exists();rows=[]
        for job in read(OUT/'jobs.json'):
            folder=OUT/'episodes'/job['key'];state=read(folder/'state.json');assert state['status'] not in ['ready','awaiting_review','model_pending']
            files=[];basehash={};patch=''
            for name in state['changed']:
                before=s.COHORT/'public'/job['instance_id']/'source'/name;after=m.source_path(folder,name)
                if sha(before)==sha(after):continue
                files.append(name);basehash[name]=sha(before)
                patch+=m.g.unified_patch(name,before.read_bytes().decode(),after.read_bytes().decode())
            row=dict(**job,valid=bool(patch),terminal_status=state['status'],model_turns=state['turns']-1,budget_steps=state['turns'],attempted_dispatch_count=state.get('attempted_dispatch_count',state['turns'])-1,files=files,base_file_sha256=basehash)
            if patch:
                dest=OUT/'patches'/(job['key']+'.patch');dest.parent.mkdir(exist_ok=True);dest.write_text(patch);row['patch_sha256']=sha(dest)
            else:row['invalid_reason']='No nonempty valid production patch at termination'
            rows.append(row)
        save(OUT/'patch_manifest.json',rows);save(OUT/'generation_summary.json',dict(slots=6,valid_patches=sum(r['valid'] for r in rows),model_turns=sum(r['model_turns'] for r in rows),attempted_model_requests=sum(r['attempted_dispatch_count'] for r in rows),rows=rows,api_spend_usd=0))
        (OUT/'queue/stop').touch();print(json.dumps(read(OUT/'generation_summary.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','submit','consume','execute','reconcile','finalize']);p.add_argument('--key');p.add_argument('--approval');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':
        try:w.run(OUT/'protocol.json')
        except Exception as error:
            if not (OUT/'queue/worker_failure.json').exists():save(OUT/'queue/worker_failure.json',dict(utc=w.utc(),error_type=type(error).__name__,error=str(error),policy='No retry; preserve any pending dispatch.'))
            raise
    elif a.stage=='submit':submit(a.key)
    elif a.stage=='consume':s.consume(a.key)
    elif a.stage=='execute':m.execute(a.key,a.approval)
    elif a.stage=='reconcile':s.reconcile(a.key)
    else:finalize()
