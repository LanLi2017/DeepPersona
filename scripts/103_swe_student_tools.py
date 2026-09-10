#!/usr/bin/env python3
"""Six fixed native-tool Qwen episodes using the already verified public runtime."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/swe-diversity-selection/swe-student-tools'


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


m=module('student_public_runtime100','100_swe_debugging_trajectories_v2.py')
COHORT=m.COHORT
MODEL=Path('/scratch/yirenl2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
ADAPTER_INSTRUCTION='Issue at most one tool call per response. Put any commentary before the tool call and end the response immediately after </tool_call>.'
STUDENT_SYSTEM=m.SYSTEM+'\n'+ADAPTER_INSTRUCTION
CHAT_TOOLS=[dict(type='function',function={k:t[k] for k in ['name','description','parameters']}) for t in m.TOOLS]
sha,read,save=m.sha,m.read,m.save


def messages(history):
    result=[]
    for item in history:
        if item.get('type')=='function_call':
            call=dict(type='function',function=dict(name=item['name'],arguments=json.loads(item['arguments'])))
            if result and result[-1]['role']=='assistant' and 'tool_calls' not in result[-1]:result[-1]['tool_calls']=[call]
            else:result.append(dict(role='assistant',content='',tool_calls=[call]))
        elif item.get('type')=='function_call_output':result.append(dict(role='tool',content=item['output']))
        else:
            assert item['role'] in ['system','user','assistant']
            result.append(dict(role=item['role'],content=item['content']))
    return result


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)


def render(state,tok):
    chat=messages(state['history'])
    prompt=tok.apply_chat_template(chat,tools=CHAT_TOOLS,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok.encode(prompt,add_special_tokens=False)
    return chat,prompt,ids


def prepare():
    w=module('student_worker102','102_swe_local_tool_worker.py')
    assert not OUT.exists();OUT.mkdir();tok=tokenizer()
    for name in ['requests','responses','dispatched']:(OUT/'queue'/name).mkdir(parents=True)
    sources=[Path(__file__).resolve(),ROOT/'scripts/102_swe_local_tool_worker.py',ROOT/'scripts/100_swe_debugging_trajectories_v2.py',ROOT/'scripts/92_swe_fresh_generation.py',ROOT/'scripts/93_swe_candidate_evaluation.py',ROOT/'scripts/85_swe_execution_bridge.py',COHORT/'selection.json',COHORT/'generation_context/manifest.json']
    jobs=[]
    for row in read(COHORT/'selection.json')['tasks']:
        iid=row['instance_id'];key=iid+'__0';public=COHORT/'public'/iid;packet=COHORT/'generation_context'/(iid+'.txt')
        meta=read(public/'task.json')
        for name,digest in meta['source_sha256'].items():assert sha(public/'source'/name)==digest
        sources += [public/'task.json',packet]
        folder=OUT/'episodes'/key;folder.mkdir(parents=True);shutil.copytree(public/'source',folder/'repo')
        state=dict(key=key,instance_id=iid,turns=0,status='ready',changed=[],history=[dict(role='system',content=STUDENT_SYSTEM),dict(role='user',content=packet.read_bytes().decode('utf-8'))])
        chat,prompt,ids=render(state,tok);assert len(ids)<=36864
        save(folder/'initial_input.json',dict(messages=chat,prompt=prompt,prompt_token_ids=ids));sources.append(folder/'initial_input.json')
        save(folder/'state.json',state)
        jobs.append(dict(key=key,instance_id=iid,sample=0,packet=str(packet.relative_to(ROOT)),initial_input_tokens=len(ids)))
    save(OUT/'jobs.json',jobs);sources.append(OUT/'jobs.json')
    prior=read(COHORT.parent/'swe-student-headroom/protocol.json')
    protocol=dict(utc=datetime.now(timezone.utc).isoformat(),scope='Six development tasks, one fixed native-tool greedy student episode each; headroom only, no learning/diversity effect',
        model=str(MODEL),model_sha256=prior['model_sha256'],model_revision=MODEL.name,seed=w.SEED,
        engine=w.ENGINE,generation=w.GENERATION,physical_gpus=[2,3],system=STUDENT_SYSTEM,shared_system=m.SYSTEM,adapter_instruction=ADAPTER_INSTRUCTION,tools=m.TOOLS,native_chat_tools=CHAT_TOOLS,
        max_model_turns=8,max_output_tokens=4096,max_input_tokens=36864,max_serialized_history_bytes=260000,
        precision='FP32 weights/KV; retain disclosed Triton TF32 attention arithmetic, no IEEE guarantee. TP2 versus prior single-GPU baseline; no causal one-shot/tool gain claim.',
        tool_contract='Same100source reads/search/Python probes/exact edits with exact operation review and no private evaluator feedback. Model uses its native chat/tool serialization.',
        limits='Native40960 context: terminate context_limit if input>36864 or serialized history+tools>260000bytes, never truncate or retry.',
        parsing='Exactly one native <tool_call> JSON object per turn with no trailing nonwhitespace; malformed/multiple calls terminate tool_protocol_error. Text without tool tags is final; max-token outputs terminate incomplete. All raw outputs retained.',
        smoke='First fixed episode is in-study technical smoke. Exact prompt IDs/finite token logprobs and faithful native parser/tool disposition; malformed calls and invalid/no-edit episodes are retained model outcomes, never reasons to retry or cancel remaining five.',
        headroom_gate='At least two interpretable failures across six after complete private evaluation; broken worker/runtime/missing evaluation is not scientific headroom. Report all slots.',
        no_retry=True,no_outcome_resampling=True,api_spend_usd=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        tokenizer_chat_template=tok.chat_template,enable_thinking=False,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),input_tokens=[j['initial_input_tokens'] for j in jobs])),flush=True)


def validate():
    p=read(OUT/'protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return p


def submit(key):
    with m.lock():
        validate();folder=OUT/'episodes'/key;state=read(folder/'state.json')
        assert state['status']=='ready' and state['turns']<8
        if key!=read(OUT/'jobs.json')[0]['key']:
            gate=read(OUT/'smoke_review.json');assert gate['passed'] and gate['protocol_sha256']==sha(OUT/'protocol.json')
        chat,prompt,ids=render(state,tokenizer())
        if len(ids)>36864 or len(json.dumps(dict(history=state['history'],tools=m.TOOLS),ensure_ascii=False).encode())>260000:
            state['status']='context_limit';save(folder/'state.json',state);print(json.dumps(dict(key=key,status='context_limit')));return
        n=state['turns']+1;jobid=f'{key}__t{n:02d}';request=OUT/'queue/requests'/(jobid+'.json')
        assert not request.exists()
        save(folder/'requests'/f'{n:02d}.json',dict(messages=chat,prompt=prompt,prompt_token_ids=ids))
        state.update(status='model_pending',pending_jobid=jobid);save(folder/'state.json',state)
        save(request,dict(jobid=jobid,prompt_token_ids=ids,max_tokens=4096))
        print(json.dumps(dict(jobid=jobid,input_tokens=len(ids))),flush=True)


def consume(key):
    with m.lock():
        validate();folder=OUT/'episodes'/key;state=read(folder/'state.json');assert state['status']=='model_pending'
        path=OUT/'queue/responses'/(state['pending_jobid']+'.json');assert path.exists(),'Request not yet completed; inspect same worker, do not resubmit'
        row=read(path);n=state['turns']+1
        assert row['jobid']==state['pending_jobid'] and row['protocol_sha256']==sha(OUT/'protocol.json')
        assert row['request_sha256']==sha(OUT/'queue/requests'/(state['pending_jobid']+'.json'))
        assert row['technical_checks_passed'] and all(row['technical_checks'].values())
        save(folder/'responses'/f'{n:02d}.json',row)
        state['turns']=n;state.pop('pending_jobid')
        if row.get('error') or row.get('error_type'):
            state['status']='worker_error';save(folder/'state.json',state);return
        expected=read(folder/'requests'/f'{n:02d}.json')['prompt_token_ids']
        assert row['prompt_token_ids']==expected
        import math
        assert len(row['generated_token_ids'])==len(row['chosen_token_logprobs'])>0
        assert all(math.isfinite(v) for v in row['chosen_token_logprobs'])
        text=row['output_text']
        if row['finish_reason']!='stop':state['status']='incomplete';save(folder/'state.json',state);return
        if '<tool_call>' not in text and '</tool_call>' not in text:
            state['history'].append(dict(role='assistant',content=text));state['status']='finished';save(folder/'state.json',state)
        else:
            try:
                assert text.count('<tool_call>')==text.count('</tool_call>')==1
                match=re.search(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S);assert match
                call=json.loads(match.group(1));assert set(call)=={'name','arguments'} and isinstance(call['arguments'],dict)
                assert call['name'] in {t['name'] for t in m.TOOLS}
                assert not text[match.end():].strip()
                visible=text[:match.start()].strip()
                call=dict(type='function_call',name=call['name'],arguments=json.dumps(call['arguments']),call_id=f'{key}:{n}')
                state['history'].append(dict(role='assistant',content=visible));state['history'].append(call)
                save(folder/'state.json',state);m.stage_action(folder,state,call)
            except (AssertionError,ValueError,KeyError,TypeError):
                state['status']='tool_protocol_error';save(folder/'state.json',state)
        print(json.dumps(dict(key=key,turn=n,status=read(folder/'state.json')['status'])),flush=True)


def reconcile(key):
    with m.lock():
        validate();folder=OUT/'episodes'/key;state=read(folder/'state.json')
        assert state['status']=='model_pending'
        jobid=state['pending_jobid'];request=OUT/'queue/requests'/(jobid+'.json')
        dispatched=OUT/'queue/dispatched'/(jobid+'.json');response=OUT/'queue/responses'/(jobid+'.json')
        if not request.exists() and not dispatched.exists() and not response.exists():
            data=read(folder/'requests'/f"{state['turns']+1:02d}.json")
            save(request,dict(jobid=jobid,prompt_token_ids=data['prompt_token_ids'],max_tokens=4096))
            print('Recovered identical undispatched queue request; no generation retry',flush=True);return
        assert (OUT/'queue/worker_failure.json').exists()
        pid=read(OUT/'queue/worker_started.json')['pid']
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError('Worker PID still exists; confirm terminal process before closing slot')
        if response.exists():
            row=read(response)
            valid=(row.get('technical_checks_passed') and row.get('jobid')==jobid and row.get('protocol_sha256')==sha(OUT/'protocol.json') and row.get('request_sha256')==sha(request))
            assert not valid, 'Completed valid response must be consumed normally'
            save(folder/'failed_response.json',row)
            state['failed_response_sha256']=sha(response)
        state.update(status='worker_error' if dispatched.exists() else 'not_run_worker_failure',failure=read(OUT/'queue/worker_failure.json'),attempted_dispatch_count=state['turns']+int(dispatched.exists()))
        save(folder/'state.json',state)
        print(json.dumps(dict(key=key,status=state['status'],generation_retried=False)),flush=True)


def finalize():
    with m.lock():
        validate();assert not (OUT/'patch_manifest.json').exists();records=[]
        for job in read(OUT/'jobs.json'):
            folder=OUT/'episodes'/job['key'];state=read(folder/'state.json')
            assert state['status'] not in ['ready','awaiting_review','model_pending']
            files=[];basehash={};patch=''
            for name in state['changed']:
                base=COHORT/'public'/job['instance_id']/'source'/name;after=m.source_path(folder,name)
                if sha(base)==sha(after):continue
                files.append(name);basehash[name]=sha(base)
                patch+=m.g.unified_patch(name,base.read_bytes().decode(),after.read_bytes().decode())
            row=dict(**job,valid=bool(patch),terminal_status=state['status'],model_turns=state['turns'],attempted_dispatch_count=state.get('attempted_dispatch_count',state['turns']),files=files,base_file_sha256=basehash)
            if patch:
                dest=OUT/'patches'/(job['key']+'.patch');dest.parent.mkdir(exist_ok=True);dest.write_text(patch);row['patch_sha256']=sha(dest)
            else:row['invalid_reason']='No nonempty valid production patch at termination'
            records.append(row)
        save(OUT/'patch_manifest.json',records);save(OUT/'generation_summary.json',dict(slots=6,valid_patches=sum(r['valid'] for r in records),model_turns=sum(r['model_turns'] for r in records),attempted_model_requests=sum(r['attempted_dispatch_count'] for r in records),rows=records,api_spend_usd=0))
        (OUT/'queue/stop').touch();print(json.dumps(read(OUT/'generation_summary.json')),flush=True)


m.OUT=OUT
m.validate=validate
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','submit','consume','execute','reconcile','finalize']);ap.add_argument('--key');ap.add_argument('--approval');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='submit':submit(a.key)
    elif a.stage=='consume':consume(a.key)
    elif a.stage=='execute':m.execute(a.key,a.approval)
    elif a.stage=='reconcile':reconcile(a.key)
    else:finalize()
