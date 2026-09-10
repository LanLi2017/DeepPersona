#!/usr/bin/env python3
"""Six fixed directed native-tool controls; no repair scoring or retries."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/swe-diversity-selection/swe-tool-competence'

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

s=module('student103','103_swe_student_tools.py')
w=module('worker102','102_swe_local_tool_worker.py')
m=s.m
sha,read,save=s.sha,s.read,s.save

def validate():
    p=read(OUT/'protocol.json')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return p

m.OUT=OUT;m.validate=validate
w.OUT=OUT;w.QUEUE=OUT/'queue'

def parse(text):
    assert text.count('<tool_call>')==text.count('</tool_call>')==1
    match=re.search(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S);assert match
    assert not text[match.end():].strip()
    call=json.loads(match.group(1))
    assert set(call)=={'name','arguments'} and isinstance(call['arguments'],dict)
    return call

def compliant(call,case):
    if call['name']!=case['expected_tool']:return False
    actual=call['arguments'];expected=case['expected_arguments']
    if set(actual)!=set(expected):return False
    if call['name']=='python_probe':
        return ast.dump(ast.parse(actual['code']))==ast.dump(ast.parse(expected['code']))
    return all(type(actual[k]) is type(v) and actual[k]==v for k,v in expected.items())

def prepare():
    assert not OUT.exists();OUT.mkdir()
    casepath=s.OUT/'tool_competence_cases.json';spec=read(casepath);tok=s.tokenizer()
    for name in ['requests','responses','dispatched']:(OUT/'queue'/name).mkdir(parents=True)
    sources=[Path(__file__).resolve(),ROOT/'scripts/103_swe_student_tools.py',ROOT/'scripts/102_swe_local_tool_worker.py',ROOT/'scripts/100_swe_debugging_trajectories_v2.py',ROOT/'scripts/92_swe_fresh_generation.py',ROOT/'scripts/93_swe_candidate_evaluation.py',ROOT/'scripts/85_swe_execution_bridge.py',casepath]
    jobs=[]
    for case in spec['cases']:
        iid=case['instance_id'];public=s.COHORT/'public'/iid
        packet=s.COHORT/'generation_context'/(iid+'.txt')
        meta=read(public/'task.json');sources.extend([packet,public/'task.json'])
        for name,digest in meta['source_sha256'].items():assert sha(public/'source'/name)==digest,name
        for condition in spec['conditions']:
            key=case['case_id']+'__'+condition;folder=OUT/'episodes'/key;folder.mkdir(parents=True)
            shutil.copytree(public/'source',folder/'repo')
            user=(packet.read_bytes().decode()+'\n\n' if condition=='full_packet' else '')+case['cue']
            state=dict(key=key,instance_id=iid,turns=0,status='ready',changed=[],history=[dict(role='system',content=s.STUDENT_SYSTEM),dict(role='user',content=user)])
            chat,prompt,ids=s.render(state,tok);assert len(ids)<=w.MAX_INPUT_TOKENS
            save(folder/'initial_input.json',dict(messages=chat,prompt=prompt,prompt_token_ids=ids))
            sources.append(folder/'initial_input.json');save(folder/'state.json',state)
            jobs.append(dict(key=key,condition=condition,input_tokens=len(ids),**case))
    assert len(jobs)==6 and len({j['key'] for j in jobs})==6
    save(OUT/'jobs.json',jobs);sources.append(OUT/'jobs.json')
    old=read(s.OUT/'protocol.json')
    save(OUT/'protocol.json',dict(utc=w.utc(),scope='Directed operation compliance; six fixed single-turn local requests. No repair, feedback-uptake, diversity, learning or pure context-length causal endpoint.',
        model=str(w.MODEL),model_revision=w.MODEL.name,model_sha256=old['model_sha256'],git_head=old['git_head'],seed=w.SEED,engine=w.ENGINE,generation=w.GENERATION,numerical_scope=w.NUMERICAL_SCOPE,physical_gpus=[2,3],
        system=s.STUDENT_SYSTEM,native_chat_tools=s.CHAT_TOOLS,tokenizer_chat_template=tok.chat_template,enable_thinking=False,
        cases=spec,max_requests=6,no_retry=True,api_spend_usd=0,
        compliance='Normal stop, exactly one native call, no trailing content. Exact argument keys/types/values; probe AST equality permits syntax-only equivalents, including whitespace and comments. Failures retained; never repair arguments.',
        execution='Only compliant calls execute through unchanged100 runtime. Search/read automatic; Python requires separate root request/code hash review. No additional model response or private tests. Runtime outcomes descriptive, not parse correctness.',
        smoke='First fixed short literal-search slot: technical prompt/provenance/parsing/disposition check before remaining five. Model compliance does not gate continuation.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    checks=[]
    for case in spec['cases']:
        call=dict(name=case['expected_tool'],arguments=case['expected_arguments'])
        assert compliant(parse('<tool_call>'+json.dumps(call)+'</tool_call>'),case)
        bad=dict(call,arguments={**call['arguments'],'extra':True});assert not compliant(bad,case)
        checks.append(case['case_id'])
    save(OUT/'preflight.json',dict(utc=w.utc(),protocol_sha256=sha(OUT/'protocol.json'),positive_and_extra_argument_rejection=checks,model_calls=0))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),jobs=[(j['key'],j['input_tokens']) for j in jobs])),flush=True)

def submit(key):
    with m.lock():
        validate();jobs=read(OUT/'jobs.json');assert key in {j['key'] for j in jobs}
        if key!=jobs[0]['key']:
            gate=read(OUT/'smoke_review.json');assert gate['passed'] and gate['protocol_sha256']==sha(OUT/'protocol.json')
        folder=OUT/'episodes'/key;state=read(folder/'state.json');assert state['status']=='ready' and state['turns']==0
        path=OUT/'queue/requests'/(key+'.json');assert not path.exists()
        state.update(status='model_pending');save(folder/'state.json',state)
        save(path,dict(jobid=key,prompt_token_ids=read(folder/'initial_input.json')['prompt_token_ids'],max_tokens=4096))
        print(key,flush=True)

def consume(key):
    with m.lock():
        validate();case=next(j for j in read(OUT/'jobs.json') if j['key']==key)
        folder=OUT/'episodes'/key;state=read(folder/'state.json');assert state['status']=='model_pending'
        response=OUT/'queue/responses'/(key+'.json');row=read(response)
        assert row['jobid']==key and row['protocol_sha256']==sha(OUT/'protocol.json')
        assert row['request_sha256']==sha(OUT/'queue/requests'/(key+'.json'))
        assert row['technical_checks_passed'] and all(row['technical_checks'].values())
        assert row['prompt_token_ids']==read(folder/'initial_input.json')['prompt_token_ids']
        assert len(row['generated_token_ids'])==len(row['chosen_token_logprobs'])>0
        assert all(math.isfinite(x) for x in row['chosen_token_logprobs'])
        result=dict(key=key,instance_id=case['instance_id'],condition=case['condition'],compliant=False,finish_reason=row['finish_reason'],response_sha256=sha(response))
        call=None
        try:
            if row['finish_reason']=='stop':
                call=parse(row['output_text']);result.update(call=call,compliant=compliant(call,case),exact_arguments=call['arguments']==case['expected_arguments'])
        except (AssertionError,ValueError,KeyError,TypeError,SyntaxError) as error:result['parse_error']=type(error).__name__
        save(folder/'compliance.json',result);state.update(turns=1,status='not_executed_noncompliant')
        save(folder/'state.json',state)
        if result['compliant']:
            action=dict(type='function_call',name=call['name'],arguments=json.dumps(call['arguments']),call_id=key+':1')
            m.stage_action(folder,state,action)
        print(json.dumps(dict(**result,status=read(folder/'state.json')['status'])),flush=True)

def finalize():
    validate();assert not (OUT/'summary.json').exists();rows=[]
    for job in read(OUT/'jobs.json'):
        folder=OUT/'episodes'/job['key'];row=read(folder/'compliance.json');state=read(folder/'state.json')
        assert state['turns']==1 and state['status'] not in ['model_pending','awaiting_review'] and not state['changed']
        if row['compliant']:
            result=read(folder/'actions/01.result.json');row['runtime_result']=result
            row['runtime_ok']='error' not in result and (row['call']['name']!='python_probe' or (result['exit_code']==0 and not result['timed_out'] and not result['output_truncated'] and bool(result['output'].strip())))
        else:row['runtime_ok']=None
        row['input_tokens']=job['input_tokens'];row['output_tokens']=read(OUT/'queue/responses'/(job['key']+'.json'))['output_tokens']
        rows.append(row)
    assert len(list((OUT/'queue/responses').glob('*.json')))==6
    save(OUT/'summary.json',dict(utc=w.utc(),slots=6,compliant=sum(r['compliant'] for r in rows),runtime_ok=sum(r['runtime_ok'] is True for r in rows),rows=rows,api_spend_usd=0,scope=read(OUT/'protocol.json')['scope']))
    (OUT/'queue/stop').touch();print(json.dumps(dict(slots=6,compliant=sum(r['compliant'] for r in rows),runtime_ok=sum(r['runtime_ok'] is True for r in rows))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','submit','consume','execute','finalize']);p.add_argument('--key');p.add_argument('--approval');a=p.parse_args()
    if a.stage=='worker':
        try:w.run(OUT/'protocol.json')
        except Exception as error:
            if not (OUT/'queue/worker_failure.json').exists():save(OUT/'queue/worker_failure.json',dict(utc=w.utc(),error_type=type(error).__name__,error=str(error),policy='No retries; root review required.'))
            raise
    elif a.stage=='prepare':prepare()
    elif a.stage=='submit':submit(a.key)
    elif a.stage=='consume':consume(a.key)
    elif a.stage=='execute':m.execute(a.key,a.approval)
    else:finalize()
