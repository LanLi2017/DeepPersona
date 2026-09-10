#!/usr/bin/env python3
"""Replay the first genuine debugging loop without model calls or private tests."""
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('trajectory97_replay',ROOT/'scripts/97_swe_debugging_trajectories.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ORIGINAL=m.OUT


def comparable(result,repo):
    result={k:v for k,v in result.items() if k not in ['seconds','raw_log_sha256']}
    if 'output' in result:result['output']=result['output'].replace(str(repo),'<REPO>')
    return result


def main():
    m.validate()
    selected=None
    for job in m.read(ORIGINAL/'jobs.json'):
        folder=ORIGINAL/'episodes'/job['key']
        state=m.read(folder/'state.json')
        if state['status'] in ['ready','awaiting_review']:continue
        actions=[(int(p.name.split('.')[0]),m.read(p)) for p in sorted((folder/'actions').glob('*.request.json'))]
        executed=[(n,r) for n,r in actions if (folder/'actions'/f'{n:02d}.result.json').exists()]
        probes=[n for n,r in executed if r['name']=='python_probe' and 'exit_code' in m.read(folder/'actions'/f'{n:02d}.result.json')]
        edits=[n for n,r in executed if r['name']=='edit_source' and 'edited' in m.read(folder/'actions'/f'{n:02d}.result.json')]
        if any(a<e<z for a in probes for e in edits for z in probes):
            selected=(job,folder,actions);break
    assert selected,'No eligible completed source-context/reproducer/edit/rerun episode'
    job,old,actions=selected
    m.OUT=ORIGINAL/'replay';assert not m.OUT.exists();m.OUT.mkdir()
    shutil.copyfile(ORIGINAL/'protocol.json',m.OUT/'protocol.json')
    folder=m.OUT/'episodes'/job['key'];folder.mkdir(parents=True)
    public=m.COHORT/'public'/job['instance_id']
    shutil.copytree(public/'source',folder/'repo')
    state=dict(key=job['key'],instance_id=job['instance_id'],turns=0,status='ready',changed=[],history=[])
    records=[]
    for n,request in actions:
        state['turns']=n
        m.stage_action(folder,state,dict(name=request['name'],arguments=json.dumps(request['arguments']),call_id=request['call_id']))
        state=m.read(folder/'state.json')
        if state['status']=='awaiting_review':
            approval=old/'actions'/f'{n:02d}.review.json'
            assert approval.exists(),'Unreviewed executable action cannot replay'
            m.execute(job['key'],str(approval))
        original=m.read(old/'actions'/f'{n:02d}.result.json')
        replay=m.read(folder/'actions'/f'{n:02d}.result.json')
        same=comparable(original,old/'repo')==comparable(replay,folder/'repo')
        records.append(dict(action=n,name=request['name'],request_sha_match=m.sha(old/'actions'/f'{n:02d}.request.json')==m.sha(folder/'actions'/f'{n:02d}.request.json'),result_match=same,
            original_result_sha256=m.sha(old/'actions'/f'{n:02d}.result.json'),replay_result_sha256=m.sha(folder/'actions'/f'{n:02d}.result.json')))
        state=m.read(folder/'state.json')
    hashes={name:dict(original=m.sha(old/'repo'/name),replay=m.sha(folder/'repo'/name)) for name in state['changed']}
    passed=all(r['request_sha_match'] and r['result_match'] for r in records) and all(v['original']==v['replay'] for v in hashes.values())
    m.save(m.OUT/'summary.json',dict(passed=passed,key=job['key'],selection='First completed eligible episode in frozen metadata/sample order',
        normalization=m.read(ORIGINAL/'protocol.json')['replay'],actions=records,final_file_sha256=hashes,api_calls=0,private_tests_accessed=False,
        source_sha256=m.sha(Path(__file__)),protocol_sha256=m.sha(ORIGINAL/'protocol.json')))
    print(json.dumps(m.read(m.OUT/'summary.json')),flush=True)
    assert passed


if __name__=='__main__':main()
