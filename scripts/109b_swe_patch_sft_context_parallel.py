#!/usr/bin/env python3
"""Execution-only parallelism for the frozen109 public context builder."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import multiprocessing
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/swe-diversity-selection/swe-sympy-patch-sft'
CONTEXT = OUT / 'context'


def frozen():
    spec = importlib.util.spec_from_file_location('context109', ROOT/'scripts/109_swe_patch_sft_context.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    m = frozen()
    m.validate()
    assert not (CONTEXT/'manifest.json').exists()
    tasks = m.read(OUT/'selection.json')['tasks']
    completed = [t for t in tasks if (CONTEXT/(t['instance_id']+'.json')).exists()]
    pending = [t for t in tasks if t not in completed]
    for task in pending:
        iid = task['instance_id']
        assert not (CONTEXT/(iid+'.packet.txt')).exists() and not (CONTEXT/(iid+'.prompt.txt')).exists(), 'Partial writes require explicit disposition; no overwrite'
    manifest = m.read(OUT/'export_manifest.json')
    assert len(manifest['tasks']) == 52 and manifest['ready'] == 52
    paths = [Path(__file__).resolve(), ROOT/'scripts/109_swe_patch_sft_context.py',CONTEXT/'protocol.json',OUT/'selection.json',OUT/'export_manifest.json',CONTEXT/'build_all.log']
    preserved = {}
    for task in completed:
        for suffix in ['.json','.packet.txt','.prompt.txt']:
            path = CONTEXT/(task['instance_id']+suffix)
            if path.exists():
                preserved[str(path)] = m.sha(path)
    m.save(CONTEXT/'parallel_execution_amendment.json', {
        'utc':datetime.now(timezone.utc).isoformat(),'change':'Serial to four spawned CPU processes; no ranking, rendering, tokenizer, input, record schema or failure-policy change.',
        'workers':4,'serial_pid':784401,'serial_session':62533,'serial_terminal_exit_code':1,
        'serial_stop':'Explicit SIGINT for execution optimization; interrupted slot had no record/text artifacts and is unfinished, not a model/outcome retry.',
        'original_protocol_sha256':m.sha(CONTEXT/'protocol.json'),
        'source_sha256':{str(p):m.sha(p) for p in paths},'preserved_artifact_sha256':preserved,
        'completed_ids':[t['instance_id'] for t in completed],'pending_ids':[t['instance_id'] for t in pending],
        'worker_method':'Import unchanged109; read wrapper narrows only selection.tasks to one already assigned slot; save wrapper suppresses only per-worker aggregate manifest. Original109build(False) writes exact task record. Parent alone writes final52 manifest.',
        'parity_before_parallel':'Rebuild first completed smoke packet, full retrieval manifest, native prompt and IDs; compare without rewriting original artifacts.',
        'private_inputs_read':False,'api_spend_usd':0,'gpu_used':False})


def validate():
    m = frozen()
    m.validate()
    p = m.read(CONTEXT/'parallel_execution_amendment.json')
    for path,value in p['source_sha256'].items():
        assert m.sha(Path(path)) == value, path
    for path,value in p['preserved_artifact_sha256'].items():
        assert m.sha(Path(path)) == value, path
    return m,p


def parity():
    m,p = validate()
    iid = p['completed_ids'][0]
    row = m.read(CONTEXT/(iid+'.json'))
    enc = m.NativeEncoding(m.tokenizer())
    packet,retrieval = m.load_ranking().build_task(OUT/'public'/iid,enc,m.BUDGET)
    retrieval.update(tokenizer=str(m.MODEL),token_count_is_model_specific_guarantee=True,
        token_count_scope='Complete native system/user/generation prompt; not bare packet tokens')
    assert packet == (CONTEXT/(iid+'.packet.txt')).read_bytes().decode('utf-8')
    assert retrieval == row['retrieval']
    assert enc.render(packet) == (CONTEXT/(iid+'.prompt.txt')).read_bytes().decode('utf-8')
    assert enc.encode(packet) == row['prompt_token_ids']
    m.native_check(enc,packet,row['prompt_token_ids'])
    m.save(CONTEXT/'parallel_parity.json', {'all_checks_passed':True,'instance_id':iid,
        'amendment_sha256':m.sha(CONTEXT/'parallel_execution_amendment.json'),'packet_retrieval_prompt_ids_all_exact':True,
        'original_artifacts_rewritten':False,'private_inputs_read':False})


def worker(iid):
    m = frozen()
    original_read,original_save = m.read,m.save
    def read(path):
        value = original_read(path)
        if path == OUT/'selection.json':
            value = {**value,'tasks':[t for t in value['tasks'] if t['instance_id']==iid]}
            assert len(value['tasks']) == 1
        return value
    def save(path,value):
        if path == CONTEXT/'manifest.json':
            return
        original_save(path,value)
    m.read,m.save = read,save
    assert not (CONTEXT/(iid+'.json')).exists()
    m.build(False)
    row = original_read(CONTEXT/(iid+'.json'))
    return {'instance_id':iid,'status':row['status'],'prompt_tokens':row.get('prompt_tokens')}


def run():
    m,p = validate()
    gate = m.read(CONTEXT/'parallel_parity.json')
    assert gate['all_checks_passed'] and gate['amendment_sha256']==m.sha(CONTEXT/'parallel_execution_amendment.json')
    assert not (CONTEXT/'parallel_started.json').exists() and not (CONTEXT/'manifest.json').exists()
    for iid in p['pending_ids']:
        assert not (CONTEXT/(iid+'.json')).exists()
    m.save(CONTEXT/'parallel_started.json',{'utc':datetime.now(timezone.utc).isoformat(),'amendment_sha256':m.sha(CONTEXT/'parallel_execution_amendment.json')})
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker,iid) for iid in p['pending_ids']]
        for future in as_completed(futures):
            print(future.result(),flush=True)
    for path,value in p['preserved_artifact_sha256'].items():
        assert m.sha(Path(path)) == value,path
    tasks = m.read(OUT/'selection.json')['tasks']
    rows = [m.read(CONTEXT/(t['instance_id']+'.json')) for t in tasks]
    m.save(CONTEXT/'manifest.json',{'protocol_sha256':m.sha(CONTEXT/'protocol.json'),'selection_sha256':m.sha(OUT/'selection.json'),
        'assigned_slots':len(tasks),'completed_slots':len(rows),'ready_slots':sum(r['status']=='ready' for r in rows),
        'tasks':[{'instance_id':r['instance_id'],'role':r['role'],'status':r['status'],'record_sha256':m.sha(CONTEXT/(r['instance_id']+'.json'))} for r in rows],
        'private_inputs_read':False,'api_spend_usd':0,'gpu_used':False})
    m.save(CONTEXT/'parallel_completed.json',{'utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
        'pending_slots_completed':len(p['pending_ids']),'preserved_completed_slots':len(p['completed_ids']),
        'manifest_sha256':m.sha(CONTEXT/'manifest.json'),'amendment_sha256':m.sha(CONTEXT/'parallel_execution_amendment.json')})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','parity','run'])
    args=parser.parse_args()
    {'prepare':prepare,'parity':parity,'run':run}[args.stage]()
