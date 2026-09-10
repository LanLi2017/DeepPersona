#!/usr/bin/env python3
"""Fixed additive MBPP expansion: remaining88 training tasks, eight samples each."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

spec = importlib.util.spec_from_file_location('pool52', Path(__file__).with_name('52_code_learning_pool.py'))
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)
ORIGINAL = pool.ROOT
OUT = Path('runs/swe-diversity-selection/code-learning-expansion')
pool.ROOT = OUT


def prepare():
    import pyarrow.parquet as pq
    assert not (OUT / 'protocol.json').exists(), 'Expansion already frozen'
    original = json.loads((ORIGINAL / 'protocol.json').read_text())
    rows = pq.read_table(pool.DATA / 'train-00000-of-00001.parquet').to_pylist()
    rows = sorted([r for r in rows if r['task_id'] not in original['train_task_ids']],
                  key=lambda r: pool.digest(f'{pool.SEED}:train:{r["task_id"]}'))
    assert len(rows) == 88 and not ({r['task_id'] for r in rows} & set(original['train_task_ids']))
    inputs, targets = [], []
    for row in rows:
        signatures = pool.interface(row)
        prompt = (row['prompt'] + '\n\nImplement the following Python interface:\n'
                  + '\n'.join(signatures)
                  + '\n\nReturn the complete solution in a single ```python code block.')
        inputs.append({'split': 'train', 'task_id': row['task_id'], 'prompt': prompt,
                       'interfaces': signatures, 'interface_source': 'Reference function signatures selected by function-name calls in test AST; no function bodies or assertions included.'})
        targets.append({'split': 'train', **row})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in [('inputs.jsonl', inputs), ('evaluator_targets.jsonl', targets)]:
        (OUT / name).write_text(''.join(json.dumps(r) + '\n' for r in content))
    protocol = {**{k: original[k] for k in ['seed','dataset','config','dataset_revision','model','model_revision','k','max_new_tokens','temperature','top_p','enable_thinking','logprobs','generation_seed_formula','grader']},
                'train_count': 88, 'train_task_ids': [r['task_id'] for r in rows],
                'excluded_original32_task_ids': original['train_task_ids'],
                'selection': 'All remaining88 official train tasks, determined by original frozen32 IDs only; no outcome-conditioned replacements or further sampling.',
                'original_protocol_sha256': pool.digest((ORIGINAL / 'protocol.json').read_text()),
                'frozen_inputs_sha256': pool.digest((OUT / 'inputs.jsonl').read_text()),
                'targets_sha256': pool.digest((OUT / 'evaluator_targets.jsonl').read_text()),
                'runner_sha256': pool.digest(Path(__file__).read_text()),
                'helper_sha256': pool.digest(Path(pool.__file__).read_text()),
                'physical_gpu': 2, 'smoke': 'First2 remaining tasks x4, technical duplicates, not additional full-pool candidates',
                'api_spend_usd': 0, 'scope': 'Additive code mechanism corpus; not SWE repair evidence',
                'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    pool.save(OUT / 'protocol.json', protocol)
    (OUT / 'generation_script.py').write_text(Path(__file__).read_text())
    (OUT / 'helper52_script.py').write_text(Path(pool.__file__).read_text())
    print(json.dumps({'frozen_train_tasks': len(rows), 'full_attempts': 704, 'gpu': 2}), flush=True)


def verify(smoke):
    from transformers import AutoTokenizer
    pool.summarize(smoke)
    prefix = 'smoke_' if smoke else ''
    raw = [json.loads(l) for l in (OUT / f'{prefix}raw.jsonl').open()]
    grades = [json.loads(l) for l in (OUT / f'{prefix}grades.jsonl').open()]
    tok = AutoTokenizer.from_pretrained(str(pool.MODEL),local_files_only=True)
    by = collections.defaultdict(list)
    for r in raw:
        assert tok.encode(r['rendered_prompt']) == r['prompt_token_ids']
        assert len(r['generated_token_ids']) == len(r['chosen_token_logprobs'])
        assert all(math.isfinite(x) and x <= 0 for x in r['chosen_token_logprobs'])
    for r in grades:
        by[r['task_id']].append(r['correct'])
    assert len(raw) == (8 if smoke else 704)
    assert len(by) == (2 if smoke else 88)
    assert all(len(v) == (4 if smoke else 8) for v in by.values())
    support = [{'task_id': k, 'correct': sum(v), 'n': len(v),
                'balanced_k4_subsets': math.comb(sum(v),2)*math.comb(len(v)-sum(v),2)
                if min(sum(v),len(v)-sum(v)) >= 2 else 0} for k,v in by.items()]
    result = {'technical_checks_passed': True, 'prompt_token_reencodings': len(raw),
              'finite_complete_logprob_arrays': len(raw), 'correct_count_histogram': dict(collections.Counter(sum(v) for v in by.values())),
              'mixed_tasks': sum(0<sum(v)<len(v) for v in by.values()),
              'balanced_k4_tasks': sum(s['balanced_k4_subsets']>0 for s in support),
              'balanced_k4_subsets': sum(s['balanced_k4_subsets'] for s in support),
              'per_task_support': support, 'api_spend_usd': 0,
              'calibration_measurement_test_generated_or_scored': False}
    pool.save(OUT / f'{prefix}validation.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_task_support'}),flush=True)
    if not smoke:
        files = ['protocol.json','inputs.jsonl','evaluator_targets.jsonl','generation_script.py','helper52_script.py','generation_manifest.json','raw.jsonl','grades.jsonl','summary.json','validation.json']
        pool.save(OUT / 'completed_manifest.json', {'phase':'generation complete','sha256':{f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files},'api_spend_usd':0})


def generate(smoke):
    from importlib.metadata import version
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'Expansion assigned physicalGPU2 only'
    for filename,key in [('inputs.jsonl','frozen_inputs_sha256'),('evaluator_targets.jsonl','targets_sha256')]:
        assert pool.digest((OUT / filename).read_text()) == protocol[key]
    assert pool.digest(Path(__file__).read_text()) == protocol['runner_sha256']
    assert pool.digest(Path(pool.__file__).read_text()) == protocol['helper_sha256']
    if not smoke:
        gate=json.loads((OUT / 'smoke_summary.json').read_text())
        assert gate['rows']==8 and gate['valid_python_rate']>=.5
        assert json.loads((OUT / 'smoke_validation.json').read_text())['technical_checks_passed']
    rows=[json.loads(l) for l in (OUT/'inputs.jsonl').open()][:2 if smoke else 88]
    targets={r['task_id']:r for l in (OUT/'evaluator_targets.jsonl').open() if (r:=json.loads(l))}
    prefix='smoke_' if smoke else ''
    rawpath,gradepath=OUT/f'{prefix}raw.jsonl',OUT/f'{prefix}grades.jsonl'
    done={(r['task_id'],r['sample']) for l in rawpath.open() if (r:=json.loads(l))} if rawpath.exists() else set()
    tok=AutoTokenizer.from_pretrained(str(pool.MODEL),local_files_only=True)
    jobs=[]
    for row in rows:
        prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        for sample in range(4 if smoke else 8):
            if (row['task_id'],sample) not in done:
                jobs.append({'task_id':row['task_id'],'sample':sample,'user_prompt':row['prompt'],'rendered_prompt':prompt})
    manifest={**protocol,'smoke':smoke,'dtype':'bfloat16','python':sys.version,
              'versions':{p:version(p) for p in ['torch','transformers','vllm']},
              'chat_template_sha256':pool.digest(tok.chat_template),'pid':os.getpid(),
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    pool.save(OUT/('smoke_manifest.json' if smoke else 'generation_manifest.json'),manifest)
    if not jobs:
        return verify(smoke)
    llm=LLM(model=str(pool.MODEL),tokenizer=str(pool.MODEL),dtype='bfloat16',gpu_memory_utilization=.8,max_model_len=2048,max_num_seqs=8,enforce_eager=True,disable_log_stats=True)
    start=time.time()
    with rawpath.open('a') as f, gradepath.open('a') as g:
        for offset in range(0,len(jobs),8):
            batch=jobs[offset:offset+8]
            params=[SamplingParams(temperature=1.,top_p=1.,max_tokens=768,seed=pool.SEED+j['task_id']*100+j['sample'],logprobs=1) for j in batch]
            replies=llm.generate([j['rendered_prompt'] for j in batch],params,use_tqdm=False)
            for job,reply in zip(batch,replies):
                seq=reply.outputs[0]
                assert seq.logprobs is not None
                lp=[float(p[t].logprob) for t,p in zip(seq.token_ids,seq.logprobs)]
                assert len(lp)==len(seq.token_ids)
                record={**job,'prompt_token_ids':reply.prompt_token_ids,'generation':seq.text,
                        'generated_token_ids':list(seq.token_ids),'chosen_token_logprobs':lp,
                        'finish_reason':seq.finish_reason,'stop_reason':seq.stop_reason}
                f.write(json.dumps(record)+'\n');f.flush()
                g.write(json.dumps({'task_id':job['task_id'],'sample':job['sample'],**pool.grade(seq.text,targets[job['task_id']])})+'\n');g.flush()
            print(json.dumps({'completed':len(done)+offset+len(batch),'total':len(done)+len(jobs),'seconds':round(time.time()-start,1)}),flush=True)
    verify(smoke)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['prepare','generate','verify'])
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='generate':generate(args.smoke)
    else:verify(args.smoke)
