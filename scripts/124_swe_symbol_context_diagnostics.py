#!/usr/bin/env python3
"""Posthoc coverage diagnosis using previously audited source change locations."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/swe-diversity-selection'
OUT=BASE/'swe-sympy-symbol-context'
PRIOR=BASE/'swe-sympy-retrieval-policies'
CAPACITY=BASE/'swe-sympy-context-capacity'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (OUT/'coverage_diagnostics.json').exists()
    audit=read(OUT/'independent_policy_score_audit.json')
    assert audit['all_checks_passed']
    assert audit['score_summary_sha256']==sha(OUT/'score_summary.json')
    assert audit['build_manifest_sha256']==sha(OUT/'build_manifest.json')
    summary=read(OUT/'score_summary.json')
    build=read(OUT/'build_manifest.json')
    old_path=CAPACITY/'changed_region_diagnostics.json'
    old=read(old_path)
    milestone=read(BASE/'paper-program/milestone_nine.json')
    assert sha(old_path)==milestone['evidence_sha256'][str(old_path.relative_to(ROOT))]
    for name,value in old['inputs_sha256'].items():assert sha(name)==value,name
    for name,value in read(OUT/'completed_manifest.json').items():assert sha(name)==value,name
    prior={r['instance_id']:r for r in old['tasks']}
    previous=read(PRIOR/'score_summary.json')
    for name,digest in read(PRIOR/'completed_manifest.json').items():assert sha(Path(name))==digest,name
    best=next(p for p in previous['policies'] if p['policy']=='production_files')
    best_ids={r['instance_id'] for r in best['tasks'] if r['prequalified']}
    assert len(best_ids)==12
    baseline={i for i,r in prior.items() if r['full_anchors_visible'] and r['completion_fits']}
    assert len(baseline)==8
    outputs=[]
    inputs={str(p):sha(p) for p in [Path(__file__),old_path,OUT/'independent_policy_score_audit.json',OUT/'score_summary.json',OUT/'build_manifest.json',OUT/'completed_manifest.json',PRIOR/'score_summary.json',PRIOR/'completed_manifest.json']}
    for policy in summary['policies']:
        rows=[]
        for score in policy['tasks']:
            iid=score['instance_id'];r=prior[iid]
            cp=OUT/policy['policy']/(iid+'.json');c=read(cp)
            assert sha(cp)==build['artifact_sha256'][str(cp)]
            inputs[str(cp)]=sha(cp)
            assert c['status']=='ready' and c['role']=='source'
            intervals=defaultdict(list)
            for span in c['retrieval']['source_spans']:
                intervals[span['path']].append((span['start_line'],span['end_line']))
            def visible(path,line):return any(a<=line<=b for a,b in intervals[path])
            missing_files=[f['path'] for f in r['files'] if not intervals[f['path']]]
            missing_deletions=[dict(path=f['path'],line=line) for f in r['files'] for line in f['deleted_lines'] if not visible(f['path'],line)]
            missing_boundaries=[dict(path=f['path'],boundary=b['boundary_after_old_line']) for f in r['files'] for b in f['insertion_boundaries'] if not any(visible(f['path'],line) for line in b['adjacent_old_lines'])]
            missing_anchors=[a for a in r['anchors'] if not any(x<=a['start_line']<=a['end_line']<=y for x,y in intervals[a['path']])]
            full=r['conversion_supported'] and not missing_anchors
            assert full==score['anchor_visible']
            category=('conversion_unsupported' if not r['conversion_supported'] else 'missing_changed_file' if missing_files else 'missing_anchor_region_in_retrieved_file' if not full else 'all_anchors_visible')
            locality=not (missing_files or missing_deletions or missing_boundaries)
            accepted=[e['path'] for e in c['retrieval']['selection_events'] if e.get('kind')=='symbol_block']
            rows.append(dict(instance_id=iid,category=category,prequalified=score['prequalified'],missing_changed_files=missing_files,missing_deletions=missing_deletions,missing_insertion_boundaries=missing_boundaries,missing_anchors=missing_anchors,locality_and_completion_fit=locality and r['completion_fits'] is True,full_anchor_visible=full,accepted_symbol_blocks=accepted,prompt_tokens=c['prompt_tokens'],seed_nodes=len(c['retrieval']['symbol_graph']['seeds']),neighbor_nodes=len(c['retrieval']['symbol_graph']['one_hop_neighbors']),attempt_limit_reached=c['retrieval']['symbol_graph']['attempt_limit_reached']))
        passing={r['instance_id'] for r in rows if r['prequalified']}
        outputs.append(dict(policy=policy['policy'],categories=dict(Counter(r['category'] for r in rows)),prequalified=len(passing),gained_vs_previous_best=sorted(passing-best_ids),lost_vs_previous_best=sorted(best_ids-passing),gained_vs_mixed_chunks=sorted(passing-baseline),lost_vs_mixed_chunks=sorted(baseline-passing),locality_and_completion_fit=sum(r['locality_and_completion_fit'] for r in rows),accepted_symbol_block_events=sum(len(r['accepted_symbol_blocks']) for r in rows),zero_seed_tasks=sum(r['seed_nodes']==0 for r in rows),attempt_limit_tasks=sum(r['attempt_limit_reached'] for r in rows),tasks=rows))
    result=dict(posthoc=True,source_slots=32,policies=outputs,inputs_sha256=inputs,interpretation='Reuses audited114b original change locations; no new private data. Locality is a proxy, not unique anchors or semantic sufficiency. These are exposed-source development results.',contexts_modified=False,evaluation_private_values_read=False,api_spend_usd=0,gpu_used=False)
    with (OUT/'coverage_diagnostics.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in p.items() if k!='tasks'} for p in outputs]))

if __name__=='__main__':main()
