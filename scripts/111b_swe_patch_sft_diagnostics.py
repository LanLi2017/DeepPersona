#!/usr/bin/env python3
"""Posthoc source-only diagnosis; never changes retrieval, targets or admission."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/swe-diversity-selection/swe-sympy-patch-sft'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=read(OUT/'source_targets/summary.json')
assert summary['technical_failures']==0 and summary['prequalified']==5
rows=[]
for task in summary['tasks']:
    iid=task['instance_id'];record=read(OUT/'source_targets'/iid/'record.json')
    if not record['conversion_supported']:
        rows.append(dict(instance_id=iid,category='conversion_unsupported',errors=record['errors']))
        continue
    context=read(OUT/'context'/(iid+'.json'));visible=defaultdict(list)
    for s in context['retrieval']['source_spans']: visible[s['path']].append((s['start_line'],s['end_line']))
    anchors=[]
    for edit in record['edits']:
        data=(OUT/'public'/iid/'source'/edit['path']).read_bytes();old=edit['old'].encode()
        assert data.find(old)>=0 and data.find(old)==data.rfind(old)
        start=data.index(old);end=start+len(old)
        a=data[:start].count(b'\n')+1;b=data[:end-1].count(b'\n')+1
        fits=any(x<=a<=b<=y for x,y in visible[edit['path']])
        anchors.append(dict(path=edit['path'],start_line=a,end_line=b,file_retrieved=bool(visible[edit['path']]),fully_visible=fits))
    missing=sorted({a['path'] for a in anchors if not a['file_retrieved']})
    category='missing_changed_file' if missing else ('missing_anchor_region_in_retrieved_file' if not all(a['fully_visible'] for a in anchors) else 'all_anchors_visible')
    assert all(a['fully_visible'] for a in anchors)==record['anchor_visible']
    rows.append(dict(instance_id=iid,category=category,missing_changed_files=missing,anchors=anchors,
        completion_tokens=record['target_tokens_including_eos'],completion_fits=record['completion_fits'],prequalified=record['prequalified']))
result=dict(posthoc=True,scope='Only already-opened32source targets. Exact anchor visibility; no claim that every unobserved anchor byte changes behavior. No evaluation gold, retrieval rerank, context expansion, training or replacement.',
    categories=dict(Counter(r['category'] for r in rows)),tasks=rows,
    source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),OUT/'source_targets/summary.json',OUT/'source_targets/completed_manifest.json',OUT/'context/manifest.json']},
    interpretation='A missing-file case cannot be repaired solely by adding more of its already retrieved files. Within-file anchor misses can reflect omitted changed code or surrounding unchanged context required by this exact converter; this analysis does not distinguish those possibilities.',api_spend_usd=0,gpu_used=False)
with (OUT/'source_targets/retrieval_diagnostics.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['categories']))
