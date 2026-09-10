#!/usr/bin/env python3
"""Independent fixed-slot, provenance and token-pricing audit; no model calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'runs/swe-diversity-selection/swe-fresh-development'
GEN = COHORT / 'generation'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = GEN / 'independent_generation_audit.json'
    assert not out.exists()
    protocol = read(GEN / 'protocol.json')
    ledger = read(GEN / 'ledger.json')
    jobs = read(GEN / 'jobs.json')
    patches = read(GEN / 'patch_manifest.json')
    assert len(jobs) == len(patches) == len(ledger['requests']) == 48
    assert len({j['instance_id'] for j in jobs}) == 6
    assert all(sum(j['instance_id'] == task for j in jobs) == 8 for task in {j['instance_id'] for j in jobs})
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    rates = protocol['pricing_per_million']
    records = []
    for job, patch in zip(jobs, patches):
        assert job['key'] == patch['key']
        row = ledger['requests'][job['key']]
        path = GEN / 'responses' / (job['key'] + '.json')
        assert sha(path) == row['response_sha256']
        assert row['state'] == 'accounted' and not row['bound_violation']
        response = read(path)
        assert response['response_status'] == 'completed' and response['returned_model'] == protocol['model']
        raw = response['response']
        assert raw['service_tier'] == 'default'
        usage = raw['usage']
        inp, output = usage['input_tokens'], usage['output_tokens']
        details = usage['input_tokens_details']
        cache, writes = details.get('cached_tokens', 0), details.get('cache_write_tokens', 0)
        assert all(isinstance(x, int) and x >= 0 for x in [inp, output, cache, writes])
        uncached = inp - cache - writes
        assert uncached >= 0
        cost = (uncached*rates['input'] + cache*rates['cached_input'] + writes*rates['cache_write_input_upper'] + output*rates['output']) / 1e6
        upper = (inp*rates['cache_write_input_upper'] + output*rates['output']) / 1e6
        assert abs(upper-row['conservative_usage_usd']) < 1e-12
        assert cost <= upper + 1e-12 <= job['reservation_usd'] + 1e-12
        if patch['valid']:
            assert sha(GEN / 'patches' / (job['key']+'.patch')) == patch['patch_sha256']
        records.append({'key':job['key'], 'input_tokens':inp, 'cached_input_tokens':cache,
            'cache_write_input_tokens':writes, 'uncached_input_tokens':uncached, 'output_tokens':output,
            'reasoning_tokens':usage['output_tokens_details']['reasoning_tokens'],
            'usage_pricing_estimate_usd':cost, 'ledger_conservative_usage_usd':upper,
            'elapsed_seconds':response['elapsed_seconds']})
    estimate = sum(r['usage_pricing_estimate_usd'] for r in records)
    upper = sum(r['ledger_conservative_usage_usd'] for r in records)
    budget = read(COHORT.parent / 'paper-program/budget.json')
    before = read(GEN / 'budget_before.json')
    assert abs(budget['new_recorded_usd'] - before['new_recorded_usd'] - upper) < 1e-9
    assert budget['new_outstanding_reservations_usd'] == before['new_outstanding_reservations_usd']
    result = {'all_checks_passed':True, 'slots':48, 'tasks':6, 'completed_responses':48,
        'constructed_patches':sum(r['valid'] for r in patches), 'unknown_usage_requests':0,
        'usage_pricing_estimate_usd':estimate, 'conservative_ledger_new_usd':upper,
        'conservative_program_total_usd':budget['authorized_total_usd']-budget['conservative_remaining_usd'],
        'remaining_authorized_usd':budget['conservative_remaining_usd'],
        'pricing_source':protocol['price_source'], 'pricing_note':'Estimate from recorded standard-tier token usage, explicitly including cache writes; not an invoice. Frozen92 standard_pricing_estimate omits cache-write uplift; use this corrected estimate or its conservative ledger upper.',
        'records':records, 'source_sha256':sha(Path(__file__)), 'candidate_correctness_read':False,
        'api_calls':0, 'gpu_used':False}
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
