#!/usr/bin/env python3
"""Prospective fixed40 autonomous inference slots; no private evaluator interface."""
import argparse
from contextlib import nullcontext
from collections import defaultdict
import difflib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'runs/swe-diversity-selection/swe-localize-repair-support'
OUT = SUPPORT / 'autonomous_inference'
EVAL = SUPPORT / 'evaluation_readiness'
READY = SUPPORT / 'evaluation_readiness_versions'
TRAIN = SUPPORT / 'full_training'
PUBLIC = SUPPORT.parent / 'swe-sympy-patch-sft/public'
GPU_UUID = 'GPU-b375cb5f-fe0d-e6d0-3366-e32427b12c66'
LIMITS = dict(global_wall_seconds=7200, episode_wall_seconds=180, max_assistant_turns=8,
              max_tool_attempts=7, max_completion_tokens=2048, max_action_tokens=256,
              max_prompt_tokens=24576, max_history_tokens=26624,
              max_episode_completion_tokens=16384, total_completion_tokens=655360)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


k = load('inference125', ROOT / 'scripts/125_swe_localize_repair_support.py')
ops = load('inference_ops', ROOT / 'scripts/swe_localize_learner_ops.py')
sha, read = k.sha, k.read


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def verify(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT / name) == digest, name


def schedule(ids):
    return [dict(slot=i*2+j, instance_id=iid, arm=arm,
                 episode_id=f'{iid}__{arm}')
            for i, iid in enumerate(ids)
            for j, arm in enumerate(('base', 'trained') if i % 2 == 0 else ('trained', 'base'))]


def unique_object(pairs):
    obj = {}
    for name, value in pairs:
        if name in obj:
            raise ValueError('duplicate_json_key')
        obj[name] = value
    return obj


def parse_response(text):
    match = re.fullmatch(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    value = json.loads(match.group(1) if match else text, object_pairs_hook=unique_object)
    if match:
        if not isinstance(value, dict) or set(value) != {'name', 'arguments'}:
            raise ValueError('tool_schema')
        return 'tool', value
    if not isinstance(value, dict) or list(value) != ['rationale', 'edits'] or value['rationale'] != '':
        raise ValueError('final_schema')
    if not isinstance(value['edits'], list) or not 1 <= len(value['edits']) <= 20:
        raise ValueError('edit_count')
    for edit in value['edits']:
        if not isinstance(edit, dict) or list(edit) != ['path', 'old', 'new']:
            raise ValueError('edit_schema')
        if not all(isinstance(edit[key], str) for key in edit):
            raise ValueError('edit_type')
    if k.TARGETS.canonical_target(value['edits']) != text:
        raise ValueError('noncanonical_final')
    return 'final', value


def action_allowed(action, initial, tiles):
    name, args = action['name'], action['arguments']
    if name not in ('outline_source', 'read_source') or not isinstance(args, dict):
        return False
    keys = {'path'} if name == 'outline_source' else {'path', 'start', 'end'}
    if set(args) != keys or not isinstance(args['path'], str) or args['path'] not in initial['production_files']:
        return False
    return name == 'outline_source' or (type(args['start']) is int and type(args['end']) is int
                                       and (args['path'], args['start'], args['end']) in tiles)


def unified_patch(lines):
    return ''.join(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n' for line in lines)


def episode(slot, initial, tok, generate, folder, clock=time.monotonic):
    """generate(ids, episode_deadline) returns raw generated IDs; tool execution is read-only."""
    started = clock()
    deadline = started + LIMITS['episode_wall_seconds']
    history = list(initial['messages'])
    tiles, spans = set(), defaultdict(list)
    turns, attempts, total = [], 0, 0
    status, patch, detail = 'turn_limit', None, None
    if not initial['initial_index_fits']:
        status = 'initial_overflow'
    else:
        for index in range(8):
            prompt, ids = k.render(history, tok)
            if index == 0:
                assert ids == initial['prompt_token_ids'] and prompt == initial['prompt']
            if len(ids) > 24576 or len(ids) + 2048 > 26624:
                status = 'history_overflow'
                break
            if clock() >= deadline:
                status = 'episode_timeout'
                break
            ids_out = generate(ids, deadline)
            total += len(ids_out)
            body_ids = ids_out[:-1] if ids_out and ids_out[-1] == tok.eos_token_id else ids_out
            text = tok.decode(body_ids, skip_special_tokens=False)
            turn = dict(turn=index, prompt_token_ids=ids, prompt_sha256=k.digest(prompt.encode()),
                        completion_token_ids=ids_out, text=text, elapsed_seconds=clock()-started)
            turns.append(turn)
            save(folder / f'turn_{index:02d}.json', turn)
            if clock() >= deadline:
                status = 'episode_timeout'
                break
            if total > 16384 or len(ids_out) > 2048 or len(ids)+len(ids_out) > 26624:
                status = 'completion_overflow'
                break
            if not ids_out or ids_out[-1] != tok.eos_token_id or tok.eos_token_id in ids_out[:-1]:
                status = 'incomplete_response'
                break
            try:
                kind, value = parse_response(text)
            except (ValueError, TypeError, KeyError):
                attempts += int(attempts < 7)
                status = 'invalid_response'
                break
            if kind == 'final':
                try:
                    names = {edit['path'] for edit in value['edits']}
                    if not names.issubset(initial['production_files']):
                        raise ValueError('unsupported_edit_path')
                    base = {name: k.source_bytes(PUBLIC/slot['instance_id'], name, initial) for name in names}
                    changed = k.TARGETS.apply_edits(base, value['edits'])
                    patch = ''.join(unified_patch(difflib.unified_diff(
                        base[name].decode().splitlines(keepends=True), changed[name].decode().splitlines(keepends=True),
                        fromfile='a/'+name, tofile='b/'+name)) for name in sorted(names))
                    # Whole-old-anchor visibility is a diagnostic, not an additional SYSTEM rule.
                    try:
                        k.TARGETS.validate_visibility(base, value['edits'], {name:k.merge_intervals(intervals) for name,intervals in spans.items()})
                        visible = True
                    except k.TARGETS.TargetError:
                        visible = False
                    save(folder/'final_edits.json', dict(**value, all_old_anchors_read=visible,
                         patched_sha256={name:k.digest(data) for name,data in changed.items()}))
                    (folder/'candidate.patch').write_bytes(patch.encode())
                    status = 'final_patch' if patch else 'empty_patch'
                except (k.TARGETS.TargetError, ValueError) as exc:
                    status, detail = 'invalid_final', str(exc)
                break
            if attempts == 7:
                status = 'tool_budget_exhausted'
                break
            attempts += 1
            if len(ids_out) > 256 or not action_allowed(value, initial, tiles):
                status = 'invalid_tool'
                break
            observation = k.tool_output(PUBLIC/slot['instance_id'], value, initial, tiles)
            stats = k.observation_stats(observation, tok)
            save(folder/f'observation_{index:02d}.json', dict(action=value, output=observation, **stats,
                 delivered=stats['fits'] and clock() < deadline))
            if clock() >= deadline:
                status = 'episode_timeout'
                break
            if not stats['fits']:
                status = 'observation_overflow'
                break
            if value['name'] == 'outline_source':
                tiles.update((value['arguments']['path'], tile['start'], tile['end']) for tile in observation['tiles'])
            else:
                args = value['arguments']
                spans[args['path']].append((args['start'], args['end']))
            history += [{'role':'assistant', 'content':'', 'tool_calls':[{'type':'function', 'function':value}]},
                        {'role':'tool', 'content':stats['text']}]
    result = dict(**slot, status=status, detail=detail, turns=len(turns), tool_attempts=attempts,
                  completion_tokens=total, elapsed_seconds=clock()-started,
                  deadline_seconds=180, episode_deadline_overrun=clock() > deadline,
                  patch_sha256=k.digest(patch.encode()) if patch is not None else None,
                  private_feedback_read=False, retry=False)
    if result['episode_deadline_overrun']:
        result['status'] = 'episode_timeout'
        result['patch_eligible_for_evaluation'] = False
    else:
        result['patch_eligible_for_evaluation'] = status == 'final_patch'
    save(folder/'result.json', result)
    return result


def freeze():
    manifest = read(EVAL/'initial_manifest.json')
    assert manifest['all20_public_initials_frozen'] and not manifest['evaluation_private_values_read']
    verify(manifest['source_sha256'])
    public_protocol = read(EVAL/'initial_protocol.json')
    verify(public_protocol['source_sha256'])
    rows = manifest['tasks']
    assert len(rows) == len({r['instance_id'] for r in rows}) == 20
    checks = read(OUT/'synthetic_checks.json')
    assert checks['all_checks_passed'] and checks['source_sha256'] == sha(Path(__file__))
    launcher_checks = read(OUT/'launcher_checks.json')
    assert launcher_checks['all_checks_passed'] and launcher_checks['launcher_sha256'] == sha(ROOT/'scripts/136_swe_localize_inference_launcher.py')
    files = [Path(__file__), ROOT/'scripts/136_swe_localize_inference_launcher.py', OUT/'launcher_checks.json', OUT/'audit_launcher.py',
             ROOT/'scripts/125_swe_localize_repair_support.py', ROOT/'scripts/110_swe_patch_sft_targets.py',
             ROOT/'scripts/swe_localize_learner_ops.py', ROOT/'scripts/132_swe_localize_full_train.py',
             EVAL/'initial_manifest.json', EVAL/'initial_protocol.json', OUT/'synthetic_checks.json',
             ROOT/'docs/swe_localize_inference_plan.md', ROOT/'docs/swe_localize_candidate_scoring_plan.md']
    files += [ROOT/name for name in public_protocol['source_sha256']]
    software = read(SUPPORT/'learner_smoke/protocol.json')['software']
    assert {name:importlib.metadata.version(name) for name in software} == software
    tok = k.tokenizer()
    for row in rows:
        path = ROOT / row['path']
        assert sha(path) == row['sha256']
        init = read(path)
        assert init['role'] == 'evaluation' and init['instance_id'] == row['instance_id']
        assert k.render(init['messages'], tok) == (init['prompt'], init['prompt_token_ids'])
        assert init['messages'][0] == dict(role='system', content=k.SYSTEM)
        files.append(path)
    files += sorted(p for p in k.MODEL.iterdir() if p.is_file())
    save(OUT/'protocol.json', dict(
        assigned_evaluation_slots=20, arms=['base','trained'], episode_slots=40,
        evaluation_initial_manifest_sha256=sha(EVAL/'initial_manifest.json'),
        prospective_checkpoint_path=str(TRAIN/'run/final_adapter'),
        source_sha256={str(p):sha(p) for p in files}, schedule=schedule([r['instance_id'] for r in rows]),
        limits=LIMITS, software=software, initial_manifest_tasks=rows, hardware=dict(gpu=2, uuid=GPU_UUID),
        precision='BF16 base, FP32 LoRA, FLASH-only SDPA; TF32 off; no autocast',
        decoding=dict(do_sample=False, num_beams=1, use_cache=True, eos_token_id=tok.eos_token_id),
        episode_deadline='Cooperative check before/after forwards and tools; any overrun invalidates the episode. No retry.',
        external_timeout=dict(term_seconds=7195, kill_grace_seconds=5, hard_seconds=7200),
        final_visibility='Report old-anchor read coverage; not a final-patch rejection rule.',
        failure_policy='Invalid tools terminate; repeated valid calls consume slots; all40 slots retained.',
        model_input_scope='Only frozen public initial plus actual outline/read observations; no quality fields.',
        checkpoint_binding='Separate launch_manifest after132 completes; no checkpoint selection.',
        api_spend_usd=0))


def launch_bind():
    p = read(OUT/'protocol.json')
    verify(p['source_sha256'])
    assert not (OUT/'launch_manifest.json').exists() and not (OUT/'run').exists()
    summary = read(TRAIN/'run/summary.json')
    assert summary['training_complete'] and summary['technical_checks_passed'] and summary['full_training_run']
    verify(read(TRAIN/'run/completed_manifest.json'))
    terminal = read(TRAIN/'outer_terminal.json')
    assert terminal['exit_code'] == 0 and not terminal['timed_out'] and terminal['error'] is None
    training_protocol = read(TRAIN/'protocol.json')
    assert terminal['protocol_sha256'] == sha(TRAIN/'protocol.json')
    assert training_protocol['inference_protocol_sha256'] == sha(OUT/'protocol.json')
    charged = [summary['charged_prior_seconds'], summary['total_elapsed_seconds'], terminal['elapsed_seconds']]
    assert all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in charged)
    assert charged[0] == training_protocol['charged_prior_seconds']
    assert charged[0] + max(charged[1:]) <= 3600, 'Actual total training budget exceeded'
    assert summary['shared_budget_seconds'] == training_protocol['total_shared_gpu_budget_seconds'] == 3600
    ready = read(READY/'summary.json')
    assert ready['all20_endpoints_ready'] and ready['readiness_gate_passed']
    files = [TRAIN/'protocol.json', TRAIN/'outer_terminal.json', TRAIN/'run/summary.json',
             TRAIN/'run/completed_manifest.json', TRAIN/'run/final_adapter/adapter_model.safetensors',
             TRAIN/'run/final_adapter/adapter_config.json', READY/'summary.json',
             SUPPORT/'learner_smoke/run/initial_adapter.pt']
    assert summary['final_adapter_sha256'] == sha(files[4])
    assert summary['protocol_sha256'] == sha(TRAIN/'protocol.json')
    assert summary['standard_peft_saved_reloaded_exact'] and summary['initial_reset_exact']
    assert summary['final_adapter_config_sha256'] == sha(files[5])
    save(OUT/'launch_manifest.json', dict(protocol_sha256=sha(OUT/'protocol.json'),
         checkpoint_and_readiness_sha256={str(path):sha(path) for path in files}, no_checkpoint_selection=True))


def worker():
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
    from torch.nn.attention import sdpa_kernel, SDPBackend
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2' and os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert os.environ['SWE_INFERENCE_EXTERNAL_TIMEOUT'] == '7195+5'
    p, launch = read(OUT/'protocol.json'), read(OUT/'launch_manifest.json')
    assert launch['protocol_sha256'] == sha(OUT/'protocol.json')
    verify(p['source_sha256'])
    assert {name:importlib.metadata.version(name) for name in p['software']} == p['software']
    verify(launch['checkpoint_and_readiness_sha256'])
    run = OUT/'run'
    run.mkdir()
    save(run/'started.json', dict(protocol_sha256=sha(OUT/'protocol.json'), launch_sha256=sha(OUT/'launch_manifest.json')))
    ops.configure_precision(GPU_UUID)
    tok = k.tokenizer()
    model = ops.load_base(k.MODEL)
    initial = torch.load(SUPPORT/'learner_smoke/run/initial_adapter.pt', map_location='cpu', weights_only=True)
    model, params = ops.attach_initial(model, initial)
    saved = load_file(str(TRAIN/'run/final_adapter/adapter_model.safetensors'))
    set_peft_model_state_dict(model, saved, adapter_name='default')
    actual = get_peft_model_state_dict(model)
    assert saved.keys() == actual.keys() and all(torch.equal(saved[n], v.detach().cpu()) for n,v in actual.items())
    assert all(v.dtype == torch.float32 for v in params.values())
    assert any(torch.count_nonzero(v) for n,v in saved.items() if '.lora_B.' in n)
    model.eval()
    before = ops.base_hash(model)
    trained_summary = read(TRAIN/'run/summary.json')
    assert before == trained_summary['frozen_base_parameter_sha256_before'] == trained_summary['frozen_base_parameter_sha256_after']
    signature = ops.storage_signature(model)
    save(run/'model_loaded.json', dict(checkpoint_loaded_exact=True,
         adapter_sha256=sha(TRAIN/'run/final_adapter/adapter_model.safetensors'),
         initial_adapter_sha256=sha(SUPPORT/'learner_smoke/run/initial_adapter.pt'),
         base_parameter_sha256=before, gpu_uuid=GPU_UUID,
         adapter_dtype='torch.float32', base_dtype='torch.bfloat16',
         base_arm='disable_adapter context', trained_arm='default active adapter',
         protocol_sha256=sha(OUT/'protocol.json')))
    trace, hooks = ops.dtype_hooks(model)
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=2048,
             eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, use_cache=True)

    class Stop(StoppingCriteria):
        def __init__(self, start, deadline):
            self.start, self.deadline = start, deadline
        def __call__(self, input_ids, scores, **kwargs):
            suffix = input_ids[0, self.start:]
            action = tok.decode(suffix.tolist(), skip_special_tokens=False).startswith('<tool_call>')
            return time.monotonic() >= self.deadline or (action and len(suffix) >= 256)

    def generate(ids, deadline):
        tokens = torch.tensor([ids], device='cuda', dtype=torch.long)
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                  generation_config=config, logits_to_keep=1,
                  stopping_criteria=StoppingCriteriaList([Stop(len(ids), deadline)]))
        return output[0, len(ids):].tolist()

    rows = {r['instance_id']:r for r in p['initial_manifest_tasks']}
    for slot in p['schedule']:
        folder = run/'episodes'/slot['episode_id']
        folder.mkdir(parents=True)
        save(folder/'started.json', dict(**slot, monotonic_started=time.monotonic()))
        context = model.disable_adapter() if slot['arm'] == 'base' else nullcontext()
        try:
            with context:
                episode(slot, read(ROOT/rows[slot['instance_id']]['path']), tok, generate, folder)
        except BaseException as exc:
            save(folder/'technical_failure.json', dict(type=type(exc).__name__, message=str(exc)))
            raise
    assert before == ops.base_hash(model) and signature == ops.storage_signature(model)
    if trace:
        ops.verify_dtypes(trace)
    for handle in hooks:
        handle.remove()
    save(run/'worker_summary.json', dict(all40_dispatched=True, checkpoint_loaded_exact=True,
         base_hash_before=before, base_hash_after=before, storage_version_dtype_unchanged=True,
         dtypes=trace, greedy=True, quality_feedback_read=False))


def collect():
    p = read(OUT/'protocol.json')
    rows = []
    for slot in p['schedule']:
        folder = OUT/'run/episodes'/slot['episode_id']
        if (folder/'result.json').exists():
            row = read(folder/'result.json')
        else:
            row = dict(**slot, status=('technical_failure' if (folder/'technical_failure.json').exists() else
                               'interrupted' if (folder/'started.json').exists() else 'unstarted'),
                       patch_eligible_for_evaluation=False)
        row['completion_tokens'] = sum(len(read(path)['completion_token_ids']) for path in folder.glob('turn_*.json'))
        rows.append(row)
    save(OUT/'summary.json', dict(assigned_tasks=20, assigned_episodes=40, episodes=rows,
         all_slots_retained=True, accuracy_scored=False, api_spend_usd=0,
         actual_total_completion_tokens=sum(r.get('completion_tokens', 0) for r in rows),
         outer_terminal_sha256=sha(OUT/'outer_terminal.json')))
    save(OUT/'completed_manifest.json', {str(p):sha(p) for p in OUT.rglob('*') if p.is_file()})


def checks():
    import tempfile
    tests = []
    def check(name, condition):
        assert condition, name
        tests.append(dict(name=name, passed=True))
    init = dict(production_files=['sympy/a.py'])
    action = dict(name='read_source', arguments=dict(path='sympy/a.py', start=1, end=160))
    check('read_requires_prior_outline', not action_allowed(action, init, set()))
    check('exact_shown_tile_allowed', action_allowed(action, init, {('sympy/a.py',1,160)}))
    wrong = dict(name='read_source', arguments=dict(path='sympy/a.py', start=True, end=160))
    check('bool_line_rejected', not action_allowed(wrong, init, {('sympy/a.py',1,160)}))
    for text in ['{"rationale":"","rationale":"","edits":[]}', '<tool_call>{}</tool_call><tool_call>{}</tool_call>',
                 '{"rationale":"hint","edits":[]}', '{"rationale":"","edits":[]}']:
        try:
            parse_response(text)
        except (ValueError, TypeError):
            tests.append(dict(name='malformed_or_noncanonical_rejected', passed=True))
        else:
            raise AssertionError(text)
    sched = schedule([str(i) for i in range(20)])
    check('fixed40_balanced_arm_order', len(sched)==40 and sum(r['arm']=='base' for r in sched)==20
          and [r['arm'] for r in sched[:4]] == ['base','trained','trained','base'])
    tok = k.tokenizer()
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        public = temp/'fake'
        (public/'source/sympy').mkdir(parents=True)
        blob = b'def f():\n    return 1\n'
        (public/'source/sympy/a.py').write_bytes(blob)
        save(public/'task.json', dict(base_commit='synthetic', source_sha256={'sympy/a.py':k.digest(blob)}))
        (public/'problem_statement.md').write_text('Fix public f. No locations supplied.')
        initial = k.initial(public, tok)
        global PUBLIC
        old_public = PUBLIC
        PUBLIC = temp
        outline = '<tool_call>\n'+k.compact(dict(name='outline_source', arguments=dict(path='sympy/a.py'))) +'\n</tool_call>'
        final = k.TARGETS.canonical_target([dict(path='sympy/a.py', old='return 1', new='return 2')])
        def run_case(name, texts, expected):
            queue = iter(texts)
            def generator(ids, deadline):
                return tok.encode(next(queue), add_special_tokens=False)+[tok.eos_token_id]
            folder = temp/name
            folder.mkdir()
            result = episode(dict(episode_id=name,instance_id='fake',arm='base'), initial, tok, generator, folder)
            check(name, result['status']==expected)
            return result
        try:
            r = run_case('seven_repeats_then_final', [outline]*7+[final], 'final_patch')
            check('seven_repeats_consumed', r['tool_attempts']==7 and r['turns']==8)
            r = run_case('eighth_tool_not_executed', [outline]*8, 'tool_budget_exhausted')
            check('no_eighth_observation', not (temp/'eighth_tool_not_executed/observation_07.json').exists())
            run_case('invalid_terminates', ['<tool_call>'+k.compact(action)+'</tool_call>'], 'invalid_tool')
            reading = '<tool_call>'+k.compact(dict(name='read_source', arguments=dict(path='sympy/a.py', start=1,end=2)))+'</tool_call>'
            run_case('outline_read_final', [outline, reading, final], 'final_patch')
            check('exact_public_observation', read(temp/'outline_read_final/observation_01.json')['output']['text'] == blob.decode())
            check('final_without_read_not_extra_rejection', read(temp/'seven_repeats_then_final/final_edits.json')['all_old_anchors_read'] is False)
            check('read_coverage_diagnostic_true', read(temp/'outline_read_final/final_edits.json')['all_old_anchors_read'] is True)
            folder = temp/'deadline'
            folder.mkdir()
            now = [0.0]
            def delayed(ids, deadline):
                now[0] = 181.0
                return tok.encode(final,add_special_tokens=False)+[tok.eos_token_id]
            expired = episode(dict(episode_id='deadline',instance_id='fake',arm='base'),initial,tok,delayed,folder,lambda:now[0])
            check('overrun_invalidates_emitted_final',expired['status']=='episode_timeout' and not expired['patch_eligible_for_evaluation'])
            folder = temp/'no_eos'
            folder.mkdir()
            incomplete = episode(dict(episode_id='no_eos',instance_id='fake',arm='base'),initial,tok,
                lambda ids, deadline:tok.encode(final,add_special_tokens=False),folder)
            check('missing_eos_terminal',incomplete['status']=='incomplete_response')
            folder = temp/'oversize_action'
            folder.mkdir()
            expanded = '<tool_call>'+(' '*3000)+k.compact(dict(name='outline_source',arguments=dict(path='sympy/a.py')))+(' '*3000)+'</tool_call>'
            # Token count is checked, not character count; force it beyond256 without invalidating JSON.
            while len(tok.encode(expanded,add_special_tokens=False)) <= 256:
                expanded = expanded.replace('<tool_call>','<tool_call> '+('\n '*300))
            over = episode(dict(episode_id='oversize_action',instance_id='fake',arm='base'),initial,tok,
                lambda ids, deadline:tok.encode(expanded,add_special_tokens=False)+[tok.eos_token_id],folder)
            check('oversize_action_not_executed',over['status'] in ('invalid_tool','completion_overflow') and not list(folder.glob('observation*')))
            check('no_newline_patch_marker', '\\ No newline at end of file' in unified_patch(['-old','+new']))
            check('initial_unchanged', initial == k.initial(public, tok))
            folder = temp/'initial_overflow'
            folder.mkdir()
            oversized = dict(initial, initial_index_fits=False)
            result = episode(dict(episode_id='initial_overflow',instance_id='fake',arm='base'),oversized,tok,
                lambda ids,deadline: (_ for _ in ()).throw(AssertionError('Must not generate')),folder)
            check('initial_overflow_kept_without_generation', result['status']=='initial_overflow' and result['turns']==0)
            folder = temp/'history_overflow'
            folder.mkdir()
            real_render = k.render
            def expanded_history(messages, tokenizer, generate=True):
                return real_render(messages,tokenizer,generate) if len(messages)==2 else ('synthetic overlimit', [1]*24577)
            k.render = expanded_history
            try:
                result = episode(dict(episode_id='history_overflow',instance_id='fake',arm='base'),initial,tok,
                    lambda ids,deadline:tok.encode(outline,add_special_tokens=False)+[tok.eos_token_id],folder)
                check('history_overflow_no_second_generation',result['status']=='history_overflow' and result['turns']==1)
            finally:
                k.render = real_render
            big_blob = (b'# '+b'x'*200+b'\n')*160
            (public/'source/sympy/a.py').write_bytes(big_blob)
            (public/'task.json').write_text(json.dumps(dict(base_commit='synthetic',source_sha256={'sympy/a.py':k.digest(big_blob)})))
            big_initial = k.initial(public,tok)
            folder = temp/'observation_overflow'
            folder.mkdir()
            requests = iter([outline, '<tool_call>'+k.compact(action)+'</tool_call>'])
            result = episode(dict(episode_id='observation_overflow',instance_id='fake',arm='base'),big_initial,tok,
                lambda ids,deadline:tok.encode(next(requests),add_special_tokens=False)+[tok.eos_token_id],folder)
            check('observation_overflow_not_delivered',result['status']=='observation_overflow' and not read(folder/'observation_01.json')['delivered'])
            check('native_prefix_exact', k.render(initial['messages'], tok)[1] == initial['prompt_token_ids'])
        finally:
            PUBLIC = old_public
    save(OUT/'synthetic_checks.json', dict(all_checks_passed=True, source_sha256=sha(Path(__file__)),
         tests=tests, model_loaded=False, gpu_used=False, actual_task_targets_read=False))
    print(json.dumps(dict(all_checks_passed=True, checks=len(tests))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['checks','freeze','bind','worker','collect'])
    globals()[parser.parse_args().stage]()
