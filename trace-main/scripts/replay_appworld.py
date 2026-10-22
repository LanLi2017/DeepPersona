#!/usr/bin/env python3
"""Replay a logged AppWorld api_calls.jsonl to recover responses, then re-evaluate.

Logged access tokens are stale on replay, so each app's token is swapped for the one
returned by the replayed /auth/token call.
  APPWORLD_ROOT=envs/appworld_root envs/appworld/bin/python scripts/replay_appworld.py <cfg_dir> <task_id> <out.json>
"""
import json, sys
from pathlib import Path
from appworld import AppWorld

cfg_dir, tid, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
calls = [json.loads(l) for l in open(cfg_dir / 'tasks' / tid / 'logs' / 'api_calls.jsonl')]
tokens, steps = {}, []
with AppWorld(task_id=tid, experiment_name=f'replay/{cfg_dir.name}', raise_on_failure=False) as w:
    for c in calls:
        app = c['url'].strip('/').split('/')[0]
        data = dict(c.get('data') or {})
        if 'access_token' in data and app in tokens:
            data['access_token'] = tokens[app]
        w.requester.reset_request_count()  # the 1000-request cap is per interaction; replay is one interaction
        try:
            r = getattr(w.requester, c['method'])(c['url'], data=data, raise_on_failure=False)
        except Exception as e:  # some APIs raise instead of returning an error payload; the agent saw this as an error
            r = {'error': f'{type(e).__name__}: {e}'}
        if c['url'].endswith('/auth/token') and isinstance(r, dict) and 'access_token' in r:
            tokens[app] = r['access_token']
        steps.append({'method': c['method'], 'url': c['url'],
                      'data': {k: v for k, v in (c.get('data') or {}).items() if k not in ('access_token', 'password')},
                      'response': r})
    w.execute("apis.supervisor.complete_task()")
    ok = w.evaluate().success
out.parent.mkdir(parents=True, exist_ok=True)
json.dump({'task_id': tid, 'config': cfg_dir.name, 'instruction': w.task.instruction, 'replay_success': ok, 'steps': steps},
          open(out, 'w'), default=str)
print(tid, cfg_dir.name, 'replay_success', ok, len(steps))
