#!/usr/bin/env python3
"""Combined G1 E2 report: natural prevalence audit + plan-conditioned AppWorld batches.

Label/quality = majority of gpt56/kimi/opus; human spot-check quality flips override the majority.
Writes data/g1/g1_pair_labels.jsonl (one row per judged pair) and reports/G1_E2_GENERATION_SUMMARY.md.
  ../.venv/bin/python scripts/summarize_g1_generation.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGES = ['gpt56', 'kimi', 'opus']
SETS = [('prevalence', 'data/g1/prevalence_audit', None), ('batch1', 'data/g1/generated_audit', 'pilot'),
        ('batch2', 'data/g1/generated_audit_b2', 'batch2')]
PRICE = {'gpt56': (4.0, 20.0), 'haiku': (1.0, 5.0)}
maj = lambda xs: (lambda c: c[0][0] if c[0][1] * 2 > len(xs) else 'split')(Counter(xs).most_common())
human = {json.loads(l)['pair_id']: json.loads(l) for l in open(ROOT / 'data/g1/review_digest/judge_spotcheck_Lan.jsonl')}

rows = []
for name, d, _ in SETS:
    d = ROOT / d
    hidden = {x['pair_id']: x for x in json.load(open(d / 'provenance_hidden.json'))['items']}
    J = {j: {json.loads(l)['pair_id']: json.loads(l) for l in open(d / 'judgments' / f'{j}.jsonl')} for j in JUDGES}
    for pid, h in hidden.items():
        js = {j: J[j][pid] for j in JUDGES}
        r = {'set': name, 'domain': 'telecom' if pid.startswith('tc') else 'appworld', 'pair_id': pid, 'task_id': h['task_id'],
             'A': h['A'], 'B': h['B'], 'proxy': h['proxy'], 'votes': {j: v['label'] for j, v in js.items()},
             'label': maj([v['label'] for v in js.values()]),
             'a_hq': maj([v['a_high_quality'] for v in js.values()]), 'b_hq': maj([v['b_high_quality'] for v in js.values()]),
             'human': None}
        if pid in human:
            hu = human[pid]
            r['human'] = {k: hu.get(k) for k in ('label_review', 'a_quality_review', 'b_quality_review', 'note')}
            for s in 'ab':
                if hu.get(f'{s}_quality_review') == 'disagree':
                    r[f'{s}_hq'] = {'yes': 'no', 'no': 'yes'}.get(r[f'{s}_hq'], r[f'{s}_hq'])
            if hu['label_review'] in ('E0', 'E1', 'E2'):
                r['label'] = hu['label_review']
        r['hq_e2'] = r['label'] == 'E2' and r['a_hq'] == r['b_hq'] == 'yes'
        rows.append(r)
with open(ROOT / 'data/g1/g1_pair_labels.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')

L = ['# G1 E2 数据：自然分布与定向生成汇总', '',
     '由 `scripts/summarize_g1_generation.py` 生成，逐对标签见 `data/g1/g1_pair_labels.jsonl`。',
     '标签和质量都取 GPT-5.6 / Kimi / Opus 三个 judge 的多数票；人工抽查过的对，按人工意见覆盖质量判断。',
     'HQ-E2 = 多数票为 E2，且两条轨迹都是高质量。', '']

L += ['## 1. 自然分布（公开基线轨迹）', '',
      '每个领域抽 24 对 proxy-E2（写操作集合不同）和 6 对 proxy-E0/E1 对照，每个任务最多一对。', '',
      '| 领域 | proxy 组 | 对数 | 多数票 E2 | HQ-E2 |', '|---|---|---|---|---|']
for dom in ('appworld', 'telecom'):
    for px in ('E2', 'E0/E1'):
        s = [r for r in rows if r['set'] == 'prevalence' and r['domain'] == dom and r['proxy'] == px]
        L.append(f"| {dom} | {px} | {len(s)} | {sum(r['label'] == 'E2' for r in s)} | {sum(r['hq_e2'] for r in s)} |")
L += ['', '结论：AppWorld 自然 HQ-E2 约占成功轨迹对的 0.5–0.9%（见 experiment_registry G1-PREV-001），telecom 为 0。', '']

L += ['## 2. 定向生成（方案驱动的 AppWorld rollout）', '',
      '流程：GPT-5.6 为每个任务写最多 3 个决策不同的方案 → GPT-5.6 和 Haiku 4.5 分别在"无方案"和"每个方案"条件下执行 →',
      'AppWorld 状态测试判成功 → 用写操作集合（含布尔参数实际值）初筛 → 三方 judge 盲审。', '',
      '| 步骤 | 脚本（`trace-main/scripts/`） | 命令示例（batch2） |', '|---|---|---|',
      '| 1. 写方案 | `appworld_targeted_rollout.py --stage plans` | `APPWORLD_ROOT=envs/appworld_root envs/appworld/bin/python scripts/appworld_targeted_rollout.py --stage plans --run batch2 --only other --axes v2` |',
      '| 2. 按方案执行（造 E2 的核心） | `appworld_targeted_rollout.py --stage rollouts` | `... --stage rollouts --run batch2 --only other --workers 4` |',
      '| 3. 成功率与 proxy 初筛 | `summarize_appworld_targeted.py` | `../.venv/bin/python scripts/summarize_appworld_targeted.py --run batch2` |',
      '| 4. 挑选并打包轨迹对 | `build_generated_review.py` | `... scripts/build_generated_review.py --run batch2 --out generated_audit_b2 --prefix g2` |',
      '| 5. 三方 judge | `run_prevalence_judge.py` | `... scripts/run_prevalence_judge.py --judge {gpt56,kimi,opus} --input data/g1/generated_audit_b2/judge_packets.jsonl` |',
      '| 6. 中文摘要 + 人工抽查页 | `build_judge_digest.py` | `... scripts/build_judge_digest.py`（页面 `annotation/judge_review.html`） |',
      '| 7. 本报告 | `summarize_g1_generation.py` | `... scripts/summarize_g1_generation.py` |', '',
      'batch1 用 `--run pilot --only route`（默认 `--axes v1`：route / information_source / recovery）；',
      'batch2 用 `--axes v2`（route / subgoal_order / write_parameters / recovery）。自然分布部分用 `replay_appworld.py`、`build_prevalence_audit.py`、`summarize_prevalence_audit.py`。', '']
for name, _, run in SETS[1:]:
    rd = ROOT / 'outputs/appworld_targeted' / run
    plans = json.load(open(rd / 'plans.json'))
    recs = [json.load(open(f)) for f in sorted(rd.glob('*/*/rollout.json'))]
    recs = [r for r in recs if (r['task_id'] in plans) and (plans[r['task_id']]['route_task'] == (name == 'batch1'))]
    L += [f'### {name}（{len(recs)} 次 rollout，{len({r["task_id"] for r in recs})} 个任务）', '',
          '| 模型 | 条件 | 成功 | 费用 |', '|---|---|---|---|']
    for m in ('gpt56', 'haiku'):
        rs = [r for r in recs if r['model'] == m]
        cost = sum(r['usage']['in'] * PRICE[m][0] + r['usage']['out'] * PRICE[m][1] for r in rs) / 1e6
        for c, sub in (('无方案', [r for r in rs if not r['plan']]), ('有方案', [r for r in rs if r['plan']])):
            L.append(f"| {m} | {c} | {sum(r['success'] for r in sub)}/{len(sub)} | {'$%.2f' % cost if c == '无方案' else ''} |")
    s = [r for r in rows if r['set'] == name]
    L += ['', '| proxy 组 | 对数 | 多数票 E2 | HQ-E2 |', '|---|---|---|---|']
    for px in sorted({r['proxy'] for r in s}):
        t = [r for r in s if r['proxy'] == px]
        L.append(f"| {px} | {len(t)} | {sum(r['label'] == 'E2' for r in t)} | {sum(r['hq_e2'] for r in t)} |")
    L.append('')

hq = [r for r in rows if r['hq_e2']]
# no trajectory content here: AppWorld derivatives may only be redistributed encrypted, and this repo is public
L += ['## 3. 全部 HQ-E2 对', '', '| 来源 | pair | 任务 | A | B | 票（gpt56/kimi/opus） |', '|---|---|---|---|---|---|']
for r in hq:
    v = ''.join({'E0': '0', 'E1': '1', 'E2': '2'}.get(x, '?') for x in r['votes'].values())
    L.append(f"| {r['set']} | {r['pair_id']} | {r['task_id']} | {r['A']} | {r['B']} | {v} |")
scen = Counter(r['task_id'].split('_')[0] for r in hq)
L += ['', f"共 {len(hq)} 对 HQ-E2，来自 {len({r['task_id'] for r in hq})} 个任务、{len(scen)} 类场景：{dict(scen)}。", '',
      '## 4. 要点', '',
      '- 自然状态下几乎全是 E1，HQ-E2 不到 1%；E0 未单独统计（需要每一步操作完全相同，自然轨迹中几乎没有）。',
      '- 给方案能稳定造出 E2，但只在少数任务上有效：多数任务 GPT-5.6 认为只有一种合理做法。',
      '- E2 种类集中：大部分是文件系统的不同写法（移动 vs 复制+删除、暂存目录、原地写 vs 删除重建）。',
      '- 写操作集合 proxy 会漏掉不改变写集合的决策差异（如 g2-12：只关注未关注的歌手 vs 全部关注），也会误报。',
      '- judge 标签与人工抽查 12/12 一致（人工先看到了 judge 结论，可能偏高）；单个 judge 里 Opus 最接近人工，GPT-5.6 偏向判 E2。',
      '- 原始 rollout（`outputs/`）和 AppWorld 衍生数据（轨迹包、judge 输出、摘要）只在本地，不入库：AppWorld 许可要求其数据和衍生内容只能加密公开。', '',
      '## 5. 下一步（待批准）', '',
      '- 用现有 HQ-E2 + 配对 E1 做小规模三元组试验，检验语义距离指标是否优于文本/动作基线。',
      '- 试点"锚点改写"：从成功轨迹出发，确定性改出 E1，在决策点分叉改出 E2，重放验证。', '']
(ROOT / 'reports/G1_E2_GENERATION_SUMMARY.md').write_text('\n'.join(L))
print(len(rows), 'pairs;', len(hq), 'HQ-E2;', dict(Counter(r['set'] for r in hq)))
