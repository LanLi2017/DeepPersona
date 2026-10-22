# G1 E2 数据：自然分布与定向生成汇总

由 `scripts/summarize_g1_generation.py` 生成，逐对标签见 `data/g1/g1_pair_labels.jsonl`。
标签和质量都取 GPT-5.6 / Kimi / Opus 三个 judge 的多数票；人工抽查过的对，按人工意见覆盖质量判断。
HQ-E2 = 多数票为 E2，且两条轨迹都是高质量。

## 1. 自然分布（公开基线轨迹）

每个领域抽 24 对 proxy-E2（写操作集合不同）和 6 对 proxy-E0/E1 对照，每个任务最多一对。

| 领域 | proxy 组 | 对数 | 多数票 E2 | HQ-E2 |
|---|---|---|---|---|
| appworld | E2 | 24 | 13 | 3 |
| appworld | E0/E1 | 6 | 0 | 0 |
| telecom | E2 | 24 | 20 | 0 |
| telecom | E0/E1 | 6 | 4 | 0 |

结论：AppWorld 自然 HQ-E2 约占成功轨迹对的 0.5–0.9%（见 experiment_registry G1-PREV-001），telecom 为 0。

## 2. 定向生成（方案驱动的 AppWorld rollout）

流程：GPT-5.6 为每个任务写最多 3 个决策不同的方案 → GPT-5.6 和 Haiku 4.5 分别在"无方案"和"每个方案"条件下执行 →
AppWorld 状态测试判成功 → 用写操作集合（含布尔参数实际值）初筛 → 三方 judge 盲审。

| 步骤 | 脚本（`trace-main/scripts/`） | 命令示例（batch2） |
|---|---|---|
| 1. 写方案 | `appworld_targeted_rollout.py --stage plans` | `APPWORLD_ROOT=envs/appworld_root envs/appworld/bin/python scripts/appworld_targeted_rollout.py --stage plans --run batch2 --only other --axes v2` |
| 2. 按方案执行（造 E2 的核心） | `appworld_targeted_rollout.py --stage rollouts` | `... --stage rollouts --run batch2 --only other --workers 4` |
| 3. 成功率与 proxy 初筛 | `summarize_appworld_targeted.py` | `../.venv/bin/python scripts/summarize_appworld_targeted.py --run batch2` |
| 4. 挑选并打包轨迹对 | `build_generated_review.py` | `... scripts/build_generated_review.py --run batch2 --out generated_audit_b2 --prefix g2` |
| 5. 三方 judge | `run_prevalence_judge.py` | `... scripts/run_prevalence_judge.py --judge {gpt56,kimi,opus} --input data/g1/generated_audit_b2/judge_packets.jsonl` |
| 6. 中文摘要 + 人工抽查页 | `build_judge_digest.py` | `... scripts/build_judge_digest.py`（页面 `annotation/judge_review.html`） |
| 7. 本报告 | `summarize_g1_generation.py` | `... scripts/summarize_g1_generation.py` |

batch1 用 `--run pilot --only route`（默认 `--axes v1`：route / information_source / recovery）；
batch2 用 `--axes v2`（route / subgoal_order / write_parameters / recovery）。自然分布部分用 `replay_appworld.py`、`build_prevalence_audit.py`、`summarize_prevalence_audit.py`。

### batch1（84 次 rollout，12 个任务）

| 模型 | 条件 | 成功 | 费用 |
|---|---|---|---|
| gpt56 | 无方案 | 12/12 | $10.90 |
| gpt56 | 有方案 | 29/30 |  |
| haiku | 无方案 | 9/12 | $5.36 |
| haiku | 有方案 | 21/30 |  |

| proxy 组 | 对数 | 多数票 E2 | HQ-E2 |
|---|---|---|---|
| E1-diffplan | 6 | 2 | 1 |
| E2 | 10 | 8 | 5 |
| noplan-ctl | 2 | 0 | 0 |

### batch2（64 次 rollout，12 个任务）

| 模型 | 条件 | 成功 | 费用 |
|---|---|---|---|
| gpt56 | 无方案 | 10/12 | $10.11 |
| gpt56 | 有方案 | 14/20 |  |
| haiku | 无方案 | 3/12 | $5.55 |
| haiku | 有方案 | 10/20 |  |

| proxy 组 | 对数 | 多数票 E2 | HQ-E2 |
|---|---|---|---|
| E1-diffplan | 4 | 1 | 1 |
| E2 | 11 | 9 | 8 |
| noplan-ctl | 2 | 1 | 1 |

## 3. 全部 HQ-E2 对

| 来源 | pair | 任务 | A | B | 票（gpt56/kimi/opus） |
|---|---|---|---|---|---|
| prevalence | aw-08 | 9bf2c8a_1 | react_llama3_test_challenge | full_code_refl_gpt4turbo_test_challenge | 222 |
| prevalence | aw-19 | d194965_2 | react_gpt4turbo_test_normal | react_llama3_test_normal | 221 |
| prevalence | aw-22 | 522e5e5_2 | full_code_refl_gpt4o_test_normal | plan_exec_gpt4o_test_normal | 221 |
| batch1 | gn-02 | 68ee2c9_1 | gpt56:P2 | haiku:P3 | 222 |
| batch1 | gn-04 | 68ee2c9_2 | gpt56:P2 | gpt56:P1 | 222 |
| batch1 | gn-05 | 34d9492_3 | haiku:P1 | haiku:P3 | 222 |
| batch1 | gn-09 | 68ee2c9_2 | gpt56:P1 | gpt56:P3 | 212 |
| batch1 | gn-10 | 34d9492_2 | gpt56:P2 | gpt56:noplan | 222 |
| batch1 | gn-14 | 34d9492_1 | haiku:P3 | haiku:P2 | 222 |
| batch2 | g2-00 | 6c2c621_2 | haiku:P3 | gpt56:noplan | 212 |
| batch2 | g2-02 | 6c2c621_2 | gpt56:P2 | haiku:P3 | 222 |
| batch2 | g2-03 | 6c2c621_2 | gpt56:P3 | haiku:P2 | 222 |
| batch2 | g2-04 | 6c2c621_2 | gpt56:noplan | gpt56:P2 | 221 |
| batch2 | g2-05 | 6c2c621_2 | haiku:P2 | haiku:P3 | 222 |
| batch2 | g2-07 | 6c2c621_2 | gpt56:P3 | gpt56:noplan | 222 |
| batch2 | g2-08 | 6c2c621_2 | gpt56:P2 | gpt56:P1 | 221 |
| batch2 | g2-10 | 6c2c621_2 | haiku:P3 | gpt56:P3 | 221 |
| batch2 | g2-12 | aa8502b_3 | gpt56:noplan | haiku:noplan | 222 |
| batch2 | g2-14 | 60d0b5b_1 | gpt56:noplan | haiku:P3 | 221 |

共 19 对 HQ-E2，来自 11 个任务、8 类场景：{'9bf2c8a': 1, 'd194965': 1, '522e5e5': 1, '68ee2c9': 3, '34d9492': 3, '6c2c621': 8, 'aa8502b': 1, '60d0b5b': 1}。

## 4. 要点

- 自然状态下几乎全是 E1，HQ-E2 不到 1%；E0 未单独统计（需要每一步操作完全相同，自然轨迹中几乎没有）。
- 给方案能稳定造出 E2，但只在少数任务上有效：多数任务 GPT-5.6 认为只有一种合理做法。
- E2 种类集中：大部分是文件系统的不同写法（移动 vs 复制+删除、暂存目录、原地写 vs 删除重建）。
- 写操作集合 proxy 会漏掉不改变写集合的决策差异（如 g2-12：只关注未关注的歌手 vs 全部关注），也会误报。
- judge 标签与人工抽查 12/12 一致（人工先看到了 judge 结论，可能偏高）；单个 judge 里 Opus 最接近人工，GPT-5.6 偏向判 E2。
- 原始 rollout（`outputs/`）和 AppWorld 衍生数据（轨迹包、judge 输出、摘要）只在本地，不入库：AppWorld 许可要求其数据和衍生内容只能加密公开。

## 5. 下一步（待批准）

- 用现有 HQ-E2 + 配对 E1 做小规模三元组试验，检验语义距离指标是否优于文本/动作基线。
- 试点"锚点改写"：从成功轨迹出发，确定性改出 E1，在决策点分叉改出 E2，重放验证。
