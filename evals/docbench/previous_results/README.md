# 历史结果与评测方法 / Previous results

本目录保存已有实验与评审，后续新结果整理到相邻 `results/`。这只是历史资料改名归档，
原始结果和统计 JSON 的内容不变，也不会被 runner 当作复现实验输入。

这里发布审阅后的统计、评审及逐题回答/执行/评分证据；不发布 PDF、原题/参考答案字段、
未经清理的请求与文档正文、数据库或本机配置。完整原件另存且不改写，本目录不由 runner
自动写入。隐私检查不代表裁判结论已成为可靠真值。

## 逐题公开证据 / Per-case evidence

- [showcase_125](showcase_125/README.md)：上轮 125 题首跑及 3+5 补跑，共 133 次执行。
  仅对应这轮的 [analysis](showcase_125/analysis/README.md) 放在旧实验目录内。
- [gate_off_highland235b_123](gate_off_highland235b_123/README.md)：同批排除 `26:4`、`37:4`，
  关闭语义验证门并改用 Highland 235B VLM，共 123 次执行，原 DeepSeek 裁判 97/123。
  [双方评审与 badcase 分析](gate_off_highland235b_123/reviews/README.md) 与旧实验隔离。

每组按 `runs/<批次>/cases/<题目>/result.json`、同题 `trajectory.json` 与批次 `scoring/` 分层。
不覆盖首跑失败，不把补跑当作新题；两道排除题没有伪造本轮结果。原文件 ID、工具错误、回答和
记录开销保留，省略的输入/文档正文带原因及原文哈希。公开轨迹哈希与原件哈希分开标识。
具体清理范围见各组 README；原始 metrics 不能按公开文本重新计算，也不是完整费用账单。

这两个不可变公开包采用精确 publication manifest 与文件哈希审阅；不是给整个 results/ 取消保护。

### 本地原始归档（不随 Git 分发）

本机可在 `private_runs/showcase_125/` 保留首跑与补跑的原始答案、已导出的轨迹、评分及复核
材料，按 `runs/<批次>/cases/<case_id>/` 分层；跨批次复核独立放在 `reviews/`。
整个 `private_runs/` 被 Git 忽略，不是公开数据包；克隆仓库不会获得这些内容。
复制不删除源档案，不改变原始结果或轨迹哈希，也不把它们作为当前 runner 的运行状态。

- `result.json`：单题结果单，包含最终回答 `reply`、执行状态、耗时、统计与轨迹引用。
- `state/artifacts/trajectory.json`：已有轨迹的完整导出；`steps` 保存步骤及正文引用，
  `blobs` 保存去重正文，`integrity` 保存完整性计数。它不是 `result.json` 的内嵌字段。
- `scoring/summary.json`：该批次原始裁判记录；不把复核分数改写成原裁判分数。

本地归档不复制 PDF、SQLite、运行缓存或模型权重；也不承诺恢复完整运行环境。
它保留的是当时已记录的内容，不会补造原先缺失的轨迹。原始文件仍可能含题目、参考答案、
文档片段和本机路径，未经另外审阅和脱敏不得强制加入 Git 或发布。

## 本轮 125 题与 8 次定向补跑

[`showcase_125.summary.json`](showcase_125.summary.json) 包含 125 个 benchmark case、
133 条实际 execution、3 轮独立运行的来源身份，以及分列的 DeepSeek、Codex 和 CC 评分。
它是内容剥离后的公开投影，不是原始轨迹。

| 统计口径 | 正确数 / 固定分母 | 比例 |
| --- | ---: | ---: |
| 首跑 DeepSeek 裁判 | 91 / 125 | 72.8% |
| 补跑替换的ds裁判 | 96 / 125 | 76.8% |
|  astra 复核：首跑 | 99 / 125 | 79.2% |
|  astra 复核：补跑合并 | 104 / 125 | 83.2% |
|  opus 复核：首跑 | 100 / 125 | 80.0% |
|  opus 复核：补跑合并 | 105 / 125 | 84.0% |


astra/opus 覆盖、125 题评分表及双方链路意见见 [评审与结果分析](showcase_125/analysis/README.md)。


### 样本与评分方法

- 固定集合为 125 题、125 个不同 benchmark 文档。academia、finance、government、laws、news 各 25 题。
- text 50 题，multimodal、metadata、unanswerable 各 25 题，不是全量 DocBench。
- 主模型为 `deepseek-chat`，视觉模型为 `qwen3-vl-30b-a3b-instruct`，裁判为
  `deepseek-chat`。


### 三轮来源及补跑策略

| ordinal / phase | 不可变源 run_id | 题数 | 原始裁判正确数 |
| --- | --- | ---: | ---: |
| 1 / first_pass | `l1-gpu-deadline25-125-20260910-a` | 125 | 91 |
| 2 / transport_retry | `l1-gpu-transport-retry-3-20260911-a` | 3 | 2 |
| 3 / execution_retry | `l1-gpu-execution-retry-5-20260911-a` | 5 | 3 |

第二轮只补跑首轮的 `203:1`、`206:0`、`207:2` 三个传输失败 case。

第三轮只补跑 `40:0`、`117:0`、`129:0`、`148:2`、`183:0` 五个其他执行失败 case。

八题原裁判均为 0，补跑得到 5 个 1，因此合并增加 5 分。其余 117 题保持首跑结果。

每题 `selected_run_ordinal` 指向实际选择的最新补跑，不取多次生成中的最高分。

三轮共用冻结 source SHA-256
`7a6043f6465a887328363891a3edc963c05cd81d2d78868558f2c17398e38da1`，
当时 Git revision 为 `81d7fd761f98a968c24feee388c2b665c01d12df`，`worktree_dirty=true`。

重跑方法见 [评测执行入口](../reproduce_or_run_script/README.md)，新跑分必须绑定新的 source/config 身份。


### 执行状态、耗时与 token

| 运行 | 轨迹记载 input tokens | 轨迹记载 output tokens | 各题耗时之和（秒） |
| --- | ---: | ---: | ---: |
| 首跑 125 | 14,306,914 | 248,421 | 31,739.979 |
| 传输补跑 3 | 382,129 | 6,151 | 584.320 |
| 执行补跑 5 | 897,000 | 15,875 | 1,560.064 |
| 合计 133 executions | 15,586,043 | 270,447 | 33,884.363 |
