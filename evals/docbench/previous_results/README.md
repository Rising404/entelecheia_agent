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

| 统计口径 | 正确数 / 固定分母 | 比例 | 含义 |
| --- | ---: | ---: | --- |
| 首跑原始 DeepSeek 裁判 | 91 / 125 | 72.8% | 所有首跑题，包含执行失败 |
| 8 次补跑替换对应题后的原始裁判 | 96 / 125 | 76.8% | 不是一次干净首跑，也不是 133 题正确率 |
| 存档 Codex 复核：首跑 | 99 / 125 | 79.2% | archived/provisional，不是独立人工金标 |
| 存档 Codex 复核：补跑合并 | 104 / 125 | 83.2% | archived/provisional，有未解决的争议 |
| 存档 CC 复核：首跑 | 100 / 125 | 80.0% | 另行归档的模型复核，非盲审、非人类金标 |
| 存档 CC 复核：补跑合并 | 105 / 125 | 84.0% | 与 Codex 有 3 道分歧，不能合成最高分口径 |

推荐对外同时报告 **72.8% 首跑、76.8% 定向补跑后**，并明确补跑策略。83.2% 只能标为存档的
暂定复核口径，不能用它覆盖原裁判列或宣传为无争议最终成绩。CC 的正式逐题档案现已找到并核对；
当前 JSON 的 `cc_review_status=archived_provisional` 表示已归档、仍有争议；不表示人类金标。
CC 覆盖层、125 题评分表及双方链路意见见 [评审与结果分析](showcase_125/analysis/README.md)。
84.0% 同样只是存档模型复核口径；本次只原位补齐公开 review 元数据和评分列，
没有改写原始运行、原裁判评分或 Codex 复核列，也没有重新调用模型。

### 样本与评分方法

- 固定集合为 125 题、125 个不同 benchmark 文档。academia、finance、government、laws、news
  各 25 题；text 50 题，multimodal、metadata、unanswerable 各 25 题。不是全量 DocBench，
  也不是保持原数据分布的简单随机样本。
- JSON 同时保留 `source_question_type` 与归并后的 `question_type`。`text-only` → text；
  `multimodal-t` / `multimodal-f` → multimodal；`meta-data` → metadata；
  `unanswerable` / `una-web` → unanswerable。归并仅用于汇报，不改原题或标签。
- 主模型为 `deepseek-chat`，视觉模型为 `qwen3-vl-30b-a3b-instruct`，裁判为
  `deepseek-chat`。使用存档的 DocBench prompt-compatible 评分产物；裁判替换及抽样设置
  意味着 `official_comparable=false`，不能当作原论文配置或官方榜单分数。
- 同一模型家族生成与裁判可能产生相关偏差。没有为此次公开迁移重新请求模型、改参考答案或重判。
  内容已剥离，因此公开 JSON 能复算三套主评分、样本分组、补跑与执行汇总，
  不能让读者仅凭 JSON 独立判定答案是否正确，也不能复算未公开逐项字段的全部链路归因。

### 三轮来源及补跑策略

| ordinal / phase | 不可变源 run_id | 题数 | 原始裁判正确数 |
| --- | --- | ---: | ---: |
| 1 / first_pass | `l1-gpu-deadline25-125-20260910-a` | 125 | 91 |
| 2 / transport_retry | `l1-gpu-transport-retry-3-20260911-a` | 3 | 2 |
| 3 / execution_retry | `l1-gpu-execution-retry-5-20260911-a` | 5 | 3 |

第二轮只补跑首轮的 `203:1`、`206:0`、`207:2` 三个传输失败 case；第三轮只补跑 `40:0`、
`117:0`、`129:0`、`148:2`、`183:0` 五个其他执行失败 case。公开标识以 `docbench:` 为前缀。
八题原裁判均为 0，补跑得到 5 个 1，因此合并增加 5 分。其余 117 题保持首跑结果。
每题 `selected_run_ordinal` 指向实际选择的最新补跑，不取多次生成中的最高分。

原始物理归档曾被重命名为 showcase-01、showcase-02、showcase-03；目录名不是实验身份。
公开 run 记录保留原 `run_id`，以及 config、selection、frozen cases、source、environment、
manifest、generation report、scoring summary 和 judge prompt 的 SHA-256；case 记录保留
PDF 与 QA 文件的 SHA-256。哈希用于关联和完整性复核，不包含原文件，也不能代替原文件实现复现。

三轮共用冻结 source SHA-256
`7a6043f6465a887328363891a3edc963c05cd81d2d78868558f2c17398e38da1`，
当时 Git revision 为 `81d7fd761f98a968c24feee388c2b665c01d12df`，`worktree_dirty=true`。
这个 revision 本身不足以恢复被测代码；公开迁移后的当前源码也不等同于当时冻结快照。
本成绩不能作为后来代码修改的验证结果。重跑方法见
[评测执行入口](../reproduce_or_run_script/README.md)，新跑分必须绑定新的 source/config 身份。

### 裁判与复核分列

`executions[].judge_score` 永远记录对应那一轮存档的原始裁判分。
`cases[].original_judge_score` 与 `retry_merged_judge_score` 分别引用首轮和选定轮；
`codex_archived_first_score` 与 `codex_archived_selected_score` 保留既有 Codex 复核覆盖层；
`cc_archived_first_score` 与 `cc_archived_selected_score` 分列记录 CC 存档复核。
三套分数使用相同 case ID 和 `selected_run_ordinal`，未补跑题的 CC 两列必须一致。
本次只按白名单投影既有档案，没有改写原始评分。

`cc_review.reviewer_kind=model`、`blind_review=false` 明示模型评审且非盲审；
`source_check_basis=reviewer_self_report` 与每题 `cc_source_checked_reported` 只记录 CC 自报
是否核对原 PDF，不是本轮重新核验的事实证明。`cc_debatable` 保留 CC 的争议标记，
不覆盖 Codex 的 `review_status`，也不把未标争议的题自动认证为正确。

CC 当前归档与 Codex 的三个分歧为 `80:5`、`178:1`、`220:1`，净差 +1 分。
CC 的 8 道可争议与下面 Codex 的 9 道未决是不同集合，不能互相覆盖；详细规则与
不含原题答的逐题覆盖层见 [评分复核](showcase_125/analysis/EVALUATION_REVIEW.md)。

存档 Codex 复核确认的 8 处 0 → 1 是 `42:5`、`68:8`、`78:6`、`85:0`、`112:2`、
`157:5`、`158:1`、`167:0`，与补跑的 8 题不重叠。仍有 9 题带
`unresolved_retained_original`：`26:4`、`37:4`、`53:3`、`67:4`、`80:5`、`111:1`、
`144:3`、`199:8`、`220:1`。保留原分不表示确认原裁判正确。

其余状态为 `original_review_retained`（100 题）、`new_answer_checked`（8 题）及
`confirmed_correction`（8 题）。这些只是存档状态，不表示本轮已重新逐题验证。
`review_artifact_sha256` 仍对应当时 Codex 合并复核文件；`cc_review.artifact_sha256` 单独对应
CC 原始评审档案。二者都不是新的评测运行或权威金标，原件不随公开投影分发。

### 执行状态、耗时与 token

答案正确与进程清洁退出分别记录，不以一个指标覆盖另一个。首跑 `execution_ok` 为 117/125，
但有 14 个非零 worker exit；其中 6 个在产出答案后异常退出，不能直接按“没有答案”计分。
补跑合并后 `execution_ok` 为 125/125，仍有原首跑保留下来的 6 个非零 exit。
这些数值不等于 post-commit 历史落库、窗口结算等收尾工作都已成功。

| 运行 | 轨迹记载 input tokens | 轨迹记载 output tokens | 各题耗时之和（秒） |
| --- | ---: | ---: | ---: |
| 首跑 125 | 14,306,914 | 248,421 | 31,739.979 |
| 传输补跑 3 | 382,129 | 6,151 | 584.320 |
| 执行补跑 5 | 897,000 | 15,875 | 1,560.064 |
| 合计 133 executions | 15,586,043 | 270,447 | 33,884.363 |

这些是 generation report 从轨迹成功 provider-response observation 累加的计数，不把 rejected
诊断或 terminal failure 汇总重复算成另一条成功响应；不包含独立裁判阶段，不等价于完整账单。
`elapsed_s` 是对应 case 的 runner 耗时，run 层为逐题求和，不是并发批次的墙钟时间。
失败请求、未记录 usage、工具内独立 provider 调用和 GPU 编码等可能有额外成本；
本投影没有凭空补齐它们，也没有以 `execution_ok` 过滤掉消耗。

### 公开 schema 与校验边界

现行 schema 标识仍为 `entelecheia-docbench-public-evaluation`，不添加另一份 summary 或内部代际。
它是封闭字段集合：只接受明确的 benchmark 标识、来源哈希、受限枚举、数字和布尔状态。
隐私检查同时验证字段类型、0/1 分数、有限非负计数、重复身份、case/run 引用、最新选定轮、
题型归并、逐轮数量/分数/耗时、三套总分及 CC 声明计数；未知字段、原始内容、私有路径和敏感值
仍然拒绝。当前工作树和暂存区必须提供完整的 CC 归档层，不能只改状态或继续发布 pending 投影。

首次导出的 pending 投影仅作为历史保存。更新不改变原 runs、executions、DS/Codex 分数或来源
哈希，但会改变整份公开 JSON 的哈希；新旧投影身份见 [来源索引](showcase_125/analysis/README.md#来源身份)。
提交历史检查 `--tree` 仅对**精确路径与精确旧文件哈希**相符的那一份已审阅快照保留接纳，
以便 pre-push 扫描包含旧导出的历史；它仍经过通常的私有字段、路径、自由文本与秘密扫描。
此边界不适用于工作树或暂存区，不恢复通用旧 schema，也不接纳改动后的 pending 文件。

这是一道内容边界与统计一致性检查，不验证 PDF 版权、不证明裁判正确，也不能阻止人为伪造一整套
自洽统计。发布前仍需要审阅实际文件集合。源 PDF、QA 与第三方再分发条件没有因哈希投影而获得授权。

`showcase_125/analysis/` 只对应旧 125 题；新 123 题评审位于
`gate_off_highland235b_123/reviews/`。两组公开清单分别登记所属分析/评审文件及哈希，
不开放任意原始评审或轨迹。正文仍经过私有路径与秘密扫描。

## 较早的工程冒烟（与本轮分开）

- [`local_bge_m3_retrieval_3.summary.json`](local_bge_m3_retrieval_3.summary.json)：3 例本地
  retrieval-only 工程冒烟，覆盖文档收录、Dense / learned sparse / BM25、authority reread
  与 BGE reranker；不是 DocBench 答案分数。
- [`live_bge_m3_l1_3.summary.json`](live_bge_m3_l1_3.summary.json)：3 例真实 L1 工程冒烟，
  未调用 judge，答案分数为空。
- [`live_bge_m3_l1_20.summary.json`](live_bge_m3_l1_20.summary.json)：早期 20 例，18/20 通过
  产品交付门，prompt-compatible 11/20，交付子集 10/18。仅 7/20 具备 ACTIVE / READY
  三方法 generation，5 例实际执行 BGE 查询；不能当作全量 BGE-RAG 成绩，也不并入本轮 125 题。

这些不可变旧 artifact 的既有 schema 标识保留用于读取，不代表新增第二条生产实现。
