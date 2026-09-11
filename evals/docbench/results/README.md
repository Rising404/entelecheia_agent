# DocBench results and methodology

这里只发布人工审阅过字段边界的脱敏统计，不发布 PDF、题目、参考答案、模型回复、裁判原文、
prompt、完整 trajectory、数据库或本机配置。原始运行档案在仓库外保存；本目录不由 runner
自动写入。统计文件通过隐私检查不代表裁判结论已成为可靠真值。

## 本轮 125 题与 8 次定向补跑

[`showcase_125.summary.json`](showcase_125.summary.json) 包含 125 个 benchmark case、
133 条实际 execution 和 3 轮独立运行的来源身份。它是内容剥离后的公开投影，不是原始轨迹。

| 统计口径 | 正确数 / 固定分母 | 比例 | 含义 |
| --- | ---: | ---: | --- |
| 首跑原始 DeepSeek 裁判 | 91 / 125 | 72.8% | 所有首跑题，包含执行失败 |
| 8 次补跑替换对应题后的原始裁判 | 96 / 125 | 76.8% | 不是一次干净首跑，也不是 133 题正确率 |
| 存档 Codex 复核：首跑 | 99 / 125 | 79.2% | archived/provisional，不是独立人工金标 |
| 存档 Codex 复核：补跑合并 | 104 / 125 | 83.2% | archived/provisional，有未解决的争议 |

推荐对外同时报告 **72.8% 首跑、76.8% 定向补跑后**，并明确补跑策略。83.2% 只能标为存档的
暂定复核口径，不能用它覆盖原裁判列或宣传为无争议最终成绩。CC 的正式复核仍为 `pending`；
本文件没有将讨论消息推造成 CC 的逐题评分，也没有将未完成复核等价为零分。

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
  内容已剥离，因此公开 JSON 能复算统计，不能让读者仅凭 JSON 独立判定答案是否正确。

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
`codex_archived_first_score` 与 `codex_archived_selected_score` 是独立的既有复核覆盖层。
本次只按白名单投影既有档案，没有改写原始评分。

存档 Codex 复核确认的 8 处 0 → 1 是 `42:5`、`68:8`、`78:6`、`85:0`、`112:2`、
`157:5`、`158:1`、`167:0`，与补跑的 8 题不重叠。仍有 9 题带
`unresolved_retained_original`：`26:4`、`37:4`、`53:3`、`67:4`、`80:5`、`111:1`、
`144:3`、`199:8`、`220:1`。保留原分不表示确认原裁判正确。

其余状态为 `original_review_retained`（100 题）、`new_answer_checked`（8 题）及
`confirmed_correction`（8 题）。这些只是存档状态，不表示本轮已重新逐题验证。
`review_artifact_sha256` 对应当时合并复核文件，不是一个新的评测运行或权威金标。

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

唯一新增 schema 标识为 `entelecheia-docbench-public-evaluation`，不添加内部 v1/v2 代际。
它是封闭字段集合：只接受明确的 benchmark 标识、来源哈希、受限枚举、数字和布尔状态。
隐私检查同时验证字段类型、0/1 分数、有限非负计数、重复身份、case/run 引用、最新选定轮、
题型归并、逐轮数量/分数/耗时及总分算术；未知字段、原始内容、私有路径和敏感值仍然拒绝。

这是一道内容边界与统计一致性检查，不验证 PDF 版权、不证明裁判正确，也不能阻止人为伪造一整套
自洽统计。发布前仍需要审阅实际文件集合。源 PDF、QA 与第三方再分发条件没有因哈希投影而获得授权。

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
