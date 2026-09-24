# 纯检索评测：数据与当前状态（讨论稿）

更新日期：2026-09-24。本文记录现有冻结数据、检索结果和已核实的问题，作为后续讨论的基准；不是一次新的实验报告。运行命令与实现说明见 [README](README.md)，详细排名与耗时分析见[本轮分析](results/analysis/retrieval-repaired75-hybrid-20260923.md)。

**当前结论：修复后的 75 题全部可计分，完整 hybrid 已跑通，最终 Hit@5 为 74/75（98.67%）。这说明在已知文档、限定模态的条件下，现有链路通常能找到已标注证据；还不能据此证明历史 Agent 当时拿到了同样的材料，也不能把剩余问答错误统一归因于幻觉。**

## 1. 文件各自负责什么

| 内容 | 位置 | 当前用途 |
| --- | --- | --- |
| 本说明 | `DATA_AND_STATUS.md` | 随讨论更新解释、状态与待办，不覆盖冻结结果 |
| 现行数据 | [repaired75-20260923](dataset/repaired75-20260923/README.md) | 75 题均有已审 primary 标签，当前比较基准 |
| 早期数据 | [reused75-20260922](dataset/reused75-20260922/README.md) | 66 题可计分、9 题待审的历史快照；不是当前覆盖率 |
| 逐题查看 | [按需下载](README.md#按需下载)后打开快照内 `INDEX.html` | 查看问题、候选截图、描述和标签 |
| 纠错记录 | [CORRECTIONS.md](dataset/repaired75-20260923/CORRECTIONS.md) | 参考答案与 evidence 的修订依据 |
| 完整索引 | [索引目录说明](indexes/README.md) | `repaired75-bge-m3-mps-fp32` 的无损分片及还原方法 |
| 完整 hybrid 结果 | [REPORT.md](results/retrieval-repaired75-hybrid-20260923/REPORT.md) / [report.json](results/retrieval-repaired75-hybrid-20260923/report.json) | 六阶段指标、数据/模型/源码身份和参数 |
| 排名与诊断 | [rankings.jsonl](results/retrieval-repaired75-hybrid-20260923/rankings.jsonl) / [diagnostics.jsonl](results/retrieval-repaired75-hybrid-20260923/diagnostics.jsonl) | 逐题、逐阶段排名，截断与装包状态 |
| 历史 Agent 回答与评审 | [123 题评审索引](../docbench/previous_results/gate_off_highland235b_123/reviews/README.md) | 保持在原 DocBench 目录，用于对照与失败分析 |

数据集的权威文件是下载快照根目录内的 `dataset.sqlite`；同目录 `exports/` 下的 corpus、queries、qrels，以及 HTML 审阅页都是派生视图。三者分离：**corpus 是可检索正文，queries 是问题与范围，qrels 是计分标签**。

主仓只保留代码、说明、小型元数据与全部检索结果，普通 clone 不下载语料或索引分片。三个完整快照通过 [Release 附件](https://github.com/Rising404/entelecheia_agent/releases/tag/docbench-retrieval-20260923) 按需取得，步骤见 [下载说明](README.md#按需下载)。Git 内保留的评测产物由 [publication_manifest.json](publication_manifest.json) 锁定，附件及其完整文件集合由 [release_assets.json](release_assets.json) 分别锁定。结果中原完整 publication 的冻结身份保持不变，本次仅调整分发方式，没有重跑或改分。后续若修正标签、语料或检索配置，应产生新的快照/运行目录并保留旧结果；本文可以追加解释，不能把解释当成已执行的修复。源码入口分别为 [数据构建](retrieval_dataset.py)、[hybrid 接入](retrieval_hybrid.py)、[计分与报告](retrieval_eval.py)。

## 2. 数据是什么，修复到了哪一步

本批复用之前评测过的题目，取 **50 道文本题、11 道表格题、14 道图表题，共 75 题、75 份文档**。遵循本轮约定，不纳入元数据题和不可回答题。

| 项目 | 当前数量 / 状态 |
| --- | --- |
| 可计分题目 | 75/75；待审 0 |
| 文本 chunk | 16,459 |
| 视觉单元 | 6,589：table 1,601、figure 2,105、vector_graphics 2,883 |
| 索引语料总量 | 23,048 个单元 |
| 图片资产 | 6,526 个；图片资产数与视觉单元数不要求相等 |
| 证据标签 | 110 条 primary、3 条 support；当前指标只使用 primary |

数量与来源以[数据清单](dataset/repaired75-20260923/manifest.json)为准。DocBench 原有 evidence 是题目层的作证说明，不会天然对应本项目的 chunk/视觉单元；本地 qrels 是在具体解析结果上另行定位、审阅得到的。这里的“已审”包含模型辅助审阅，**不是独立人工穷尽标注**。

语料复用了历史解析快照，没有重新解析全部 PDF。其中 74 份取自最新的 `l1-gate-off-highland235b-123-20260914-a`，文档 ID 为 63 的文档使用更早的 showcase 首跑解析结果补齐。问题文本保持不变，文档身份可由源 PDF 哈希核对；语料经过修复，不能视作历史 Agent 当时所见内容的原样回放。

视觉候选使用检测区域的 PDF 文字层及经核验的转录，保留截图引用。它们覆盖文档内的视觉 inventory，也包含干扰项和检测碎片。此次没有新调 VLM 生成描述，没有使用图像 embedding；历史随问题生成的 `picture_observations` 不进入语料，避免答案泄漏。

从早期 66 题可计分提升到 75 题，依靠的是转录、证据定位和标注修复，不是删除困难题。修复还包括 `85:0` 的表格转录派生文本 chunk，以及以下参考内容校准：

| 题目 | 已记录的修订 |
| --- | --- |
| `167:0` | 原参考遗漏文档中的例外条件，修订 answer 与 evidence |
| `36:2` | 补上原问题明确要求、参考答案漏写的 BLEU 数值 |
| `53:3` | 区分 Finance income 与 Total income，保留问题歧义说明 |
| `67:4` | 原答案正确，evidence 错写了最大收入部门，修订 evidence |
| `80:5` | 区分品牌与产品类别的比较口径，修订 answer 与 evidence |
| `85:0` | 两个表格单元格中的 4 被拼成 44，恢复列语义 |

原值、新值和来源保存在数据库 `curation_events`，可读说明见[纠错记录](dataset/repaired75-20260923/CORRECTIONS.md)。原始 QA 文件和历史裁判结果没有被这些修订覆盖；本次也没有按修订后的答案重新运行三裁判。

## 3. 当前测了哪条链路

每题输入一次原问题，执行：

```text
原问题 + 已知文档/模态范围
    → Dense / Learned Sparse / BM25 各路召回
    → RRF 融合 → reranker 重排 → packing 装包 → 对照 primary qrels 计分
```

评测通过薄适配层调用生产 `RetrievalService.retrieve_file_query_batch`，复用生产编码器、方法存储、融合、重排和装包。没有 Agent loop、查询扩写、回答生成或裁判调用；答案、evidence、qrels 和审阅理由不作为索引正文或检索模型输入。

| 条件 | 本轮实际设置 |
| --- | --- |
| 检索范围 | 各路 top-k 之前限定到题目所属文档；文本题只查 chunk，多模态题查全部三种视觉类型的文字描述 |
| 设备与模型 | Apple GPU（MPS）、FP32；本地离线 BGE-M3 与 bge-reranker-v2-m3，revision 固定在报告中 |
| 候选数量 | 每路最多 128；RRF `k=60`、保留最多 64；每题一个 query |
| 重排长度 | 每个 query/passage pair 最多 1,024 tokens |
| 装包预算 | 最多 96 个单元、96,000 estimated tokens；本轮条数还受 RRF 的 64 条上限约束 |

三路索引各覆盖全部 23,048 个单元；本批题目允许范围的并集为 13,725 个单元，其余单元仍参与全库 BM25 统计。单题候选数最少 1、中位数 31、最多 2,361；46/75 题不超过 64 个候选。因此当前成绩衡量的是**文档内证据定位**，不是跨整个语料库、不提供文档身份的搜索能力。

### 已完成的运行与结果

完整运行状态为 `complete`，75/75 题计分，未发生检索/重排降级，packing 没有省略候选。以下为最终 `packed` 阶段；它与本轮 `reranker` 排名完全相同。

| 范围 | 题数 | Hit@1 | Hit@5 | 宏平均 Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 75 | 67/75（89.33%） | 74/75（98.67%） | 96.04% | 0.9296 |
| 文本 | 50 | 49/50（98.00%） | 50/50（100.00%） | 97.67% | 0.9840 |
| 多模态合计 | 25 | 18/25（72.00%） | 24/25（96.00%） | 92.80% | 0.8207 |
| 其中：表格 | 11 | 7/11（63.64%） | 11/11（100.00%） | 92.73% | 0.7879 |
| 其中：图表 | 14 | 11/14（78.57%） | 13/14（92.86%） | 92.86% | 0.8464 |

Hit@k 表示前 k 条至少命中一个 primary；Recall@k 表示各题 primary 覆盖比例的平均值；MRR 使用各题首条 primary 的倒数排名。三者均不等于最终答案正确率。多模态合计按 25 题汇总，不是将表格、图表两个百分比直接平均。

RRF 阶段 Hit@5 为 96.00%，重排后为 98.67%；RRF 完整候选集已经对 75/75 题命中至少一个 primary，但只覆盖 105/110 条 primary。**“每题找到一条证据”不等于“每题所需证据全部找到”。** 六阶段完整指标见[结果报告](results/retrieval-repaired75-hybrid-20260923/REPORT.md)。

BGE 编码前检查的总量为 7,682,852 tokens，最长单元 2,981，没有触及 8,192-token passage 上限。重排另有 163/3,016 对输入在截断前超过 1,024 tokens，涉及 17 题；未排第一的 8 题，其指定 gold 均未被重排截断。最终每题装包估算为 1,029–65,862 tokens，本轮没有继续交给回答模型。

完整索引已归档为 5 个分片，还原后的数据库与原件哈希一致。成功轮次建库 59.21 分钟、查询 18.11 分钟；此前三次中断构建另耗时 113.96 分钟。复用历史表示仍需写入统一索引，逐条连接/事务开销较大，不能把建库时间全算成 embedding 或 reranker 时间。后续相同配方可还原索引复用，详细记录见[耗时分析](results/analysis/retrieval-repaired75-hybrid-20260923.md)。

### 已知的标签与描述问题

逐题原文与截图在下载快照的 `review/docbench-67-4.html`、`review/docbench-80-5.html`，可从其中的 `INDEX.html` 打开。

唯一 Hit@5 未命中是 `67:4`：冻结 primary 排第 59，描述缺少图中关键名称和数值；但 Top1 的另一个视觉区域实际包含可回答问题的数据。这里同时有描述损失与替代证据漏标，不能将其表述为“完全没有召回答案”。

`80:5` 的 Top1 也有未进入 qrels 的有效表格区域，冻结 primary 排第 3。上述发现目前只记入分析，**没有事后改标签抬高本轮分数**。平铺 qrels 还不能表达“任选一个替代证据组”与“必须联合命中多个证据”；未标注不等于已确认无关。

## 4. 与历史三裁判得分怎样对照

下表取最新 **123 题单跑中的同一批 75 道文本/多模态题**，不是旧 125 题首跑或补跑合并分数。最新运行同时关闭了语义验证门、改用 Highland `qwen3-vl-235b-a22b-instruct`，不能将运行差异单独归因于关门。

| 历史回答评分口径 | 文本 50 题 | 多模态 25 题 | 其中：表格 11 题 | 其中：图表 14 题 |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek 原裁判 | 46/50（92%） | 19/25（76%） | 9/11（81.82%） | 10/14（71.43%） |
| Codex 按原文校准的复核 | 47/50（94%） | 20/25（80%） | 9/11（81.82%） | 11/14（78.57%） |
| Claude / Opus 复核 | 47/50（94%） | 22/25（88%） | 10/11（90.91%） | 12/14（85.71%） |

来源：[原裁判汇总][judge-summary]、[Codex 逐题判决][codex-cases]、[Claude 逐题判决][claude-cases]。Codex 这里采用 document-grounded 口径，而非保留原参考目标的另一列。归档只把该评审者标为 Codex，未冻结 Astra 模型 ID，所以本文不把它写成可核验的 Astra 成绩。DeepSeek 当时未开启 thinking；这是配置事实，不能仅因分低就认定其判决全部不可靠。Claude 复核可见 Codex 结论，两份复核都不是独立盲审的人类金标。

当前检索与历史问答的题目、源 PDF 身份相同，但解析来源、修复后的描述和标签并非完全相同。对照结果可用于筛查问题，**不能用 Hit@5 减去问答正确率，直接计算“模型理解导致的损失”**。

例如，Codex 与 Opus 都判错的文本题 `63:2`、`165:0`、`206:0`，以及多模态题 `32:1`、`53:3`、`190:2`，在当前检索中均 Hit@5 命中。这只是当前材料可被检索到的证据；历史失败仍需回到当时的 trajectory，确认工具请求、返回内容、VLM observation 与最终回答。

## 5. “运行完成”“工具成功”“入库成功”分别是什么意思

| 状态层 | 能证明什么 | 不能据此推出什么 |
| --- | --- | --- |
| case `completed` / `execution_ok` / 正式交付 | 执行链完成并产生了交付 | 所有工具有效、文件已入库、回答正确 |
| tool_call 外层 `ok` | 工具按执行契约返回结果 | 结果中的每份文件都 ready |
| tool_call `failed` / `rejected` | 本次调用失败或被拒绝 | 服务一定崩溃；参数/范围校验也会产生这些状态 |
| 文件结果 `ready` / `not_ingested` / `unavailable` | 具体文件的业务状态 | 外层调用状态可以代替此字段 |
| 索引 generation / 三路覆盖 | 可用于当前检索的索引是否存在并覆盖单元 | Agent 一定调用了检索或选中了正确文件 |
| 裁判得分 | 按某个评分口径评估最终回答 | 能单独确定错误发生在哪一层 |

逐一统计历史最新 123 题公开 trajectory 中 `kind == "tool_call"` 的记录，共 **794 次：`ok` 711、`failed` 50、`rejected` 33**。这是调用次数，不是失败题数；同一题可重试多次。`ok` 还包含返回文件 `unavailable` 的情况，所以也不能将 711 次称为“711 次有效取得材料”。计数入口是[该运行的 cases 目录][historical-cases]，与当前 75 题纯检索实验分开。

### 已核实的 `63:2`：先发生来源访问问题，未进入有效准备/检索

根据[当时 trajectory][case63-trajectory]和[result][case63-result]：

1. 首次模型输入的 `file_catalog` 已提供附件的正确 file ID、version ID 与相对路径。
2. 模型请求 `check_files_state`、`prepare_files` 时，构造的路径混用了 file ID 与 version ID 的片段；随后 5 次调用持续使用同一错误路径。
3. 5 次外层状态都是 `ok`，但内部文件结果均为 `status: unavailable`、`reason_code: file_source_unavailable`，`ready_indices` 为空，document/file/version 身份为空。
4. 没有调用 `retrieve_files`；result 中 `active_generation` 为空、三路 `unit_count` 均为 0，索引 outbox 的 `attempt_count` 为 0。
5. 最终交付的是无法读取文件的说明；case 仍记为完成。

因此本例支持的解释是：**工具参数错误导致来源不可用，未完成有效文件准备，也未进行文档检索。** 这不是“已成功召回后模型理解错了”，目前也没有证据将其定性为解析器、embedding 或数据库写入故障。当前检索集使用更早的成功解析补齐文档 63，所以当前 Hit@5 命中并不与历史失败矛盾。

契约可从[文件工具适配器](../../src/personagraph/tools/files/file_adapter.py)、[来源解析](../../src/personagraph/workspace/files/access.py)、[工具执行](../../src/personagraph/tools/execution.py)和[轨迹状态投影](../../src/personagraph/trajectory/recorder.py)核查：适配器可以合法返回文件不可用，外层仍记录执行成功。`file_source_unavailable` 本身是归一化错误码，单靠该码无法区分路径、权限等原因；本例的路径错误另有实际输入作证。

### 其他错误需要按实际证据分别判断

历史评审索引已记录：`190:2` 的错误人数出现在 VLM observation 中，支持视觉理解错误传播到答案的判断；`32:1` 的视觉返回内容不完整，但缺少 finish reason，不能断言一定是 token 上限截断。`53:3` 涉及参考答案/题意口径，`206:0` 则不能仅因最终不作答就叫作幻觉。具体证据与限定见[评审说明](../docbench/previous_results/gate_off_highland235b_123/reviews/README.md)。

## 6. 待讨论事项与更新约定

| 待讨论事项 | 已知依据 | 尚未实施的后续工作 |
| --- | --- | --- |
| 替代证据和联合证据如何计分 | `67:4`、`80:5` 有有效但未标注的候选；平铺 qrels 表达有限 | 审阅替代证据组/必需证据组的标签方案，再发布新快照 |
| 如何衡量描述损失 | 当前只能检索图片的文字描述 | 区分图片本身含证据、描述含证据、描述可被召回三个层面 |
| 怎样与历史 QA 做归因 | 当前 Hit 高，历史仍有访问、感知、作答和评分口径问题 | 逐题核对当时材料，标记最早可证实的失败环节；轨迹语料化另行讨论 |
| 如何报告工具可靠性 | 外层 `ok` 不保证文件可用 | 增加业务状态、失败调用数、受影响题数和重试恢复的分别统计 |
| 是否扩展检索难度 | 已知文档，且部分候选池较小 | 继续分候选规模报告；是否增加跨文档设置另定 |
| 是否优化建库速度 | 逐条写库成本已观察到 | 讨论连接复用与批量事务；本轮未改变生产写库算法 |

当前先保留上述数据与指标作为基准。后续决定应在本文写明“已确认 / 待验证 / 已实施”，新增实验链接到独立结果目录；不回写历史排名、裁判分数或冻结语料来替代一次新的验证。

[judge-summary]: ../docbench/previous_results/gate_off_highland235b_123/runs/l1-gate-off-highland235b-123-20260914-a/scoring/summary.json
[codex-cases]: ../docbench/previous_results/gate_off_highland235b_123/reviews/codex/case_reviews.json
[claude-cases]: ../docbench/previous_results/gate_off_highland235b_123/reviews/claude/case_reviews.json
[historical-cases]: ../docbench/previous_results/gate_off_highland235b_123/runs/l1-gate-off-highland235b-123-20260914-a/cases/
[case63-trajectory]: ../docbench/previous_results/gate_off_highland235b_123/runs/l1-gate-off-highland235b-123-20260914-a/cases/docbench-63-2/trajectory.json
[case63-result]: ../docbench/previous_results/gate_off_highland235b_123/runs/l1-gate-off-highland235b-123-20260914-a/cases/docbench-63-2/result.json
