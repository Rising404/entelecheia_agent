# Entelecheia L1：DocBench 评测与评分复核 / Evaluation review

整理日期：2026-09-11。范围：125 题首跑、3 题传输失败补跑、5 题其他执行失败补跑。
本文合并既有评测报告、Codex 评分覆盖层与 CC（Claude）逐题复核；本轮只核对档案与算术，
不新增被测系统调用或裁判调用。原始分数与原档案保持不变。

## 1. 先分清六个成绩

| 评分来源 | 首跑 /125 | 首跑比例 | 8 题补跑替换后 /125 | 合并比例 |
| --- | ---: | ---: | ---: | ---: |
| 原始 DeepSeek 裁判 | 91 | 72.8% | 96 | 76.8% |
| Codex 存档复核 | 99 | 79.2% | 104 | 83.2% |
| CC 存档复核 | 100 | 80.0% | 105 | 84.0% |

对外基线仍应同时列原判首跑 72.8% 与定向补跑后 76.8%，并披露补跑策略。
两套复核是模型辅助的事后评分意见，不是人类金标、官方榜单或新的系统运行。
不能挑选每题最高的评审分数合成一个更高成绩；CC 比 Codex 高 1 分也不是 agent 能力提高。

[公开统计](../../showcase_125.summary.json) 现已分列 DeepSeek、Codex 和 CC 的逐题评分，
CC 状态为 `archived_provisional`。原位补齐复核层会更新整份公开投影的哈希，
不改变原始运行、裁判分数或 Codex 列；更新前 pending 投影仅保留为历史身份。
新旧哈希见 [来源索引](README.md#来源身份)，CC 字段的可读对应表见第 8 节。

## 2. 样本、配置与复现边界

### 2.1 样本构成

125 个 case 对应 125 份不同 PDF，每份抽取 1 题；8 次补跑复用其中 8 题，
所以是 125 个独立题目、133 次实际执行。五领域各 25 题，
Text 50 题，其余三类各 25 题，但“领域 × 题型”并不逐格均衡。

| 领域 | Text | Multimodal | Metadata | Unanswerable | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Academia | 5 | 12 | 3 | 5 | 25 |
| Finance | 7 | 12 | 3 | 3 | 25 |
| Government | 13 | 0 | 7 | 5 | 25 |
| Laws | 12 | 0 | 7 | 6 | 25 |
| News | 13 | 1 | 5 | 6 | 25 |
| 合计 | 50 | 25 | 25 | 25 | 125 |

24/25 的 Multimodal 来自学术和金融，不能把领域分差简单当成领域难度排名。
原报告确认与紧邻的 20 题开发检查重叠 4 道题、12 份文档；
同题 ID 为 123:0、156:4、170:3、219:3。它没有穷尽更早的全部开发接触。
会话和目录隔离不等于严格未接触的 hold-out 测试。

### 2.2 被测配置

| 项目 | 存档配置与限制 |
| --- | --- |
| 主模型 / 裁判 | 请求名均为 `deepseek-chat`，同一模型家族与服务可能带来相关偏差 |
| 视觉模型 | 请求名 `qwen3-vl-30b-a3b-instruct`，不是主模型自带视觉 |
| 检索 | BGE-M3 Dense + learned sparse + BM25 → RRF → BGE reranker |
| 本地设备 | Apple MPS，FP32；DeepSeek 主模型通过远程 API 调用 |
| 检索策略 | 三路 required，strict；配置本身不证明每题都执行或命中相关证据 |
| 执行 | 每批单 worker 串行；Turn 1500 秒，case 外层 1800 秒 |
| 评分 | DocBench prompt-compatible 二元评分，`official_comparable=false` |

BGE-M3 revision 为 `5617a9f61b028005a4858fdac845db406aefb181`；
reranker revision 为 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。
远程模型名是请求标识，不保证底层权重版本完全冻结。三轮 source 身份见
[来源索引](README.md#来源身份)，当前公开仓库不能仅凭旧 Git commit 精确重建脏工作树快照。

每次独立执行（首跑或补跑）使用独立 Session、状态和 Project，不继承上一次执行的回答、笔记或历史索引；
同一次执行内部的 L1 Attempt 仍共享该会话并延续笔记。
QA/参考材料不挂入 agent 工作目录；不同执行不按最高分挑选答案。
这份公开分析不附原 PDF、题答或轨迹，因此不能让读者仅凭本文独立重新判定全部答案。

## 3. 补跑策略与结果

| 补跑类别 | Case ID（省略 `docbench:` 前缀） | 补跑结果 |
| --- | --- | --- |
| 传输失败 3 题 | 203:1、206:0、207:2 | 三题成功交付，203:1 与 207:2 判 1；206:0 判 0 |
| 其他执行失败 5 题 | 40:0、117:0、129:0、148:2、183:0 | 五题成功交付，40:0、129:0、183:0 判 1；117:0、148:2 判 0 |

三套评分口径对这 8 个新答案的二元标签一致：5 题由 0 变 1，3 题仍为 0。
其余 117 题沿用首跑答案。没有剔除再次答错的题，也没有将 133 次执行当成 133 道独立题。
“并入补跑”是选择性恢复结果，不是新的一次全量首跑，更不是普遍 best-of-N 指标。

## 4. 分项成绩

所有分项均以同一 case ID 集合复算。下面明确标注首跑与合并，避免把不同答案集合直接比较。
Codex 与 CC 的比较表示评分规则/意见差异，不是两个独立 agent 的性能差异。

### 4.1 分领域

| 领域 | DeepSeek 首跑 | CC 首跑 | DeepSeek 合并 | Codex 合并 | CC 合并 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Academia | 18/25 | 19/25 | 19/25 | 20/25 | 20/25 |
| Finance | 18/25 | 22/25 | 18/25 | 21/25 | 22/25 |
| Government | 21/25 | 22/25 | 22/25 | 23/25 | 23/25 |
| Laws | 16/25 | 20/25 | 16/25 | 19/25 | 20/25 |
| News | 18/25 | 17/25 | 21/25 | 21/25 | 20/25 |
| 总计 | 91/125 | 100/125 | 96/125 | 104/125 | 105/125 |

### 4.2 分题型

| 题型 | DeepSeek 首跑 | CC 首跑 | DeepSeek 合并 | Codex 合并 | CC 合并 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 40/50 | 44/50 | 43/50 | 46/50 | 47/50 |
| Multimodal | 18/25 | 20/25 | 19/25 | 20/25 | 21/25 |
| Metadata | 17/25 | 16/25 | 17/25 | 17/25 | 16/25 |
| Unanswerable | 16/25 | 20/25 | 17/25 | 21/25 | 21/25 |

Text 的合并复核分不是纯 RAG 召回率；Metadata 的误差包括读取、页数定义、统计口径和模型整合。
题型名也不等于该题实际只使用同名工具。具体链路证据见 [失败归因](FAILURE_ATTRIBUTION.md)。

## 5. 两位评审如何判断，以及哪里不同

### 5.1 方法

Codex 对原判 34 个零分做复核，对 91 个一分做语义筛查，并针对争议检查原文和轨迹；
这不意味着穷尽核查每个 PDF 或每条附加事实。
CC 逐题给出首跑及补跑评分，其档案声明有 13 题回到原 PDF，其余 112 题未标记全文核对。
CC 评审时能看到 Codex 字段，所以“另行判断”不等于盲评或统计独立。

两者均采用偏向核心问项的口径：核心正确时可保留 1，并把附加事实错误另记。
CC 将物理 PDF 页数作为页数评分约定，并在参考与原文冲突时优先原文；
材料不足以把该约定宣称为所有 DocBench 题的官方统一计页规则。

### 5.2 与 Codex 不同的三题

| Case ID | Codex | CC | 分歧理由（不复制原题答） |
| --- | ---: | ---: | --- |
| 80:5 | 0 | 1 | 问项类别与参考类别边界不一致；CC 按文档分类接受答案，但仍标可争议 |
| 178:1 | 0 | 1 | 核心内容正确却混有多条错误陈述；CC 给核心分并单列质量缺陷，仍标可争议 |
| 220:1 | 1 | 0 | 识别了相关事实，但最终结论转移问项；是否可因隐含信息给分存在分歧 |

三题在首跑和合并口径下均不同，净差为 +1。CC 对 80:5、178:1 标记原 PDF 核对，
220:1 未标记；本文不把最后一项表述为本轮亲自验证原文后的定论。

CC 同意 Codex 已改判的 8 题：42:5、68:8、78:6、85:0、112:2、157:5、158:1、167:0。
其中 Codex 将六题归为语义判分问题，将 85:0、167:0 归为参考内容问题；
这两类不能全部算作“DeepSeek 理解错误”。CC 在这八题中仅对 167:0 标记本次原 PDF 核对，
不能写成两位评审均独立复查过八份原件。

相对 DeepSeek，CC 共改动 11 个标签：上述 8 题加 80:5、178:1 为 0→1，220:1 为 1→0。
这些调整不回写原裁判文件。

### 5.3 两套争议集合都保留

- Codex 的 9 道未决：26:4、37:4、53:3、67:4、80:5、111:1、144:3、199:8、220:1。
- CC 的 8 道可争议：35:3、53:3、67:4、80:5、111:1、144:3、178:1、199:8。
- 二者并集为 11 题。新评审不自动清除旧评审的不确定性。

CC 的争议类型包括摘要核心是否充分、统计科目/类别口径、物理与印刷页数、
封面日期是否代表发布日期、推断能否回答事实问项，以及正确核心与错误附加内容的取舍。
Codex 另保留两道闭域动态外部事实题和一题结论错位的争议。

CC 标记的 8 题当前为 3 个 1、5 个 0。若把这 8 题一律置 0 或置 1，
首跑得到 97–105/125，即 77.6%–84.0%。这只是**指定争议题的评分敏感性范围**，
不是置信区间，不保证其他题没有错判，也不是系统真实准确率上下界。
同法对合并答案计算为 102–110/125；不把范围上端作为成绩宣传。

26:4、37:4 属于归并后的 Unanswerable 中的 `una-web`，涉及闭域条件不提供的外部/动态事实。
固定 125 分母仍保留这两题。CC 提到的 100/123（约 81.3%）只是移除两题后的附加口径，
不能替代 100/125 或与其他未改分母的成绩直接比较。

## 6. 执行可靠性与成本不能由总分覆盖

首跑实质交付 117/125；8 次补跑均实质交付。首跑另有 6 次保存答案后 worker exit=-6，
与 8 次执行失败不重叠。提交正文与评测答案一致不等于历史索引、窗口释放或进程退出正常。

首跑逐题耗时累计 31,739.979 秒，平均 253.920 秒（约 4 分 14 秒），中位 175.628 秒，
P90 519.846 秒，最长 1,501.125 秒。133 次执行累计 33,884.363 秒；
这是 case 耗时之和，不包含完整预检、题间和裁判开销。

三批记录主模型 input 15,586,043、output 270,447，合计 15,856,490 token；
不含独立视觉、裁判或缺失 usage 的潜在计费。失败诊断不能作为第二份成功响应重复累加。
完整分批表与统计边界保持在 [结果说明](../../README.md#执行状态耗时与-token)。

所有三批存档 `baseline_eligible=false`，不因为存在复核高分就改成正式基线验收通过。
没有同条件的 BM25-only、裸模型、DSH 或 gate 开关对照，不能据此归功于某个模块。

## 7. CC 首跑零分的分类

以下是 CC 对首跑 25 个零分的互斥整理，不是合并后 20 个零分的分类，
也不是每题只有一个因果因素。特别是 151:3 的“解析不完整”解释存在不同证据，
须结合 [覆盖与计数的修订](FAILURE_ATTRIBUTION.md#4-文档读取覆盖与统计) 阅读。

| 分类 | 数量 | Case ID |
| --- | ---: | --- |
| 正确核心候选被验证链拦至未交付 | 2 | 183:0、40:0 |
| 模型传输失败或工具完成未确认 | 4 | 203:1、206:0、207:2、129:0 |
| 计数、页数、词数 | 6 | 69:4、111:1、117:0、148:2、151:3、160:2 |
| 视觉解释错误 | 3 | 17:8、32:1、190:2 |
| 文档有无信息的判断错误 | 2 | 140:0、153:2 |
| 题意、参考口径或回答覆盖 | 6 | 35:3、53:3、67:4、199:8、220:1、222:0 |
| 闭域外部事实 `una-web` | 2 | 26:4、37:4 |

这张表保留 CC 的归类，而不将其视为已验证的故障机制。
例如 32:1 还存在视觉生成不完整和结果复用问题，40:0 同时有机械引用错误，
不能统一归为模型能力不足或某一层 Host 单独导致。

## 8. CC 逐题评分覆盖层

来源是 `claude_review_scores.json`，SHA-256：
`de04415ef7871877e16ab6d0ce5cbf74deba9a13c497723a10b69af21bbe5820`。
只投影 ID、0/1 标签和两个声明标记；不复制其自由文本说明、路径、原题或答案。

下面 125 个唯一 ID 及四个 CC 字段与 [公开 JSON](../../showcase_125.summary.json) 完全一致，
分别对应 `cc_archived_first_score`、`cc_archived_selected_score`、`cc_debatable` 和
`cc_source_checked_reported`。
两列求和分别为 100 和 105；只有 5 个 0→1，其余 120 个标签相同。
“可争议”和“原 PDF 核对”是 CC 档案中的布尔声明，**不是本轮重新核对 PDF 的证明**。
JSON 中 `cc_review` 明示模型评审、非盲审、PDF 核验为评审者自报。
三套主评分及声明计数均可直接从 JSON 复算；归因与原文正确性不由这些字段证明。
原判与 Codex 两列使用同一个 case ID 联结，禁止按表格行号猜测对应关系。

| Case ID | CC 首跑 | CC 合并 | CC 可争议 | CC 原 PDF 核对 |
| --- | ---: | ---: | --- | --- |
| docbench:1:1 | 1 | 1 | false | false |
| docbench:2:0 | 1 | 1 | false | false |
| docbench:4:0 | 1 | 1 | false | false |
| docbench:6:0 | 1 | 1 | false | false |
| docbench:7:2 | 1 | 1 | false | false |
| docbench:10:3 | 1 | 1 | false | false |
| docbench:11:4 | 1 | 1 | false | false |
| docbench:13:3 | 1 | 1 | false | false |
| docbench:17:8 | 0 | 0 | false | true |
| docbench:19:4 | 1 | 1 | false | false |
| docbench:20:4 | 1 | 1 | false | false |
| docbench:21:0 | 1 | 1 | false | false |
| docbench:23:3 | 1 | 1 | false | false |
| docbench:26:4 | 0 | 0 | false | false |
| docbench:29:0 | 1 | 1 | false | false |
| docbench:30:2 | 1 | 1 | false | false |
| docbench:31:2 | 1 | 1 | false | false |
| docbench:32:1 | 0 | 0 | false | false |
| docbench:35:3 | 0 | 0 | true | false |
| docbench:36:2 | 1 | 1 | false | false |
| docbench:37:4 | 0 | 0 | false | false |
| docbench:38:4 | 1 | 1 | false | false |
| docbench:39:9 | 1 | 1 | false | false |
| docbench:40:0 | 0 | 1 | false | false |
| docbench:42:5 | 1 | 1 | false | false |
| docbench:50:1 | 1 | 1 | false | false |
| docbench:51:3 | 1 | 1 | false | false |
| docbench:53:3 | 0 | 0 | true | true |
| docbench:54:4 | 1 | 1 | false | false |
| docbench:55:2 | 1 | 1 | false | false |
| docbench:60:2 | 1 | 1 | false | false |
| docbench:61:6 | 1 | 1 | false | false |
| docbench:63:2 | 1 | 1 | false | false |
| docbench:64:1 | 1 | 1 | false | false |
| docbench:66:3 | 1 | 1 | false | false |
| docbench:67:4 | 0 | 0 | true | false |
| docbench:68:8 | 1 | 1 | false | false |
| docbench:69:4 | 0 | 0 | false | true |
| docbench:73:4 | 1 | 1 | false | false |
| docbench:75:0 | 1 | 1 | false | false |
| docbench:76:4 | 1 | 1 | false | false |
| docbench:78:6 | 1 | 1 | false | false |
| docbench:80:5 | 1 | 1 | true | true |
| docbench:81:2 | 1 | 1 | false | false |
| docbench:82:3 | 1 | 1 | false | false |
| docbench:83:3 | 1 | 1 | false | false |
| docbench:85:0 | 1 | 1 | false | false |
| docbench:86:8 | 1 | 1 | false | false |
| docbench:87:9 | 1 | 1 | false | false |
| docbench:88:9 | 1 | 1 | false | false |
| docbench:92:1 | 1 | 1 | false | false |
| docbench:93:3 | 1 | 1 | false | false |
| docbench:94:0 | 1 | 1 | false | false |
| docbench:103:1 | 1 | 1 | false | false |
| docbench:104:0 | 1 | 1 | false | false |
| docbench:105:2 | 1 | 1 | false | false |
| docbench:106:0 | 1 | 1 | false | false |
| docbench:108:2 | 1 | 1 | false | false |
| docbench:109:3 | 1 | 1 | false | false |
| docbench:110:2 | 1 | 1 | false | false |
| docbench:111:1 | 0 | 0 | true | false |
| docbench:112:2 | 1 | 1 | false | false |
| docbench:113:4 | 1 | 1 | false | false |
| docbench:114:2 | 1 | 1 | false | false |
| docbench:117:0 | 0 | 0 | false | true |
| docbench:118:1 | 1 | 1 | false | false |
| docbench:119:2 | 1 | 1 | false | false |
| docbench:120:4 | 1 | 1 | false | false |
| docbench:122:3 | 1 | 1 | false | false |
| docbench:123:0 | 1 | 1 | false | false |
| docbench:124:2 | 1 | 1 | false | false |
| docbench:125:1 | 1 | 1 | false | false |
| docbench:128:2 | 1 | 1 | false | false |
| docbench:129:0 | 0 | 1 | false | false |
| docbench:132:2 | 1 | 1 | false | false |
| docbench:137:1 | 1 | 1 | false | false |
| docbench:140:0 | 0 | 0 | false | true |
| docbench:144:3 | 1 | 1 | true | false |
| docbench:148:2 | 0 | 0 | false | false |
| docbench:149:0 | 1 | 1 | false | false |
| docbench:151:3 | 0 | 0 | false | true |
| docbench:152:1 | 1 | 1 | false | false |
| docbench:153:2 | 0 | 0 | false | true |
| docbench:156:4 | 1 | 1 | false | false |
| docbench:157:5 | 1 | 1 | false | false |
| docbench:158:1 | 1 | 1 | false | false |
| docbench:159:3 | 1 | 1 | false | false |
| docbench:160:2 | 0 | 0 | false | true |
| docbench:161:0 | 1 | 1 | false | false |
| docbench:164:1 | 1 | 1 | false | false |
| docbench:165:0 | 1 | 1 | false | false |
| docbench:166:2 | 1 | 1 | false | false |
| docbench:167:0 | 1 | 1 | false | true |
| docbench:168:3 | 1 | 1 | false | false |
| docbench:169:2 | 1 | 1 | false | false |
| docbench:170:3 | 1 | 1 | false | false |
| docbench:171:5 | 1 | 1 | false | false |
| docbench:174:3 | 1 | 1 | false | false |
| docbench:176:1 | 1 | 1 | false | false |
| docbench:178:1 | 1 | 1 | true | true |
| docbench:179:5 | 1 | 1 | false | false |
| docbench:183:0 | 0 | 1 | false | false |
| docbench:187:0 | 1 | 1 | false | false |
| docbench:189:2 | 1 | 1 | false | false |
| docbench:190:2 | 0 | 0 | false | true |
| docbench:191:1 | 1 | 1 | false | false |
| docbench:192:4 | 1 | 1 | false | false |
| docbench:193:1 | 1 | 1 | false | false |
| docbench:194:1 | 1 | 1 | false | false |
| docbench:198:4 | 1 | 1 | false | false |
| docbench:199:8 | 0 | 0 | true | false |
| docbench:202:2 | 1 | 1 | false | false |
| docbench:203:1 | 0 | 1 | false | false |
| docbench:206:0 | 0 | 0 | false | false |
| docbench:207:2 | 0 | 1 | false | false |
| docbench:208:1 | 1 | 1 | false | false |
| docbench:211:1 | 1 | 1 | false | false |
| docbench:212:2 | 1 | 1 | false | false |
| docbench:214:4 | 1 | 1 | false | false |
| docbench:217:2 | 1 | 1 | false | false |
| docbench:219:3 | 1 | 1 | false | false |
| docbench:220:1 | 0 | 0 | false | false |
| docbench:221:0 | 1 | 1 | false | false |
| docbench:222:0 | 0 | 0 | false | false |
| docbench:224:2 | 1 | 1 | false | true |
