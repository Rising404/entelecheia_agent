# Entelecheia L1：DocBench 评测与评分复核 / Evaluation review

整理日期：2026-09-11。本文统计共同的 123 题，排除两道依赖外部信息的题目
`docbench:26:4`、`docbench:37:4`；保留首跑、3 题传输失败补跑与 5 题其他执行失败补跑的结果。
任务主模型与 DeepSeek 裁判均为 **DeepSeek Flash，未开启思考模式**；另两位裁判为 **Astra** 与 **Opus 5**。
档案中的 Codex、CC 分别对应 Astra、Opus 5；逐题数据见 [评分汇总](../../showcase_125.summary.json)。

## 1. 首跑与补跑合并成绩

| 评分方 | 首跑 /123 | 首跑比例 | 8 题补跑替换后 /123 | 合并比例 |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek Flash | 91 | 73.98% | 96 | 78.05% |
| Astra | 99 | 80.49% | 104 | 84.55% |
| Opus 5 | 100 | 81.30% | 105 | 85.37% |

## 2. 样本、配置与复现边界

### 2.1 样本构成

原始档案保存 125 份 PDF 各 1 题及 8 次补跑，共 133 次实际执行。
本篇排除两道联网题后，统计 123 道题、131 次执行。学术领域 23 题，其余领域各 25 题；
Text 50 题，Multimodal、Metadata 各 25 题，Unanswerable 23 题。“领域 × 题型”并不逐格均衡。

| 领域 | Text | Multimodal | Metadata | Unanswerable | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Academia | 5 | 12 | 3 | 3 | 23 |
| Finance | 7 | 12 | 3 | 3 | 25 |
| Government | 13 | 0 | 7 | 5 | 25 |
| Laws | 12 | 0 | 7 | 6 | 25 |
| News | 13 | 1 | 5 | 6 | 25 |
| 合计 | 50 | 25 | 25 | 23 | 123 |

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

## 3. 补跑策略与结果

| 补跑类别 | Case ID（省略 `docbench:` 前缀） | 补跑结果 |
| --- | --- | --- |
| 传输失败 3 题 | 203:1、206:0、207:2 | 三题成功交付，203:1 与 207:2 判 1；206:0 判 0 |
| 其他执行失败 5 题 | 40:0、117:0、129:0、148:2、183:0 | 五题成功交付，40:0、129:0、183:0 判 1；117:0、148:2 判 0 |

## 4. 分项成绩

### 4.1 分领域

| 领域 | DeepSeek Flash 首跑 | Astra 首跑 | Opus 5 首跑 | DeepSeek Flash 合并 | Astra 合并 | Opus 5 合并 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Academia | 18/23 | 19/23 | 19/23 | 19/23 | 20/23 | 20/23 |
| Finance | 18/25 | 21/25 | 22/25 | 18/25 | 21/25 | 22/25 |
| Government | 21/25 | 22/25 | 22/25 | 22/25 | 23/25 | 23/25 |
| Laws | 16/25 | 19/25 | 20/25 | 16/25 | 19/25 | 20/25 |
| News | 18/25 | 18/25 | 17/25 | 21/25 | 21/25 | 20/25 |
| 总计 | 91/123 | 99/123 | 100/123 | 96/123 | 104/123 | 105/123 |

### 4.2 分题型

| 题型 | DeepSeek Flash 首跑 | Astra 首跑 | Opus 5 首跑 | DeepSeek Flash 合并 | Astra 合并 | Opus 5 合并 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text | 40/50 | 43/50 | 44/50 | 43/50 | 46/50 | 47/50 |
| Multimodal | 18/25 | 19/25 | 20/25 | 19/25 | 20/25 | 21/25 |
| Metadata | 17/25 | 17/25 | 16/25 | 17/25 | 17/25 | 16/25 |
| Unanswerable | 16/23 | 20/23 | 20/23 | 17/23 | 21/23 | 21/23 |

## 5. 两位评审如何判断，以及哪里不同

### 5.1 方法

两位评审基于已生成的答案与既有评分材料进行非盲分析。

Astra 对共同范围内原判 32 个零分做复核，对 91 个一分做语义筛查，并针对争议检查原文和轨迹。

Opus 5 逐题给出首跑及补跑评分，其档案声明有 13 题回到原 PDF，其余 110 题未标记全文核对。

两者均采用偏向核心问项的口径：核心正确时可保留 1，并把附加事实错误另记。

### 5.2 与 Astra 不同的三题

| Case ID | Astra | Opus 5 | 分歧理由 |
| --- | ---: | ---: | --- |
| 80:5 | 0 | 1 | 问项类别与参考类别边界不一致；Opus 5 按文档分类接受答案，但仍标可争议 |
| 178:1 | 0 | 1 | 核心内容正确却混有多条错误陈述；Opus 5 给核心分并单列质量缺陷，仍标可争议 |
| 220:1 | 1 | 0 | 识别了相关事实，但最终结论转移问项；是否可因隐含信息给分存在分歧 |

三题在首跑和合并口径下均不同，净差为 +1。Opus 5 对 80:5、178:1 标记原 PDF 核对，
220:1 未标记原 PDF 核对，保留为评审判断。

Opus 5 同意 Astra 已改判的 8 题：42:5、68:8、78:6、85:0、112:2、157:5、158:1、167:0。
其中 Astra 将六题归为语义判分问题，将 85:0、167:0 归为参考内容问题。
Opus 5 在这八题中仅对 167:0 标记本次原 PDF 核对，其余沿用既有材料。

相对 DeepSeek，Opus 5 共改动 11 个标签：上述 8 题加 80:5、178:1 为 0→1，220:1 为 1→0。

### 5.3 两套争议集合都保留

- Astra 的 7 道未决：53:3、67:4、80:5、111:1、144:3、199:8、220:1。
- Opus 5 的 8 道可争议：35:3、53:3、67:4、80:5、111:1、144:3、178:1、199:8。
- 二者并集为 9 题。新评审不自动清除旧评审的不确定性。

Opus 5 的争议类型包括摘要核心是否充分、统计科目/类别口径、物理与印刷页数、
封面日期是否代表发布日期、推断能否回答事实问项，以及正确核心与错误附加内容的取舍。
Astra 另保留一题结论错位的争议。

Opus 5 标记的 8 题当前为 3 个 1、5 个 0。若把这 8 题一律置 0 或置 1，
首跑得到 97–105/123，即 78.86%–85.37%。这只是**指定争议题的评分敏感性范围**，
不是置信区间，不保证其他题没有错判，也不是系统真实准确率上下界。
同法对合并答案计算为 102–110/123，即 82.93%–89.43%。

排除的 26:4、37:4 均属于 Unanswerable 中的 `una-web`，依赖闭域条件未提供的外部或动态信息。

## 6. Opus 5 首跑零分的分类

以下按 Opus 5 的主要归因将首跑 23 个零分分组。关于解析覆盖与计数口径的分歧，见
[文档读取与统计分析](FAILURE_ATTRIBUTION.md#4-文档读取覆盖与统计)。

| 分类 | 数量 | Case ID |
| --- | ---: | --- |
| 正确核心候选被验证链拦至未交付 | 2 | 183:0、40:0 |
| 模型传输失败或工具完成未确认 | 4 | 203:1、206:0、207:2、129:0 |
| 计数、页数、词数 | 6 | 69:4、111:1、117:0、148:2、151:3、160:2 |
| 视觉解释错误 | 3 | 17:8、32:1、190:2 |
| 文档有无信息的判断错误 | 2 | 140:0、153:2 |
| 题意、参考口径或回答覆盖 | 6 | 35:3、53:3、67:4、199:8、220:1、222:0 |

部分题目涉及多个环节：32:1 还存在视觉生成不完整和结果复用问题，40:0 同时有机械引用错误。

## 7. Opus 5 逐题评分

| Case ID | Opus 5 首跑 | Opus 5 合并 | Opus 5 可争议 | Opus 5 原 PDF 核对 |
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
| docbench:29:0 | 1 | 1 | false | false |
| docbench:30:2 | 1 | 1 | false | false |
| docbench:31:2 | 1 | 1 | false | false |
| docbench:32:1 | 0 | 0 | false | false |
| docbench:35:3 | 0 | 0 | true | false |
| docbench:36:2 | 1 | 1 | false | false |
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
