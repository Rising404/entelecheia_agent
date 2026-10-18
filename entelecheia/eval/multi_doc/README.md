# 多文档 L1 问答用例

输入为 **多份原创合成文档 + 1 个用户问题**，输出保存真实模型回复、工具执行情况、
提交后收尾状态和人工复核参考。用例不属于 DocBench，也不作为模型总体能力基准。

## 现行 PDF 样本集

默认 `pdf` 套件遵循 **一道问题 + 四份 PDF，只有一份 PDF 包含完整答案证据** 的约束。
每题都提供全部四份 PDF，另三份为干扰文档；不需要跨文档拼接答案。

- 四份原创中文活动执行手册，每份四页，共十六页，正文合计约 7100 字符。
- 文件名为 `document_01.pdf` 至 `document_04.pdf`，不标记目标与干扰角色。
- 内容分别对应青岚北站、青岚南站、澄川北站、澄川南站；项目、站点和表格结构相近，
  交接地点、时间、人员、计数规则和设备状态不同，检查是否串用其他文档的事实。
- 十二题：交接信息定位四题、同一 PDF 内规则与名单计算四题、同一 PDF 内规则与设备表筛选四题。
  四份 PDF 各作为三道题的唯一目标。证据分别在物理第 2 页，或第 3–4 页。
- 模型应给出答案、来源文件名和物理页码；金标保存逐条原文证据与人工复核标准。

这是四份共享文档上的十二道定向样本，并非十二组独立语料；题型有意保持相对简单。
PDF 含可搜索中文文字层和真实表格，尚不覆盖扫描件、OCR 或复杂视觉理解。
`smoke`、`text`、`complex` 保留为历史诊断套件，其多来源或混合格式任务不属于本次 PDF 基准。

离线导出可直接使用的文件，输出必须在仓库之外，且目录不能已存在：

```bash
.venv/bin/python -m entelecheia.eval.multi_doc.export_pdf \
  --output /private/tmp/entelecheia-pdf-benchmark
```

导出目录包含 `attachments/` 下的四份 PDF、`questions.json`、`gold.json` 和哈希清单
`manifest.json`。模型只能接收四份 PDF 与当前问题，不能挂载包含金标的导出根目录。
导出与离线测试不读取模型凭据、不调用 API；实际推理另需显式启用。

中文字体会以子集嵌入 PDF。macOS 默认使用 `/System/Library/Fonts/STHeiti Light.ttc`；
其他环境需将 `ENTELECHEIA_EVAL_CJK_FONT` 设置为已有中文 TrueType 字体的绝对路径。
生成过程不下载字体；清单记录字体哈希，同一字体与依赖环境下 PDF 可重复生成。
更换字体会改变 PDF 字节和附件哈希，应作为一次新的样本导出记录。

## 历史诊断套件

`smoke` 套件每题 4 份附件：

| 用例 | 检查目标 |
| --- | --- |
| `single_source` | 答案仅在一份文档中，其余文档包含相近地点和时间 |
| `cross_document` | 合并材料规则与人数名单，排除候补和历史数据，计算数量 |
| `distractor_revision` | 区分同名项目、站点、失效通知与正式验收日期 |
| `missing_information` | 所有文档都缺少所问信息，避免编造或混用停车费 |
| `visual_source` | 从库存图片读取本周数据，再结合文本规则计算补货量 |

`text` 套件保留上述 4 道纯文本题，再加入 8 道短文档题，共 12 题。
新增题每题 4–6 份中文 `.txt` / `.md`，合计 250–2000 字符；没有 PDF、图片或长文本填充。
题目覆盖唯一来源定位、局部修订、名单去重、单位换算、条件筛选、计划与实际状态区分、
部分信息缺失，以及同级有效文件冲突。新增题在 `text_cases.py` 维护原文证据与人工复核标准。
本套件用于观察短文本上的当前链路，不改变原有 smoke / complex 题目或校验要求。

`complex` 套件每题 16 份附件：10 份各 6 页的 PDF、4 份长文本、2 张库存图片。
文本合计约 26.3 万字符、3.76 万英文词；PDF 有可提取文字层，图片状态以颜色和图例表示。
四题共享同一套合成语料，但各自在独立状态中运行；参考事实分散在 PDF 深页、文本第 14 节
和当前图片中。旧版规则、未签草稿、近似项目与无关数字作为干扰。

| 用例 | 检查目标 |
| --- | --- |
| `deep_lookup` | 定位第 6 页正式通知，排除撤回版、草稿与相似项目 |
| `visual_replenishment` | 联合 PDF 规则、图片颜色状态、预留库存和整包订购计算 |
| `deduplicated_materials` | 跨文档人数去重、排除教师/候补，正确应用全场一次预留 |
| `partial_answer_abstention` | 回答已知物料数，同时拒绝用撤回估价、其他活动费或停车费填补报名费 |

这些是中文问题、英文材料的定向压力用例。长度主要来自模板化日常记录，不能代表真实业务
文档的全部复杂性；没有扫描 PDF、复杂版式 OCR 或低清图像覆盖。

每题有独立进程、Project、Session 和 SQLite 状态。只上传生成的附件，问题通过正常消息
提交；参考答案、用例源码、评测结果均不位于模型工作目录中。实际执行复用现行
`sessions.chat_turn` → Entry → L1、附件上传、文档收录和工具实现。

## 使用

从仓库根目录使用当前 checkout 的独立运行时：

```bash
# 离线查看用例；不读取模型凭据，不调用 API。
.venv/bin/python -m entelecheia.eval.multi_doc.run --list

# 分别使用现行任务与视觉适配器发送最小真实请求。
.venv/bin/python -m entelecheia.eval.multi_doc.probe --allow-live

# 先自动探测 API，再运行现行 PDF 套件的十二题。
.venv/bin/python -m entelecheia.eval.multi_doc.run --allow-live --suite pdf

# 单独运行或在更新视觉凭据后补跑；每次产生新的运行目录。
.venv/bin/python -m entelecheia.eval.multi_doc.run --allow-live --suite smoke --cases visual_source

# 扩充后的短文本套件：4 道原题 + 8 道新增题。
.venv/bin/python -m entelecheia.eval.multi_doc.run --suite text --list
.venv/bin/python -m entelecheia.eval.multi_doc.run --allow-live --suite text

# 查看并实际运行复杂混合格式套件。
.venv/bin/python -m entelecheia.eval.multi_doc.run --suite complex --list
.venv/bin/python -m entelecheia.eval.multi_doc.run --allow-live --suite complex
```

`--allow-live` 显式启用联网和可能收费的真实推理。复用本机已配置的 Router/L1 与视觉端点，
不写回或切换 GUI 配置。可达性探测验证实际推理请求，不以 TCP 连通或模型列表代替。
任务端点探测失败时不执行问答；视觉探测失败时保留文本题，视觉题记为 `blocked_preflight`。

文档解析使用 native reader，检索沿用当前环境的配置与已有本地模型资产，不自动下载。
禁止网页工具：`pdf`、`smoke` 和 `text` 每题最多 12 个 L1 决策步骤、600 秒运行预算；`complex` 为 32 步、1800 秒。
复杂集首次基线使用 24 步、1500 秒；L1 续读与引用修复后的重跑增加上述余量，
附件、问题和评分标准保持不变。每步最多 8 次工具调用、每次视觉分析最多 3 页。
语义检查保持 `always`。
其他功能使用产品默认值，实际完整 features 与检索模式记录在结果中。
子进程上限比运行预算多 240 秒，包含模型加载与提交后收尾。

## 结果

结果保存在本目录 `runs/<UTC时间>-<随机后缀>/`：

- `report.md`：逐题问题、模型原回复、参考答案和工具列表。
- `results.json`：API 探测和所有用例的状态汇总。
- `<case_id>.json`：运行过的题目的详细结果，回答先落盘，再等待后台收尾。
- `<case_id>.trace.json`：工具参数、完整返回证据和 L1 决策中的引用，支持按文件和物理页码复核。
- `<case_id>.log`：本地诊断日志。

汇总记录源代码哈希；逐题记录附件哈希、输入规模、参考证据和复核标准。真实运行中不修改
Python 源码。人工复核应逐项对照原文、图片、工具返回和最终回复，分别记录找对文档、证据
定位、计算准确性、无依据断言与执行失败；不能仅凭流程完成或出现关键词判定通过。

合成附件、模型可见 Project 和应用状态放在新的系统临时目录；路径记在
`results.json` 的 `private_state_root` 中。它们保留用于排查，不访问已有 GUI 会话数据库。
`runs/` 整目录忽略，不自动提交原始模型回复、日志或状态。

`chain_complete` 要求 L1 返回 completed、无运行错误或终止失败通知，且后台作业 applied、
窗口释放；它仍不等于答案正确。API 的 completed 本身可能只表示一条失败通知已收尾，
因此必须检查 `error_code`、`end_reason` 和回复内容；例如 `MODEL_OUTPUT_INVALID` 应记录为执行失败。
`literal_fact_coverage` 仅检查关键字符串是否出现，无法识别否定、错误推导或幻觉，
不得当作正确率。缺失信息题尤其需要人工复核。视觉题被跳过时，不能宣称图像链路通过。

离线检查：

```bash
.venv/bin/python -m ruff check entelecheia/eval/multi_doc
.venv/bin/python -m pytest -q entelecheia/eval/multi_doc
```
