# 多文档 L1 问答用例

输入为 **多份文档 + 1 个用户问题**，输出保存真实模型回复、工具执行情况、
提交后收尾状态和人工复核参考。`pdf` 是原创合成套件；`docbench` 使用本地 DocBench
原题与原 PDF，并添加三份干扰文档。两者都不是模型总体能力基准。

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

### 直接使用已提供的样本

| 内容 | 文件 |
| --- | --- |
| 十二道完整问题及每题附件列表 | [questions.json](dataset/questions.json) |
| 参考答案、唯一目标文件、物理页码与复核标准 | [gold.json](dataset/gold.json) |
| 四份固定输入 PDF | [document_01.pdf](dataset/attachments/document_01.pdf)、[document_02.pdf](dataset/attachments/document_02.pdf)、[document_03.pdf](dataset/attachments/document_03.pdf)、[document_04.pdf](dataset/attachments/document_04.pdf) |
| 文件大小、SHA-256 与字体身份 | [manifest.json](dataset/manifest.json) |

这些文件是两轮十二题评测使用的原创合成样本，按原字节迁入；四份 PDF 由全部题目共享。
用户已授权公开此精确样本包，仓库隐私门禁校验七个文件的完整集合和哈希。
在 GUI 中每题新建 Session，绑定仓库外工作目录，上传四份 PDF，再输入一个问题。
模型只能接收 PDF 和当前问题；`gold.json` 留给人工复核，不能挂载整个样本目录。
原始模型回复、轨迹与数据库不随样本包发布。

### 重新生成

离线导出可直接使用的文件，输出必须在仓库之外，且目录不能已存在：

```bash
.venv/bin/python -m evals.multi_doc.export_pdf \
  --output /private/tmp/entelecheia-pdf-benchmark
```

导出目录包含 `attachments/` 下的四份 PDF、`questions.json`、`gold.json` 和哈希清单
`manifest.json`。模型只能接收四份 PDF 与当前问题，不能挂载包含金标的导出根目录。
导出与离线测试不读取模型凭据、不调用 API；实际推理另需显式启用。

中文字体会以子集嵌入 PDF。macOS 默认使用 `/System/Library/Fonts/STHeiti Light.ttc`；
其他环境需将 `ENTELECHEIA_EVAL_CJK_FONT` 设置为已有中文 TrueType 字体的绝对路径。
生成过程不下载字体；清单记录字体哈希，同一字体与依赖环境下 PDF 可重复生成。
更换字体会改变 PDF 字节和附件哈希，应作为一次新的样本导出记录。
Ubuntu CI 显式安装 `fonts-wqy-zenhei` 并设置字体路径，运行 PDF 再生成与原生读取检查。

## DocBench 四附件套件

[选题与附件组合](docbench_selection.json)锁定 12 道原题、12 份不同目标文档，以及各题
三份干扰 PDF。总计使用 14 份不同的完整 PDF，覆盖学术、金融、政府、法律和新闻；
9 道文本题、2 道表格题、1 道图示题。每题为四附件，物理页数合计 4–242 页。
不裁页、不改写问题、不把目标文件名或参考答案加进模型输入；文件显示名统一为
`document_01.pdf` 至 `document_04.pdf`，目标位置在 12 题中均衡分布。

[来源适配](docbench.py)复用正式 DocBench selection loader，校验原选集、QA 与 PDF 哈希。
原题、答案和上游 evidence 按原文读取；物理证据页码是本套件额外核查的复核信息。
每次复制到隔离 Project 前再次核验 PDF 字节。模型仅得到通用回答／引用要求、当前原题和四份 PDF。
此设置改变了官方单文档输入，结果不得当作官方 DocBench 成绩或与历史 125 题直接比较。

先按 [DocBench 数据准备](../docbench/reproduce_or_run_script/README.md)取得授权的本地完整来源，
然后离线导出可浏览的问题和附件包：

```bash
.venv/bin/python -m evals.multi_doc.docbench \
  --data-root "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/source/data"
```

导出默认位于 `evals/multi_doc/local/docbench/`，包含 `questions.json`、`gold.json`、
`manifest.json` 和 `attachments/<case-id>/` 下的每题四份 PDF。
该目录被 Git 忽略并由隐私门禁禁止发布；只提交无原题正文的选题身份与代码。
`questions.json` 是这批原题及附件列表的入口；`gold.json` 只供人工复核。
已有导出不会覆盖；如需保留另一份导出，使用 `--output` 指定仓库外新目录。

从同一冻结来源启动真实测试，入口会先探测当前任务模型与 VLM：

```bash
.venv/bin/python -m evals.multi_doc.run --suite docbench --list \
  --data-root "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/source/data"
.venv/bin/python -m evals.multi_doc.run --suite docbench --allow-live \
  --data-root "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/source/data"
```

可用 `--cases docbench-1-1` 定向运行。每题仍使用独立子进程、Project 和 Session，
沿用当前模型配置与 L1 产品链路。长文档采用现有 complex 套件的 32 次 attempt、
1800 秒整轮预算；引用／语义审查要求不变。输出保存在本目录下被忽略的 `runs/`，
显式记录来源选集哈希、完成状态与收尾时间；字面匹配不是正确率。

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
.venv/bin/python -m evals.multi_doc.run --list

# 分别使用现行任务与视觉适配器发送最小真实请求。
.venv/bin/python -m evals.multi_doc.probe --allow-live

# 先自动探测 API，再运行现行 PDF 套件的十二题。
.venv/bin/python -m evals.multi_doc.run --allow-live --suite pdf

# 单独运行或在更新视觉凭据后补跑；每次产生新的运行目录。
.venv/bin/python -m evals.multi_doc.run --allow-live --suite smoke --cases visual_source

# 扩充后的短文本套件：4 道原题 + 8 道新增题。
.venv/bin/python -m evals.multi_doc.run --suite text --list
.venv/bin/python -m evals.multi_doc.run --allow-live --suite text

# 查看并实际运行复杂混合格式套件。
.venv/bin/python -m evals.multi_doc.run --suite complex --list
.venv/bin/python -m evals.multi_doc.run --allow-live --suite complex
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
.venv/bin/python -m ruff check evals/multi_doc
.venv/bin/python -m pytest -q evals/multi_doc
```
