# DocBench hybrid 检索优化

这是与 `evals/docbench/` 并列的独立召回评测包。DocBench 原有的 L1 执行、裁判和历史 QA 成绩继续由原目录持有；此处管理冻结检索语料、派生索引和检索指标。

当前有两个语料快照：`repaired75-20260923` 为现行 75 题可计分数据，`reused75-20260922` 为 66 题可计分、9 题待审的历史快照。独立 CLI 支持 SQLite FTS5 BM25 基线和生产 hybrid 路由。**75 题完整 hybrid 评测已完成：MPS、FP32，最终 Hit@5 为 74/75（98.67%），宏平均 Recall@5 为 96.04%，MRR 为 0.9296。** 这是已知文档范围内的证据召回指标；详见[结果与耗时分析](results/analysis/retrieval-repaired75-hybrid-20260923.md)。归档的 9 题 BGE CPU 检查仍只是小候选池验证。

```text
evals/docbench_hybrid_retrieval_optimize/
├── DATA_AND_STATUS.md         # 数据、当前成绩与失败归因边界（讨论稿）
├── __main__.py / cli.py        # 独立命令入口
├── paths.py                   # 产物写入位置边界
├── retrieval_dataset.py       # 语料构建、转录纠错、标注、导出
├── retrieval_eval.py          # BM25 基线、排序计分、报告
├── retrieval_hybrid.py        # 冻结语料接入生产 hybrid 路由、索引与六阶段评测
├── encoding_cache.py          # 同正文、同编码器表示的校验与复用
├── index_archive.py           # 完整索引的无损分片保存与还原
├── release_assets.py / .json   # 按需下载入口与附件/文件哈希目录
├── dataset/                   # 数据说明与小型元数据；完整语料按需下载
├── indexes/                   # 索引配方与分片清单；分片按需下载
├── results/                   # 全部检索结果，随源码提供
└── publication_manifest.json  # Git 内已审产物的 SHA-256 与大小
```

- [数据与当前状态（讨论稿）](DATA_AND_STATUS.md)
- [现行数据集](dataset/repaired75-20260923/README.md) · [下载后逐题浏览](#按需下载) · [纠错记录](dataset/repaired75-20260923/CORRECTIONS.md)
- [结果目录](results/README.md) · [索引目录](indexes/README.md)
- 测试位于 `tests/evals/docbench_hybrid_retrieval_optimize/`，复用生产 `personagraph.retrieval.relevance` 指标。

## 按需下载

普通 `git clone` 只取得代码、说明、小型数据/索引清单及全部检索结果（约 4.8 MB），不会下载语料和索引分片。完整快照在 [DocBench retrieval Release](https://github.com/Rising404/entelecheia_agent/releases/tag/docbench-retrieval-20260923) 独立公开；只看成绩无需下载附件。

| Asset ID（文件名为 ID + `.tar.gz`） | 内容 | 压缩下载大小 |
| --- | --- | ---: |
| `repaired75-20260923` | 当前 75 题语料、SQLite、导出、图片和审阅页 | 776,540,465 bytes（约 776.5 MB） |
| `reused75-20260922` | 修复前 66 题可计分的历史快照 | 182,739,609 bytes（约 182.7 MB） |
| `repaired75-bge-m3-mps-fp32` | 已建三路索引的 5 个分片与还原清单 | 170,409,289 bytes（约 170.4 MB） |

从仓库根目录执行。下载入口只依赖 Python 标准库，不会加载模型：

```bash
python3 -m evals.docbench_hybrid_retrieval_optimize.release_assets list
python3 -m evals.docbench_hybrid_retrieval_optimize.release_assets fetch \
  --asset repaired75-20260923 \
  --output /absolute/path/to/repaired75-20260923
```

`--output` 是**尚不存在的仓库外快照目录本身**，不是它的父目录。成功后该目录直接包含 `dataset.sqlite`、`assets/`、`exports/`、`INDEX.html` 和 `review/`；打开其中 `INDEX.html` 即可逐题浏览。入口按 [release_assets.json](release_assets.json) 验证压缩包及全部解压文件的大小和 SHA-256，拒绝覆盖已有目录。

也可先从 Release 手动下载，再离线验证与解包：

```bash
python3 -m evals.docbench_hybrid_retrieval_optimize.release_assets fetch \
  --asset repaired75-20260923 \
  --archive /absolute/path/to/repaired75-20260923.tar.gz \
  --output /absolute/path/to/repaired75-20260923
```

两个语料附件相互独立，常规复现只需当前快照；希望省去建库时再下载索引附件。附件保留冻结快照的原始说明，仓库中的 README 维护现行下载方式。结果中冻结的数据、索引与原完整 publication 身份保持不变；调整分发方式没有重新运行评测或修改分数。

取得语料后，无需配置原 DocBench runner 的外部数据根即可复跑 BM25：

```bash
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize --help
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize evaluate-retrieval \
  --dataset /absolute/path/to/repaired75-20260923/dataset.sqlite \
  --output evals/docbench_hybrid_retrieval_optimize/results/CHOOSE_A_NEW_RUN
```

每次输出使用新目录，拒绝覆盖历史运行。上述命令默认执行离线 BM25。下文的 `--backend hybrid --allow-live` 显式启用本地编码器和重排模型；两种入口均不启动 Agent、视觉模型或裁判。

## 公开资料与来源

本批按用户明确授权公开检索语料、标注、区域图片和结果。Git 仅跟踪说明、清单和结果；大型语料及索引分片由 Release 附件分发。本地已有语料保留，但对应大文件不再进入 Git。`publication_manifest.json` 锁定 Git 内保留产物，`release_assets.json` 另行锁定完整附件及其中每个文件；增加或修改公开产物须重新审阅相应清单，凭据检查仍执行。模型权重、原始 PDF、原始 Agent 状态与轨迹不在这些附件中。

发布副本清理了本机绝对路径。`docbench-source://data/…` 与 `docbench-run://…` 是来源身份标记，不是可直接打开的磁盘路径或下载地址；来源 PDF 哈希、页码、解析指纹和原运行身份用于溯源。原始构建时的代码哈希与原实验指标保留；数据库/清单因路径清理产生的新哈希与原哈希分别记录，迁移不代表重新运行评测。读取语料、图片和复跑 BM25 无需访问原机器上的状态库。

第三方文档与题目的来源沿用 [DocBench 上游清单](../docbench/upstream.manifest.json)，不将这些第三方内容的权利归属改写为本项目代码许可。

## 完整 hybrid 评测

`retrieval_hybrid.py` 将冻结语料接入生产 `RetrievalService.retrieve_file_query_batch`，复用生产 SQLite 方法存储、BGE 编码、RRF、重排、上下文装包和相关性指标实现。每题只发送一次未经改写的原问题，没有查询扩展、Agent loop、答案生成或裁判调用。答案、evidence、qrels 和审阅理由不进入模型输入或索引正文。

本次索引附件为 `repaired75-bge-m3-mps-fp32`，配方与分片清单保存在 `indexes/repaired75-bge-m3-mps-fp32/`，完整结果为 [`results/retrieval-repaired75-hybrid-20260923`](results/retrieval-repaired75-hybrid-20260923/REPORT.md)。成功轮次建库用时 59.2 分钟，查询 18.1 分钟；此前中断构建的耗时及原因单列在分析中。构建和评测先写临时目录，完整校验成功后才发布最终目录。

公开索引附件采用无损分片归档。先下载并解包，再将分片还原到另一个仓库外新目录，最后复用完整索引并写入新的结果目录：

```bash
python3 -m evals.docbench_hybrid_retrieval_optimize.release_assets fetch \
  --asset repaired75-bge-m3-mps-fp32 \
  --output /absolute/path/to/repaired75-bge-m3-mps-fp32
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize.index_archive restore \
  --archive /absolute/path/to/repaired75-bge-m3-mps-fp32 \
  --output /absolute/path/to/restored-docbench-index
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize evaluate-retrieval \
  --backend hybrid \
  --dataset /absolute/path/to/repaired75-20260923/dataset.sqlite \
  --index /absolute/path/to/restored-docbench-index \
  --output evals/docbench_hybrid_retrieval_optimize/results/CHOOSE_A_NEW_HYBRID_RUN \
  --device mps --batch-size 8 --allow-live
```

已有索引必须与数据、模型、生产代码和批次配方完全一致才会复用；不完整或不匹配时要求另选新的索引目录。换用 `cpu` 或 `cuda:0` 时也应指定新的索引目录。查询阶段使用索引的临时副本，已发布索引保持不变。`--batch-size` 控制索引编码输入批次；生产重排器保留默认的 8 对输入批次。

新建索引时可通过 `--encoding-cache /absolute/path/to/verified-cache` 复用已经校验的表示。查找键始终为目标正文 SHA-256，完整编码器指纹必须一致；当前 tokenizer 重新生成 BM25 token IDs，所有单元仍交给生产索引写入器。历史来源必须具备同一 generation 的模型身份及完整物理表示；中断快照只提供已提交、三路完整的单元，不能当作完整索引。

缓存构建默认仅接受原文哈希相等。显式启用 `allow_strip_token_equivalence` 时，还可尝试仅删除目标原文首尾空白：只有变换后的正文哈希命中历史表示，且固定 BGE tokenizer 的完整模型输入逐字段相等、没有依赖截断掩盖差异时才复用。原文、检索引用和 reranker 输入不变，不使用模糊文本匹配。缓存清单记录 tokenizer 身份、验证摘要、接受/拒绝数量、来源哈希和冲突策略；不承诺不同历史 batch 或设备运算的浮点结果位级一致。索引与评测报告分别记录复用、新编码数量及缓存清单哈希。最终索引复用不依赖私有缓存或原 Agent 数据库，也不复用历史问题排名。

索引覆盖整个快照的 **23,048 个单元**，保留原 unit 身份。查询时先限制到题目所属文档：50 道文本题只检索 `chunk`；11 道表格题与 14 道图表题都检索 `table`、`figure`、`vector_graphics` 的文字描述，并返回图片引用。过滤发生在各路 top-k 之前。本批 75 题允许范围的并集是 13,725 个单元，110 条 primary 引用全部在范围内；其余单元仍参与全库 BM25 统计。图片不直接送入编码器或重排器。

| 阶段 | 生产配方与输出 |
| --- | --- |
| `dense` | BGE-M3 1,024 维向量，每题最多 128 个候选 |
| `learned_sparse` | BGE-M3 learned sparse，每题最多 128 个候选；query 取最高 256 个正权重，passage 保留全部正权重 |
| `bm25` | SQLite FTS5，以 BGE token-ID shadow terms 建索引；query 前 256 tokens 去重后用 OR 匹配，每题最多 128 个候选 |
| `rrf` | 生产 RRF，`k=60`，每题最多保留 64 个融合候选 |
| `reranker` | BGE reranker v2-m3 对融合候选重排；query 上限 256 tokens，每个 query/passage 对含特殊标记共限 1,024 tokens |
| `packed` | 按生产规则装包，最多 96 个单元、96,000 estimated tokens；本轮每题只有一个 query，因此实际数量还受前面的 64 个融合候选上限约束 |

模型默认身份为 `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181` 和 `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，仅使用已经准备好的本地资产。运行设置离线环境变量，加载采用 `local_files_only=True`；实际资产、设备和配方身份写入报告。完整三路索引、重排或 token 估算不可用时失败，不把降级结果标成完整 hybrid。

MPS 运行在实际发生新编码的索引批次和每题查询结束后复用生产设备清理函数，释放未使用的缓存；全命中表示缓存的批次无需执行 GPU 清理。清理失败会中止。进度和 `execution.device_memory` 记录清理前后的 active/driver bytes、次数和边界观测峰值，这些不是 forward 内部的瞬时峰值。该策略不改变模型、FP32 精度、候选范围或检索排序配方。首次全量编码耗时明显高于索引复用后的查询，报告分别记录建库与查询耗时。

BGE 编码器 passage 上限为 8,192 tokens，query 配置上限为 512；生产 query guard 先执行更严格的 256-token 校验。运行前逐条核查候选长度并确认原问题不会被 guard 改写，超过 passage 上限时停止。重排的 1,024 上限独立于 embedding 长度；`diagnostics.jsonl` 记录每题实际评分对数和截断前超过上限的对数，报告汇总最大长度。

`rankings.jsonl` 保存六阶段的逐题候选顺序，`diagnostics.jsonl` 保存路径状态、重排长度和装包遗漏；`report.json` 与 `REPORT.md` 保存各阶段及分题型的 Hit/Recall/MRR/nDCG（cutoffs 为 1、3、5、10）。报告同时锁定数据、索引、模型、代码和参数身份。本批是文档内检索评测，平铺 qrels 衡量证据单元覆盖，不是 DocBench 官方 QA 分数；生产 BM25 的分词与旧 unicode61 基线不同，须分别报告。

## 从历史运行建立召回集 / Retrieval dataset

此入口复用已完成运行中的解析快照，不重跑 L1、不调用裁判或视觉模型。当前选择
`text-only`、`multimodal-t`、`multimodal-f`；`balanced_125_v1.json` 对应 75 题。
每份文档限一条题目。依次检查 `--source-run`，选择第一个源 PDF 哈希匹配且有 chunks 的版本，
把实际来源及 parser/chunker 指纹写入清单；历史数据库（含 WAL）不修改。
视觉候选默认覆盖所有所选文档；可重复指定 `--visual-doc-id` 冻结显式文档范围，
范围独立于题型。`--curation` 可在新快照发布前导入经过原文核验的转录和参考答案纠错。

```bash
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize build-retrieval \
  --selection evals/docbench/selections/balanced_125_v1.json \
  --data-root /absolute/private/docbench/source/data \
  --source-run /absolute/private/docbench/runs/preferred-completed-run \
  --source-run /absolute/private/docbench/runs/earlier-completed-run \
  --output /absolute/path/to/reuse-release
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize annotate-retrieval \
  --dataset /absolute/path/to/reuse-release/dataset.sqlite \
  --annotations /absolute/private/reviewed-labels.json
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize evaluate-retrieval \
  --dataset /absolute/path/to/reuse-release/dataset.sqlite \
  --output evals/docbench_hybrid_retrieval_optimize/results/retrieval-baseline-run
```

数据集 authority 是 `dataset.sqlite`，存储文档来源、文本/图表候选、原题与证据引用。
`assets/` 为原 PDF 检测区域截图；`INDEX.html` 为逐题入口。`review/` 显示已导入标签和
明确区分的自动证据定位建议，自动建议不能当作已确认标签；标注导入后刷新浏览投影。
`manifest.json` 锁定来源、数据库和标注批次哈希。构建入口仍支持本包的产物目录与仓库外路径；建议将新语料和索引写到仓库外。公开大型产物时使用经过审阅、哈希锁定的 Release 附件，不重新加入 Git。

视觉候选包含 `table`、`figure`、`vector_graphics`，保留原始检测类型。
描述使用最终截图区域中的 PDF 原始文字层；没有可提取文字时只用单元自身原文，
不按元素序号拼接页尾文本。相邻表格条带扩展为包含上下文的区域，大范围矢量单元可能
使用整页，实际粒度在 locator 中明确记录。没有新 VLM 描述。
历史 `picture_observations` 不进入候选语料，避免把旧问题和答案泄漏到索引。
候选来自范围内整份文档的视觉 inventory，包含干扰项和检测碎片；不是按原题挑选的正确图片合集。
旧解析器版本、检测区域碎片、裁剪边界和描述缺失均属于这个快照的已知限制。

`exports/corpus.jsonl` 提供稳定 `id`、`text`、文档身份、模态、来源版本、内容哈希、页码和图片引用；
可直接用 `text` 构建 embedding 索引。`queries.jsonl` 单独保存问题及文档/模态范围，
`qrels.jsonl` 单独保存已审 primary 标签。答案、evidence 和审阅理由不出现在索引语料中。
查询时先应用文档/模态过滤再取 top-k；未标为相关的候选不等于已经确认不相关。
导出清单记录数据库和三个派生文件的哈希。构建和标注 CLI 自动刷新这些文件；
这一步不生成 embedding，也不运行 GPU 或外部模型。

curation 使用 `docbench-retrieval-curation-v1`：`units` 按 `unit_id` 指定完整原文转录，
要求来源 PDF 哈希、页码、reviewer、note；新增文本单元须用 `derived_from_unit_id` 绑定
同文档已有视觉单元。`corrections` 按 case_id 记录 answer/evidence 的 old_value、new_value、
source_unit_id、reviewer 和 note。原值必须匹配，问题文本不改写；原值和新值都保存于
`curation_events`，原始 QA 文件保持不变。不能只把针对问题生成的答案写成候选描述。

标注 JSON 是列表，每项包含 `case_id`、`primary_unit_ids`、可选 `support_unit_ids`、
`reviewer` 和 `note`。primary 必须是同文档且适合该题检索模态的候选；support 允许同文档其他模态。
`status: "pending"` 可记录尚未解决的证据缺口，不赋予 gold；默认 reviewed 必须有 primary。
原参考答案与 PDF 冲突时通过上述显式 curation 纠错、保留原值；不能按错误参考答案强行赋标签。
明确记录审阅者是模型还是人，不把模型辅助审阅称为人工 gold。

基线仅索引 `units.content`，只用原问题作为 query。文本题检索 chunk；多模态题检索视觉单元描述，
返回截图引用。问题、答案、evidence、审阅理由不进索引。指标复用
`src/personagraph/retrieval/relevance.py` 的 Hit/Recall/MRR/nDCG，分列标注覆盖率与题型结果。
未审题仍输出排名，但不计入质量指标分母；空召回不能补成命中。
qrels 是平铺的候选集合：Hit 表示至少命中一个 primary，Recall 衡量 primary 候选覆盖，
不能表达多段证据必须联合命中或替代证据组，因此不等同于答案完整性。
BM25 的 IDF 和长度统计来自整个快照，按文档和模态限制候选时不重新计算统计。
这是离线词法基线，不能称为生产 BGE-M3/RRF/reranker 成绩，也不能称为 DocBench 官方 QA 分数。
`retrieval_eval.score_rankings` 可对其他路径输出的同一快照引用计分。
