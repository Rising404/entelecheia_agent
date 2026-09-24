# DocBench 75 题召回评测集

已落库 75 题：50 道文本、11 道表格、14 道图表。当前 66 题有可计分标签，9 题保留待处理状态。

按[下载说明](../../README.md#按需下载)获取 `reused75-20260922.tar.gz`（约 182.7 MB），打开下载快照内 `INDEX.html` 查看原题、来源证据、截图、primary 标签及审阅理由。Git 仅保留本页和小型元数据，不默认下载完整语料。 [基线报告](../../results/retrieval-reused75-bm25-20260922/REPORT.md) 包含分题型结果。

## 数据与标注

- `dataset.sqlite` 是唯一事实库：documents 保存来源，units 保存候选，queries 保存题目与状态，qrels 保存标签。
- `manifest.json` 记录来源优先级、处理器/切块指纹及数据库、来源、标注批次哈希。
- `annotations/reviewed75.json` 是已经导入的标注批次，供审计；不要再次应用同一批次。
- `assets/` 是 4,971 个原 PDF 检测区域截图，包含表格/图表碎片和装饰，不代表 4,971 张完整图表。
- `review/` 和 `INDEX.html` 是派生浏览视图；自动候选建议与已核对的 primary 标签有明确区别。
- 共复用 16,458 个文本 chunk；25 份多模态文档提供图表候选，候选不按题目或答案筛选。
- 多模态描述采用 source_context：原表格文字、标题及同页邻近文本；没有新增 VLM 描述，也不采用历史上受问题条件影响的 picture observations。
- 本批标注由 Codex 辅助原文/截图核对，并非独立人工 gold。标签不保证穷尽所有等价证据。

## 首轮离线结果

文档内 SQLite FTS5 BM25，仅以原问题检索候选正文。答案、evidence、审阅理由不进入索引。该结果不代表生产 BGE-M3/RRF/重排器，也不代表 DocBench 官方问答分数。

| 题型 | 可计分 / 全部 | Hit@5 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| 整体 | 66 / 75 | 89.4% | 85.6% | 92.5% |
| 文本 | 49 / 50 | 98.0% | 92.9% | 95.6% |
| 表格 | 8 / 11 | 62.5% | 62.5% | 77.5% |
| 图表 | 9 / 14 | 66.7% | 66.7% | 88.9% |

未计分的 9 题不当作成功或失败，覆盖率 88% 必须与条件指标一起阅读。历史解析缺口会影响覆盖率，不能用这 66 题的分数推断全部 75 题的端到端能力。

Hit 表示至少命中一个已标 primary；Recall 是已标 primary 单元覆盖比例。当前是平铺 qrels，不表达必须联合命中的证据组与互为替代的证据组，不能直接当作答案完整性。BM25 的 IDF/长度统计取自整个语料快照，候选按题目文档与模态过滤。

## 待处理的 9 题

以下为数据库中保存的审阅理由，原始参考答案保持不改写：

### docbench:51:3

Verified original PDF physical page 97 (printed 93): employees-by-region donut lists Europe 43,181, North America 28,586, Asia 22,674, etc. There is no table/figure visual unit on this page. Nearby candidates on pages 95/96 are unrelated photographs. Original evidence exists, but the frozen visual candidate corpus lacks it.

### docbench:53:3

Verified original PDF page 66 Financial results table: 2021/2020 columns; Total income is 48/42. Frozen candidates are three individual total-row strips without the year header. No current visual candidate retains the necessary column-year mapping, so do not promote the overlap-matching Total income strip to gold. Also note wording ambiguity: Finance income alone is 4; original answer/evidence intend Total income 48.

### docbench:54:4

Verified original PDF physical page 52 (printed 48), Major facilities table: Basel, Switzerland - St. Johann is 589,000 square meters, largest listed site. No table/figure visual unit exists on physical page 52 in the frozen dataset. Query has real evidence but candidate extraction omitted the complete table.

### docbench:55:2

Original evidence points to physical PDF page 34 consolidated operating-profit waterfall. That chart is absent from the visual candidates; page 34 exports only unrelated narrow table strips. Also inspected 12 alternative candidate crops containing 4,553 on pages 51/52/59: none preserves both Operating profit and the 2020 year/column header (page 51 preserves the year but omits metric headings). Keep pending until a sufficient complete chart/table candidate exists.

### docbench:61:6

Verified original PDF physical page 6 (printed 4) left FY2020 Revenue donut: Consumer $33.3 billion is highest; Global Banking $19.0B, Global Markets $18.8B, GWIM $18.6B. No table/figure visual candidates exist on physical pages 6 or 7, so the target donut is missing from this frozen candidate corpus.

### docbench:66:3

Verified original PDF physical page 66 (printed 64), Gender diversity statistics: Holdings Board has 9 male (64%) and 5 female (36%). Its candidate ntu_3f2a0309ff2b8fc275c195fc contains bar fragments but crops away category labels, percentage labels and gender legend. Other page 23 crops likewise omit crucial labels. Cannot mark these incomplete fragments as sufficient gold.

### docbench:80:5

Verified original PDF physical page 38 (printed 28) PMI Shipment Volume by Brand table: Heated Tobacco Units increased 27.6% (2020 76,111 vs 2019 59,652), above Next 4.4% and other negative changes. Seven candidate crops are single-row strips and omit the 2020/2019/Change column headers and complete comparison context; no sufficient complete-table visual unit is present.

### docbench:85:0

需挂起：已视觉核验原PDF ASX_AJY_2020.pdf物理第6页，Dominic行两个不同列分别为held=4与attended=4，答案应为4，参考44是列值粘连。复用语料该doc仅25个chunk，无table/figure unit；当前块只有董事简介、表前引导、表后文字，未保留数字表。不能把相关简介伪标为回答证据。需要新增准确表格unit并记录参考纠正后再审阅；本次未改原题答案或DB。

### docbench:88:9

Verified original PDF physical page 53 (printed 52) Ownership Breakdown donut: financial institutions/brokerages 1,079,803 thousand shares (38.98%) is largest, above other corporations 25.18%, foreign corporate entities 23.88%, individuals 11.96%. Only a Major Shareholders table-header strip and navigation-header strip are exported on this page; the ownership donut is absent.

## 复跑与后续

从源码仓库根目录运行（输出需为尚不存在的新目录）：

```bash
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize evaluate-retrieval \
  --dataset /absolute/path/to/reused75-20260922/dataset.sqlite \
  --output evals/docbench_hybrid_retrieval_optimize/results/CHOOSE_A_NEW_RUN_NAME
```

本批是快速建立评测能力的开发集，未划分独立保留测试集。改进检索后应在同一冻结快照上对比；修复解析缺口或修改已审标签时创建明确的新快照并重新计分，不改写这次报告。无需重跑 L1 即可使用已有 score_rankings 对同一快照的其他检索排序计分。

此公开副本包含题目、参考答案、语料、截图与标注；完整源 PDF 和历史运行数据库保留为上游来源，不随包分发。此次未执行新的付费模型、GPU 推理或完整 L1 运行。

## 公开迁移说明

这是原实验的路径元数据脱敏投影，没有重新运行评测。候选正文、问题、答案、evidence、标签及截图保持不变；数据库只调整来源定位并清理空闲页，因此数据库与 manifest 哈希已更新。原哈希见 manifest 的 `origin_*` 与 [MIGRATION.json](MIGRATION.json)。`docbench-source://data/...` 和 `docbench-run://<run-id>/...` 是逻辑来源标识，需要本地取得上游资料后自行映射，不能作为现成本机路径打开。已有数据可直接做检索评测，无需重新获取原 PDF。

分发更新：大型语料改为 Release 附件，数据与指标未重算。附件保存冻结快照原文件（包括原 README）；本 Git 页面维护当前下载方式，两者的文件哈希分别由附件目录与 Git publication 清单锁定。
