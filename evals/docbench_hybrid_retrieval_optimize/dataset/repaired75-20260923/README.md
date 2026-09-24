# DocBench 75 题：修复后的召回语料

当前快照：`repaired75-20260923`。75/75 题已标注可计分：50 道文本、11 道表格、14 道图表；共有 110 条 primary 相关性引用。旧快照与上游原始 PDF/QA 保持不变。

[下载后逐题浏览](../../README.md#按需下载) · [答案与证据纠错](CORRECTIONS.md) · [BM25 基线](../../results/retrieval-repaired75-bm25-20260923/REPORT.md) · [真实 BGE CPU 检查](../../results/retrieval-repaired75-dense-smoke-20260923/README.md)

## 可直接接入语义索引的文件

完整内容在 Release 的 `repaired75-20260923.tar.gz`，约 776.5 MB；[下载说明](../../README.md#按需下载)提供校验与解包命令。Git 保留本页及小型清单。以下路径均相对于**下载后的快照根目录**；逐题入口是其中的 `INDEX.html`。

- `exports/corpus.jsonl`：23,048 个单元。仅将 `text` 送入 embedding；`id`、`doc_id`、`kind`、页码、来源版本及内容哈希用于定位和过滤。`asset_path` 相对于本数据集根目录。
- `exports/queries.jsonl`：75 个原始问题，包含 `doc_id` 和 `allowed_kinds`。检索时先应用这两个范围过滤再取 top-k。
- `exports/qrels.jsonl`：110 条已审 primary 标签，二值 score=1。相关性标签与答案不能写进候选正文。未标候选不等于已确认不相关。
- `exports/manifest.json`：导出行数、文件哈希和母数据库哈希。数据库是唯一事实库，JSONL 与网页均为派生视图。

语料包括 16,459 个 chunk、2,105 个 figure、1,601 个 table、2,883 个 vector_graphics。视觉范围显式覆盖 25 份多模态文档和文本题出席表所在文档；其余文本题复用原有 chunks。

## 本次修复

1. 恢复被旧导出误排除的矢量图形；零宽/高的原生线段按合法几何保留，非法反向和非有限坐标仍拒绝。
2. 图表描述改为最终截图区域内的 PDF 原始文字；表格条带合并上下文，保留表头。相同类型、区域和文本的候选精确去重，源别名留在 metadata。区域/整页粒度在 locator 中明确记录。
3. 对原来 9 道待处理题，依据原 PDF 核对并转录完整图表/表格，包括所有关键类别、表头、单位、时期和数值。出席表新增为可检索的文本 chunk。
4. 修订 6 题共 10 处 answer/evidence：85 的 44→4；53 区分 Finance income 与 Total income；80 区分产品类别与具名品牌；167 补足原文中的有条件允许；36 补齐 BLEU 20.96；67 修正证据中的业务部门名称。问题文本未改写。

## 验证

全量 tokenizer 使用本地 BAAI/bge-m3 的固定 revision。23,048 个单元最长 2,981 tokens，P95 为 671，超过 8,192 的单元为 0。精确计数与各长度阈值见下载快照内 `TOKEN_PROFILE.json`。不同模型/自定义较短 max_length 仍须另行检查截断。

| 类别 | 可计分 | Hit@5 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| 全部 | 75 | 94.7% | 90.0% | 93.8% |
| 文本 | 50 | 98.0% | 93.0% | 95.7% |
| 表格 | 11 | 90.9% | 86.4% | 86.4% |
| 图表 | 14 | 85.7% | 82.1% | 92.9% |

上述为文档内 SQLite FTS5 BM25 全量基线。候选库、标注和计分题数都发生变化，不能把它与旧版 66 题成绩直接解释成同条件检索算法提升。

另对 9 道修复题做了真实本地 BGE-M3 CPU 检查：45 个候选（每题 1 个已标相关单元及 4 个确定性候选）、9 个问题；1,024 维 dense 向量，max_length=1,446，0 截断。编码器初始化、54 条编码与产物准备计时 41.80 秒；9 题的目标证据均位于各自 5 个候选的第 1 位。这个候选池含人工确保纳入的相关证据，因此只能验证语义接入，不是全库 Recall 或生产混合检索成绩。

CPU 检查使用仓库 canonical BgeM3Encoder，离线读取已有权重，无 GPU、网络或新付费模型调用；编码器同时产生但未使用 sparse 输出，最终只保存并按 dense cosine 排序。向量和复跑脚本保存在独立报告目录。

## 边界与审阅记录

- 标注、完整转录来自模型辅助原文核对，不是独立人工 gold，也不保证穷尽所有等价证据。
- 图表候选仍包含装饰、线段或较大页面区域；可计分不意味着每个候选都是完整独立图表。
- 学术题 7、32、42 的图像证据可用，但图内部分文字不在 PDF 文字层中，描述仍主要依靠图注；此限制写入逐题 notes。
- 51 的截图顶部总人数局部裁切，但各地区数据完整；53 截图保留完整财务表，转录中的解释/财年/币种另依据同份原 PDF 原文，审阅理由记录了补充来源。
- 参考纠错保留在 curation_events。原文内部的轻微舍入/合计差异照录并注明，不擅自改动原始数值。
- flat qrels 的 Hit 表示至少命中一个 primary，Recall 表示 primary 单元覆盖。等价替代与联合必需证据未分组，不能将其直接视为答案完整性。
- 本批是开发用途的召回集，未划分独立保留测试集。此公开副本包含题目、参考答案、候选正文、截图和标签；完整源 PDF 与历史运行数据库不随包分发。

## 重建与复跑

输入和代码哈希见下载后快照中的 `inputs/build_recipe.json`（[下载说明](../../README.md#按需下载)）。构建时传 `--curation inputs/curation.json`（使用本快照内文件的绝对路径）及清单中的全部 `--visual-doc-id`；在新目录创建快照后，再用 `annotate-retrieval` 导入 `inputs/reviewed-labels.json`。不要对当前已审快照重复应用标注。

复跑 BM25（输出需为不存在的新目录）：

```bash
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize evaluate-retrieval \
  --dataset /absolute/path/to/repaired75-20260923/dataset.sqlite \
  --output evals/docbench_hybrid_retrieval_optimize/results/CHOOSE_A_NEW_REPORT_DIRECTORY
```

原快照构建时的验证：429 项相关 pytest 通过；ruff、diff 格式检查、仓库隐私门禁通过。这些是当时的记录，不表示分发方式调整后重新执行了全部验证或模型评测。

## 公开迁移说明

这是原实验的路径元数据脱敏投影，没有重新运行评测。候选正文、问题、答案、evidence、标签及截图保持不变；数据库只调整来源定位并清理空闲页，因此数据库与 manifest 哈希已更新。原哈希见 manifest 的 `origin_*` 与 [MIGRATION.json](MIGRATION.json)。`docbench-source://data/...` 和 `docbench-run://<run-id>/...` 是逻辑来源标识，需要本地取得上游资料后自行映射，不能作为现成本机路径打开。已有数据可直接做检索评测，无需重新获取原 PDF。

分发更新：大型语料改为 Release 附件，数据与指标未重算。附件保存冻结快照原文件（包括原 README）；本 Git 页面维护当前下载方式，两者的文件哈希分别由附件目录与 Git publication 清单锁定。
