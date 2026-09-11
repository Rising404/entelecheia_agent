# Benchmark 候选与历史清单

当前 `evals/` 只把 DocBench 作为已接入 benchmark。下列项目暂时只保留名称与用途摘要，不保留空 adapter、suite 或 results 目录；需要恢复时，应先确认数据许可、运行成本和可执行适配器。

## 候选 benchmark

| 名称 | 概要 |
| --- | --- |
| `pg.document_core.v1` | 原内部文档链路回归集，覆盖 PDF、图片、DOCX、PPTX、表格及迭代式文档任务。 |
| `pg.taskgraph_revision.v1` | 原内部 TaskGraph 回归集，覆盖规划、修订、语义校验、N→N+1 与交付重放。 |
| `pg.product_entry.v1` | 原内部产品入口回归集，覆盖上传或挂载、L1/L2 结算、流式中断与重放。 |
| MMLongBench-Doc | 长 PDF、多页证据与文本/视觉联合推理候选集。arXiv:2407.01523，NeurIPS 2024 D&B Track；1,062 题 / 130 篇 / 均 49.4 页，33.2% 跨页、22.8% 不可答。无 "V2" 版本。 |
| DocVQA 2026 | 复杂版面与视觉文档问答候选集；接入前需复核数据条款。 |
| OfficeComprehensionBench | 原生 DOCX/PPTX 理解候选集；代码与数据许可需分别核实。 |
| OmniDocBench | 文档解析、版面、表格和公式能力的诊断集，不应直接作为 Agent 总分。 |
| Workspace-Bench Lite | 模糊文件发现、多文件推理与工具选择候选集。 |
| BFCL V4 | 函数选择、多轮工具调用、成本与延迟评测候选集。 |
| GAIA | 开放域助理、网页检索和多步工具使用候选集；数据访问受限。 |
| PaperBench | 长周期论文复现与产物质量评测候选集，运行成本较高。 |
| `local_longdoc_qa` | 曾用于本地长文档真实链路探测，不是正式公开 benchmark。 |

## 历史组合门

| 名称 | 概要 |
| --- | --- |
| `phase0_core_v1` | 曾组合 document core、TaskGraph revision 与 product entry 三套内部回归；它是发布门，不是独立 benchmark。 |

普通代码回归测试仍归 `tests/` 管理；本清单不会自动注册、下载或运行任何 benchmark。
