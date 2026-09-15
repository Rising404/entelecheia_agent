# 逐题公开证据 / Public per-case evidence

这是原始档案的公开投影，不是原样副本、模型重跑或重新评分。

- `runs/<batch>/cases/<case>/result.json`：回答、状态、统计及轨迹引用。
- 同题 `trajectory.json`：全部已记录步骤、调用参数、返回及单次指标；正文仍通过 blobs 关联。
- `runs/<batch>/scoring/`：原裁判逐题记录与汇总，裁判 token 与生成阶段分开。
- `run_manifest.json`：原运行身份、模型及配置/源码哈希，不含完整本机配置。

仅省略原题/参考答案字段、原始用户请求、文档 text/content/snippet 和无法判别来源的展开文本叶；
这些位置保留 publication_omitted、原字节数与 SHA-256。模型生成的回答、视觉观察、
文件 ID、工具参数、错误和性能指标保留；私有绝对路径替换为 <local-path>，凭据不公开。
模型上下文不是原样请求，不能按公开文本重算原 token 或直接重放；已有 assistant 摘要不补写。
模型回答/观察可能包含引文，未另行将源 PDF 或 QA 数据集打包。来源身份用于追溯，不授予第三方版权。

`publication.source_sha256` 指向私有原件；result.trajectory.sha256 指向这里的公开轨迹，
parts 的 source_blob_sha256 保留被修改前的正文身份。原始 incomplete、失败和零分全部保留。
路径中的冒号改为连字符便于跨平台 checkout；JSON 内的 case_id 不变。

## 批次 / Runs

- `l1-gpu-deadline25-125-20260910-a`：125 次执行，原裁判 91/125。
- `l1-gpu-transport-retry-3-20260911-a`：3 次执行，原裁判 2/3。
- `l1-gpu-execution-retry-5-20260911-a`：5 次执行，原裁判 3/5。

DocBench prompt-compatible、非官方可比成绩；补跑不覆盖首跑，不能直接相加作为独立题目准确率。
