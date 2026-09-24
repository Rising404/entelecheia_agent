# 检索评测结果

- [完整 75 题生产 hybrid](retrieval-repaired75-hybrid-20260923/REPORT.md) · [中文结果、失败与耗时分析](analysis/retrieval-repaired75-hybrid-20260923.md)
- [修复后 75 题 BM25](retrieval-repaired75-bm25-20260923/REPORT.md)
- [9 题 BGE CPU 接入检查](retrieval-repaired75-dense-smoke-20260923/README.md)
- [修复前 66 题 BM25](retrieval-reused75-bm25-20260922/REPORT.md)

这里归档检索结果，与原 DocBench Agent 端到端 QA 成绩分开。完整 hybrid 是本仓库实际新运行的结果；其余三个历史结果未因迁移重新计算，公开副本仅清理机器路径并更新关联哈希。历史候选库与标签不同，分数不能直接解释为同条件算法提升。每次新评测另建目录，原始结果清单冻结后不改写，补充解释放在 `analysis/`。

全部结果（约 4.8 MB）继续保存在 Git，可直接阅读。完整语料和索引通过 [Release 按需下载](../README.md#按需下载)。报告中的数据、索引、源码和原完整 publication 身份保留为当时的冻结记录；当前 Git publication 清单只覆盖仓库保留产物，附件目录另行锁定完整下载包。此次分发调整没有重新计分或重跑模型。
