# 检索数据集

- [repaired75-20260923](repaired75-20260923/README.md)：现行 75 题可计分快照，23,048 个候选单元、110 条 primary 标注。
- [reused75-20260922](reused75-20260922/README.md)：原始历史快照，66 题可计分、9 题待处理。

SQLite 是每个快照的唯一事实库；JSONL、网页为派生视图。Git 只保留说明与小型元数据，SQLite、导出、图片、审阅页及 token 长度明细通过 [Release 按需下载](../README.md#按需下载)。下载输出目录就是快照根目录，直接包含 `dataset.sqlite` 和 `INDEX.html`。公开文件已清理本机路径；Git 产物哈希见 [publication_manifest.json](../publication_manifest.json)，完整附件与文件哈希见 [release_assets.json](../release_assets.json)。
