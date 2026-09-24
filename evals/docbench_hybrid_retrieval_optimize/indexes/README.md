# 检索派生索引

[`repaired75-bge-m3-mps-fp32/`](repaired75-bge-m3-mps-fp32/manifest.json) 在 Git 内保存现行 75 题索引的配方与分片清单。完整分片位于同名 [Release 附件](../README.md#按需下载)，压缩包约 170.4 MB，不随 clone 下载。它对应完整三路索引：23,048 个单元，Dense、learned sparse、BM25 各覆盖全部单元。模型为固定 revision 的 BGE-M3，设备 MPS、精度 FP32。

数据库原始大小为 308,203,520 bytes。公开归档按原字节拆为 5 个不超过 64 MiB 的文件，`archive.json` 锁定分片和整库 SHA-256；原 `manifest.json` 保持不变。下载入口先校验压缩包和全部文件，得到分片目录；分片不直接作为 SQLite 使用，须再还原到另一个仓库外的新目录：

```bash
python3 -m evals.docbench_hybrid_retrieval_optimize.release_assets fetch \
  --asset repaired75-bge-m3-mps-fp32 \
  --output /absolute/path/to/repaired75-bge-m3-mps-fp32
.venv/bin/python -m evals.docbench_hybrid_retrieval_optimize.index_archive restore \
  --archive /absolute/path/to/repaired75-bge-m3-mps-fp32 \
  --output /absolute/path/to/restored-docbench-index
```

还原目录包含 `retrieval.sqlite` 和 `manifest.json`，将该目录传给评测 CLI 的 `--index`。还原会校验全部分片、完整数据库及 manifest 的哈希，拒绝覆盖已有目录。本归档已实际还原并验证与原库逐字节一致。

复用要求语料、模型、代码和批次配方一致；换设备或实现后不能把旧索引当作新配置的结果。索引可重建，不作为语料正文的第二事实库。模型权重、源 PDF、Agent 状态库和中间编码缓存不放在这里。
