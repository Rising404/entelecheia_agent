# DocBench 执行入口 / Runner entry

这是仓库唯一的 DocBench 执行包，不依赖作者桌面的脚本。从仓库根目录执行：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
```

示例换成仓库外绝对路径，在 Python 启动前设置。参见[安装](../../../README.md)、[评测总览](../../README.md)、[操作手册](../docs/formal_l1_eval_runbook.md)、[状态隔离](../docs/state_isolation.md)。

## 命令做什么 / Commands

| 命令 | 职责 |
| --- | --- |
| `prepare-data` | 显式联网下载 QA 或 balanced125 PDF，需对应选择参数 |
| `build-selection --balanced` | 确定性生成选题清单并严格重载，写入新的指定文件 |
| `validate` | 检查仓库 L1 YAML 与 schema，不调用模型 |
| `list` | 列出配置 ID 和规范化哈希，不调用模型 |
| `readiness` | 离线只读检查配置、文件哈希、凭据是否存在、本地模型资产 |
| `run` | 创建隔离子进程，执行真实 L1，需 `--allow-live` |
| `retry-failed` | 对指定失败题补跑，保留旧尝试，需 `--allow-live` |
| `score` | 使用固定 prompt 单独调用裁判，需 `--allow-live` |

`run` **不会自动评分**。`--allow-live` 表示同意必要外发和潜在费用；文档修改无需付费调用验收。

## 同题关门实验 / Semantic-gate ablation

`l1_balanced_125_gate_off.yaml` 与 `l1_balanced_125.yaml` 使用相同的 125 题、Provider 来源、检索设置、预算和裁判 prompt，仅切换 features 文件，使 `l1_semantic_verification_mode: "off"`。语义 reviewer 源码和产品默认 `always` 保留，schema、引用、授权和 closed-world 检查仍开着。`readiness` 在 `effective_policy` 中报告模式。

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script run \
  --config evals/docbench/configs/l1_balanced_125_gate_off.yaml \
  --run-id l1-semantic-gate-off-125-EXPERIMENT_ID --allow-live
.venv/bin/python -m evals.docbench.reproduce_or_run_script score \
  --config evals/docbench/configs/l1_balanced_125_gate_off.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/l1-semantic-gate-off-125-EXPERIMENT_ID" \
  --allow-live
```

`EXPERIMENT_ID` 换成唯一标识。首跑对首跑，补跑另列；若源码、模型等不同，同题历史成绩只能作背景，不能称严格单变量消融。上述命令不会自动启动另一组对照。

更大 VLM 可选 `l1_balanced_125_gate_off_highland_235b.yaml`，从环境读取 `HIGHLAND_API_KEY`，只在关门配置上替换视觉端点，不改主模型和裁判来源。见[配置说明](../configs/README.md)。

## 代码归属 / Canonical owners

| 文件 | 职责 |
| --- | --- |
| `cli.py` | 参数分派、退出状态，不另建 runner |
| `config.py` | 配置校验、路径解析、规范化指纹 |
| `download_selection.py`、`selection.py`、`derived_selection.py` | 数据获取、选题和身份绑定 |
| `readiness.py` | 只读前置检查 |
| `runner.py` | 单题 worker、产品 Turn 入口、生成检查、续跑/补跑 |
| `post_commit.py` | 等待 runtime 收尾并投影报告 |
| `scorer.py` | 裁判协议、检查点、汇总 |
| `provenance.py`、`retrieval_observability.py` | 源码/环境身份、检索观测 |
| `export_results.py` | 只读私有归档，导出逐题公开投影；不运行 Agent、不评分、不覆盖既有输出 |

`configs/`、`selections/`、`config.schema.json` 与实现包分开。路径写法参与配置身份，不能修改冻结配置后仍声称是原实验。

## 路径与数据边界 / Paths and isolation

`PERSONAGRAPH_BENCH_EVAL_DIR` 指向外部父目录，其 `docbench/` 保存私有数据和状态；`bench://source/data`、`bench://source/evaluation_prompt.txt`、`bench://runs` 在这里解析。没有默认桌面位置；`--help` 不要求变量，`validate` / `list` 要求变量但不要求目录存在。完全显式提供所需路径的命令（如 `build-selection --data-root ... --output ...`）不要求一个实际不用的外部根。

可自定义外部评测根、数据目录、下载映射和选题输出。但活跃 run 必须是 `<外部根>/docbench/runs/<run-id>`，并非任意 `output_root` 都能运行。源码 `results/` 供事后整理，旧资料在 `previous_results/`；runner 不读取这些档案，也不能将它们绑定为 Project。

项目锁已含 `gdown`；BGE 资产用 `scripts/prepare-local-models.py` 准备，不静默下载。`prepare-data` 不准备裁判 prompt、上游 checkout 或所有回归 PDF，须按清单补齐。

默认前缀仅要求使用附件回答，不规定工具顺序。参考答案只交给裁判，不挂载给 Agent。closed-world 移除网页工具，工程成功、生成检查、答案正确率分别报告。

## 导出公开证据 / Publication projection

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script.export_results \
  --archive /absolute/path/to/private_archive \
  --output /absolute/path/to/new_public_export
```

输入使用 `runs/<batch>/cases/`、`scoring/` 的阅读归档结构；输出必须是新目录。保留回答、
已记录调用/错误和指标；原题字段、原始用户请求与文档文本叶以省略标记/哈希取代，凭据及私有路径清理。
这是公开投影而非原样请求或可重放状态，具体范围见生成的 README。导出成功不等于自动批准提交：
新增文件仍须检查正文、核对身份与统计，再更新精确公开清单；不要放宽整个私有目录或跳过隐私门禁。
