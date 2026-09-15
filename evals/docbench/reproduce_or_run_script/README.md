# DocBench 执行入口 / Runner entry

这个执行包提供 DocBench 的数据准备、配置检查、运行和评分命令。从仓库根目录执行：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
```

将示例路径换成自己的仓库外绝对路径，在 Python 启动前设置。首次运行可直接按[操作手册](../docs/formal_l1_eval_runbook.md)完成数据、模型、执行和评分准备。

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

`run` **不会自动评分**，生成完成后单独执行 `score`。`--allow-live` 启用真实 Provider 调用，可能发送文档文本或页面图像并产生费用。

## 同题关门实验 / Semantic-gate ablation

`l1_balanced_125_gate_off.yaml` 与 `l1_balanced_125.yaml` 使用相同的 125 题、Provider 来源、检索设置、预算和裁判 prompt，仅切换 features 文件，使 `l1_semantic_verification_mode: "off"`。产品默认模式为 `always`；关门配置仍执行 schema、引用、授权和 closed-world 检查。`readiness` 在 `effective_policy` 中报告模式。

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script run \
  --config evals/docbench/configs/l1_balanced_125_gate_off.yaml \
  --run-id l1-semantic-gate-off-125-EXPERIMENT_ID --allow-live
.venv/bin/python -m evals.docbench.reproduce_or_run_script score \
  --config evals/docbench/configs/l1_balanced_125_gate_off.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/l1-semantic-gate-off-125-EXPERIMENT_ID" \
  --allow-live
```

将 `EXPERIMENT_ID` 换成唯一标识。两组实验需分别运行；比较语义门影响时，源码、模型、环境和其他参数保持相同，首跑与补跑分开统计。不同条件下的历史成绩可作为背景对照。

更大 VLM 可选 `l1_balanced_125_gate_off_highland_235b.yaml`，从环境读取 `HIGHLAND_API_KEY`，只在关门配置上替换视觉端点，不改主模型和裁判来源。见[配置说明](../configs/README.md)。

## 源码入口 / Implementation

| 文件 | 职责 |
| --- | --- |
| `cli.py` | 参数分派、退出状态 |
| `config.py` | 配置校验、路径解析、规范化指纹 |
| `download_selection.py`、`selection.py`、`derived_selection.py` | 数据获取、选题和身份绑定 |
| `readiness.py` | 只读前置检查 |
| `runner.py` | 单题 worker、产品 Turn 入口、生成检查、续跑/补跑 |
| `post_commit.py` | 等待 runtime 收尾并投影报告 |
| `scorer.py` | 裁判协议、检查点、汇总 |
| `provenance.py`、`retrieval_observability.py` | 源码/环境身份、检索观测 |
| `export_results.py` | 从已有归档生成逐题公开投影，写入新目录 |

配置、选题和 schema 分别位于相邻的 `configs/`、`selections/` 和 `config.schema.json`。路径写法也参与配置身份；改动配置后，续跑时的身份校验会拒绝沿用原实验。

## 路径与数据边界 / Paths and isolation

`PERSONAGRAPH_BENCH_EVAL_DIR` 指向仓库外的父目录，其 `docbench/` 保存数据和状态；`bench://source/data`、`bench://source/evaluation_prompt.txt`、`bench://runs` 在这里解析。`--help` 不要求该变量，`validate` / `list` 要求变量但不要求目录存在。完全显式提供所需路径的命令（如 `build-selection --data-root ... --output ...`）可直接使用这些路径。

可自定义外部评测根、数据目录、下载映射和选题输出。活跃 run 的位置限定为 `<外部根>/docbench/runs/<run-id>`。仓库内的 `results/` 和 `previous_results/` 用于保存实验报告，与运行输入和 Project 分开。

项目锁已含 `gdown`；BGE 资产由 `scripts/prepare-local-models.py` 显式下载。`prepare-data` 下载 QA 和 balanced125 PDF，裁判 prompt、其他回归附件和模型需要单独准备。

默认前缀仅要求使用附件回答，不规定工具顺序。参考答案只交给裁判，不挂载给 Agent。closed-world 移除网页工具，工程成功、生成检查、答案正确率分别报告。

## 导出公开证据 / Publication projection

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script.export_results \
  --archive /absolute/path/to/private_archive \
  --output /absolute/path/to/new_public_export
```

输入使用 `runs/<batch>/cases/`、`scoring/` 的阅读归档结构；输出写入新目录，已有目录会被拒绝。
导出过程只读取归档，无需再次调用 Agent 或裁判。公开文件保留回答、已记录调用/错误和指标；
原题字段、原始用户请求与文档文本叶以省略标记或哈希取代，凭据及私有路径清理。
生成的 README 说明具体投影范围；这些文件用于查阅过程和统计，恢复运行仍需要完整私有状态。
