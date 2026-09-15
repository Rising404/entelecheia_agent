# 评测入口

当前正式接入的是 **DocBench 的 L1 产品链路**：真实 Session、Project、附件入库、检索和 Agent 执行。不会把参考答案注入上下文，也不指定工具顺序；L2 不在本评测范围。其他候选见 [Benchmark 清单](BENCHMARK_BACKLOG.md)。

完成[项目安装](../README.md)后，从仓库根目录执行：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
```

复现时请将示例换成自己的**仓库外绝对路径**，在启动 Python 前设置。`--help` 不要求环境变量，`validate` / `list` 要求变量，但不要求目录或数据已存在。

<br>

## 目录归属

| 位置 | 放什么 |
| --- | --- |
| [reproduce_or_run_script/](docbench/reproduce_or_run_script/README.md) | 唯一的准备、选题、运行、补跑、评分实现 |
| [configs/](docbench/configs/README.md)、`selections/`、`config.schema.json` | 实验配置、无题答正文的选题清单、配置契约 |
| [results/](docbench/results/README.md) | 后续新结果的整理区 |
| [previous_results/](docbench/previous_results/README.md) | 历史统计、评审和已清理的逐题执行证据；原件在被忽略的 `private_runs/` 中 |
| [操作手册](docbench/docs/formal_l1_eval_runbook.md)、[状态隔离](docbench/docs/state_isolation.md) | 准备步骤、命令与进程边界 |
| `<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/` | 为复现提供管理的 `source/`、`runs/`、`workspaces/` |

- 结果自动写到外部 `docbench/runs/<run-id>/`，不会自动复制回仓库 `results/`。`output_root: bench://runs` 就指这里。可换外部根目录，但当前 runner 不支持把活跃状态任意写入源码目录。

<br>

## 数据与依赖

docbench的PDF、QA 和固定裁判提示词需先单独准备，默认在 `docbench/source/`。YAML 的 `dataset.data_root`、`scoring.prompt` 也可显式指定数据路径。详见[数据准备](docbench/sources/README.md)和[上游清单](docbench/upstream.manifest.json)。

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --qa-catalog-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --balanced-pdfs-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script readiness
```

前两条会联网，只准备 QA 目录和 balanced125 所需 PDF。`readiness` 离线检查整个配置目录，缺材料会影响整体状态。

依赖使用项目锁定环境（包含 `gdown`）；BGE encoder / reranker 使用项目提供的模型准备入口。

<br>

## 选择实验

- `l1_bge_m3_live_1.yaml`：单题 CPU 工程检查；文档 104 包含在 balanced125 下载范围。
- `l1_storage_smoke_3.yaml`、`l1_stage_20.yaml` 及小样本配置：链路或定向回归稳定性测试。
- `l1_balanced_125.yaml`：原有 125 题配置，语义验证开启，检索设备为 MPS。
- `l1_balanced_125_gate_off.yaml`：同题同设置，仅关闭语义验证。
- `l1_balanced_125_gate_off_highland_235b.yaml`：在关门配置上更改视觉模型为 `qwen3-vl-235b-a22b-instruct`，读取 `HIGHLAND_API_KEY`，不切换 GUI 活跃配置。

该 VLM 已用于存档的同题 123 题实验（原 125 题排除两道 una-web，但本agent暂未提供联网检索能力），结果见[逐题证据](docbench/previous_results/gate_off_highland235b_123/README.md)。字段与示例见[配置说明](docbench/configs/README.md)。

<br>

## 运行与解释 / Run and interpret

按[操作手册](docbench/docs/formal_l1_eval_runbook.md)分别执行 `run` 和 `score`。`run`、`retry-failed`、`score` 都需 `--allow-live`。同名 run 默认拒绝覆盖；续跑要求身份一致，补跑使用新状态并保留旧结果。


开启验证门一轮的历史被测源码标识为 `source_sha256=7a6043f6465a887328363891a3edc963c05cd81d2d78868558f2c17398e38da1`，见[公开统计](docbench/previous_results/showcase_125.summary.json)。当前代码重跑应记录为新实验。
