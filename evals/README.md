# 评测入口 / Evaluations

当前正式接入的是 **DocBench 的 L1 产品链路**：真实 Session、Project、附件入库、检索和 Agent 执行。不会把参考答案注入上下文，也不指定工具顺序；L2 不在本评测范围。其他候选见 [Benchmark 清单](BENCHMARK_BACKLOG.md)。

完成[项目安装](../README.md)后，从仓库根目录执行：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
```

将示例换成自己的**仓库外绝对路径**，在启动 Python 前设置。无需桌面包装脚本，入口是仓库模块。没有默认桌面位置；`--help` 不要求环境变量，`validate` / `list` 要求变量，但不要求目录或数据已存在。

## 目录归属 / Directory ownership

| 位置 | 放什么 |
| --- | --- |
| [reproduce_or_run_script/](docbench/reproduce_or_run_script/README.md) | 唯一的准备、选题、运行、补跑、评分实现 |
| [configs/](docbench/configs/README.md)、`selections/`、`config.schema.json` | 实验配置、无题答正文的选题清单、配置契约 |
| [results/](docbench/results/README.md) | 后续新结果的整理区，不是 runner 默认写入目录 |
| [previous_results/](docbench/previous_results/README.md) | 历史统计、评审和已清理的逐题执行证据；原件在被忽略的 `private_runs/` 中 |
| [操作手册](docbench/docs/formal_l1_eval_runbook.md)、[状态隔离](docbench/docs/state_isolation.md) | 准备步骤、命令与进程边界 |
| `<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/` | 各复现者独立管理的 `source/`、`runs/`、`workspaces/` |

**结果自动写到外部 `docbench/runs/<run-id>/`，不会自动复制回仓库 `results/`。** `output_root: bench://runs` 就指这里。可换外部根目录，但当前 runner 不支持把活跃状态任意写入源码目录。历史归档不参与选题、Provider 配置或状态恢复，不会污染别人的新运行。

## 数据与依赖 / Prerequisites

PDF、QA 和固定裁判提示词需先单独准备，默认在 `docbench/source/`；这里的 `source` 指数据，不是被测代码快照。YAML 的 `dataset.data_root`、`scoring.prompt` 也可显式指定数据路径。详见[数据准备](docbench/sources/README.md)和[上游清单](docbench/upstream.manifest.json)。

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --qa-catalog-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --balanced-pdfs-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script readiness
```

前两条会联网，只准备 QA 目录和 balanced125 所需 PDF；不下载所有回归附件、上游代码、裁判提示词或模型权重，没有 `prepare-data --config` 参数。`readiness` 离线检查整个配置目录；其他回归集缺材料也会影响整体状态，不等于所选配置一定不可运行，更不是付费 Provider 冒烟。

依赖使用项目锁定环境（包含 `gdown`）；BGE encoder / reranker 使用项目模型准备入口，不静默下载缺失权重。原生文档处理是基线，安装可选 layout 依赖不等于资产已就绪。上游 revision、哈希和许可审阅不构成再分发授权，原 PDF、QA 和上游材料不随源码发布。

## 选择实验 / Experiment configuration

- `l1_bge_m3_live_1.yaml`：单题 CPU 工程检查；文档 104 包含在 balanced125 下载范围。
- `l1_storage_smoke_3.yaml`、`l1_stage_20.yaml` 及小样本配置：链路或定向回归，不是总体正确率。
- `l1_balanced_125.yaml`：原有 125 题配置，语义验证开启，检索设备为 MPS。
- `l1_balanced_125_gate_off.yaml`：同题同设置，仅关闭语义验证。
- `l1_balanced_125_gate_off_highland_235b.yaml`：在关门配置上只改视觉端点为 Highland 的 `qwen3-vl-235b-a22b-instruct`，读取 `HIGHLAND_API_KEY`，不切换 GUI 活跃配置。

该 VLM 已用于存档的同题 123 题实验（原 125 题排除两道 una-web），结果见[逐题证据](docbench/previous_results/gate_off_highland235b_123/README.md)；不据此保证当前端点可用或性能提升。更换视觉模型后不能再称为“只关闭验证门”。字段与示例见[配置说明](docbench/configs/README.md)。

## 运行与解释 / Run and interpret

按[操作手册](docbench/docs/formal_l1_eval_runbook.md)分别执行 `run` 和 `score`。`run`、`retry-failed`、`score` 都需 `--allow-live`，可能付费并外发必要文本、页面图像。同名 run 默认拒绝覆盖；续跑要求身份一致，补跑使用新状态并保留旧结果。

工程成功、L1 路由正确、无待答用户问题、来源稳定和答案正确是不同指标。机械 generation gate 不是正确率；裁判为 `docbench_prompt_compatible`，`official_comparable=false`，不是官方榜单成绩。

历史 125 题首跑与 3+5 补跑共用冻结 `source_sha256=7a6043f6465a887328363891a3edc963c05cd81d2d78868558f2c17398e38da1`，三个 run ID 见[历史统计](docbench/previous_results/showcase_125.summary.json)。8 次补跑不是 8 个新问题，应另列策略。新代码运行不能继承历史分数；精确复现还需相同源码、配置、环境和经授权数据。

除既有统计外，公开明确审阅的逐题回答、调用/失败、评分及开销投影；保留文件 ID，清理私有路径、凭据、原始请求与文档正文。私有原件不变，公开清单与文件 SHA-256 锁定；`.json` 或 `.summary.json` 文件名本身不代表可公开。
