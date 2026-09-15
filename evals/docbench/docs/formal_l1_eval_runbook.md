# DocBench L1 操作手册 / Runbook

全部命令从仓库根执行，先完成[项目安装](../../../README.md)。本文是操作说明，不代表已替你下载模型、执行真实 API 或验证成绩。

## 1. 设置自己的评测目录 / Evaluation root

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
```

示例换成仓库外绝对路径，在 Python 启动前设置。无需外部包装脚本，也没有默认桌面回退。`bench://` 和默认数据/映射路径依赖该变量；`--help` 不需要，`validate` / `list` 需要变量但不要求目录或数据已存在。变量本身不能代替 worker 的 import 前隔离检查。

```text
<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/
  source/       PDF、QA、下载映射和固定裁判提示词
  runs/         各次运行的清单、单题尝试、私有状态与评分
  workspaces/   Host 为各题创建的 Session Project
  local_config/ 可选的评测专用私有 Provider 配置
```

可换外部根目录；当前活跃 run 必须在这里的 `runs/<run-id>` 下。不能把 `source/` 或 `runs/` 绑定给 Agent；见[状态隔离](state_isolation.md)。仓库 `results/` 是新结果整理区，`previous_results/` 是历史归档，二者不参与运行输入或恢复。

## 2. 准备数据和模型 / Prerequisites

[上游清单](../upstream.manifest.json) 固定 revision 与哈希。单独获取经授权的数据，在 `docbench/source/evaluation_prompt.txt` 放固定裁判提示词。仓库不再分发上游 checkout、prompt、PDF 或 QA；许可审阅和哈希不是再分发授权。

使用项目锁定依赖（含 `gdown`）和根 README 的模型准备命令。配置自己的主模型/视觉 Provider；如需与 GUI 凭据隔离，在启动父进程前设置仓库外的 `PERSONAGRAPH_LOCAL_CONFIG_DIR`。准备和评测时保持自定义 Hugging Face 缓存设置一致。

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --qa-catalog-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --balanced-pdfs-only
```

这两条会联网，只取 QA 和 balanced125 PDF，不准备所有回归附件、裁判 prompt、上游 checkout 或模型权重。单题配置用到的文档 104 包含在其中，其他选题可能需要额外 PDF；没有 `prepare-data --config` 参数。

也可用已有数据，修改 YAML 的 `dataset.data_root` 指向自己的目录，并通过 `scoring.prompt` 指向固定 prompt；数据仍须符合选题清单的路径结构与哈希。下载器支持显式 `--data-root` / `--mapping`，见[数据说明](../sources/README.md)。

独立重建选题时写入新文件，不覆盖已有清单：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script build-selection \
  --balanced \
  --data-root "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/source/data" \
  --output "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/rebuilt-balanced-125.json"
```

对照仓库选题与哈希，不要用重建清单悄悄接受数据漂移。公开 selection 不含题目、参考答案或证据正文。

## 3. 校验并选择配置 / Configuration

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
.venv/bin/python -m evals.docbench.reproduce_or_run_script readiness
```

这些命令不调用 Provider。`validate` 校验契约；`readiness` 额外检查整个配置目录的源哈希、prompt、凭据是否存在及检索资产，可能被其他回归集缺材料影响。它不证明真实 Provider/GPU 推理或答案正确。

最小真实检查是 `l1_bge_m3_live_1.yaml`（CPU）；`l1_balanced_125.yaml` 是 125 题配置（MPS）。小回归只说明链路，不能估总体准确率。换设备、模型或字段会改变实验身份。完整目录见[配置说明](../configs/README.md)。

新增 `l1_balanced_125_gate_off_highland_235b.yaml` 保持同题、关门、MPS 和原预算，视觉端点明确指定 Highland 的 `qwen3-vl-235b-a22b-instruct`。先在自己的 shell 或私有凭据环境设置 `HIGHLAND_API_KEY`；不要把真实值写入 YAML 或提交到 Git。旧关门配置仍保留，不改 GUI 活跃模型。

## 4. 运行、续跑与补跑 / Execution

以下操作可能产生费用，并发送必要文档文本或页面图像，请主动选择后执行。

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script run \
  --config evals/docbench/configs/l1_bge_m3_live_1.yaml \
  --run-id my-l1-smoke --allow-live
```

正式运行使用新的 run ID；`--require-clean` 要求干净工作树与可识别 Git revision：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script run \
  --config evals/docbench/configs/l1_balanced_125.yaml \
  --run-id my-balanced-125 --require-clean --allow-live
```

同名目录已存在时默认拒绝覆盖。仅对相同配置、run ID 的未完成运行使用 `run --resume`；配置、选题、源码或环境身份漂移会被拒绝，已保存失败也不会自动覆盖。

只补跑某类失败时显式指定错误码：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script retry-failed \
  --config evals/docbench/configs/l1_balanced_125.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/my-balanced-125" \
  --max-workers 1 --error-code MODEL_TRANSPORT_FAILURE --allow-live
```

`--error-code` 可重复；省略则选择全部失败题。每次补跑使用新的 state 和 Host Project，旧结果保存在 attempts 与补跑报告中，不丢弃不利尝试。

## 5. 单独评分 / Scoring

生成和评分是两步，`run` 不自动调用裁判：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script score \
  --config evals/docbench/configs/l1_balanced_125.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/my-balanced-125" \
  --resume --allow-live
```

使用固定 DocBench prompt 与所选裁判身份，报告 `scoring_protocol=docbench_prompt_compatible`、`official_comparable=false`。答案分数与执行成功不同；尝试完成、execution、L1 lane、待答用户问题、源码稳定等机械 gate 字段另列，`baseline_eligible` 还涉及来源和干净工作树要求。

## 6. 查阅结果，不干扰下一次复现 / Results

原始结果自动保存在本人的外部 `runs/<run-id>/`：单题 `result.json` 保存最终 `reply`、状态、耗时及轨迹引用；`state/artifacts/trajectory.json` 保存已记录的过程，`scoring/` 保存裁判输出。

后续可将审阅后的统计整理到仓库 `results/`，或把原始阅读副本放在其被忽略的 `private_runs/<实验名>/`。这不是自动同步，也不是恢复运行状态。已有历史材料在 [previous_results](../previous_results/README.md)，不会被 `run` 或选题器读取。原始材料不能直接提交公开。

两种结果目录都不参与被测源码指纹，整理归档不会造成 `source_changed_during_run`。
`--require-clean` 的 Git 干净工作树要求仍保留；配置、选题、runner 和产品源码仍参与身份核验。

历史 125 首跑和 3+5 补跑共用冻结 `source_sha256=7a6043f6465a887328363891a3edc963c05cd81d2d78868558f2c17398e38da1`；三个 run ID 见[历史统计](../previous_results/showcase_125.summary.json)。8 次补跑不是 8 个新问题，同题新实验也不能继承旧成绩。保留运行、配置、选题、源码、模型身份和补跑来源。

除 ID、哈希和统计外，允许发布明确审阅的生成回答、工具调用/失败、评分与开销投影，见历史目录中的两个逐题证据包。PDF、QA、参考答案、未经清理的请求/轨迹、数据库和本机配置仍不进 Git；复制不等于脱敏，发布前必须核验精确文件清单、SHA-256 和隐私边界。
