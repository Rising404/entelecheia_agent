# DocBench L1 操作手册 / Runbook

本手册按准备目录、下载数据与模型、检查配置、执行和评分的顺序说明一次完整评测。全部命令从仓库根执行，Python 环境按[快速上手](../../../QUICKSTART.md)安装。

## 1. 设置自己的评测目录 / Evaluation root

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
```

将示例换成自己的仓库外绝对路径，在 Python 启动前设置。`bench://` 和默认数据/映射路径依赖该变量；`--help` 不需要，`validate` / `list` 需要变量但不要求目录或数据已存在。

```text
<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/
  source/       PDF、QA、下载映射和固定裁判提示词
  runs/         各次运行的清单、单题尝试、私有状态与评分
  workspaces/   Host 为各题创建的 Session Project
  local_config/ 可选的评测专用私有 Provider 配置
```

每题尝试在新子进程中运行，拥有独立 state 和 Session Project。活跃 run 位于 `runs/<run-id>`，Agent 只访问本题 Project 中挂载的附件；`source/` 和 `runs/` 存有 QA、裁判材料和私有状态，不作为 Project 开放。配置与状态路径在子进程启动前设置，具体检查见[状态隔离](state_isolation.md)。

## 2. 准备数据和模型 / Prerequisites

[上游清单](../upstream.manifest.json) 记录来源、revision、许可和哈希。PDF、QA 和固定裁判提示词需要单独获取；裁判提示词放在 `docbench/source/evaluation_prompt.txt`，其哈希与 YAML 中的 `scoring.prompt_sha256` 匹配。

数据下载使用项目锁定的 `gdown` 依赖：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --qa-catalog-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --balanced-pdfs-only
```

这两条会联网，只取 QA 和 balanced125 PDF，不准备所有回归附件、裁判 prompt、上游 checkout 或模型权重。单题配置用到的文档 104 包含在其中，其他选题可能需要额外 PDF；没有 `prepare-data --config` 参数。

也可用已有数据，修改 YAML 的 `dataset.data_root` 指向自己的目录，并通过 `scoring.prompt` 指向固定 prompt；数据需要符合选题清单的路径结构与哈希。下载器的 `--data-root` 和 `--mapping` 可分别指定数据目录与下载映射文件。

独立重建选题时写入新文件，不覆盖已有清单：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script build-selection \
  --balanced \
  --data-root "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/source/data" \
  --output "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/rebuilt-balanced-125.json"
```

选题清单记录题目 ID、文件定位与内容哈希，不含题目、参考答案或证据正文。与仓库清单的差异会改变选题身份。

检索模型可先做本地资源检查；缺少资产时再下载：

```bash
.venv/bin/python scripts/prepare-local-models.py check
.venv/bin/python scripts/prepare-local-models.py download
.venv/bin/python scripts/prepare-local-models.py check
```

`check` 不下载或加载权重，输出 `status: ready` 表示资源齐备；`download` 从 Hugging Face 获取固定 revision 的 BGE-M3 和 reranker，约需 4–5 GB 空间，不调用任务 Provider。Docling 布局/OCR 模型另行准备。若自定义 `HF_HOME` 或 `HF_HUB_CACHE`，准备和评测进程使用相同设置。

主模型和视觉模型默认读取安装级 Provider 配置。如需与 GUI 使用不同凭据，在启动评测父进程前，将 `PERSONAGRAPH_LOCAL_CONFIG_DIR` 指向自己的仓库外配置目录。

## 3. 校验并选择配置 / Configuration

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
.venv/bin/python -m evals.docbench.reproduce_or_run_script readiness
```

这些命令不调用 Provider。`validate` 检查配置契约，`list` 显示配置 ID 和哈希，`readiness` 检查整个配置目录所需的源文件、prompt、凭据是否存在及检索资产。其他回归集缺材料也会出现在报告中。其检查范围是本地准备情况，真实 Provider 和检索推理通过后续单题运行检查。

| 配置 | 用途 |
| --- | --- |
| `l1_bge_m3_live_1.yaml` | 单题 CPU 检查，适合先确认执行链路 |
| `l1_balanced_125.yaml` | 125 题评测，语义门开启，MPS |
| `l1_balanced_125_gate_off.yaml` | 相同 125 题，仅关闭语义门 |
| `l1_balanced_125_gate_off_highland_235b.yaml` | 相同 125 题关门，并指定 Highland 235B 视觉模型 |

换设备、模型或配置字段会改变实验身份。单题检查用于确认链路，总体准确率按完整选题统计。Highland 配置从 `HIGHLAND_API_KEY` 读取密钥，视觉模型为 `qwen3-vl-235b-a22b-instruct`，主模型和裁判来源保持不变；它不切换 GUI 的活跃模型。

## 4. 运行、续跑与补跑 / Execution

`run`、`retry-failed` 和 `score` 通过 `--allow-live` 启用真实 Provider 调用，可能发送文档文本或页面图像并产生费用。先运行单题配置检查链路：

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

`--error-code` 可重复；省略则选择全部失败题。每次补跑使用新的 state 和 Host Project，旧结果保存在 attempts 与补跑报告中。

## 5. 单独评分 / Scoring

生成和评分是两步，`run` 不自动调用裁判：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script score \
  --config evals/docbench/configs/l1_balanced_125.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/my-balanced-125" \
  --resume --allow-live
```

评分使用固定 DocBench prompt 与所选裁判，报告 `scoring_protocol=docbench_prompt_compatible`、`official_comparable=false`，即提示词兼容评测，非官方同条件结果。

答案正确率和执行状态分别记录：尝试完成、execution、L1 lane、待答用户问题、源码稳定等 gate 字段描述链路状态，`baseline_eligible` 还检查来源与干净工作树。进程完成和答案判对是两项独立指标。

## 6. 查阅结果 / Results

原始结果自动保存在本人的外部 `runs/<run-id>/`：单题 `result.json` 保存最终 `reply`、状态、耗时及轨迹引用；`state/artifacts/trajectory.json` 保存已记录的过程，`scoring/` 保存裁判输出。

仓库内的 `results/` 和 [previous_results](../previous_results/README.md) 保存整理后的报告与公开逐题记录，runner 不读取这些目录作为运行输入或恢复状态。原始阅读副本可另存到 `results/private_runs/<实验名>/`，该目录由 Git 忽略。

两种结果目录都不参与被测源码指纹，整理归档不会造成 `source_changed_during_run`。
`--require-clean` 的 Git 干净工作树要求仍保留；配置、选题、runner 和产品源码仍参与身份核验。

公开逐题记录包含生成回答、工具调用/失败、评分、token 和耗时；原题、参考答案与文档文本被省略或替换为哈希，完整请求、轨迹和数据库保存在本地运行目录。导出方式见[执行包的导出命令](../reproduce_or_run_script/README.md#导出公开证据--publication-projection)。

<details>
<summary>历史实验的复现记录</summary>

历史 125 首跑和 3+5 补跑共用冻结 `source_sha256=7a6043f6465a887328363891a3edc963c05cd81d2d78868558f2c17398e38da1`；三个 run ID、配置和模型身份见[历史统计](../previous_results/showcase_125.summary.json)。8 次补跑对应原选题中的 8 个问题，合并统计仍是 125 题。当前源码重跑会记录新的源码身份，成绩与历史运行分别保存。

</details>
