# 评测

当前有两类独立评测：**RAG 纯检索**只测证据召回、排序与装包；**L1 Agent 端到端**由 Agent 自主选择工具并生成回答。前者按 qrels 计算检索指标，后者由裁判评估答案，不能混用分数或实验设置。

## 原创多文档 PDF 样本

[四份 PDF、十二道问题及参考证据](multi_doc/README.md)用于定向检查 L1 的文档定位、
跨页计算、条件筛选和引用。每题提供同一组四份 PDF，仅一份包含完整答案；样本文件已随仓库提供。
它属于 Agent 端到端诊断，不是 DocBench，也不能作为总体能力或视觉理解成绩。

## RAG 纯检索结果

2026-09-23 完成 75 题完整 hybrid 运行，最终 **Hit@5 = 74/75（98.67%）、宏平均 Recall@5 = 96.04%、MRR = 0.9296**。同轮 RRF 阶段分别为 96.00%、92.04%、0.8648。每题使用一次原问题，先限定所属文档和模态，再运行 Dense + learned sparse + BM25 → RRF → reranker → packing。

**不运行 Agent、答案生成、答案验证门或 LLM 裁判；这不是验证门开关实验。** 视觉候选以文字描述参与检索，本轮不新增 VLM 解读。Hit/Recall 不等于问答正确率，也不是无文档范围约束的全库检索成绩。

- [实验首页与全部已跑结果](docbench_hybrid_retrieval_optimize/README.md)：75 题六阶段与分题型指标、两个 BM25 基线和 9 题接入检查。
- [中文结果、失败与耗时分析](docbench_hybrid_retrieval_optimize/results/analysis/retrieval-repaired75-hybrid-20260923.md)。
- [语料、索引与复跑说明](docbench_hybrid_retrieval_optimize/README.md#按需下载)：大型语料与索引通过 Release 按需下载，脚本、报告和校验清单随主仓提供。

## Agent 端到端问答结果

以下两轮评测覆盖 **L1 文档任务链路**：每题创建独立 Session 和 Project，由 Agent 自主选择读取、检索与视觉工具完成任务。参考答案只用于评分。

[两轮 DocBench 实验](docbench/previous_results/README.md) 汇总各轮次、五个文档领域、四种题型的成绩，以及 DeepSeek Flash、Astra、Opus 5 三位裁判的判断。关闭验证门一轮在共同 123 题上全部完成回答交付与后台收尾，三位裁判正确率分别为 **78.86%、83.74%、86.18%**。

- [开启验证门：评分与错题分析](docbench/previous_results/first_gate_on_125/analysis/EVALUATION_REVIEW.md)、[链路失败分析](docbench/previous_results/first_gate_on_125/analysis/FAILURE_ATTRIBUTION.md)。
- [关闭验证门：分析与评分分歧](docbench/previous_results/gate_off_highland235b_123/reviews/README.md)、[Astra](docbench/previous_results/gate_off_highland235b_123/reviews/codex/REVIEW.md)、[Opus 5](docbench/previous_results/gate_off_highland235b_123/reviews/claude/REVIEW.md)。
- [开启验证门逐题统计](docbench/previous_results/showcase_125.summary.json) 提供评分、补跑选择与运行信息。

两道需要联网的题 `docbench:26:4`、`docbench:37:4` 超出本轮仅使用本地文档工具的范围，成绩对比统一排除这两题。

## 自己运行 Agent 问答评测

完成[项目安装](../QUICKSTART.md)后，从仓库根目录执行：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
```

将示例替换为自己的仓库外绝对路径。评测数据放在其 `docbench/source/` 下，运行输出放在 `docbench/runs/<run-id>/` 下。

完整的数据准备、模型配置、运行、评分与补跑步骤见[操作手册](docbench/docs/formal_l1_eval_runbook.md)。

| 配置 | 用途 |
| --- | --- |
| `l1_bge_m3_live_1.yaml` | 单题 CPU 链路检查 |
| `l1_balanced_125.yaml` | 开启语义验证门，MPS 检索，原选集 125 题 |
| `l1_balanced_125_gate_off.yaml` | 使用相同选集与设置，关闭语义验证门 |
| `l1_balanced_125_gate_off_highland_235b.yaml` | 关闭语义验证门，并使用 Qwen3-VL 235B；视觉密钥读取 `HIGHLAND_API_KEY` |

`run`、`retry-failed`、`score` 需要 `--allow-live`，会调用配置的模型服务并可能产生费用。字段说明见[实验配置](docbench/configs/README.md)，其余命令见[CLI 说明](docbench/reproduce_or_run_script/README.md)。

尚未接入的评测集见 [Benchmark 候选清单](BENCHMARK_BACKLOG.md)。
