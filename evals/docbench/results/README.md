# 新实验结果

本目录用于整理后续实验的报告与汇总。已完成的两轮 DocBench 实验见[结果总览](../previous_results/README.md)。

运行结果自动保存到自己的外部评测目录：

```text
<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/runs/<run-id>/
  run_manifest.json                         运行配置与环境
  generation_report.json                    生成阶段汇总
  cases/<case-id>/result.json                回答、状态和指标
  cases/<case-id>/state/artifacts/trajectory.json  执行过程
  scoring/summary.json                      裁判评分
```

仓库中的报告由运行完成后整理，runner 的原始输出仍保存在上述目录。运行、评分和补跑步骤见[操作手册](../docs/formal_l1_eval_runbook.md)。
