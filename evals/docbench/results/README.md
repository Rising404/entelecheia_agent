# 新实验结果 / Current results

本目录用于整理**后续新实验**的结果。已有 125 题、补跑、早期工程测试与评审分析已归入
[previous_results](../previous_results/README.md)，不混作新代码的成绩。

## 自动输出到哪里 / Runner output

当前 runner 默认写到你指定的外部目录：

```text
<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/runs/<run-id>/
  run_manifest.json
  generation_report.json
  cases/<case-id>/result.json
  cases/<case-id>/state/artifacts/trajectory.json
  scoring/summary.json
```

**不会自动写回本目录**。外部根可自定义；活跃运行仍须在它的 `docbench/runs/` 下，保持
状态与源码、GUI 分离。详见[操作手册](../docs/formal_l1_eval_runbook.md)。

## 事后整理 / Archiving

- 可公开的统计放本目录直接下的 `*.summary.json` / `*.baseline.json`，必须通过现有 schema
  与隐私检查；后缀本身不构成发布许可。新的分析文档需逐文件审阅后再加入白名单。
- 原始答案、轨迹和评分阅读副本按 `private_runs/<实验名>/runs/<批次>/cases/<case-id>/` 分层，
  跨批次复核单独放在该实验的 `reviews/`。整个 `private_runs/` 被 Git 忽略。
- 不把原始 SQLite、模型权重或缓存当阅读材料搬入；原件留在外部运行目录。
- 不覆盖历史成绩；新配置、新模型或新代码使用新 run ID，补跑与首跑分别保留。
- 经明确授权的逐题公开包另行清理，按 result/trajectory/scoring 组织；当前已审阅的两组见
  [历史证据](../previous_results/README.md)。新增公开包必须重新审阅清单，不能直接解除原件的 ignore。

本目录与 `previous_results/` 都不是数据输入或 Session Project。克隆仓库不会获得被忽略的
原始档案，也不会自动恢复作者的会话；他人使用自己的外部根和新 run ID 即可独立复现。

新旧结果目录均不参与被测源码指纹；整理归档不会被误判为执行代码变动。
但 `--require-clean` 仍要求 Git 工作树干净；修改受 Git 跟踪的公开报告后，仍需先完成自己的提交。
