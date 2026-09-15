# 关闭验证门一轮：实验档案

本轮关闭语义验证门，并将视觉模型换为 Qwen3-VL 235B。任务主模型与 DeepSeek 裁判均为 **DeepSeek Flash，未开启思考模式**。

原选集排除两道需要联网的题 **`docbench:26:4`、`docbench:37:4`** 后共 123 题。单次运行全部完成回答交付、后台收尾与进程正常退出；三位裁判成绩为 **DeepSeek Flash 97/123（78.86%）、Astra 103/123（83.74%）、Opus 5 106/123（86.18%）**。

## 分析

- [分析概要与评分分歧](reviews/README.md)：两轮比较、工具使用、视觉错误与评审分歧。
- [Astra 分析](reviews/codex/REVIEW.md)、[Opus 5 分析](reviews/claude/REVIEW.md)：逐题判断与后续建议。
- [两轮实验对比](../README.md)：轮次、文档领域和题型的完整表格。

## 查阅逐题记录

运行批次为 [`l1-gate-off-highland235b-123-20260914-a`](runs/l1-gate-off-highland235b-123-20260914-a/)。

| 文件 | 内容 |
| --- | --- |
| `cases/<题目>/result.json` | 回答、执行状态、耗时与调用统计 |
| 同题 `trajectory.json` | 工具调用、返回结果和执行过程 |
| `scoring/` | DeepSeek 的逐题评分与汇总 |
| `run_manifest.json` | 模型、配置和运行信息 |
| [Astra 逐题评分](reviews/codex/case_reviews.json) · [Opus 5 逐题评分](reviews/claude/case_reviews.json) | 两位裁判的判断、理由和对应证据 |

原 PDF 和 QA 数据需另行准备，公开轨迹省略了部分原始输入与文档正文。
