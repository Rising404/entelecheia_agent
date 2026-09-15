# DocBench 状态隔离 / State isolation

后端评测不使用 GUI/API 的 Session 数据库或 Project 命名空间。隔离单位为：**一次题目尝试 → 一个新子进程 → 一份私有状态 → 一个 Host 创建的 Session Project**。

## 外部根 / External root

启动 Python **之前**，把 `PERSONAGRAPH_BENCH_EVAL_DIR` 设为仓库外绝对路径。从仓库根运行 `python -m evals.docbench.reproduce_or_run_script`，无需外部 launcher。

```text
<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/
  source/                benchmark 原始输入，整个目录不向 Agent 开放
  runs/<run-id>/         单题输入、私有状态、轨迹、评分
  workspaces/<run-id>/   Host 分配给 Agent 的 Session Project 根
```

每位复现者可选自己的外部根。源码中的 `results/`、`previous_results/` 是整理区，不参与运行定位。当前 runner 不支持将活跃状态直接写入源码目录。

## 子进程边界 / Process boundary

首跑由父进程分配 `runs/<run-id>/cases/<case-id>/state/`，补跑在该题 `attempts/` 下创建新状态。父进程在子解释器启动前，将精确 state、case-local config、benchmark root、run workspace 写入环境。产品路径在 import 时冻结，不能靠导入后赋值假装隔离。

worker 在写状态或请求 Provider 前核对路径。私有状态包括 Project/Session 库、解析内容、索引、Turn/Task、轨迹、工具和模型账本、授权及配额记录。

## Project 分配 / Allocation

同一 run 的 worker 将 `workspaces/<run-id>` 用作 `PERSONAGRAPH_DEFAULT_PROJECTS_DIR`。真正创建和绑定 Project 的是产品 Session service，不是评测替代实现。每题独立 Session，补跑使用新 Project。

runner 验证 Project 在本 run workspace 下，在私有结果中记录 DocBench 根相对定位。**不能把 `source/` 或私有 `runs/` 绑定为 Project**，否则文件工具可能看到其他题目、参考答案、裁判材料或 SQLite 状态。模型只获得本题获准挂载的材料。

## 配置复用不等于共享状态 / Configuration vs. state

父进程可读取安装级活跃 Provider profile 并冻结端点；worker 启动前替换状态/配置根，不沿用 GUI 默认 Project 根。要独立凭据，可在启动父进程前将 `PERSONAGRAPH_LOCAL_CONFIG_DIR` 指向另一份仓库外私有配置目录。

不要把真实 profile、`.env`、数据库或 GUI Project 放入 Git。Session 名称不是隔离边界，数据库根和 Project 授权才是。隔离不等于禁止外发：真实 Provider 请求仍可能发送已授权的文档内容。

## 回归约束 / Regression obligations

1. 子进程 state 在该 run 的精确 case 树下。
2. import 后的 state/config 常量等于父进程指定路径。
3. 默认 Project 根等于本 run workspace。
4. workspace 与私有 run/state 不重叠。
5. 新 Session Project 位于本 run workspace 下。
6. 每次补跑新建 state 和 Host Project。
7. GUI catalog/Session 库不新增评测记录。

对应测试归 `tests/evals/`；真实运行前置条件见[操作手册](formal_l1_eval_runbook.md)。
