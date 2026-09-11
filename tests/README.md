# Test Suite Layout

`tests/` 按被验证的系统层分类。新增测试优先放到对应主题目录，根目录只保留
`conftest.py` 和本说明文件。

## Directory Map

| Directory | Scope |
|---|---|
| `api/` | 本地 API service/server、前端契约、HTTP/stream 投影 |
| `context/`、`context_budget/` | 轻量 token 估算，以及 provider envelope 的测量、校验和硬准入 |
| `documents/` | DocSet、文档 ingest、artifact spill/read |
| `retrieval/` | 文件/历史检索、索引、召回、重排与相关持久化行为 |
| `evals/` | DocBench 实验、正式 runner/scorer 与授权/复现边界 |
| `l2/task_graph/` | TaskGraph 合同、验证、修订迁移与生产门 |
| `runtime/` | trusted Turn 入口、Ingress/L0-L2、TaskGraph/AuxiliaryGraph/WorkRun 合同、TurnExecutionWindow、事件、恢复与 Host Guard |
| `configuration/` | 路径配置、敏感路径 deny rules 与调用方向围栏 |
| `session/` | 会话管理、working memory、历史持久化 |
| `session_context/` | SessionContext 契约、抽取、生命周期、当前持久化与 Repair |
| `tools/` | ToolSpec、effect-aware ToolPolicy、Catalog/Registry 及具体工具；WorkRun/Attempt 接线归 `runtime/` |

## Placement Rules

- 测试某个包的内部行为时，放到对应主题目录，例如 `src/personagraph/retrieval/*`
  对应 `tests/retrieval/`。
- 测试工具协议、Policy、Registry 或具体工具时，放到 `tests/tools/`；测试 Runtime 如何选择和执行工具时，放到 `tests/runtime/`。
- 测试 API 返回给前端的 contract 时，放到 `tests/api/`，即使它间接读写 memory/session。
- 跨层测试放到最能解释用户入口的目录；产品 Turn 入口测试统一从 `runtime.entry` 或 `/api/chat` 进入。
- SessionContext 只保留当前 persistence/API 和维护入口覆盖。
- 不再向 `tests/` 根目录新增 `test_*.py`。

## Runtime Migration Rule

- 新测试不得导入 `personagraph.graph`、`runtime.turn/checkpoints/event_journal`、旧 supervisor/shadow 或旧 pending-review/resume 编排。
- 旧图已验证过的领域行为只有在迁入新所有者后，才从 `runtime.entry`、TaskGraph/WorkRun application service 或领域 facade 重建测试。
- 已退役模块由 `runtime/test_test_suite_runtime_boundaries.py` 防止重新进入活跃测试和源码边界。

## Common Commands

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/retrieval
.venv/bin/python -m pytest tests/session_context
.venv/bin/python -m pytest tests/evals
.venv/bin/python -m pytest tests/l2/task_graph/test_production_gate.py
.venv/bin/python -m pytest tests/tools/test_tool_platform.py
```

当前没有 persona/角色卡测试目录，因为权威 Runtime 不加载或注入 persona。持久化
`persona_id` 的测试只能验证兼容元数据，不得把它解释为角色选择。历史阶段报告中可能仍引用
已删除的 persona、旧图或旧 suite 测试；它们只说明当时的实现证据，不是当前回归入口。
