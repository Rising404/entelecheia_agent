# Entelecheia 架构导览

Entelecheia 将模型的下一步决策接入文件、检索和持久状态，组成面向文档任务的本地单用户工作台。
本文聚焦当前公开的 L1 路径：在绑定 Project 的 Session 中完成一次有时间与步骤预算的任务。
这里的 **Host** 指应用运行时，负责权限、执行与状态；模型负责提出计划、选择工具和组织回答。

建议用约 15 分钟沿“一次任务 → 四个设计取舍 → 源码与评测证据”阅读。
先观察产品可从 [文档问答示例](examples/document_qa/README.md) 开始；安装与功能范围见 [首页](README.md)。
下文测试链接用于说明可检查的行为，不表示本次文档整理重新运行了这些测试。

## 一次文档任务如何执行

```text
Electron / Vue → 本地 API → Turn 入口 → L1 决策循环
                                       ├─ 模型请求 → Provider
                                       ├─ 工具请求 → 文件准备 / 读取 / 检索 / 视觉 / 回读
                                       └─ 候选回答 → 引用与语义检查
                          Turn 入口 ← 已通过当前检查的答复
                               ↓
                    提交 Session 消息 → 摘要、历史索引与窗口收尾
```

1. 用户选择 Project、创建 Session 并提交问题和附件。API 校验请求及本地服务身份，
   将文件和会话操作交给各自的应用服务；它不在 HTTP 层实现另一套 Agent 循环。
2. [Turn 入口](src/personagraph/runtime/entry/) 接受这轮请求，确定执行路径，
   冻结相关模型、工具、来源与预算身份，并为执行和恢复建立持久记录。
3. [L1 控制器](src/personagraph/runtime/l1/controller.py) 读取当前计划、笔记和可用材料，
   向模型请求下一步决定。模型可以调用一批工具或提交候选答复；新计划和修订由 Host 校验、保存。
4. 如果需要准备尚未就绪的文件，[收录任务](src/personagraph/workspace/ingestion/worker.py)
   协调解析、内容入库、索引和覆盖检查。此后模型按任务需要选择直接读取、检索或视觉分析，
   并非每题都固定经过所有工具。工具结果保存后，循环继续；步骤数、工具批次和整轮时间均有边界。
5. 候选答复接受引用等机械检查，并按冻结的执行配置进行语义检查。拒绝可以形成下一步反馈，
   也可能在预算耗尽时终止；检查通过不保证事实正确。控制器返回答复，正式消息提交归 Turn 入口。
6. 消息提交后，摘要、Session 历史索引和窗口释放仍有各自的状态与恢复逻辑。
   “已保存答案”“进程退出正常”“后台收尾完成”必须分别判断。

其中 **L1 attempt 是同一轮任务中的一次决策步骤**，不是新的独立 Session，也不是一次 HTTP 重试。
同一次模型逻辑请求可能有多次物理请求；benchmark 的一次 case 重跑则使用新的进程和状态。

## 设计一：模型提出动作，Host 持有执行权

模型看到的是任务内容与可用工具，并不直接选择数据库、Session 路由或任意私有目录。
Host 将动作绑定到当前授权范围，检查参数、来源身份、调用预算和状态，再执行并保存结果。
文件 ID、版本 ID 与工具结果 ID 各有用途；模型提到某个 ID，不等于它获得了对应权限。

| 决策或状态 | 责任所在 |
| --- | --- |
| 下一步查什么、读什么、怎样回答 | 模型提出计划、工具参数、工作笔记和候选答复 |
| 路由、预算、执行身份、消息提交 | `runtime/entry/` 与 `runtime/l1/` 的编排和持久端口 |
| 模型请求与工具分派的逻辑/物理调用记录 | `runtime/model_calls/`、`runtime/tool_calls/` |
| Provider 方言、结构化输出解码与模型配置合同 | `model_io/` |
| 工具定义、资源绑定、授权和领域适配 | `tools/`；模型展示归 `tools/model_interface/` |
| Session 与 Project 的存储结构和事务 | `session/`、`workspace/`，通过显式连接和端口访问 |

这一分工让模型可以调整任务步骤，同时把有副作用的操作放在可检查的边界内。
代价是必须维护模型协议、来源身份和状态机之间的一致性；反馈含糊或模型混用身份仍会造成失败。
机械检查保证的是结构与授权等条件，语义检查本身也依赖模型，两者都不能代替事实核验。

源码入口：[模型内容投影](src/personagraph/runtime/l1/model_view/projection.py)、
[工具调用状态](src/personagraph/runtime/tool_calls/authority.py)。
对应测试：[输入边界](tests/runtime/test_attempt_input_projection_boundary.py)、
[工具调用权威与重复分派](tests/runtime/test_runtime_tool_call_authority.py)。

## 设计二：缩小当前上下文，保留可回读的记录

模型上下文是持久状态的一份有界展示，不是执行状态的唯一副本。
默认只将紧邻前一步的完整工具批次纳入下一步的结果投影，再按工具协议展示业务内容；
更早结果保留在存储中。计划与当前笔记工作集持续进入上下文，但同样受展示预算限制。

| 内容 | 用途与范围 | 不应混淆之处 |
| --- | --- | --- |
| 当前计划与持久笔记 | 保存本轮目标、验收项与每步公开工作记录，供后续步骤继续任务 | 笔记可以写错；它不是自动核实过的原文证据 |
| 工具历史回读 | `list_tool_results` / `read_tool_result` 分页读取当前执行已保存的结果，包括失败结果 | 回读不重跑原工具；回读回执不能替代原始结果身份，失败内容也不是成功证据 |
| Session 历史检索 | 在 Host 冻结的当前 Session、已提交轮次截止点和可用索引范围内检索历史 | 不因共享 Project 而读到其他 Session，也不是当前执行工具结果目录 |
| Project 文档检索 | 从获准访问的文件版本、内容单元与检索表示中寻找材料 | 文档证据与模型生成的笔记、历史答复有不同来源 |

收益是无需每步重复携带所有旧工具正文，同时仍保留回查入口；这不是无损压缩或 token 降幅的实测结论。
代价是模型需要判断何时回读、如何使用分页与结果路径，旧笔记也可能持续影响判断。
“没有放进当前 prompt”“原结果未保存”“历史索引尚未就绪”是不同问题。

源码入口：[前一步结果投影](src/personagraph/runtime/l1/tool_context/projection.py)、
[笔记保存](src/personagraph/runtime/l1/execution_notes.py)、
[工具历史绑定](src/personagraph/runtime/l1/tool_context/history_binding.py)、
[Session 历史检索组合](src/personagraph/runtime/l1/history_retrieval_composition.py)。
对应测试：[正文移出上下文后的回读](tests/runtime/l1/test_tool_history_flow.py)、
[历史范围与完整性](tests/session/persistence/l1/test_l1_tool_history.py)、
[笔记流程](tests/runtime/test_l1_execution_notes_flow.py)。

## 设计三：Project 复用文档，Session 保留独立状态

```text
私有状态根/
  project_catalog.sqlite                    Project / Session 定位
  projects/<project-id>/documents.sqlite     文件版本、内容块、检索表示
  sessions/<session-id>/session.sqlite       历史、运行状态、授权、调用账本

用户 Project 目录/                           用户文件与生成产物
私有配置根/                                  设置、模型配置、凭据
```

同一 Project 的多个 Session 可以使用共享的文档表示，不必将文档库变成每段对话的私有副本。
但文件挂载权限和会话历史不自动共享；Project 归属和 Session 授权仍要在实际读取时核验。
用户工作目录与 Host 私有配置、状态分开，模型工具不能访问后者；新建产物能力不等于任意覆盖文件权限。

文档链路也按这个边界分工：[格式解析](src/personagraph/input_processing/documents/) 负责解释输入，
[持久收录](src/personagraph/workspace/ingestion/) 负责任务与检查点，
[检索](src/personagraph/retrieval/) 负责表示版本、查询和排名。
固定 BGE-M3 资产构建 Dense、learned sparse，加上 BM25，经 RRF 融合与 reranker 重排。
实际来源、版本、方法覆盖和设备身份约束可使用的索引；修改配置不能自动证明旧索引可继续使用。

收益是明确文档复用与会话隔离的位置，并把解析结果和检索表示绑定到文件版本。
代价是收录发布、权限变化、源文件变化与索引更新需要协调。页面齐全不保证每个字符都解析正确，
索引就绪也不保证相关材料被召回、进入上下文或被模型正确使用。视觉分析还有独立的授权和模型调用边界。

源码入口：[收录 worker](src/personagraph/workspace/ingestion/worker.py)、
[候选融合](src/personagraph/retrieval/orchestration/ranking.py)、
[来源查询执行](src/personagraph/retrieval/orchestration/source_execution.py)。
对应测试：[Project 文档隔离](tests/retrieval/test_project_documents_context.py)、
[收录共享事务](tests/runtime/test_document_ingest_shared_transaction.py)、
[检索表示生命周期](tests/retrieval/lifecycle/test_generation.py)。

## 设计四：恢复依据持久状态，而不是重新猜测进度

TurnRun、Attempt 和调用账本记录执行事实，内存变量只是当前投影，trajectory 用于诊断。
恢复需要重新核对执行身份和租约，并沿用原有预算；重新进入控制器不能获得一份新的整轮额度。

| 已保存的情况 | 恢复边界 |
| --- | --- |
| 调用已成功结算，具有可验证结果 | 可以回放已保存结果，避免仅因客户端重连再次分派 |
| 已知可重试失败 | 还需相应重试授权与剩余次数，不能只凭“失败”无限重试 |
| 已分派但尚未结算，或外部结果不确定 | 需要核对外部完成情况；不能直接当作“从未执行”再做一次 |
| 有精确绑定的持久文件准备请求 | 满足恢复资格时可继续观察原任务；未知副作用、无法证明的状态不据此获得重放权限 |
| 正文已提交，历史索引或窗口收尾未完成 | 由提交后的任务与恢复流程继续处理，答案交付状态单独保留 |

收益是把“恢复原执行”“新的一次重试”“重复发送已有结果”区分开。
代价是状态机、租约与后台任务增加了故障组合；保守停止也可能需要后续诊断。
本地幂等与回放机制不承诺远程请求 exactly-once，也不证明每种崩溃点都已恢复成功。

源码入口：[L1 恢复资格与租约](src/personagraph/runtime/l1/recovery.py)、
[恢复调度](src/personagraph/runtime/l1/recovery_worker.py)、
[调用结果回放](src/personagraph/runtime/tool_calls/authority.py)、
[提交后任务](src/personagraph/runtime/post_commit/)。
对应测试：[恢复 worker](tests/runtime/test_l1_recovery_worker.py)、
[终局消息持久化](tests/session/persistence/l1/test_l1_terminal_delivery_persistence.py)、
[Session 检索恢复](tests/runtime/post_commit/test_session_retrieval_recovery.py)。

## 用评测检查这些设计，而不是用架构图证明效果

[DocBench runner](evals/docbench/reproduce_or_run_script/README.md) 调用同一条 L1 产品路径。
每个 case 首跑或补跑使用独立子进程、状态和 Host 创建的 Project；它不使用 GUI Session 数据库，
也不把 QA、参考答案目录挂载给模型。具体边界见 [评测隔离](evals/docbench/docs/state_isolation.md)。

[评分复核](evals/docbench/previous_results/showcase_125/analysis/EVALUATION_REVIEW.md) 分列首跑、定向补跑与模型评审意见。
[失败归因](evals/docbench/previous_results/showcase_125/analysis/FAILURE_ATTRIBUTION.md) 则展示这些边界受到的实际考验：
引用身份混用、工具历史路径错误、文档覆盖与计数口径分歧、视觉结果不完整，以及提交后的生命周期问题。
报告区分源证据、评审推断和调整建议，没有同条件消融就不能把总分归功于 RRF、笔记或验证门。

历史成绩绑定当时的冻结源码、配置与环境，不验证之后的当前实现，也不证明长期历史检索的跨轮收益。
公开仓库保存统计、配置、来源哈希，以及经审阅和哈希锁定的逐题回答、工具调用、失败与评分投影。
原 PDF、QA、未经清理的完整 prompt/轨迹和数据库不随源码分发；公开投影不等同于私有原件或可恢复状态。
新实验应使用新的运行身份；操作入口见 [评测说明](evals/README.md)。

## 维护边界

每项职责保留一个现行实现；依赖从 API、工具适配和应用编排流向领域合同与显式存储/Provider 端口。
模型展示不另建执行实现，controller 不拥有数据库 schema 或 Provider 传输。
数据库 schema 与目录工厂是源码，实际数据库内容是私有运行数据。

L2 任务图仍被路由、存储和回归引用，超出本文的公开 L1 功能与成绩范围，不能因默认关闭就视为可直接删除。
内部 Python 包名 `personagraph` 与 `persona_id` 存储兼容字段保留；公开会话没有角色选择或角色注入路径。
本地界面外观与受范围约束的 Session 上下文是另外的现行功能。
状态、配置和 token 路径覆盖必须是源码 checkout 外的绝对路径。
修改规则与验证要求见 [AGENTS.md](AGENTS.md)，定向测试入口见 [测试说明](tests/README.md)。
