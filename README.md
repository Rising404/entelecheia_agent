# Entelecheia

> *From intent to actuality.*

Entelecheia 是一个面向文档研究与问答的桌面 Agent 项目。模型自主选择检索、文本读取和页面视觉分析工具；应用运行时负责权限、预算、执行记录与答案交付。文档和状态保存在本地，模型推理可使用远程 API。桌面端使用 Electron + Vue，后端使用 Python。

当前重点是 **L1 文档任务**：围绕一次用户请求，模型通过多步工具调用收集材料、组织回答。项目仍在开发中。

[开始使用](QUICKSTART.md) · [合成文档示例](examples/document_qa/README.md) · [架构与源码导览](ARCHITECTURE.md) · [评测结果与失败分析](evals/docbench/previous_results/README.md)

## 可以用它做什么

- 绑定本地工作目录或上传文档，围绕 PDF、DOCX、PPTX、Markdown 和文本文件提问。
- 由模型选择搜索文档、读取原文、定位页面；配置视觉模型后，可进一步分析页面和图表。
- 在对话中查看工具活动，结合持久执行记录核对来源、调用和错误；模型回答仍需判断，不把笔记自动当作原文证据。
- 在授权 Project 的 `output/` 下新建文本产物；当前默认能力不允许覆盖已有用户文件。

第一次体验可使用仓库提供的[三页合成手册](examples/document_qa/README.md)：完成一次信息查找、一次跨页综合，再检查模型是否能如实说明文档未提供的信息。它不需要 DocBench 数据，也不是准确率评测。

## 核心设计

| 设计 | 实际做法 | 值得检查的取舍 |
| --- | --- | --- |
| 模型决策与运行时执行分开 | 模型提出计划和工具调用；运行时（Host）检查权限、输入与执行状态，再调用工具和记录结果 | 可约束外部操作，但格式/语义校验也可能拒绝本可交付的回答 |
| 有界上下文与持久结果 | 默认只投影前一步工具批次的正文；计划、工作笔记维持连续性，更早的完整结果可按需回读 | 减少重复携带正文，但依赖模型正确记录、找到并整合旧材料 |
| 文档共享、会话隔离 | 同一 Project 复用文件版本、解析块和索引；各 Session 分别保存历史、权限与执行状态 | 文档复用不等于会话之间自动共享记忆或授权 |
| 检索与失败可检查 | BGE-M3 Dense / learned sparse / BM25 召回，RRF 融合及 BGE reranker 重排；记录实际方法、来源版本与失败 | 索引就绪、召回证据和最终答对是不同指标，不能相互替代 |

实现入口和设计代价见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 快速开始

目前完整锁定安装面向 **macOS Apple Silicon（arm64）**。需要 Git、网络和磁盘空间；两个检索模型约占 4–5 GB，此外还需 Python/前端依赖与缓存。

```bash
git clone https://github.com/Rising404/entelecheia_agent.git
cd entelecheia_agent
./scripts/bootstrap-local-runtime.sh
```

安装器为此 checkout 准备独立的 Python、Node、pnpm 和依赖，并完成运行时版本与归属检查；不需要 Codex、ChatGPT 或作者的旧环境。版本见 [runtime-versions.conf](runtime-versions.conf)，依赖闭包见 [安装约束](scripts/DEPENDENCIES.md)。这不是现成的桌面安装包。

准备真实文档检索所需的本地模型，然后启动基础文档路径：

```bash
.venv/bin/python scripts/prepare-local-models.py download
.venv/bin/python scripts/prepare-local-models.py check
PERSONAGRAPH_DOCUMENT_ENGINE=native ./frontend/start-electron.command
```

接下来在“设置 → 任务模型配置”填写并启用自己的模型配置，再上传[合成示例](examples/document_qa/README.md)提问。默认 `mock` 只返回占位结果，不能代替真实问答。页面视觉分析需要另行配置视觉模型。

**使用前注意**：Provider 密钥保存在本地明文配置中；任务所需的文本摘录、prompt 或页面图像可能外发到你配置的服务商，当前不逐次弹窗确认。真实调用可能收费，请先使用无敏感信息的样例。

完整的首次配置、独立状态、模型推理检查、成功标志和常见错误见 [QUICKSTART.md](QUICKSTART.md)。只想预览界面、暂不下载模型，也请使用其中的全新隔离状态步骤。

### 平台与输入边界

| 范围 | 当前状态 |
| --- | --- |
| macOS arm64 | 提供完整依赖锁与安装流程；模型资源、Provider 和实际推理需分别检查 |
| macOS Intel / Linux | 有 `--runtime-only` 路径，但不安装应用依赖；完整桌面文档流程未验收 |
| 原生 Windows | 有启动器，但无完整安装器；Project 文件收录、读取、上传和输出的安全文件接口尚未适配，暂不支持完整文档工作流 |
| 文本型 PDF、DOCX、PPTX、MD/TXT | 基础文档范围；解析有资源与覆盖限制，不保证复杂排版无损 |
| 扫描页、图片和复杂图表 | 需要相应视觉能力；请求成功不等于内容完整或理解正确 |
| Docling / 旧 Office / XLSX / L2 | Docling 增强解析是实验路径；旧 `.doc` / `.ppt`、XLSX 不属基础承诺；L2 图式编排保留部分代码，但不在本次展示与评测范围 |

Windows 限制不只是缺少命令：当前文件操作使用尚无 Windows 替代实现的 POSIX 接口，见 [文件读取](src/personagraph/workspace/files/observation.py) 与 [产物写入](src/personagraph/workspace/files/outputs.py)。平台依赖细节见 [快速教程](QUICKSTART.md)。

## 评测：做过什么，尚不能证明什么

已归档一轮 DocBench 开发子集：125 题、125 份不同文档，五个领域各 25 题；Text 50 题，其余三类各 25 题。主模型和裁判为 `deepseek-chat`，视觉模型为 `qwen3-vl-30b-a3b-instruct`。

| 原始 DeepSeek 裁判口径 | 正确数 | 比例 |
| --- | ---: | ---: |
| 125 题首跑，包含执行失败 | 91 / 125 | 72.8% |
| 对 8 个执行失败题定向补跑并替换对应结果 | 96 / 125 | 76.8% |

补跑后不是一次干净首跑，也不是 133 道独立题。该集合用于开发，非严格未见测试集；抽样与裁判配置不等同于官方榜单。历史被测源码快照与当前公开代码不同，不能用旧成绩证明后续修改有效。

公开材料包括[逐项统计、回答及执行证据](evals/docbench/previous_results/README.md)、[模型复核分歧](evals/docbench/previous_results/showcase_125/analysis/EVALUATION_REVIEW.md)和[链路失败归因](evals/docbench/previous_results/showcase_125/analysis/FAILURE_ATTRIBUTION.md)。复核不是独立人工金标，不用最高复核分覆盖原始裁判分。
关闭语义验证门、改用 235B VLM 的后续 123 题实验另见[本轮双方评审与 badcase 分析](evals/docbench/previous_results/gate_off_highland235b_123/reviews/README.md)，不与旧实验分析混放。

失败分析涵盖格式/语义拒绝、旧结果回读、视觉结果完整性、长尾超时和回答后的收尾问题。目前没有与 DSH、裸模型或 BM25-only 的同配置对照，也没有严格单变量的验证门/记忆模块消融实验。

### 如何自己核查

- **不调用模型**：阅读公开摘要、逐题回答、工具调用和评分，复算统计与检查失败链路；独立重判仍需另取原 PDF/QA，公开副本省略文档正文。
- **先跑一题**：按[评测运行手册](evals/docbench/docs/formal_l1_eval_runbook.md)准备授权数据、模型和凭据，使用仓库内入口分别 `run`、`score`。生成不会自动评分。
- **扩大实验**：使用新 run ID 和隔离的仓库外评测目录；同题当前代码重跑是新实验，不是历史源码的精确复现。

原始 PDF、QA、未经清理的完整轨迹与数据库不随仓库分发；回答、调用/失败、评分和开销以审阅后的公开副本提供。数据准备、配置核验、真实生成和评分分别有自己的前提；`validate` 成功不代表已能跑真实 API。

## 继续阅读与开发

| 你想了解 | 入口 |
| --- | --- |
| 第一次安装、配置与文档问答 | [QUICKSTART.md](QUICKSTART.md) |
| 一个可公开、可重建的小型文档任务 | [examples/document_qa](examples/document_qa/README.md) |
| 执行主干、状态边界和源码阅读路线 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 安装闭包、锁文件与本地模型 | [scripts/DEPENDENCIES.md](scripts/DEPENDENCIES.md) |
| 评测入口、隔离及复现边界 | [evals/README.md](evals/README.md) |
| 修改与定向验证规范 | [CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) |

源码按职责分为 `frontend/`、`src/personagraph/`、`evals/`、`tests/` 与 `scripts/`。默认测试与真实 API/设备检查分开，不把 mock 通过当作实际模型验证。发布前还需执行仓库隐私检查；`.gitignore` 不能清除已经提交的秘密。

### 内部命名与许可

内部包名 `personagraph`、`PERSONAGRAPH_*` 环境变量和部分存储标识仍保留。它们不是角色功能入口：项目不支持 persona subsystem，`persona_id` 仅是内部存储兼容字段，不提供角色选择或 persona prompt 注入。界面背景属于本地外观设置。

源码使用 [Apache-2.0](LICENSE)，第三方说明见 [NOTICE](NOTICE)。模型权重与第三方文档遵守各自许可，不因本仓库开源而自动获得再分发许可。
