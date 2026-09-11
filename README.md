# Entelecheia

> *From intent to actuality.*

Entelecheia 是面向文档任务的本地单用户 Agent 工作台，将 Project 文件库、Session 对话、文档解析与检索、模型调用和工具执行组织为可观察、可恢复的工作流。桌面界面使用 Electron + Vue，后端使用 Python。

当前公开重点是有界 L1 文档任务，仍处于开发阶段。内部 Python 包名 `personagraph`、`PERSONAGRAPH_*` 环境变量及部分持久化标识保持兼容；它们不是旧角色功能的入口。项目不支持 persona subsystem（角色卡加载、人格图或角色注入）。Session 的 `persona_id` 仅是内部存储兼容字段，不对客户端开放角色选择，也不用于构造模型 prompt。

## 能力与边界

- Project 级文件收录、版本和解析块；同一 Project 的 Session 共享文档表示，但历史和挂载权限隔离。
- PDF、DOCX、PPTX 和文本的有界解析；保留来源、版本指纹及覆盖缺口。扫描页、复杂图表和版式不保证完整理解。
- 固定 revision 的本地 BGE-M3 Dense / learned sparse / BM25 检索，以及 `bge-reranker-v2-m3` 重排。严格配置缺失模型或能力时会报错，不会静默伪装成 Hybrid 成功。
- L1 自主选择文件、检索、读取与视觉工具；工具结果持久化，可按需回读。模型最终回答经过运行时校验，但仍可能事实错误。
- Session/Turn、工具结果和交付的本地恢复与重放；不承诺远程模型请求 exactly-once。
- 在授权 Project 内新建有界产物，不默认允许覆盖已有用户文件；来源授权、路径与私有状态边界仍受 Host 约束。

远程视觉需要独立可用的模型配置。L2 TaskGraph 代码因现行路由/存储依赖而保留，但不是当前稳定功能或 L1 评测成绩的承诺。Docling 增强版面处理仍属实验路径：安装依赖不代表所需布局模型已准备，基础复现以 native 文档路径为准，可在进程启动前显式设置 `PERSONAGRAPH_DOCUMENT_ENGINE=native`。旧 `.doc` / `.ppt`、XLSX 和复杂视觉内容不属于基础支持承诺。

## 安装与启动

首发完整锁定安装目标为 **macOS arm64、Python 3.12**。Python、Node 与 pnpm 的精确版本见 [runtime-versions.conf](runtime-versions.conf)；Python 依赖见 [requirements-macos-arm64.lock](requirements-macos-arm64.lock)，前端依赖由 pnpm lockfile 固定。安装闭包及唯一源码构建例外见 [依赖约束](scripts/DEPENDENCIES.md)。其他平台的完整安装尚未验证，存在启动脚本不等于已经支持。

在仓库根目录执行：

```bash
./scripts/bootstrap-local-runtime.sh
.venv/bin/python scripts/verify-runtime-independence.py
```

安装需要网络与可用磁盘空间。脚本创建此 checkout 独立的 `.runtime/` 和 `.venv/`，按锁安装 Python/前端依赖；不会使用旧仓环境，也不从 Codex 或 ChatGPT Runtime 回退启动。`--runtime-only` 只准备运行时，不表示其他平台已获得完整锁定安装。真实凭据、原始 benchmark 文档和模型权重不随源码提供。

### 本地检索模型

先做离线检查；缺少模型时退出非零，不会偷偷下载：

```bash
.venv/bin/python scripts/prepare-local-models.py check
```

需要时显式下载固定的两个模型快照，预留数 GB 空间：

```bash
.venv/bin/python scripts/prepare-local-models.py download
.venv/bin/python scripts/prepare-local-models.py check
```

缓存遵守 `HF_HOME` / `HF_HUB_CACHE`。使用自定义缓存时，准备、GUI、后端和评测进程必须使用一致的环境。不要把模型缓存复制进 Git。模型快照检查只证明资源存在，不等于设备推理通过；可另行运行真实本地检查：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m personagraph.retrieval.operations.real_smoke
```

成功应同时报告三种检索方法可用及 `reranker_relevant_first=true`。本地设备/模型检查不调用任务 Provider，也不证明文档问答准确率。

### 桌面工作台

```bash
./frontend/start-electron.command
```

或通过项目 pnpm 入口启动：

```bash
./scripts/run-project-pnpm.sh --dir frontend start
```

默认模型为 `mock`，可检查界面和工作流，但回答是占位结果。真实使用时，在“设置 → 任务模型配置”填写自己的 Provider、请求方言、模型名、端点与密钥，并启用配置。首次新建 Session 时选择仓库外工作目录或上传文档。

Electron 使用项目 `.venv` 的 Python sidecar，并检查本地 API 所有权。已有旧后端占用端口时，应先退出旧应用；不要靠放宽所有权检查复用未知进程。

## 数据与外发

默认配置、状态和用户 Project 均在源码仓库外。macOS 配置/状态位于 `~/Library/Application Support/Entelecheia/`；默认 Project 位于 `~/Documents/Entelecheia/`。可在启动前设置绝对路径 `PERSONAGRAPH_LOCAL_CONFIG_DIR`、`PERSONAGRAPH_STATE_DIR`、`PERSONAGRAPH_DEFAULT_PROJECTS_DIR`，但不得指回源码 checkout。

Provider 配置和密钥存于本地明文文件。环境变量优先于用户配置目录的 `.env`，后者优先于 GUI 保存设置；仓库根 `.env` 不作为运行配置加载。Host 禁止模型工具访问配置目录与 API token，但这不替代操作系统和磁盘安全。

绑定的工作目录、上传附件及完成任务所需的 prompt、文本摘录或页面图像可能发送给所配置的远程服务商；当前外发默认同意，不逐次弹窗。使用前请自行确认文档与 Provider 的使用权限。UI 背景外观属于本地个人设置，与角色系统无关；背景文件不随源码分发。

## DocBench 评测

执行实现在本仓库，不依赖作者桌面的包装脚本。在仓库根目录、启动 Python **之前**设置仓库外评测根：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script --help
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
```

请把示例绝对路径替换为自己的目录。`validate` 只验证配置，不证明 PDF、模型或凭据齐备；`readiness` 进行进一步离线核验。下载数据需显式准备，真实 `run`、`retry-failed` 和 `score` 需 `--allow-live`，可能产生费用。

数据许可、隔离、配置与可复现边界见 [评测说明](evals/README.md) 和 [运行手册](evals/docbench/docs/formal_l1_eval_runbook.md)。第三方 PDF、QA、完整回复、原始轨迹与数据库不会分发。

125 题首跑及 3+5 题补跑属于冻结的 PersonaGraph 评测快照 `l1-gpu-deadline25-20260910-a`；补跑是重复执行，不是新增独立问题。当前 Entelecheia 包含后续改动，不能把历史成绩当作当前版本成绩。相同选集上的当前运行不是旧快照的精确复现；公开历史摘要也不等于可公开原始数据或完整私有轨迹。

## 开发与验证

```bash
.venv/bin/python -m pytest -q
./scripts/run-project-pnpm.sh --dir frontend run check
.venv/bin/python scripts/check_repository_privacy.py --worktree
```

默认测试使用隔离临时状态和 mock Provider；真实网络/模型测试需显式启用。提交前还应检查 staged/tree，不能只依赖 `.gitignore`。不把旧仓 `.git`、本地 Runtime、依赖目录、用户文档或私有状态复制为公开历史。

源码布局和职责见 [架构](ARCHITECTURE.md)，修改约束见 [贡献指南](CONTRIBUTING.md) 与 [AGENTS.md](AGENTS.md)。许可见 [LICENSE](LICENSE)，第三方边界见 [NOTICE](NOTICE)。

```text
frontend/          Electron、Vue 和前端测试
src/personagraph/  API、L1/保留的 L2、模型/工具、文档检索和持久化
evals/             DocBench 实现、冻结配置/选集与脱敏摘要
tests/             对现行行为和安全边界的回归测试
scripts/           安装、模型准备、运行时与仓库隐私检查
```
