# Entelecheia Frontend

This folder contains frontend-only code for the Entelecheia UI.

For installation, an isolated mock preview, and a first real document task, start with
the root [Quickstart](../QUICKSTART.md). This page explains frontend development and platform boundaries;
it does not provide a second installation workflow.

The boundary is intentional:

- `frontend/` owns UI layout, client-side state, and API calls.
- `src/personagraph/` owns runtime orchestration, session/document storage, tools, and API adapters.
- Frontend code must not import Python modules or read SQLite/checkpoint files directly.

## Current Slice

Current production-first slice:

- Vue 3 + Vite renderer
- Electron desktop shell
- Session and chat workspace backed by real `/api/sessions`, `/api/chat/stream`, and `/api/chat`
- Document workspace reads and metadata edits through `/api/documents`; new intake uses only durable ingest-job endpoints
- Workspace feature: folder/session tree, backend-assigned fixed workspaces, and session-scoped document summary
- Create / rename / archive / trash / restore sessions
- Server-backed session search across titles and turns
- Send chat turns through the stream event channel and handle inline approval cards
- Inspect, explain, export, and durably clear SessionContext from the chat inspector
- Record correction evidence, review backend Repair diffs, and apply only a fresh preview token
- Ingest documents, edit document title/tags/summary, and delete document indexes
- Inspect real session-scoped in-session task details emitted by the runtime
- Connection status against the local API
- Debug drawer hidden by default

The renderer is feature-sliced under `frontend/src/features/`. Pinia is deliberately limited to the workspace aggregate
(folders, sessions, current workspace selection); feature-local forms and drafts remain local rather than becoming
global stores. Chat and documents keep their selection, form, dirty/reset, and action coordination inside their own
feature composables.
`AppNav` and `FeedbackBanners` hold shared shell UI while the root retains cross-feature navigation, messages, refresh, and dirty-state coordination.

The active renderer is `frontend/src/App.vue` with API calls in `frontend/src/api.js`.

## Prerequisites And Reproducible Install

The dependency manifests are the source of truth. Do not commit `node_modules/` or copy it between machines:

| Layer | Required version | Source of truth |
|---|---|---|
| Python backend | Python 3.12.13 | root `runtime-versions.conf` and `pyproject.toml` |
| Renderer and desktop shell | Node.js 24.14.0 | root `runtime-versions.conf`, `.node-version`, and `package.json` `engines` |
| Frontend package manager | pnpm 11.9.0 | `package.json` `packageManager` and `pnpm-lock.yaml` |
| Desktop runtime | Electron 41.x, installed by pnpm for the current OS | `package.json` and `pnpm-lock.yaml` |

From the repository root, install the pinned project-owned runtimes and locked dependencies:

```bash
./scripts/bootstrap-local-runtime.sh
```

The bootstrap downloads checksum-verified Python and Node archives into the Git-ignored root `.runtime/`, binds
root `.venv` to that Python, keeps Corepack and the pnpm store under `.runtime/`, and installs the Python
dependency closure from `requirements-macos-arm64.lock` and frontend dependencies from `pnpm-lock.yaml`.
Bootstrap itself checks runtime versions and checkout ownership before reporting success.
The complete locked source installation is currently scoped to macOS arm64. Launch paths do not fall back
to Codex, ChatGPT, system interpreters, or caller-provided runtime paths.
To repair only the runtimes and virtual-environment binding while preserving installed packages:

```bash
./scripts/bootstrap-local-runtime.sh --runtime-only
```

Standalone runtime downloads cover macOS arm64/x86_64 and Linux arm64/x86_64 only;
`--runtime-only` does not install application dependencies or establish full support on those other targets.
There is **no Windows branch** in bootstrap, including runtime-only mode. The experimental Windows launcher
expects repository-owned `.runtime\\node\\node.exe` and `.venv\\Scripts\\python.exe`, but the existence of
these path checks is not a supported Windows installation procedure. Core file operations also remain blocked
by the POSIX safety boundary described below; manually supplying runtimes would not resolve that limitation.

After a fresh checkout, verify the complete renderer, Electron recovery bridge, and production bundle before opening the app:

```bash
./scripts/run-project-pnpm.sh --dir frontend run check
```

If pnpm reports missing package files, remove no source files and rerun
`./scripts/run-project-pnpm.sh --dir frontend install --frozen-lockfile` from the repository root with network access.
The lockfile pins the dependency graph; a partial `node_modules/` directory is never a valid substitute for that install.

## Run

Complete the reproducible install above first. Use pnpm rather than npm so the committed lockfile stays authoritative.

### 桌面应用（macOS arm64，独立 Electron 窗口）

- 完成支持范围内的安装后，双击本目录的 `start-electron.command`。
- Windows 的 `start-electron-hidden.vbs` 和 `.\start-electron.ps1` 仅是实验启动/排障入口，
  不是可用的完整安装教程，也不承诺无闪窗或文档任务成功。

macOS 还可以安装一个不打开 Terminal 的一键启动器：

```bash
./scripts/run-project-pnpm.sh --dir frontend run launcher:mac
```

它会生成 `~/Applications/Entelecheia.app`。以后直接双击、放进 Dock，或在已经运行时按
`Command+Shift+Space` 即可唤醒主窗口。应用菜单中的“显示主窗口”使用同一快捷键；如果本地 API
意外退出，页面顶部会出现“唤醒本地服务”按钮。应用菜单也提供重新安装启动器的入口。

Windows 实验代码还包含 `install-windows-shortcut.vbs`，设计上会创建 `Entelecheia.lnk`，
并支持 `Ctrl+Shift+Space` 冷启动/唤醒与隐藏控制台。启动日志路径为
`%LOCALAPPDATA%\Entelecheia\Logs\launcher.log`。这些是待 Windows 真机验证的设计行为，
不表示已经提供可用安装或解决下面的文件安全层限制。

macOS 启动器只使用仓库 `.runtime/node`，用本地 `node_modules` 构建渲染层并启动
Electron，**不依赖全局 Node 或 pnpm**。
Electron 复用已有本地 API 前，会先向 `http://127.0.0.1:8765/api/health` 发送一次性随机挑战，
并验证监听者用本地 API token 生成的 HMAC 所有权证明；证明通过前不会向该端口发送 Bearer token。
没有后端在跑就自动从仓库 `.venv` 拉起 Python API sidecar；身份不匹配则失败关闭。
模型 Key 在应用内「设置」里填，无需命令行。

Windows 目前只有**桌面启动预览代码，不是完整链路支持**。Electron 和本地 API 有 Windows
启动路径，但未提供对应的完整锁定安装。会话工作目录访问、把上传附件复制进工作目录、Agent
写入工作区、以及工作区产物验证依赖 Unix 安全目录句柄，在 Windows 上会主动拒绝。
这需要文件安全层适配，不是补齐解释器后跑一次 smoke 就能解决。因此不能把当前版本描述为
“Windows 可正常跑完整 L1/L2 任务”。此外，当前 `pyproject.toml` 在 Windows 跳过 Python
`ripgrep` 依赖；目录浏览/搜索要求系统 `PATH` 中有 `rg.exe`，但补上它也不解除前述阻塞。
旧式 `.doc/.ppt` 的受限 LibreOffice 转换桥只实现了 macOS
sandbox；PNG/JPG 的默认本地 OCR 使用 Apple Vision。扫描 PDF 存在非 macOS 选择 RapidOCR
的设计路径，但推理依赖、模型预取和安全落盘尚未
完成 Windows 适配/验证；配置好的外部视觉模型 HTTP 接口本身不依赖 macOS。完整支持还需要
Windows 真机 smoke 和上述文件安全层的 Windows 实现。

### 浏览器开发预览（不启动 Electron 窗口）

- macOS：双击 `start-web.command`（起 Vite，打开 `http://127.0.0.1:5174/`）

浏览器渲染同样先探 `http://127.0.0.1:8765`；想用真实数据时，另在仓库根手动起
`PERSONAGRAPH_API_ALLOW_UNAUTHENTICATED_DEV_ORIGINS=1 .venv/bin/python -m personagraph.api.server`。该开关只精确接受
`http://127.0.0.1:5174` 与 `http://localhost:5174`；`null`、`file://` 和额外配置的 Origin
仍必须提供 Bearer token。Electron 正式路径使用 main-process Bearer 注入，不需要此降级。

### API server

From the repo root:

```bash
PERSONAGRAPH_API_ALLOW_UNAUTHENTICATED_DEV_ORIGINS=1 .venv/bin/entelecheia-api --port 8765
```

Or:

```bash
PERSONAGRAPH_API_ALLOW_UNAUTHENTICATED_DEV_ORIGINS=1 .venv/bin/python -m personagraph.api.server --port 8765
```

Manual API startup is optional for Electron, but useful for browser fallback testing.

## Desktop Shell

Electron provides the desktop window and a consistent Chromium renderer. A cross-platform shell does not make
the Python runtime, file-safety layer, or dependency installation cross-platform; use the support matrix below.

The desktop shell lives in `frontend/electron/`:

- `electron/main.js`: window lifecycle and desktop menu.
- `electron/preload.js`: minimal safe bridge for renderer environment info.

Security defaults:

- `nodeIntegration: false`
- `contextIsolation: true`
- `sandbox: true`

Current platform status:

- macOS arm64: the only complete locked installation target. Dependency installation, GUI startup, local model
  inference, and paid task execution are separate acceptance steps; this documentation does not certify all four.
- macOS Intel / Linux arm64 or x86_64: runtime bootstrap targets exist, without a validated full application installation.
- Windows: experimental launcher code only; bootstrap has no Windows branch, and core POSIX file operations
  need implementation work before end-to-end validation is meaningful.

## What Is Independent UI?

There are two ways to view the current frontend:

- Electron: independent desktop window. This is the intended app shell. It should start or reuse the local API sidecar and then talk to real session/project databases.
- Browser fallback: static renderer hosted by `start-web.command`. It is useful for visual checks, but if `127.0.0.1:8765` is not running it shows an offline connection state and API errors.

So seeing `离线` or API component errors in the connection panel means the renderer is not attached to the local backend. It does not mean the project only has static HTML.

## Planned API Boundary

The frontend expects a thin HTTP API shaped around these concepts:

- `GET /api/sessions`
- `GET /api/sessions/{id}`
- `POST /api/sessions`
- `PATCH /api/sessions/{id}`
- `GET/POST /api/folders` and `PATCH/DELETE /api/folders/{id}`
- `POST /api/chat/stream`
- `POST /api/chat`
- `GET /api/sessions/{id}/insession-tasks/{task_id}`
- `GET /api/sessions/{id}/session-context`
- `GET /api/sessions/{id}/session-context/explain`
- `GET /api/sessions/{id}/session-context/export`
- `POST /api/sessions/{id}/session-context/clear`
- `POST /api/sessions/{id}/session-context/corrections`
- `POST /api/sessions/{id}/session-context/repair/preview|apply`
- `GET /api/documents`
- `GET /api/documents?session_id={session_id}` for documents mounted to a workspace session
- `GET /api/documents/{id}`
- `PATCH /api/documents/{id}`
- `DELETE /api/documents/{id}`
- `GET/POST /api/document-ingest-jobs`
- `GET /api/document-ingest-jobs/{job_id}`
- `POST /api/document-ingest-jobs/{job_id}/retry`

If the API is unavailable, the renderer shows connection errors/offline status rather than silently pretending the real agent is connected.

## Manual Check

In a real API session, the right rail should show:

- `api` mode in the connection panel.
- `HTTP API`, session/project storage, and `Agent Gateway` all healthy.
- `Session DB` counts for active / archived / trashed sessions.
- Stream events after sending a message: accepted / running / final, or fallback on older APIs.
- Inline approval cards when a turn returns `pending_review.pending_writes`.
- The chat inspector shows SessionContext state, exact evidence, and export/clear controls.
- Evidence excerpts remain hidden until explicitly requested; clear requires a reason and preservation acknowledgement.
- Correction records evidence without silently changing the view; Repair shows backend before/after and remains feature-flag gated.
- Session search should find titles and historical message snippets.
- Active sessions can be archived or moved to trash; archived sessions can be unarchived or moved to trash; trashed sessions can be restored.
- Archived/trashed sessions should be read-only in the composer.
- Documents can be ingested from a path or Electron file picker, edited, searched, and deleted from the index.
- In the 工作区 tab, creating a session leaves fixed workspace assignment to the backend; the returned `working_dir` remains visible and powers the file surface. Folder management still supports create, rename, move, archive/trash/restore, and deleting only empty folders.
- A selected workspace session shows only documents mounted to it; runtime task details appear inside the chat turn that produced them.

If the connection panel is offline, the renderer is not connected to the local API. Electron should normally start that API automatically; browser fallback requires the API server to be started manually.

When the connection panel shows `api`, failed chat requests should appear as an error card in the message stream. They should not silently turn into mock assistant replies.

When a turn needs review, approve/reject from the inline card. The old drawer remains as a secondary review surface for the current turn.
