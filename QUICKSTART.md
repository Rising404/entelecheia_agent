# 快速上手：从空白界面到一次文档任务

Entelecheia 是本地单用户文档 Agent 工作台，仍在开发中。本教程使用现有安装和运行入口，
不需要安装 Codex、ChatGPT 或依赖作者机器上的环境。内部 Python 包名仍为 `personagraph`。

先选择你要完成的体验；后一层不能用前一层的成功代替：

| 体验 | 需要什么 | 能检查什么 |
| --- | --- | --- |
| A：空白 GUI 预览 | 已安装依赖、新的本地配置和状态；无需模型权重或远程密钥 | 窗口、本机 API 连接、设置界面 |
| B：真实文档问答 | A 的安装、本地检索模型、自己的任务模型配置；视觉按需配置 | 文件准备、工具读取、回答与来源 |
| C：DocBench 新运行 | B 的模型准备、任务与视觉配置、授权数据及固定评分 prompt | 当前代码在指定题集上的新执行与评分 |

**开始 B/C 前先确认：**任务、视觉和评分可能调用你配置的远程服务并产生费用，必要的 prompt、
文档摘录或页面图像会外发；当前不逐次弹窗确认。Provider 密钥保存在源码外的本地明文配置中。
先确认材料可以交给该服务，不提交或录屏展示密钥、私有文档与运行记录。“本地工作台”不等于全部推理离线。

下文的“成功标志”是操作后要检查的结果，不表示这些步骤已经在你的机器通过。
文档说明与命令校核也不等于真实 GUI、GPU 或付费评测验收。

## 1. 安装一次，使用这个 checkout 自己的运行时

### 平台范围

| 平台 | 当前范围 |
| --- | --- |
| macOS arm64 | 本教程唯一的完整锁定安装目标；Python 3.12.13，Node/pnpm 版本由仓库固定 |
| macOS Intel、Linux x86_64/arm64 | 有独立运行时下载目标；`--runtime-only` 不安装应用依赖，不承诺完整应用可用 |
| Windows | 有桌面启动预览代码，没有完整自动安装/端到端支持；不能照本教程宣称可运行文档任务 |

Windows 的限制不只是缺一个安装脚本：工作目录访问、上传附件复制、Agent 写文件与产物验证
依赖 Unix 安全目录句柄，Windows 路径会主动拒绝。
OCR、旧 Office 转换和目录搜索还各有平台约束，详见 [前端平台说明](frontend/README.md)。

先在你希望保存源码的位置克隆仓库；需要已安装 Git。已有 checkout 时直接进入其根目录，不必再克隆：

```bash
git clone https://github.com/Rising404/entelecheia_agent.git
cd entelecheia_agent
```

后续命令都在仓库根目录执行，终端当前目录应包含 `pyproject.toml`、`frontend/` 和 `scripts/`：

```bash
./scripts/bootstrap-local-runtime.sh
```

安装需要网络，shell 需要 `curl`、`tar` 和 `shasum` 或 `sha256sum`。
脚本在当前 checkout 创建 `.runtime/`、`.venv/`，检查运行时版本与归属，按 Python hash lock 与 pnpm lock 安装依赖。
不要复制其他仓库的虚拟环境、`node_modules` 或模型缓存，也不要用系统 Python 临时补包。

**成功标志：**bootstrap 正常结束，内置版本/归属检查和依赖安装均未报错。
安装完成不证明模型、文档或 Provider 已经可用。

当前完整锁包含 torch、FlagEmbedding、Transformers 和 Docling/OCR 等较重依赖；
即使只做界面预览，也没有另一份受支持的 GUI-only 安装锁。
`--runtime-only` 不是轻量应用安装。模型权重不包含在 bootstrap 中。
安装闭包、唯一源码构建例外与维护规则见 [依赖约束](scripts/DEPENDENCIES.md)。

## 2. A：不下载模型，先看空白 GUI

这里的“无 API”指**无远程模型调用**；界面仍会访问本机 Python API。
先退出已运行的 Entelecheia 和旧版本应用。在干净 shell 中，选择从未使用过的仓库外目录，
把下面三个绝对路径替换成自己的路径，再启动：

```bash
PERSONAGRAPH_MODEL_PROVIDER=mock \
PERSONAGRAPH_DOCUMENT_ENGINE=native \
PERSONAGRAPH_LOCAL_CONFIG_DIR="/absolute/path/to/entelecheia-preview/config" \
PERSONAGRAPH_STATE_DIR="/absolute/path/to/entelecheia-preview/state" \
PERSONAGRAPH_DEFAULT_PROJECTS_DIR="/absolute/path/to/entelecheia-preview/projects" \
./frontend/start-electron.command
```

这使用同一个正式 GUI 和本机 sidecar，不是另一套演示后端。全新配置默认使用 mock；
上面的进程级设置进一步明确此次预览不使用真实任务 Provider。

**成功标志：**独立窗口打开，能查看“任务模型配置”“视觉模型配置”等设置，且本机 API 连接可用。
如果界面显示离线或 API 错误，这一步尚未完成，不能把它当作正常的 mock 状态。

此层只预览空白界面：不要上传文档、启用真实配置或把占位回复当作问答结果。
默认 strict 检索和历史处理仍可能需要 BGE 模型；没有模型时，不承诺新建会话、聊天、入库及
交付后处理的整个流程都能完成。

不能直接复用已有用户状态来保证零外发：API 启动会发现旧 Project 维护、未完成 Turn 和
post_commit 工作，并尝试恢复。新的配置/状态目录、干净环境和退出旧应用是预览边界的一部分。

## 3. B：准备一次真实 native 文档问答

### 3.1 检查并准备本地检索模型

先做完全本地的资源检查：

```bash
.venv/bin/python scripts/prepare-local-models.py check
```

**成功标志：**输出的 `status` 为 `ready`，退出码为 0。
若输出 `not_ready` 并退出 1，先读原因；新环境缺权重是正常的待准备状态，不要绕过 strict 检查。
这个命令不会下载、加载权重做推理，也不会调用任务 Provider。

确认愿意下载后，显式准备两个固定 revision 的检索模型：

```bash
.venv/bin/python scripts/prepare-local-models.py download
.venv/bin/python scripts/prepare-local-models.py check
```

这是 Hugging Face 公共模型传输，需网络和约 4–5 GB 模型空间；不是付费任务 API 调用。
它不下载 Docling 布局/OCR 模型。若使用 `HF_HOME` 或 `HF_HUB_CACHE` 自定义缓存，
准备、GUI 和评测必须在启动前使用相同设置，否则一个进程能找到的权重另一个可能找不到。

资源齐备后，可另做真正的本地检索推理检查：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m personagraph.retrieval.operations.real_smoke
```

**成功标志：**三种检索方法均可用，且输出 `reranker_relevant_first=true`。
该检查会实际使用本地模型/设备，但不调用任务 Provider，也不证明文档问答正确。
模型资源存在、设备推理成功、任务回答正确是三个不同验收结果。

### 3.2 打开 native 路径，配置自己的模型

退出预览应用，使用新的正常启动进程：

```bash
PERSONAGRAPH_DOCUMENT_ENGINE=native ./frontend/start-electron.command
```

`native` 明确选择原生文档读取。基础文本型 PDF、DOCX、PPTX 或文本任务不要求 Docling 权重；
它也不保证扫描页、复杂表格或图表都能靠文字层解决。默认完整检索仍使用前一步准备的 BGE 模型。

在“设置 → 任务模型配置”创建自己的配置，填写名称、Provider、请求方言、Base URL、Model 和 API Key。
任务 Provider 支持界面列出的 OpenAI 兼容、Anthropic 兼容协议；方言需与实际服务接口一致。
第一份配置会自动启用；已有多份时，启用你准备使用的那一份。
不要把文档里的示例模型名或端点当成已提供的免费服务。

**成功标志：**该配置显示为活动配置，当前模型状态不再是 mock。
保存配置只证明本地设置已接受，不验证远端密钥、额度或模型能力；真实任务才会暴露这类问题。

若任务需要图像理解，再在“视觉模型配置”创建并启用独立配置。
任务模型可以只处理文本；没配置视觉是受支持状态，但视觉工具会报告 unavailable，而不是自动获得视觉能力。

**视觉 URL 特别注意：**当前视觉适配器在 Base URL 后固定追加 `/v1/chat/completions`。
应填写对应服务根地址，不要再带 `/v1` 或完整请求路径；例如根地址为 `https://vision.example.com`，
实际路径会是 `https://vision.example.com/v1/chat/completions`。此示例不是可用服务。
视觉 Provider 字段是标识，不会把这条适配器自动切换成任意厂商的其他协议。

配置可能被环境覆盖：应用的非空环境变量优先，其次是用户配置目录 `.env`，再是 GUI 保存设置。
如果以前 `export` 过 mock、端点或模型参数，先撤掉相应覆盖再启动；本教程 A 的进程级赋值
本身不会在命令退出后变成 shell 的永久配置。仓库根 `.env` 不作为运行配置加载，
留空环境值也不会清除 GUI 中已有的配置。

### 3.3 生成自己的小样例，再提问

先读 [合成文档样例](examples/document_qa/README.md)。它不是 DocBench 原题或第三方 PDF。
用一个仓库外、尚未使用的输出文件路径生成样例：

```bash
.venv/bin/python examples/document_qa/generate_sample.py \
  --output /absolute/path/outside/repo/harbor_handbook.pdf
```

**成功标志：**命令完成，指定路径出现可打开的合成 PDF；内容与样例说明一致。
在 GUI 中新建 Session，选择仓库外工作目录，上传该 PDF，再使用样例说明中的任务。
不要把仓库、模型缓存、配置目录或 benchmark 的 QA/评分目录当成任务工作目录。

**成功标志：**文件被当前会话接收；任务过程能看到真实工具执行；最终回答可与样例文档及其
来源位置核对。上传成功不等于文件全文已经读完，有回复也不等于内容正确。
文件准备失败、工具报错、占位回复或窗口仍在等待处理，都应作为未完成检查记录，而不是忽略。
任务与证据核对点由样例说明维护，本教程不另外保存一套“标准答案”。

### 3.4 使用真实文档之前

Provider 密钥保存在源码外的本地明文配置中。不要录屏展示密钥，不要提交配置、用户文档、
对话、轨迹、数据库或模型缓存。默认 macOS 数据位于用户的 `Library/Application Support/Entelecheia/`，
默认 Project 位于用户的 `Documents/Entelecheia/`；可用本教程中的外部路径设置隔离。

任务 prompt、必要文本摘录和页面图像可能发送给你配置的远程服务商；当前外发默认同意，
不逐次弹窗。先确认文档可以交给该服务，再进行真实问答。所谓“本地工作台”不等于模型推理全部本地。

## 4. C：运行当前代码的 DocBench，不冒充旧成绩复现

完整数据准备、恢复、重试和评分规则以 [现行 runbook](evals/docbench/docs/formal_l1_eval_runbook.md)
为准。下面只展示已有入口的最短运行路径；不要跳过 runbook 的材料和授权检查。

在 Python 启动前设置自己的外部评测根，并明确 native 文档路径：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
export PERSONAGRAPH_DOCUMENT_ENGINE=native
.venv/bin/python -m evals.docbench.reproduce_or_run_script validate
.venv/bin/python -m evals.docbench.reproduce_or_run_script list
.venv/bin/python -m evals.docbench.reproduce_or_run_script readiness
```

**成功标志与限制：**

- `validate` 正常退出、`list` 给出配置及哈希：只证明配置契约，不检查材料与 Provider 连通性。
- `readiness` 输出 `status=ready`：进一步核对数据/选集哈希、评分 prompt、凭据存在和本地检索资源。
  它不调用 Provider；全目录中任何回归配置缺材料都可能导致整体失败，没有 `readiness --config` 参数。
- Provider 解析失败时，readiness 无法取得完整 runner 环境，检索检查也会报告不可用；
  想单独检查模型文件，使用前面的 `prepare-local-models.py check`。

DocBench 还有两个不同于普通 GUI 文本任务的前提：

- 当前 runner 即使运行最小单题，也要求**任务和视觉两套完整配置与密钥**，不能因为题目看起来是文本就省略视觉配置。
- 现行 YAML 的 `source: installation` 优先读取 GUI 已启用 profile，未找到时读取安装配置文件。
  不要认为只设置常规 `PERSONAGRAPH_API_KEY` 等环境变量就一定满足该配置。评测与 GUI 应使用同一个
  `PERSONAGRAPH_LOCAL_CONFIG_DIR`，或按 runbook 配置独立的外部评测设置。

数据、QA、固定 `evaluation_prompt.txt` 不随仓库提供。`prepare-data` 的现有选项只覆盖 QA catalog
或 balanced125 PDF 子集，不会准备所有回归 PDF、评分 prompt、upstream checkout 或模型；
没有 `prepare-data --config`。最小单题使用 doc 104，包含在 balanced125 PDF 集合中。

材料和配置齐备、确认愿意进行可能收费的调用后，执行 CPU 单题，再单独评分：

```bash
.venv/bin/python -m evals.docbench.reproduce_or_run_script run \
  --config evals/docbench/configs/l1_bge_m3_live_1.yaml \
  --run-id my-l1-smoke --allow-live

.venv/bin/python -m evals.docbench.reproduce_or_run_script score \
  --config evals/docbench/configs/l1_bge_m3_live_1.yaml \
  --run "$PERSONAGRAPH_BENCH_EVAL_DIR/docbench/runs/my-l1-smoke" \
  --resume --allow-live
```

为每次新实验换一个未使用的 run ID。不要把 `--resume` 当成覆盖失败结果或绕过来源变化检查。
**成功标志：**run 完成且生成门通过；score 单独完成并给出评分报告。执行、来源稳定性、
交付后处理和答案正确率仍需分别看；单题 smoke 不能代表总体准确率。
输出保存在外部根的 `docbench/runs/my-l1-smoke/`，不能直接复制进 Git。
`--allow-live` 允许真实调用与必要内容外发；“两条命令”不意味着准备工作只有两步或耗时有保证。

### 哪一种“复现”是你要做的？

| 目标 | 正确入口与边界 |
| --- | --- |
| 复算公开历史统计 | 阅读 [公开摘要与方法](evals/docbench/previous_results/README.md)，核对逐题分数与汇总算术；不调用裁判，也不能重新判定原答案真假 |
| 对历史答案重新评分 | 需要仓库外原始生成结果、QA、固定 prompt 和匹配身份；`score` 仍要求 `--allow-live`，缺 checkpoint 时可能产生新裁判调用 |
| 同题重跑当前代码 | 使用现行 runbook、新 run ID 和当前 source/config 身份，报告为新实验 |
| 精确复现旧冻结评测 | 还需旧冻结源码、配置、环境及授权数据；只有同一题集或 Git commit 不够，当前源码不能替代旧脏工作树 |

历史 125 题首跑加 8 次定向补跑不是 133 道独立问题，也不是迁移后当前版本的成绩。
`score --resume` 可复用匹配的完成 checkpoint，但不是一个承诺零网络的统计复算开关。

## 5. 卡住时先检查这里

| 现象 | 先检查什么 |
| --- | --- |
| 启动器找不到 Python/Node/前端依赖 | 回到唯一 bootstrap；不要改为系统解释器或复制旧环境 |
| 窗口显示离线或后端身份不匹配 | 退出旧应用，确认本机 8765 端口不是旧后端；不要关闭 token/所有权校验来接未知进程 |
| GUI 已改模型，但仍显示 mock 或旧端点 | 检查非空环境变量、用户配置目录 `.env`、活动 profile 及使用的配置目录 |
| 缺模型或 strict 检索失败 | 先看模型 check，再核对 HF 缓存是否一致；不要用关闭重排/静默降级掩盖缺资源 |
| 视觉 unavailable 或路径错误 | 检查独立视觉 profile 是否完整启用，Base URL 是否重复带 `/v1` |
| readiness 不通过 | 阅读各配置的失败项；可能是别的回归集缺材料，不等于当前 Provider 已实测故障 |
| 有回答，但任务仍异常或内容不对 | 分别检查工具证据、最终交付和 post_commit/退出状态；不要把有正文当成全链路完成 |

需要修改代码时再读 [架构](ARCHITECTURE.md) 和 [开发规则](AGENTS.md)。
根目录 `doc/`、`docs/` 是不发布的本地开发记录区，不用它们保存需要公开的教程或示例。
