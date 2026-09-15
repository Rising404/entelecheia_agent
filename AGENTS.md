# Entelecheia Repository Rules

本文件统一维护 Entelecheia 的开发规则。修改前先阅读本文件、[架构职责](ARCHITECTURE.md)
及受影响模块的源码与测试。本公开副本不依赖本机历史报告或私有文档目录。

## 单一现行实现

- 每项生产职责只保留一个 canonical owner 和一条现行执行路径。内部模块、类、函数、工具、
  controller、runtime 路径及活跃配置必须使用职责名称，不用 `V1`、`V2`、`V3`、`*_v2`、
  `next` 或 `new` 表示实现代际。
- 行为升级默认原位修改 canonical 实现，同时迁移调用方与测试，并在同一批删除被替代的实现、
  注册、feature flag、alias、facade 和重复测试。Git 历史承担回滚职责，不在生产代码中长期保留
  一套旧流程作为回滚方案。
- 不得为了降低本次修改难度而复制现有实现后另起一版，也不得把“先接新路径、以后再收口”视为
  完成。现有带版本号的内部实现属于待分域清理的历史债务，不是新增代码的命名模板。
- 只有 wire/persistence schema、数据库 migration、必须回放的不可变评测或历史 artifact，以及
  第三方正式名称可以使用版本号。版本解析和兼容只停留在系统边界；进入领域层后立即归一化为
  唯一现行模型，不得据此派生第二套业务编排或存储实现。
- 若已确认的外部或持久化消费者迫使两个协议短期共存，必须先获得用户明确同意，并记录唯一
  authority、具体消费者、退出条件和删除批次；禁止双写、隐式 fallback 和向旧路径继续添加功能。
- `L0`、`L1`、`L2` 表示不同处理层级，不是实现版本；第三方名称如 `bge-reranker-v2-m3` 也不在
  本规则的改名范围内。

不要在未获用户明确授权时对现有版本化符号做全仓机械重命名；应按职责边界逐域收口，并保持每批
可验证、可提交。

## 公开副本与测试边界

- `personagraph` 是现行 Python 内部包名，不是 persona subsystem。禁止恢复已退役的角色卡、
  人格图或 persona prompt 注入；保留无角色选择的 API/前端负向回归。
- Session 的 `persona_id` 是不公开的存储兼容字段；删除列、轮换 schema 或重命名整个包属于
  独立变更，不能作为公开文件整理顺手执行。普通 UI appearance 和 Session context 不属于旧角色系统。
- 保留来源/路径授权、私有配置隔离、契约校验、调用账本、恢复/租约、错误语义与测试覆盖。
  不得通过删断言、扩大 except、放宽 fallback 或写死成功值使测试通过。
- 测试默认只使用隔离临时状态和 mock Provider；真实 GPU、网络或收费调用须显式启用，
  mock 通过不得表述为真实环境已验证。
- 依赖与模型准备使用公开安装入口；不复制其他仓库的 `.venv`、`.runtime`、缓存或状态。
- 不提交私有文档、原始评测题目/参考答案、未经清理的 prompt/轨迹、数据库、凭据或模型权重。
  用户已批准公开 DocBench 的生成回答、工具调用与失败、评分、token 和耗时；仅发布明确审阅、
  哈希锁定的逐题投影，完整私有原件仍忽略。不把公开投影冒充原样请求或可恢复状态。
  文档不能链接到缺失的本机历史报告。发布前审阅 exact file set 并运行隐私门禁。
- 只修改被分配的文件；保留并行工作树改动。提交、推送和外部发布需要明确授权。

## 开发与验证

- 依赖从 API/工具适配、应用编排流向领域契约和显式存储/Provider 端口，禁止反向依赖。
  纯校验与 I/O 分离；可变配置、存储连接和生命周期由明确的所有者管理。
- 工具 schema、API 信封、状态枚举、持久化模型及限制各有唯一来源。未知状态、Provider
  或后端应明确拒绝；schema 变更需记录支持的起始版本、迁移与身份校验，并保留现有数据。
- 按 [快速上手](QUICKSTART.md) 准备受支持的独立运行时；依赖遵守平台锁，不用临时补包
  代替正式安装。先运行受影响模块的检查，再运行相关完整套件。
- 保留 `tests/conftest.py` 的隔离、合成 fixture、子进程与冷导入检查。测试不得继承真实
  凭据、用户数据库或后台任务；报告实际执行的检查与限制，不把 mock 通过当作真实推理验收。

常用检查在仓库根执行；测试归属与定向命令见 [tests/README.md](tests/README.md)：

```bash
.venv/bin/python -m ruff check src tests scripts evals
.venv/bin/python -m pytest -q
./scripts/run-project-pnpm.sh --dir frontend run check
```

## 文档与评测维护

- `README.md` 展示项目与实验结果，`QUICKSTART.md` 维护完整使用步骤，`ARCHITECTURE.md`
  说明职责与源码入口，本文件维护开发规则。根目录 `doc/`、`docs/` 只放不发布的本地记录。
- 公开文档只链接仓库内可访问的材料。合成示例维护在 `examples/document_qa/`，生成文件
  保存在仓库外；不将私有历史报告或用户文档恢复到公开副本。
- DocBench 使用唯一的[现行执行入口](evals/docbench/reproduce_or_run_script/README.md)。
  历史实验配置、选题与来源身份保持不变；当前源码重跑应记录为新实验。分别报告回答正确率、
  执行失败、覆盖范围与官方可比性，不能相互替代。

## 本地 hooks 与发布检查

新 clone 不会继承 Git 的本地 hooks 设置。审阅 `.githooks/` 后，启用并确认：

```bash
git config --local core.hooksPath .githooks
git config --get core.hooksPath
```

第二条命令应输出 `.githooks`。pre-commit 与 pre-push 使用 PATH 中的 `python3` 执行
隐私检查；远端 CI 不能阻止私有数据先被上传。手动检查入口为：

```bash
.venv/bin/python scripts/check_repository_privacy.py --worktree
.venv/bin/python scripts/check_repository_privacy.py --staged
```

发布前审阅具体文件集合和预期提交树。已批准的评测公开投影仍受清单与哈希约束，新增或修改
内容需要重新审阅；完整私有原件保持忽略。`.gitignore` 不能移除已跟踪的秘密，门禁也不是
任意内容均无隐私问题的证明。不得通过推送未经审阅的历史来试运行远端检查。
