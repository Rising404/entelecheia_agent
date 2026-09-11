# 本机配置

这个源码目录只保留配置边界说明，不存放真实配置。应用会在源码外的用户配置目录创建：

- `app_config.json`：当前生效的 provider、model、tier 和应用设置。
- `model_profiles.json`：已保存的模型与视觉端点 profile。
- `.env`：可选的 CLI / API 进程环境覆盖；GUI 用户通常不需要它。

macOS 默认为用户目录中的 `Library/Application Support/Entelecheia/config`；其他平台
使用各自的应用配置目录。上述文件都可能含有明文 API Key，不能复制进公开仓库。
应用管理的 JSON 通过原子替换
写入，文件权限限制为当前用户读写（`0600`），Agent 工作区读写工具也会硬拒绝该目录。
目录权限应保持为 `0700`。

如需使用其他本机配置根目录，应在进程启动前把 `PERSONAGRAPH_LOCAL_CONFIG_DIR` 设置为
位于源码 checkout 之外的绝对路径；相对路径或 checkout 内路径都会被拒绝。
运行时数据库、缓存、能力探测结果、配额账本以及本地 UI→API 协调令牌位于仓库外的
Entelecheia 用户数据目录（或显式 `PERSONAGRAPH_STATE_DIR`），不会默认写入源码目录。
