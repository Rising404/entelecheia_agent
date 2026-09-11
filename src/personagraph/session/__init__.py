"""会话层：会话/轮次持久化与管理操作。

- store           会话与轮次的 SQLite 存储（原 session_store）
- persistence     store facade 背后的 SQLite schema、dependency port 与 record CRUD（内部实现）
- service         管理操作：改名/移动/归档/回收站/导出/文件夹（原 session_service）
- context/        SessionContext 的类型、目录、store、evidence、抽取、repair 与 inspection
                 （内部按 context.models/catalog/store/... 拆分，避免和普通 session store 混淆）
"""

__all__ = [
    "context",
    "persistence",
    "service",
    "session_summary",
    "store",
]
