"""L2 任务执行领域。

此包边界刻意采用惰性加载：导入 :mod:`personagraph.l2` 不会同时导入任务图契约、
持久化或 Runtime 编排。调用方应显式导入所需的归属子包。
"""

__all__ = ("planning", "task_execution", "task_graph", "work_run")
