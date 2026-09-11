"""L0/L1/L2 共用的 Turn 运行时间策略。

本模块只保存不依赖 Entry、模型或 Store 的冷常量。具体超时覆盖值仍由冻结后的
feature snapshot 提供；这里的 Window heartbeat TTL 用于所有恢复者判断租约新鲜度。
"""

from __future__ import annotations


# 该租约必须覆盖一次 60 秒物理模型请求及 30 秒调度/持久化余量。它与可配置的
# 整体 Turn deadline 是两个独立概念。
ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S = 90.0


__all__ = ["ACTIVE_TURN_WINDOW_HEARTBEAT_TTL_S"]
