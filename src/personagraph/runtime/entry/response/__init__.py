"""Entry 的直接回复生成子域。

``model`` 只负责无需执行型 lane 的 L0/L2 有界回复调用。调用方须从职责明确的
子模块导入；本 package 保持冷导入且不提供兼容性 re-export。
"""

from __future__ import annotations

__all__: list[str] = []
