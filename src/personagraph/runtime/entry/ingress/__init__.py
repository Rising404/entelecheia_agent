"""Entry 的入口判定子域。

``contracts`` 定义可信输入与 Host 决策形状，``model_contracts`` 定义模型分类提案，
``policy`` 保持纯 Host 策略，``classification`` 只协调 ingress 与模型分类。调用方
应显式导入所属模块；本 package 不重导出内部 API，也不伪装已删除的旧模块路径。
"""

from __future__ import annotations

__all__: list[str] = []
