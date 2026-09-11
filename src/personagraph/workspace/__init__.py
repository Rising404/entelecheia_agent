"""Workspace 领域。

稳定能力由 ``binding``、``discovery`` 与 ``pictures`` 等子域分别公开；包根不聚合
具体状态或基础设施，避免普通导入启动文件扫描、数据库或后台 worker。
"""
