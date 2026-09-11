"""已执行工具参数保持原生字段，不创建额外引用表或换参协议。"""

from copy import deepcopy


def project_tool_arguments(tool_id: str, arguments: dict) -> dict:
    """凭据脱敏与长度限制由调用方在持久参数校验后统一实施。"""
    return deepcopy(arguments)
