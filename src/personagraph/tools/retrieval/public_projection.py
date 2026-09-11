"""模型可见 Retrieval Tool 定位器的小型公开投影边界。

定位器是 ``"page 7"`` 或 ``"Section 2.1"`` 等显示提示。它们绝不是来源权威信息：
形似本地路径、URI、父目录遍历或含控制字符字符串的定位器，必须在 Tool 结果到达模型前省略。
"""

from __future__ import annotations

import re
from typing import Any


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[/\\]")


def sanitize_public_locator(value: Any, *, maximum: int = 512) -> str | None:
    """返回有界显示定位器；不安全时返回 ``None``。

    此过程有意是有损的。定位器被拒绝时，调用方应保留证据或块本身；
    不透明候选项/块引用仍是权威公共选择器。
    """

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        return None
    normalized_path = normalized.replace("\\", "/")
    if (
        normalized.startswith(("/", "../", "~/", "file://"))
        or _WINDOWS_ABSOLUTE_PATH.match(normalized)
        or "/../" in normalized_path
        or normalized_path.endswith("/..")
        or any(ord(character) < 32 for character in normalized)
    ):
        return None
    return normalized


__all__ = ["sanitize_public_locator"]
