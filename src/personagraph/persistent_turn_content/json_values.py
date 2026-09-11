"""持久内容合同共享的不可变 JSON 值。

模型提案和 Host 状态都可能携带 JSON 参数。先递归冻结可以防止已校验内容在
后续步骤被原地修改，但本模块本身不负责序列化或存储。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class FrozenJsonDict(dict[str, Any]):
    """拒绝原地修改、同时保持 JSON/Pydantic 可序列化的对象。"""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen JSON objects cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def freeze_json(value: Any) -> Any:
    """返回递归不可变且兼容 JSON 的值。"""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return FrozenJsonDict(
            {key: freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


__all__ = ["FrozenJsonDict", "freeze_json"]
