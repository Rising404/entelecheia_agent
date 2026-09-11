"""L1 持久请求使用的规范 JSON 和摘要。"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


def canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: object) -> str:
    return _sha256_text(canonical_json(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "sha256_json"]
