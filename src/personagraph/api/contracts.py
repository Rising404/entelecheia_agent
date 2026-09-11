"""本地 API 边界的小型无依赖 HTTP 请求契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .service.errors import ApiError


MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024
MAX_REQUEST_TARGET_BYTES = 8 * 1024
MAX_JSON_DEPTH = 24
MAX_JSON_ITEMS = 10_000


def parse_json_object(raw: bytes) -> dict[str, Any]:
    """解码一个有界 JSON 对象，不暴露解码器内部细节。

    解码后还递归限制深度和条目数量；字段的必填、附件列表和路由策略等语义不归此层。
    服务收到 dict 只是通过 wire shape 检查，尚未形成 accepted Turn。
    """
    if len(raw) > MAX_REQUEST_BODY_BYTES:
        raise ApiError(
            "REQUEST_BODY_TOO_LARGE",
            "请求体超过 1 MiB 限制",
            status=413,
            details={"max_bytes": MAX_REQUEST_BODY_BYTES},
        )
    if not raw.strip():
        return {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ApiError("INVALID_JSON", "请求体必须是 UTF-8 JSON", status=400) from exc
    try:
        import json

        value = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ApiError("INVALID_JSON", "请求体不是合法 JSON", status=400) from exc
    if not isinstance(value, dict):
        raise ApiError("INVALID_JSON", "请求体必须是 JSON object", status=400)
    _validate_json_shape(value)
    return value


def _validate_json_shape(value: Any, *, depth: int = 0, seen_items: list[int] | None = None) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ApiError("REQUEST_TOO_COMPLEX", "JSON 嵌套层级超过限制", status=400)
    if seen_items is None:
        seen_items = [0]
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return
    for child in children:
        seen_items[0] += 1
        if seen_items[0] > MAX_JSON_ITEMS:
            raise ApiError("REQUEST_TOO_COMPLEX", "JSON 项目数量超过限制", status=400)
        _validate_json_shape(child, depth=depth + 1, seen_items=seen_items)


@dataclass(frozen=True)
class BodySchema:
    """默认宽松、但会验证已声明字段的正文契约。

    仍允许未知字段，使桌面客户端与服务层可以独立演进。已声明字段只要出现就会检查；业务级
    必填/语义验证仍由拥有它的服务负责。
    """

    fields: dict[str, type | tuple[type, ...]] = field(default_factory=dict)

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """只检查已声明且非 None 的字段类型，保留原 payload 和未知字段。

        这不是完整业务 schema：缺失/None 不在此判必填，CHAT_BODY 也不验证附件列表
        元素。chat_turn 的 required_str 与 _requested_attachment_ids 继续完成准入前检查。
        """

        for name, expected in self.fields.items():
            if name not in payload or payload[name] is None:
                continue
            if not isinstance(payload[name], expected):
                expected_names = "/".join(kind.__name__ for kind in (expected if isinstance(expected, tuple) else (expected,)))
                raise ApiError(
                    "INVALID_REQUEST_FIELD",
                    f"字段 {name} 类型不正确",
                    status=400,
                    details={"field": name, "expected": expected_names},
                )
        return payload


OBJECT_BODY = BodySchema()
SESSION_BODY = BodySchema({
    "title": str, "folder_id": str, "status_action": str,
    "working_dir": str, "client_request_id": str,
})
CHAT_BODY = BodySchema({
    # attachment_ids 是此标量 schema 无法表达的列表；它在服务中验证，同时应用逐 Turn
    # 数量边界。
    "session_id": str, "client_request_id": str, "message": str, "config": str,
    "runtime_policy": dict,
})
MODEL_PROFILE_BODY = BodySchema({
    "kind": str, "name": str, "provider": str,
    "request_dialect": str, "base_url": str, "model": str, "api_key": str,
    "quota": dict,
})

# 六档模型配置（27/002 §4）。profile_id 是指针；thinking 接受布尔或
# on/off 串，reasoning_effort 接受可移植枚举。后端再按 profile 的具体请求
# 方言校验组合，避免把一个厂商的字段误发给另一个厂商。
_TIER_CONFIG_FIELDS: dict[str, type | tuple[type, ...]] = {
    key: kind
    for name in (
        "router",
        "architect",
        "attempt",
        "node_verification",
        "final_gate",
        "l1",
    )
    for key, kind in (
        (f"tier_{name}_profile_id", str),
        (f"tier_{name}_thinking", (str, bool)),
        (f"tier_{name}_reasoning_effort", str),
    )
}

CONFIG_BODY = BodySchema({
    "default_projects_dir": str,
    "provider": str, "request_dialect": str,
    "base_url": str, "model": str, "api_key": str,
    "vision_provider": str, "vision_base_url": str,
    "vision_model": str, "vision_api_key": str,
    **_TIER_CONFIG_FIELDS,
})
FOLDER_BODY = BodySchema({"name": str, "parent_id": str, "status": str})
PROJECT_BODY = BodySchema({"path": str, "name": str, "pinned": bool})
# paths 是列表，这个标量 schema 表达不了；元素类型在 service 里校验。
PROJECT_ORDER_BODY = BodySchema()
AUTONAME_BODY = BodySchema({"user_text": str, "assistant_text": str})
DOCUMENT_BODY = BodySchema({
    "session_id": str, "title": str, "summary": str, "tags": str,
})
DOCUMENT_INGEST_JOB_BODY = BodySchema({
    "job_id": str,
    "path": str,
    "session_id": str,
    "with_summary": bool,
})
DOCUMENT_INGEST_RETRY_BODY = BodySchema({"session_id": str})
SESSION_CONTEXT_CORRECTION_BODY = BodySchema({
    "confirm": bool,
    "domain": str,
    "state_type": str,
    "key": str,
    "operation": str,
})
