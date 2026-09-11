"""SessionContext 检查与显式持久清除 API 服务。"""

from __future__ import annotations

from typing import Any

from ...configuration.features import load_features
from ...session.context import catalog as context_catalog
from ...session.context import inspection, repair
from ...session.context import reset as context_reset
from ...session.context.models import Operation, SessionDomain
from .common import require_session, required_str
from .errors import ApiError


def get_session_context(session_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    require_session(session_id)
    params = params or {}
    include_inactive = _strict_bool(params.get("include_inactive"), default=False)
    return {
        "session_context": inspection.view_session_context(
            session_id,
            include_inactive=include_inactive,
        )
    }


def explain_session_context_state(
    session_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    require_session(session_id)
    try:
        domain = SessionDomain(required_str(params, "domain"))
    except ValueError as exc:
        raise ApiError(
            "INVALID_SESSION_CONTEXT_DOMAIN",
            "domain 仅支持 user/task/interaction",
            details={"domain": params.get("domain")},
        ) from exc
    state_type = required_str(params, "state_type")
    key = required_str(params, "key")
    include_excerpt = _strict_bool(
        params.get("include_evidence_excerpt"),
        default=False,
    )
    result = inspection.explain_state(
        session_id,
        domain,
        state_type,
        key,
        include_evidence_excerpt=include_excerpt,
    )
    if result is None:
        raise ApiError(
            "SESSION_CONTEXT_STATE_NOT_FOUND",
            "当前会话中不存在该状态",
            status=404,
            details={"domain": domain.value, "state_type": state_type, "key": key},
        )
    return result


def export_session_context(
    session_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require_session(session_id)
    params = params or {}
    include_excerpt = _strict_bool(
        params.get("include_evidence_excerpt"),
        default=False,
    )
    return {
        "session_context_export": inspection.export_session_context(
            session_id,
            include_evidence_excerpt=include_excerpt,
        )
    }


def clear_session_context(
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    require_session(session_id)
    if payload.get("confirm") is not True:
        raise ApiError(
            "SESSION_CONTEXT_CLEAR_CONFIRMATION_REQUIRED",
            "清空会话状态需要显式 confirm=true",
            status=409,
            details={"session_id": session_id},
        )
    request_id = required_str(payload, "request_id")
    reason = required_str(payload, "reason")
    try:
        report = context_reset.clear_session_context(
            session_id,
            request_id=request_id,
            actor="api-user",
            reason=reason,
        )
    except context_reset.ContextResetError as exc:
        if exc.code == "request_id_collision":
            raise ApiError(
                "SESSION_CONTEXT_CLEAR_IDEMPOTENCY_CONFLICT",
                "request_id 已用于不同的清空请求",
                status=409,
                details=exc.details,
            ) from exc
        raise ApiError(
            "SESSION_CONTEXT_CLEAR_INVALID",
            "无法清空当前会话状态",
            details={"reason": exc.code, **exc.details},
        ) from exc
    return {
        "session_context_clear": report,
        "session_context": inspection.view_session_context(session_id),
    }


def create_session_context_correction(
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    require_session(session_id)
    if payload.get("confirm") is not True:
        raise ApiError(
            "SESSION_CONTEXT_CORRECTION_CONFIRMATION_REQUIRED",
            "记录用户纠正需要显式 confirm=true",
            status=409,
            details={"session_id": session_id},
        )
    slot = _required_slot(payload)
    try:
        operation = Operation(str(payload.get("operation") or Operation.SET.value))
    except ValueError as exc:
        raise ApiError(
            "INVALID_SESSION_CONTEXT_OPERATION",
            "不支持的 SessionContext correction operation",
            details={"operation": payload.get("operation")},
        ) from exc
    value_present = "value" in payload
    value = payload.get("value")
    if operation in {Operation.SET, Operation.APPEND} and not value_present:
        raise ApiError("MISSING_FIELD", "缺少必要字段", details={"field": "value"})
    try:
        record = repair.record_correction(
            session_id,
            slot[0],
            slot[1],
            slot[2],
            value,
            operation=operation,
            actor="api-user",
        )
        preview = repair.repair_session(
            session_id,
            dry_run=True,
            slot=slot,
            allow_reextract=False,
        )
    except (ValueError, repair.RepairError) as exc:
        raise _repair_api_error(exc) from exc
    return {
        "correction_evidence": {
            "id": record.id,
            "kind": record.kind.value,
            "created_at": record.created_at,
            "metadata": dict(record.metadata),
        },
        "repair": preview.to_dict(),
        "apply_enabled": _repair_apply_enabled(),
    }


def preview_session_context_repair(
    session_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require_session(session_id)
    payload = payload or {}
    if payload.get("force_reextract") is not None and payload.get("force_reextract") is not False:
        raise ApiError(
            "SESSION_CONTEXT_REEXTRACT_NOT_ALLOWED",
            "产品 Repair 暂不允许重新调用模型抽取",
            status=422,
        )
    slot = _optional_slot(payload)
    try:
        preview = repair.repair_session(
            session_id,
            dry_run=True,
            slot=slot,
            allow_reextract=False,
        )
    except (ValueError, repair.RepairError) as exc:
        raise _repair_api_error(exc) from exc
    return {
        "repair": preview.to_dict(),
        "apply_enabled": _repair_apply_enabled(),
    }


def apply_session_context_repair(
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    require_session(session_id)
    if payload.get("confirm") is not True:
        raise ApiError(
            "SESSION_CONTEXT_REPAIR_CONFIRMATION_REQUIRED",
            "应用 Repair 需要显式 confirm=true",
            status=409,
            details={"session_id": session_id},
        )
    preview_token = required_str(payload, "preview_token")
    if (
        not _repair_apply_enabled()
        and not repair.has_recorded_apply(session_id, preview_token)
    ):
        raise ApiError(
            "SESSION_CONTEXT_REPAIR_APPLY_DISABLED",
            "controlled replace 当前被 feature flag 禁用",
            status=403,
        )
    slot = _optional_slot(payload)
    try:
        result = repair.repair_session(
            session_id,
            dry_run=False,
            slot=slot,
            expected_preview_token=preview_token,
            allow_reextract=False,
            apply_actor="api-user",
        )
    except (ValueError, repair.RepairError) as exc:
        raise _repair_api_error(exc) from exc
    return {
        "repair": result.to_dict(),
        "session_context": inspection.view_session_context(session_id),
    }


def _repair_apply_enabled() -> bool:
    return bool(load_features(None).get("session_context_repair_apply_enabled", False))


def _optional_slot(payload: dict[str, Any]) -> tuple[SessionDomain, str, str] | None:
    fields = ("domain", "state_type", "key")
    present = [
        field for field in fields
        if payload.get(field) is not None and payload.get(field) != ""
    ]
    if not present:
        return None
    if len(present) != len(fields):
        raise ApiError(
            "INCOMPLETE_SESSION_CONTEXT_SLOT",
            "Repair slot 必须同时提供 domain/state_type/key",
            details={"present": present},
        )
    return _required_slot(payload)


def _required_slot(payload: dict[str, Any]) -> tuple[SessionDomain, str, str]:
    try:
        domain = SessionDomain(required_str(payload, "domain"))
    except ValueError as exc:
        raise ApiError(
            "INVALID_SESSION_CONTEXT_DOMAIN",
            "domain 仅支持 user/task/interaction",
            details={"domain": payload.get("domain")},
        ) from exc
    state_type = required_str(payload, "state_type").strip()
    key = required_str(payload, "key").strip()
    if context_catalog.policy_for(domain, state_type) is None:
        raise ApiError(
            "INVALID_SESSION_CONTEXT_TYPE",
            "state_type 不在当前 SessionContext catalog",
            details={"domain": domain.value, "state_type": state_type},
        )
    return domain, state_type, key


def _repair_api_error(exc: Exception) -> ApiError:
    reason = str(exc)
    if reason.startswith("preview_stale") or reason.startswith(
        "session_context_changed_during_preview"
    ):
        return ApiError(
            "SESSION_CONTEXT_REPAIR_PREVIEW_STALE",
            "SessionContext 已变化，请重新生成 Repair 预览",
            status=409,
            details={"reason": reason},
        )
    if reason in {"reextract_required", "reextract_not_allowed"}:
        return ApiError(
            "SESSION_CONTEXT_REEXTRACT_REQUIRED",
            "当前没有可确定性重放的 candidate/correction",
            status=422,
            details={"reason": reason},
        )
    if reason == "preview_token_required":
        return ApiError(
            "SESSION_CONTEXT_REPAIR_PREVIEW_REQUIRED",
            "应用前必须先生成 Repair 预览",
            status=409,
        )
    return ApiError(
        "SESSION_CONTEXT_REPAIR_INVALID",
        "SessionContext correction/Repair 无法执行",
        status=422,
        details={"reason": reason},
    )


def _strict_bool(value: object | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if value is True or value == "true" or value == "1":
        return True
    if value is False or value == "false" or value == "0":
        return False
    raise ApiError(
        "INVALID_BOOLEAN",
        "布尔参数仅支持 true/false/1/0",
        details={"value": value},
    )
