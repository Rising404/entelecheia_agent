"""生成不含 Source 内容的 DocBench 检索配置与就绪度投影。"""

from __future__ import annotations

from collections.abc import Mapping
import os
import re
from typing import Any


SCHEMA_VERSION = "docbench-retrieval-observability-v1"
_SAFE_REASON = re.compile(r"^[A-Za-z0-9_.:,-]{1,512}$")


def retrieval_preflight_snapshot(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """解析并预检冻结 profile，不加载模型权重或访问 Source。"""

    snapshot = _empty_snapshot()
    try:
        from personagraph.retrieval.profile import (
            DocumentRetrievalProfile,
            build_document_retrieval_runtime,
            preflight_document_retrieval_profile,
        )

        profile = DocumentRetrievalProfile.from_environment(
            os.environ if environ is None else environ
        )
        capability = preflight_document_retrieval_profile(profile)
        snapshot["requested_profile"] = _profile_snapshot(profile)
        snapshot["preflight"] = capability.diagnostic_snapshot()
        snapshot["sqlite_vec"] = {
            "ready": capability.sqlite_vec_ready,
            "reason_code": (
                None if capability.sqlite_vec_ready else "sqlite_vec_unavailable"
            ),
        }
        snapshot["degradation_reasons"] = list(capability.reason_codes)
        try:
            runtime = build_document_retrieval_runtime(profile)
        except Exception as exc:
            snapshot["status"] = "unavailable"
            _append_reason(snapshot, _safe_reason(exc, "retrieval_runtime_unavailable"))
        else:
            snapshot["effective_profile"] = _profile_snapshot(
                runtime.effective_profile
            )
            snapshot["encoder_fingerprint"] = runtime.encoder.fingerprint()
            snapshot["reranker_fingerprint"] = (
                runtime.reranker.fingerprint() if runtime.reranker is not None else None
            )
            if runtime.degraded_reason:
                _append_reason(snapshot, runtime.degraded_reason)
            snapshot["status"] = (
                "degraded" if snapshot["degradation_reasons"] else "ready"
            )
    except Exception as exc:
        snapshot["status"] = "unavailable"
        _append_reason(snapshot, _safe_reason(exc, "retrieval_preflight_unavailable"))
    return snapshot


def retrieval_runtime_snapshot(
    *,
    preflight: Mapping[str, Any] | None = None,
    composition: Any | None = None,
) -> dict[str, Any]:
    """读取当前 Project 的实际 generation 与逐方法覆盖。

    Worker 收尾必须传入本例已建立的 composition。它冻结 profile、模型指纹和
    generation 配方，但 catalog / method store 查询仍在每次调用时重新执行，因此既
    不重新探测计算设备，也不会用首次快照掩盖随后完成的索引状态。省略该参数只供
    独立 readiness/诊断入口构造一次当前 composition。
    """

    if preflight:
        snapshot = dict(preflight)
    elif composition is not None:
        snapshot = _empty_snapshot()
    else:
        snapshot = retrieval_preflight_snapshot()
    snapshot["degradation_reasons"] = list(
        snapshot.get("degradation_reasons") or ()
    )
    try:
        from personagraph.retrieval.contracts import RetrievalMethod
        from personagraph.retrieval.lifecycle.generation import document_index_methods
        from personagraph.retrieval.lifecycle.rollout import (
            generation_readiness_snapshot,
        )

        if composition is None:
            from personagraph.retrieval.operations.document_maintenance import (
                build_document_retrieval_composition,
            )

            composition = build_document_retrieval_composition()
        requested_profile = getattr(composition, "requested_profile", None)
        effective_profile = getattr(composition, "effective_profile", None)
        capability = getattr(composition, "capability", None)
        degraded_reason = getattr(composition, "degraded_reason", None)
        if requested_profile is not None:
            snapshot["requested_profile"] = _profile_snapshot(requested_profile)
        if effective_profile is not None:
            snapshot["effective_profile"] = _profile_snapshot(effective_profile)
        if capability is not None:
            snapshot["preflight"] = capability.diagnostic_snapshot()
        snapshot["encoder_fingerprint"] = composition.encoder.fingerprint()
        snapshot["reranker_fingerprint"] = (
            composition.reranker.fingerprint()
            if composition.reranker is not None
            else None
        )
        if degraded_reason:
            _append_reason(snapshot, str(degraded_reason))

        catalog = composition.foundation.catalog
        sqlite_vec = catalog.capability("sqlite_vec")
        snapshot["sqlite_vec"] = _capability_snapshot(
            sqlite_vec,
            unavailable_reason="sqlite_vec_unavailable",
        )
        if not snapshot["sqlite_vec"]["ready"]:
            _append_reason(
                snapshot,
                str(snapshot["sqlite_vec"]["reason_code"]),
            )
        active = catalog.active_data_version()
        required_methods = document_index_methods(
            composition.generation_spec.index_recipe
        )
        snapshot["active_generation"] = (
            {
                "id": active.id,
                "fingerprint": active.fingerprint,
                "role": active.role.value,
                "state": active.state.value,
                "expected_fingerprint": composition.generation_spec.fingerprint,
                "matches_runtime": (
                    active.fingerprint == composition.generation_spec.fingerprint
                ),
            }
            if active is not None
            else None
        )
        if active is None:
            snapshot["methods"] = _unobserved_methods(
                RetrievalMethod,
                required_methods,
                reason_code="active_generation_unavailable",
            )
            _append_reason(snapshot, "active_generation_unavailable")
        else:
            if active.role.value != "active" or active.state.value != "ready":
                _append_reason(snapshot, "active_generation_not_ready")
            coverages = generation_readiness_snapshot(
                catalog=catalog,
                method_store=composition.foundation.method_store,
                generation_id=active.id,
                required_methods=required_methods,
            )
            snapshot["methods"] = [
                {
                    "method": coverage.method.value,
                    "required": coverage.required,
                    "ready": coverage.ready,
                    "unit_count": coverage.unit_count,
                    "manifest_ready_count": coverage.manifest_ready_count,
                    "representation_present_count": (
                        coverage.representation_present_count
                    ),
                    "manifest_absent_count": coverage.manifest_absent_count,
                    "invalid_unit_count": coverage.invalid_unit_count,
                }
                for coverage in coverages
            ]
            if active.fingerprint != composition.generation_spec.fingerprint:
                _append_reason(snapshot, "active_generation_fingerprint_mismatch")
            for coverage in coverages:
                if coverage.required and not coverage.ready:
                    _append_reason(
                        snapshot,
                        f"{coverage.method.value}_index_not_ready",
                    )
        snapshot["status"] = (
            "degraded" if snapshot["degradation_reasons"] else "ready"
        )
    except Exception as exc:
        snapshot["status"] = "unavailable"
        _append_reason(snapshot, _safe_reason(exc, "retrieval_runtime_snapshot_unavailable"))
    snapshot["indexing_outbox"] = _current_indexing_outbox_snapshot()
    return snapshot


def _empty_snapshot() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unavailable",
        "requested_profile": None,
        "effective_profile": None,
        "encoder_fingerprint": None,
        "reranker_fingerprint": None,
        "preflight": None,
        "active_generation": None,
        "methods": [],
        "indexing_outbox": {
            "available": False,
            "reason_code": "project_documents_not_bound",
            "attempt_audit_available": False,
            "attempt_count": 0,
            "attempt_audit_truncated": False,
            "status_counts": {},
            "outcome_counts": {},
            "failure_counts": {},
            "batch_count": 0,
            "worker_instance_count": 0,
            "terminal_failures": [],
            "attempt_audits": [],
        },
        "sqlite_vec": {"ready": False, "reason_code": "not_checked"},
        "degradation_reasons": [],
    }


def _profile_snapshot(profile: Any) -> dict[str, Any]:
    return {
        "mode": profile.mode.value,
        "fingerprint": profile.fingerprint(),
        "failure_policy": profile.failure_policy.value,
        "required_methods": [method.value for method in profile.retrieval_methods],
        "index_recipe": profile.index_recipe,
        "encoder_asset_identity": profile.encoder_asset.identity,
        "reranker_mode": profile.reranker_mode.value,
        "reranker_asset_identity": (
            profile.reranker_asset.identity
            if profile.reranker_mode.value != "off"
            else None
        ),
        "device": profile.device,
        "use_fp16": profile.use_fp16,
        "local_files_only": profile.local_files_only,
    }


def _capability_snapshot(
    capability: tuple[bool, str | None] | None,
    *,
    unavailable_reason: str,
) -> dict[str, Any]:
    if capability is None:
        return {"ready": False, "reason_code": unavailable_reason}
    ready, reason_code = capability
    return {
        "ready": bool(ready),
        "reason_code": None if ready else (reason_code or unavailable_reason),
    }


def _unobserved_methods(
    retrieval_method: Any,
    required_methods: tuple[Any, ...],
    *,
    reason_code: str,
) -> list[dict[str, Any]]:
    required = set(required_methods)
    return [
        {
            "method": method.value,
            "required": method in required,
            "ready": False,
            "unit_count": 0,
            "manifest_ready_count": 0,
            "representation_present_count": 0,
            "manifest_absent_count": 0,
            "invalid_unit_count": 0,
            "reason_code": reason_code,
        }
        for method in (
            retrieval_method.DENSE,
            retrieval_method.LEARNED_SPARSE,
            retrieval_method.BM25,
        )
    ]


def _append_reason(snapshot: dict[str, Any], reason: str) -> None:
    normalized = reason.strip()
    if normalized and normalized not in snapshot["degradation_reasons"]:
        snapshot["degradation_reasons"].append(normalized)


def _safe_reason(exc: BaseException, fallback: str) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and _SAFE_REASON.fullmatch(code):
        return code
    message = str(exc).strip()
    if _SAFE_REASON.fullmatch(message):
        return message
    return f"{fallback}:{type(exc).__name__}"


def _current_indexing_outbox_snapshot() -> dict[str, Any]:
    """Project 入库发生在 Turn 前；把安全审计投影进评测结果而不伪造 Turn trajectory。"""

    try:
        from personagraph.workspace.storage.context import current
        from personagraph.retrieval.operations.diagnostics import authority_outbox_snapshot

        database = current()
        if database is None:
            return _summarize_authority_outbox(
                {
                    "available": False,
                    "reason_code": "project_documents_not_bound",
                    "attempt_audit_available": False,
                    "status_counts": {},
                    "terminal_failures": [],
                    "attempt_audits": [],
                }
            )
        return _summarize_authority_outbox(
            authority_outbox_snapshot(
                authority_db_path=database.db_path,
                limit=10_000,
            )
        )
    except Exception as exc:
        return _summarize_authority_outbox(
            {
                "available": False,
                "reason_code": _safe_reason(
                    exc,
                    "indexing_outbox_snapshot_unavailable",
                ),
                "attempt_audit_available": False,
                "status_counts": {},
                "terminal_failures": [],
                "attempt_audits": [],
            }
        )


def _summarize_authority_outbox(raw: Mapping[str, Any]) -> dict[str, Any]:
    """只保留 pointer、批次和安全错误分类，绝不复制路径或 Source 正文。"""

    attempts = [
        _safe_attempt_audit(item)
        for item in raw.get("attempt_audits", ())
        if isinstance(item, Mapping)
    ]
    terminal_failures = [
        _safe_terminal_failure(item)
        for item in raw.get("terminal_failures", ())
        if isinstance(item, Mapping)
    ]
    outcome_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    batch_ids: set[str] = set()
    worker_hashes: set[str] = set()
    for attempt in attempts:
        outcome = str(attempt.get("outcome") or "unknown")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        batch_id = str(attempt.get("batch_id") or "")
        if batch_id:
            batch_ids.add(batch_id)
        worker_hash = str(attempt.get("worker_instance_hash") or "")
        if worker_hash:
            worker_hashes.add(worker_hash)
        stage = attempt.get("failure_stage")
        code = attempt.get("safe_error_code")
        if stage or code:
            key = f"{stage or 'unknown'}:{code or 'unknown'}"
            failure_counts[key] = failure_counts.get(key, 0) + 1
    attempt_count = raw.get("attempt_audit_count", len(attempts))
    try:
        normalized_attempt_count = max(0, int(attempt_count))
    except (TypeError, ValueError):
        normalized_attempt_count = len(attempts)
    return {
        "available": bool(raw.get("available")),
        "reason_code": raw.get("reason_code"),
        "attempt_audit_available": bool(raw.get("attempt_audit_available")),
        "attempt_count": normalized_attempt_count,
        "attempt_audit_truncated": bool(
            raw.get(
                "attempt_audit_truncated",
                normalized_attempt_count > len(attempts),
            )
        ),
        "status_counts": {
            str(key): int(value)
            for key, value in dict(raw.get("status_counts") or {}).items()
        },
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "batch_count": len(batch_ids),
        "worker_instance_count": len(worker_hashes),
        "terminal_failures": terminal_failures,
        "attempt_audits": attempts,
    }


def _safe_attempt_audit(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "event_id",
            "attempt",
            "worker_kind",
            "worker_instance_hash",
            "batch_id",
            "batch_limit",
            "batch_size",
            "batch_ordinal",
            "outcome",
            "failure_stage",
            "safe_error_code",
            "occurred_at",
        )
    }


def _safe_terminal_failure(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "event_id",
            "kind",
            "source_type",
            "source_unit_id",
            "source_revision",
            "indexed_content_hash",
            "data_version_id",
            "occurred_at",
            "attempts",
            "reason_code",
            "updated_at",
        )
    }


__all__ = [
    "SCHEMA_VERSION",
    "retrieval_preflight_snapshot",
    "retrieval_runtime_snapshot",
]
