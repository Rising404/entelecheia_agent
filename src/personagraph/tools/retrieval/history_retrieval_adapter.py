"""``retrieve_history`` 的 Host 冻结 Tool adapter 与公共结果投影。

模型可见契约位于 :mod:`personagraph.tools.retrieval.retrieval_tools`，版本中立的数据面契约位于
:mod:`personagraph.retrieval.tooling.contracts`。本模块只绑定冻结 History 范围并投影公开结果。
每个模型字段只能缩小不可变 Host 范围：

* History 范围名称映射到 Host 固定的当前 Session/用户/Task 标识；
* 检索世代及来源截止点绝不是模型参数；
* 公共证据从允许列表重建，绝不原样转发端口引用元数据。

本模块自身没有 L1/L2 lane 组合副作用；各组合根按需绑定返回的注册，而不改变此契约。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ...retrieval.contracts import (
    SourceType,
)
from ...retrieval.sources.identity import parse_current_session_source_unit_id
from ...retrieval.tooling.contracts import (
    FrozenHistoryRetrievalScope,
    FrozenHistorySourceSnapshot,
    HistoryScope,
    MAX_RETRIEVAL_QUERY_CHARS,
    RetrievalCorpus,
    RetrievalStatus,
    RetrievalToolEvidence,
    RetrievalToolPort,
    RetrievalToolRequest,
    RetrievalToolResult,
    parse_current_session_cutoff,
)
from ..effects import EffectAction, EffectResource, EffectScopeKind
from ..execution import ToolBusinessFailure
from ..policy import AuthorityFacts, ScopeGrant
from ..registration import ToolRegistration
from .retrieval_tools import (
    HistoryRetrievalScope,
    MAX_EVIDENCE_TEXT_CHARS,
    RetrievalCorpus as PublicRetrievalCorpus,
    RetrievalEvidenceCoverage,
    RetrievalEvidenceEnvelope,
    RetrievalEvidenceGap,
    RetrievalEvidenceLocator,
    RetrievalEvidenceOrigin,
    RetrievalEvidenceOutcome,
    RetrievalEvidenceSource,
    RetrievalEvidenceStatus,
    RetrievalEvidence,
    build_retrieve_history_registration,
    opaque_fingerprint,
)
DEFAULT_RETRIEVAL_LIMIT = 8

_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_CANONICAL_TURN_INDEX = re.compile(r"(?:0|[1-9][0-9]{0,17})\Z")


@dataclass(frozen=True, slots=True)
class HistoryRetrievalRuntime:
    """注册及精确效果授予；没有通道组合副作用。"""

    registrations: tuple[ToolRegistration, ...]
    authority: AuthorityFacts


def build_history_retrieval_tool_registration(
    scope: FrozenHistoryRetrievalScope,
    *,
    port: RetrievalToolPort,
    source_fingerprint: str | None = None,
) -> ToolRegistration:
    """将 ``retrieve_history`` 绑定到一个冻结 History/记忆范围。"""

    if not isinstance(scope, FrozenHistoryRetrievalScope):
        raise ValueError("scope must be FrozenHistoryRetrievalScope")

    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        query, limit = _base_payload(
            payload,
            maximum_limit=scope.max_items,
            narrowing_field="scopes",
        )
        selected = _select_history_scopes(scope, payload.get("scopes"))
        request = RetrievalToolRequest(
            request_id=_request_id(
                scope.scope_snapshot_id,
                query,
                tuple(item.value for item in selected),
                limit,
            ),
            corpus=RetrievalCorpus.HISTORY,
            query=query,
            limit=limit,
            context_token_limit=scope.context_token_limit,
            session_id=scope.session_id,
            scope_snapshot_id=scope.scope_snapshot_id,
            retrieval_data_version=scope.retrieval_data_version,
            history_scopes=selected,
            current_session_cutoff=(
                scope.current_session_cutoff
                if HistoryScope.CURRENT_SESSION in selected
                else None
            ),
            long_term_task_id=(
                scope.long_term_task_id
                if HistoryScope.CURRENT_TASK in selected
                else None
            ),
            history_source_snapshots=tuple(
                item
                for item in scope.long_term_source_snapshots
                if (
                    item.source_type is SourceType.LONG_TERM_USER
                    and HistoryScope.LONG_TERM_USER in selected
                )
                or (
                    item.source_type is SourceType.LONG_TERM_TASK
                    and HistoryScope.CURRENT_TASK in selected
                )
            ),
        )
        result, deferred_audit = _invoke_history_port(port, request)
        try:
            projected = _project_result(request, result).to_dict()
        except Exception:
            if deferred_audit is not None:
                deferred_audit.record(
                    final_projection={
                        "status": "blocked",
                        "evidence": [],
                        "gaps": [
                            {
                                "code": "history_projection_exception",
                                "blocking": True,
                            }
                        ],
                    },
                    diagnostics=(
                        {
                            "stage": "public_projection",
                            "code": "history_projection_exception",
                            "status": "failed",
                            "blocking": True,
                            "known_count": 1,
                        },
                    ),
                )
            raise
        if deferred_audit is not None:
            deferred_audit.record(final_projection=projected)
        return projected

    return build_retrieve_history_registration(
        handler=handler,
        effect_scope=scope.effect_scope,
        source_fingerprint=source_fingerprint,
    )


def build_history_retrieval_runtime(
    *,
    scope: FrozenHistoryRetrievalScope,
    port: RetrievalToolPort,
    source_fingerprint: str | None = None,
) -> HistoryRetrievalRuntime:
    """构建一个冻结的 History 工具注册及其精确授权。"""

    registration = build_history_retrieval_tool_registration(
        scope,
        port=port,
        source_fingerprint=source_fingerprint,
    )
    grant = ScopeGrant(
        EffectResource.MEMORY,
        EffectAction.SEARCH,
        EffectScopeKind.SESSION,
        scope.effect_scope,
    )
    return HistoryRetrievalRuntime(
        registrations=(registration,),
        authority=AuthorityFacts(grants=(grant,)),
    )


def _invoke_port(
    port: RetrievalToolPort,
    request: RetrievalToolRequest,
) -> RetrievalToolResult:
    try:
        result = port.retrieve(request)
    except ToolBusinessFailure:
        raise
    except Exception as exc:
        raise ToolBusinessFailure(
            "retrieval_backend_unavailable",
            "The authorized retrieval corpus could not complete this request.",
        ) from exc
    if not isinstance(result, RetrievalToolResult):
        raise ToolBusinessFailure(
            "retrieval_backend_contract_violation",
            "The retrieval backend returned an invalid result contract.",
        )
    return result


def _invoke_history_port(
    port: RetrievalToolPort,
    request: RetrievalToolRequest,
) -> tuple[RetrievalToolResult, Any | None]:
    """如实现提供延迟审计扩展则使用，否则调用基础 ``retrieve`` port。"""

    deferred_retrieve = getattr(port, "retrieve_history_unrecorded", None)
    if not callable(deferred_retrieve):
        return _invoke_port(port, request), None
    try:
        result, deferred_audit = deferred_retrieve(request)
    except ToolBusinessFailure:
        raise
    except Exception as exc:
        raise ToolBusinessFailure(
            "retrieval_backend_unavailable",
            "The authorized retrieval corpus could not complete this request.",
        ) from exc
    if not callable(getattr(deferred_audit, "record", None)):
        raise ToolBusinessFailure(
            "retrieval_backend_contract_violation",
            "The retrieval backend returned an invalid deferred audit handle.",
        )
    if not isinstance(result, RetrievalToolResult):
        deferred_audit.record(
            final_projection={
                "status": "blocked",
                "evidence": [],
                "gaps": [
                    {
                        "code": "retrieval_backend_contract_violation",
                        "blocking": True,
                    }
                ],
            },
            diagnostics=(
                {
                    "stage": "dataplane",
                    "code": "retrieval_backend_contract_violation",
                    "status": "failed",
                    "blocking": True,
                    "known_count": 1,
                },
            ),
        )
        raise ToolBusinessFailure(
            "retrieval_backend_contract_violation",
            "The retrieval backend returned an invalid result contract.",
        )
    return result, deferred_audit


def _project_result(
    request: RetrievalToolRequest,
    result: RetrievalToolResult,
) -> RetrievalEvidenceEnvelope:
    if request.corpus is not RetrievalCorpus.HISTORY:
        raise ToolBusinessFailure(
            "retrieval_backend_contract_violation",
            "History projection received a non-History request.",
        )
    if not result.scope_is_current or result.scope_snapshot_id != request.scope_snapshot_id:
        return _blocked_envelope(request, "scope_snapshot_stale")
    if result.retrieval_data_version != request.retrieval_data_version:
        return _blocked_envelope(request, "retrieval_generation_stale")

    allowed_source_types = _allowed_source_types(request)
    projected: list[RetrievalEvidence] = []
    gaps: list[RetrievalEvidenceGap] = []
    projection_truncated = False
    seen_handles: set[str] = set()

    for raw in result.evidence:
        if (
            raw.source_type not in allowed_source_types
            or not _history_evidence_is_authorized(request, raw)
        ):
            return _blocked_envelope(request, "retrieval_authority_mismatch")
        locator = _project_public_history_locator(raw)
        content = raw.content.strip()
        if len(content) > MAX_EVIDENCE_TEXT_CHARS:
            content = content[:MAX_EVIDENCE_TEXT_CHARS].rstrip()
            projection_truncated = True
            gaps.append(
                RetrievalEvidenceGap(
                    code="evidence_text_truncated",
                    blocking=False,
                    source_type=_public_source_type(raw.source_type),
                )
            )
        handle = _history_evidence_handle(raw)
        if handle in seen_handles:
            continue
        seen_handles.add(handle)
        projected.append(
            RetrievalEvidence(
                handle=handle,
                source_type=_public_source_type(raw.source_type),
                origin=_public_origin(raw),
                text=content,
                content_sha256=raw.indexed_content_hash,
                source_revision_fingerprint=opaque_fingerprint(
                    raw.source_revision
                ),
                rank=raw.rank,
                estimated_tokens=raw.estimated_tokens,
                locator=locator,
            )
        )

    projected.sort(key=lambda item: (item.rank, item.handle))
    if len(projected) > request.limit:
        dropped = len(projected) - request.limit
        projected = projected[: request.limit]
        projection_truncated = True
        gaps.append(
            RetrievalEvidenceGap(
                code="port_item_limit_exceeded",
                blocking=False,
                known_count=dropped,
            )
        )

    for raw_gap in result.gaps:
        if raw_gap.source_type is not None and raw_gap.source_type not in allowed_source_types:
            return _blocked_envelope(request, "retrieval_authority_mismatch")
        gaps.append(
            RetrievalEvidenceGap(
                code=raw_gap.code,
                blocking=raw_gap.blocking,
                source_type=(
                    _public_source_type(raw_gap.source_type)
                    if raw_gap.source_type is not None
                    else None
                ),
                known_count=raw_gap.known_count,
            )
        )

    if result.status is RetrievalStatus.BLOCKED:
        if not gaps:
            gaps.append(
                RetrievalEvidenceGap(
                    code="retrieval_blocked",
                    blocking=True,
                )
            )
        projected = []
    elif result.status is RetrievalStatus.PARTIAL and not (
        gaps or result.truncated or projection_truncated
    ):
        gaps.append(
            RetrievalEvidenceGap(
                code="retrieval_partial",
                blocking=False,
            )
        )
    elif result.status is RetrievalStatus.COMPLETE and result.gaps:
        return _blocked_envelope(request, "retrieval_backend_contract_violation")

    gaps = _deduplicate_gaps(gaps)
    truncated = bool(result.truncated or projection_truncated)
    status = _public_status(
        port_status=result.status,
        evidence=projected,
        gaps=gaps,
        truncated=truncated,
    )
    return RetrievalEvidenceEnvelope(
        corpus=PublicRetrievalCorpus.HISTORY,
        status=status,
        outcome=_public_outcome(status=status, evidence=projected),
        query=request.query,
        evidence=tuple(projected),
        gaps=tuple(gaps),
        coverage=_coverage(
            request,
            returned_items=len(projected),
            configured_token_limit=min(
                request.context_token_limit,
                result.configured_token_limit,
            ),
            packed_tokens=min(
                result.packed_tokens,
                request.context_token_limit,
                result.configured_token_limit,
            ),
            encoder_fingerprint=(
                opaque_fingerprint(result.encoder_fingerprint)
                if result.encoder_fingerprint
                else None
            ),
            reranker_fingerprint=(
                opaque_fingerprint(result.reranker_fingerprint)
                if result.reranker_fingerprint
                else None
            ),
        ),
        truncated=truncated,
    )


def _blocked_envelope(
    request: RetrievalToolRequest,
    code: str,
) -> RetrievalEvidenceEnvelope:
    return RetrievalEvidenceEnvelope(
        corpus=PublicRetrievalCorpus.HISTORY,
        status=RetrievalEvidenceStatus.BLOCKED,
        outcome=RetrievalEvidenceOutcome.NOT_ESTABLISHED,
        query=request.query,
        evidence=(),
        gaps=(RetrievalEvidenceGap(code=code, blocking=True),),
        coverage=_coverage(
            request,
            returned_items=0,
            configured_token_limit=request.context_token_limit,
            packed_tokens=0,
        ),
        truncated=False,
    )


def _coverage(
    request: RetrievalToolRequest,
    *,
    returned_items: int,
    configured_token_limit: int,
    packed_tokens: int,
    encoder_fingerprint: str | None = None,
    reranker_fingerprint: str | None = None,
) -> RetrievalEvidenceCoverage:
    return RetrievalEvidenceCoverage(
        scope_fingerprint=opaque_fingerprint(request.scope_snapshot_id),
        retrieval_generation_fingerprint=opaque_fingerprint(
            request.retrieval_data_version
        ),
        requested_aliases=(),
        requested_scopes=tuple(
            HistoryRetrievalScope(item.value)
            for item in request.history_scopes
        ),
        returned_items=returned_items,
        configured_token_limit=configured_token_limit,
        packed_tokens=packed_tokens,
        encoder_fingerprint=encoder_fingerprint,
        reranker_fingerprint=reranker_fingerprint,
    )


def _base_payload(
    payload: Mapping[str, Any],
    *,
    maximum_limit: int,
    narrowing_field: str,
) -> tuple[str, int]:
    if not isinstance(payload, Mapping):
        raise ToolBusinessFailure("invalid_request", "Tool input must be an object.")
    unknown = set(payload) - {"query", narrowing_field, "limit"}
    if unknown:
        raise ToolBusinessFailure(
            "invalid_request",
            "Retrieval input contains unsupported fields.",
        )
    try:
        query = _query(payload.get("query"))
    except ValueError as exc:
        raise ToolBusinessFailure("invalid_request", str(exc)) from exc
    raw_limit = payload.get("limit", DEFAULT_RETRIEVAL_LIMIT)
    if (
        isinstance(raw_limit, bool)
        or not isinstance(raw_limit, int)
        or not 1 <= raw_limit <= maximum_limit
    ):
        raise ToolBusinessFailure(
            "invalid_request",
            f"limit must be an integer from 1 to {maximum_limit}",
        )
    return query, raw_limit


def _select_history_scopes(
    scope: FrozenHistoryRetrievalScope,
    raw_scopes: Any,
) -> tuple[HistoryScope, ...]:
    if raw_scopes is None:
        return scope.allowed_scopes
    if (
        not isinstance(raw_scopes, list)
        or not raw_scopes
        or len(raw_scopes) != len(set(map(str, raw_scopes)))
    ):
        raise ToolBusinessFailure(
            "invalid_request",
            "scopes must be a non-empty unique list.",
        )
    try:
        selected_set = {HistoryScope(value) for value in raw_scopes}
    except (TypeError, ValueError) as exc:
        raise ToolBusinessFailure(
            "history_scope_outside_frozen_scope",
            "A requested history scope is not available to this tool.",
        ) from exc
    if not selected_set.issubset(scope.allowed_scopes):
        raise ToolBusinessFailure(
            "history_scope_outside_frozen_scope",
            "A requested history scope is outside the Host-frozen scope.",
        )
    return tuple(item for item in scope.allowed_scopes if item in selected_set)


def _allowed_source_types(
    request: RetrievalToolRequest,
) -> frozenset[SourceType]:
    mapping = {
        HistoryScope.CURRENT_SESSION: SourceType.CURRENT_SESSION,
        HistoryScope.LONG_TERM_USER: SourceType.LONG_TERM_USER,
        HistoryScope.CURRENT_TASK: SourceType.LONG_TERM_TASK,
    }
    return frozenset(mapping[scope] for scope in request.history_scopes)


def _history_evidence_is_authorized(
    request: RetrievalToolRequest,
    evidence: RetrievalToolEvidence,
) -> bool:
    if evidence.source_type is SourceType.CURRENT_SESSION:
        if request.current_session_cutoff is None:
            return False
        identity = parse_current_session_source_unit_id(evidence.source_unit_id)
        assistant_turn_idx = _optional_nonnegative_int(
            evidence.citation.get("assistant_turn_idx")
        )
        cutoff = parse_current_session_cutoff(request.current_session_cutoff)
        return (
            identity is not None
            and identity.session_id == request.session_id
            and evidence.citation.get("session_id") == request.session_id
            and assistant_turn_idx is not None
            and assistant_turn_idx <= cutoff
        )

    snapshot = next(
        (
            item
            for item in request.history_source_snapshots
            if item.source_type is evidence.source_type
        ),
        None,
    )
    if snapshot is None or not any(
        binding.source_unit_id == evidence.source_unit_id
        and binding.source_revision == evidence.source_revision
        and binding.indexed_content_hash == evidence.indexed_content_hash
        for binding in snapshot.bindings
    ):
        return False
    if evidence.source_type is SourceType.LONG_TERM_TASK:
        return (
            request.long_term_task_id is not None
            and evidence.citation.get("task_id") == request.long_term_task_id
        )
    return evidence.source_type is SourceType.LONG_TERM_USER


def _public_source_type(source_type: SourceType) -> RetrievalEvidenceSource:
    return RetrievalEvidenceSource(source_type.value)


def _public_origin(
    evidence: RetrievalToolEvidence,
) -> RetrievalEvidenceOrigin:
    return {
        SourceType.CURRENT_SESSION: RetrievalEvidenceOrigin.CURRENT_SESSION,
        SourceType.LONG_TERM_USER: RetrievalEvidenceOrigin.LONG_TERM_USER,
        SourceType.LONG_TERM_TASK: RetrievalEvidenceOrigin.CURRENT_TASK,
    }[evidence.source_type]


def _project_public_history_locator(
    evidence: RetrievalToolEvidence,
) -> RetrievalEvidenceLocator:
    """投影模型可见 History 信封所使用的有界定位器。"""

    citation = evidence.citation
    if evidence.source_type is SourceType.CURRENT_SESSION:
        return RetrievalEvidenceLocator(
            user_turn_index=_optional_nonnegative_int(
                citation.get("user_turn_idx")
            ),
            assistant_turn_index=_optional_nonnegative_int(
                citation.get("assistant_turn_idx")
            ),
        )
    return RetrievalEvidenceLocator(
        memory_type=_optional_safe_code(citation.get("memory_type")),
    )


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value < 10**18 else None
    if isinstance(value, str) and _CANONICAL_TURN_INDEX.fullmatch(value):
        return int(value)
    return None


def _optional_safe_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if _SAFE_CODE.fullmatch(value) else None


def _public_status(
    *,
    port_status: RetrievalStatus,
    evidence: list[RetrievalEvidence],
    gaps: list[RetrievalEvidenceGap],
    truncated: bool,
) -> RetrievalEvidenceStatus:
    if port_status is RetrievalStatus.BLOCKED:
        return RetrievalEvidenceStatus.BLOCKED
    if port_status is RetrievalStatus.PARTIAL or gaps or truncated:
        return RetrievalEvidenceStatus.PARTIAL
    return RetrievalEvidenceStatus.COMPLETE


def _public_outcome(
    *,
    status: RetrievalEvidenceStatus,
    evidence: list[RetrievalEvidence],
) -> RetrievalEvidenceOutcome:
    if evidence:
        return RetrievalEvidenceOutcome.MATCHED
    if status is RetrievalEvidenceStatus.COMPLETE:
        return RetrievalEvidenceOutcome.NO_MATCH
    return RetrievalEvidenceOutcome.NOT_ESTABLISHED


def _deduplicate_gaps(
    values: list[RetrievalEvidenceGap],
) -> list[RetrievalEvidenceGap]:
    by_key: dict[tuple[Any, ...], RetrievalEvidenceGap] = {}
    for value in values:
        key = (
            value.code,
            value.blocking,
            value.source_type,
            value.source_alias,
            value.known_count,
        )
        by_key[key] = value
    return [by_key[key] for key in sorted(by_key, key=lambda item: tuple(str(part) for part in item))]


def _request_id(
    scope_snapshot_id: str,
    query: str,
    narrowed_scope: tuple[str, ...],
    limit: int,
) -> str:
    digest = hashlib.sha256(
        "\0".join(
            (
                scope_snapshot_id,
                RetrievalCorpus.HISTORY.value,
                query,
                "\0".join(narrowed_scope),
                str(limit),
            )
        ).encode("utf-8")
    ).hexdigest()[:32]
    return f"retrieval_request_{digest}"


def _history_evidence_handle(
    evidence: RetrievalToolEvidence,
) -> str:
    """根据一个精确私有 Source 指针推导不透明公共句柄。"""

    source_type = evidence.source_type
    source_unit_id = evidence.source_unit_id
    source_revision = evidence.source_revision
    indexed_content_hash = evidence.indexed_content_hash
    digest = hashlib.sha256(
        "\0".join(
            (
                RetrievalCorpus.HISTORY.value,
                source_type.value,
                source_unit_id,
                source_revision,
                indexed_content_hash,
            )
        ).encode("utf-8")
    ).hexdigest()[:32]
    return f"retrieval_evidence_{digest}"


def _query(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("query must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > MAX_RETRIEVAL_QUERY_CHARS:
        raise ValueError("query must be a bounded non-empty string")
    if any(ord(character) < 32 and character not in "\n\t\r" for character in normalized):
        raise ValueError("query must not contain control characters")
    return normalized


def freeze_long_term_history_source_snapshots(
    *,
    allowed_scopes: tuple[HistoryScope, ...],
    long_term_task_id: str | None,
) -> tuple[FrozenHistorySourceSnapshot, ...]:
    """对已退役的长期 History 来源采用失败关闭策略。

    该入口不构造记忆 adapter，也不读取全局 memory store。
    """

    del long_term_task_id
    if any(
        scope in {
            HistoryScope.LONG_TERM_USER,
            HistoryScope.CURRENT_TASK,
        }
        for scope in allowed_scopes
    ):
        raise RuntimeError("long-term History retrieval is retired")
    return ()


__all__ = (
    "DEFAULT_RETRIEVAL_LIMIT",
    "HistoryRetrievalRuntime",
    "build_history_retrieval_runtime",
    "build_history_retrieval_tool_registration",
    "freeze_long_term_history_source_snapshots",
)
