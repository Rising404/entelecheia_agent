"""L1 当前会话历史检索工具的生产组合。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ...tools.catalog.binding import ToolBinding
from ...tools.policy import AuthorityFacts
from ...tools.registration import ToolRegistration
from ...tools.retrieval.history_retrieval_catalog import (
    HistoryRetrievalBindingFacts,
    build_history_retrieval_tool_bindings,
    derive_history_retrieval_source_fingerprint,
)
from ...retrieval.tooling.contracts import (
    FrozenHistoryRetrievalScope,
    HistoryScope,
    format_current_session_cutoff,
)
from ...retrieval.tooling.service import RetrievalServiceToolPort
from ...tools.retrieval.history_retrieval_adapter import (
    build_history_retrieval_runtime,
)

if TYPE_CHECKING:
    from ...retrieval.sources.session.contracts import SessionRetrievalComposition


_SESSION_SCOPE_CONTRACT = "l1-session-retrieval-scope-v1"


@dataclass(frozen=True, slots=True)
class L1HistoryRetrievalToolSource:
    registrations: tuple[ToolRegistration, ...]
    authority: AuthorityFacts
    history_retrieval_bindings: tuple[ToolBinding, ...]


def build_l1_history_retrieval_tool_source(
    *,
    session_id: str,
    turn_id: str | None,
    session_retrieval_data_version: str | None = None,
    session_retrieval_assistant_turn_cutoff: int | None = None,
    store: object | None = None,
    session_composition_factory: Callable[
        [], SessionRetrievalComposition
    ] | None = None,
) -> L1HistoryRetrievalToolSource | None:
    """冻结当前会话范围，并组合唯一的 History 读取工具。"""

    session_factory = (
        build_session_retrieval_composition
        if session_composition_factory is None
        else session_composition_factory
    )
    session_composition = session_factory()
    _require_expected_generation_identity(
        expected=session_retrieval_data_version,
        recipe=session_composition.generation_spec.version_id,
    )

    cutoff = session_retrieval_assistant_turn_cutoff
    if cutoff is None:
        from ...retrieval.sources.session.composition import (
            prepare_session_retrieval_binding,
        )

        cutoff = prepare_session_retrieval_binding(
            session_id=session_id,
            store=store,
            composition=session_composition,
        ).assistant_turn_cutoff
    if cutoff is None:
        return None

    from ...retrieval.sources.session.lifecycle import ensure_session_retrieval_ready

    active_session_version = ensure_session_retrieval_ready(
        session_composition,
        session_id=session_id,
        assistant_turn_cutoff=cutoff,
    )
    history_scope = FrozenHistoryRetrievalScope(
        session_id=session_id,
        scope_snapshot_id=_derive_scope_snapshot_id(
            {
                "session_id": session_id,
                "retrieval_data_version": active_session_version,
                "assistant_turn_cutoff": cutoff,
            }
        ),
        retrieval_data_version=active_session_version,
        allowed_scopes=(HistoryScope.CURRENT_SESSION,),
        current_session_cutoff=format_current_session_cutoff(cutoff),
    )
    port = RetrievalServiceToolPort(
        history_foundation=session_composition.foundation,
        trajectory_turn_id=turn_id,
    )
    binding_facts = _history_retrieval_binding_facts(
        session_id=session_id,
        turn_id=turn_id,
        scope=history_scope,
        composition=session_composition,
    )
    runtime = build_history_retrieval_runtime(
        scope=history_scope,
        port=port,
        source_fingerprint=derive_history_retrieval_source_fingerprint(
            binding_facts
        ),
    )
    history_retrieval_bindings = build_history_retrieval_tool_bindings(
        runtime.registrations,
        facts=binding_facts,
    )
    return L1HistoryRetrievalToolSource(
        registrations=runtime.registrations,
        authority=runtime.authority,
        history_retrieval_bindings=history_retrieval_bindings,
    )


def build_session_retrieval_composition() -> SessionRetrievalComposition:
    """延迟创建组合，并保留一个窄测试注入点。"""

    from ...retrieval.sources.session.composition import (
        build_session_retrieval_composition as build,
    )
    from ...session import store as session_store

    return build(store=session_store)


def _derive_scope_snapshot_id(value: dict[str, object]) -> str:
    payload = json.dumps(
        {"contract": _SESSION_SCOPE_CONTRACT, **value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_SESSION_SCOPE_CONTRACT}:{digest}"


def _history_retrieval_binding_facts(
    *,
    session_id: str,
    turn_id: str | None,
    scope: FrozenHistoryRetrievalScope,
    composition: SessionRetrievalComposition,
) -> HistoryRetrievalBindingFacts:
    """Describe the selected History stack without persisting private values."""

    generation_sha256 = _sha256_value(
        {
            "schema_version": "history-retrieval-generation-binding-v1",
            "generation_spec_sha256": _sha256_text(
                composition.generation_spec.canonical_json
            ),
            "active_data_version_sha256": _sha256_text(
                scope.retrieval_data_version
            ),
        }
    )
    encoder_fingerprint = _bounded_fingerprint(
        composition.encoder.fingerprint(),
        "History retrieval encoder",
    )
    reranker_fingerprint = (
        None
        if composition.reranker is None
        else _bounded_fingerprint(
            composition.reranker.fingerprint(),
            "History retrieval reranker",
        )
    )
    service_sha256 = _sha256_value(
        {
            "schema_version": "history-retrieval-service-identity-v1",
            "recipe": "current-history-retrieval-service@1",
            "session_sha256": _sha256_text(session_id),
            "trajectory_turn_sha256": (
                None if turn_id is None else _sha256_text(turn_id)
            ),
            "retrieval_generation_sha256": generation_sha256,
            "retrieval_database_sha256": _sha256_text(
                str(Path(composition.foundation.catalog.db_path).resolve())
            ),
            "requested_profile_sha256": _sha256_text(
                composition.requested_profile.fingerprint()
            ),
            "effective_profile_sha256": _sha256_text(
                composition.effective_profile.fingerprint()
            ),
            "encoder_sha256": _sha256_text(encoder_fingerprint),
            "reranker_sha256": (
                None
                if reranker_fingerprint is None
                else _sha256_text(reranker_fingerprint)
            ),
            "retrieval_methods": [
                method.value
                for method in composition.foundation.retrieval_methods
            ],
        }
    )
    authority_sha256 = _sha256_value(
        {
            "schema_version": "history-retrieval-authority-binding-v1",
            "session_id": scope.session_id,
            "scope_snapshot_id": scope.scope_snapshot_id,
            "retrieval_data_version": scope.retrieval_data_version,
            "effect_scope": scope.effect_scope,
            "allowed_scopes": [item.value for item in scope.allowed_scopes],
            "current_session_cutoff": scope.current_session_cutoff,
            "long_term_task_id": scope.long_term_task_id,
            "long_term_source_snapshots": [
                item.fingerprint_payload()
                for item in scope.long_term_source_snapshots
            ],
            "max_items": scope.max_items,
            "context_token_limit": scope.context_token_limit,
        }
    )
    return HistoryRetrievalBindingFacts(
        effect_scope_sha256=_sha256_text(scope.effect_scope),
        session_sha256=_sha256_text(session_id),
        turn_cutoff_sha256=_sha256_value(
            {"current_session_cutoff": scope.current_session_cutoff}
        ),
        retrieval_generation_sha256=generation_sha256,
        allowed_scopes_sha256=_sha256_value(
            [item.value for item in scope.allowed_scopes]
        ),
        retrieval_service_sha256=service_sha256,
        retrieval_authority_sha256=authority_sha256,
    )


def _bounded_fingerprint(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 4_096
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} fingerprint is invalid")
    return value.strip()


def _sha256_text(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("SHA-256 text input must be non-empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_expected_generation_identity(
    *,
    expected: str | None,
    recipe: str,
) -> None:
    if expected is None:
        return
    if not isinstance(expected, str) or not expected or len(expected) > 512:
        raise ValueError("accepted Session retrieval generation is invalid")
    if expected != recipe:
        raise RuntimeError(
            "accepted Session retrieval generation differs from current recipe"
        )


__all__ = [
    'L1HistoryRetrievalToolSource',
    "build_l1_history_retrieval_tool_source",
]
