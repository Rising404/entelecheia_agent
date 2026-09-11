"""Current Session 检索依赖组合与 AcceptedTurn 冻结绑定。"""

from __future__ import annotations

from pathlib import Path

from ....input_processing.documents.chunking import ChunkingProfile
from ....workspace.storage.context import (
    current as current_project_documents,
)
from ...foundation import build_session_retrieval_foundation
from ...lifecycle.generation import session_generation_spec
from ...ports import BgeM3EncoderPort, RerankerPort
from ...profile import DocumentRetrievalProfile, build_document_retrieval_runtime
from .contracts import (
    SessionRetrievalComposition,
    SessionRetrievalNotReady,
    SessionStoreReadPort,
    SessionTurnRetrievalBinding,
)
from .projection import (
    CurrentSessionSourceAdapter,
    SESSION_MAX_TOKENS,
    SESSION_OVERLAP_TOKENS,
    SESSION_PROJECTION_CONTRACT,
    SESSION_TARGET_TOKENS,
    _assistant_turn_index,
)


SESSION_RETRIEVAL_DB_NAME = "session_retrieval.sqlite"


def build_session_retrieval_composition(
    *,
    retrieval_db_path: Path | str | None = None,
    profile: DocumentRetrievalProfile | None = None,
    encoder: BgeM3EncoderPort | None = None,
    reranker: RerankerPort | None = None,
    store: SessionStoreReadPort,
) -> SessionRetrievalComposition:
    """构建与 Document 使用同一 tokenizer、方法集合和 reranker 的 Session 索引。"""

    runtime = build_document_retrieval_runtime(profile)
    retrieval_encoder = encoder or runtime.encoder
    retrieval_reranker = reranker if reranker is not None else runtime.reranker
    offsets = getattr(retrieval_encoder, "token_offsets", None)
    chunking_profile = ChunkingProfile(
        target_tokens=SESSION_TARGET_TOKENS,
        max_tokens=SESSION_MAX_TOKENS,
        min_tokens=0,
        include_section_heading=False,
        split_overlap_tokens=SESSION_OVERLAP_TOKENS,
        count_tokens=lambda text: len(tuple(retrieval_encoder.token_ids(text))),
        token_offsets=offsets if callable(offsets) else None,
        tokenizer_id=retrieval_encoder.tokenizer_fingerprint(),
    )
    projection_fingerprint = (
        f"{SESSION_PROJECTION_CONTRACT}+{chunking_profile.tokenizer_id}"
        f"+atomic_max{SESSION_MAX_TOKENS}"
        f"+target{SESSION_TARGET_TOKENS}+max{SESSION_MAX_TOKENS}"
        f"+overlap{SESSION_OVERLAP_TOKENS}+role_boundary=true"
    )
    generation_spec = session_generation_spec(
        encoder_fingerprint=retrieval_encoder.fingerprint(),
        projection_fingerprint=projection_fingerprint,
        index_recipe=runtime.effective_profile.index_recipe,
    )
    source_adapter = CurrentSessionSourceAdapter(
        chunking_profile=chunking_profile,
        store=store,
    )
    db_path = (
        Path(retrieval_db_path)
        if retrieval_db_path is not None
        else _project_session_retrieval_db_path()
    )
    foundation = build_session_retrieval_foundation(
        db_path=db_path,
        source_adapter=source_adapter,
        encoder=retrieval_encoder,
        generation_spec=generation_spec,
        reranker=retrieval_reranker,
        retrieval_methods=runtime.effective_profile.retrieval_methods,
    )
    return SessionRetrievalComposition(
        foundation=foundation,
        generation_spec=generation_spec,
        chunking_profile=chunking_profile,
        source_adapter=source_adapter,
        encoder=retrieval_encoder,
        reranker=retrieval_reranker,
        requested_profile=runtime.requested_profile,
        effective_profile=runtime.effective_profile,
        capability=runtime.capability,
        degraded_reason=runtime.degraded_reason,
    )


def prepare_session_retrieval_binding(
    *,
    session_id: str,
    store: SessionStoreReadPort | None = None,
    composition: SessionRetrievalComposition | None = None,
) -> SessionTurnRetrievalBinding:
    """在 Turn 接受前冻结 generation 配方和最近完整 assistant 截止点。"""

    if composition is None and store is None:
        raise TypeError("store is required when composition is not supplied")
    if composition is None:
        assert store is not None
        selected = build_session_retrieval_composition(store=store)
    else:
        selected = composition
    if store is None:
        cutoff = selected.source_adapter.latest_assistant_turn_cutoff(session_id)
    else:
        pairs = tuple(store.list_committed_turn_pairs(session_id, limit=1))
        cutoff = _assistant_turn_index(pairs[-1]) if pairs else None
    return SessionTurnRetrievalBinding(
        composition=selected,
        data_version_id=selected.generation_spec.version_id,
        assistant_turn_cutoff=cutoff,
    )


def _project_session_retrieval_db_path() -> Path:
    database = current_project_documents()
    if database is None:
        raise SessionRetrievalNotReady(
            "session retrieval requires a bound project documents context"
        )
    return database.db_path.with_name(SESSION_RETRIEVAL_DB_NAME)


__all__ = [
    "SESSION_RETRIEVAL_DB_NAME",
    "build_session_retrieval_composition",
    "prepare_session_retrieval_binding",
]
