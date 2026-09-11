"""已提交 Turn 对到 Current Session 检索单元的确定性投影。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import re
from threading import RLock

from ....input_processing.documents.chunking import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SPLIT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    ChunkingProfile,
)
from ...contracts import (
    CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
    CURRENT_SESSION_TURN_INDEX_SCOPE_KEY,
    SourceAccess,
    SourceAvailability,
    SourceFilter,
    SourceIndexBinding,
    SourceIndexBindingSnapshot,
    SourceType,
    SourceUnit,
    SourceUnitRef,
    source_index_binding_manifest,
)
from ...lifecycle.outbox import RetrievalUpdateEvent
from ...lifecycle.sync import IndexableSourceUnit
from ..identity import (
    current_session_source_unit_id,
    parse_current_session_source_unit_id,
)
from .contracts import SessionRetrievalNotReady, SessionStoreReadPort


SESSION_PROJECTION_CONTRACT = "current-session-turn-projection-v2"
# Session 与 Document 的检索单元共享同一组目标、硬上限和相邻 overlap。Session 不采用
# Document 的 min_tokens 合并规则：完整短 Turn 对本身就是有意义的原子单元。
SESSION_TARGET_TOKENS = DEFAULT_TARGET_TOKENS
SESSION_MAX_TOKENS = DEFAULT_MAX_TOKENS
SESSION_OVERLAP_TOKENS = DEFAULT_SPLIT_OVERLAP_TOKENS
_SESSION_SENTENCE_BOUNDARY = re.compile(r"(?<=[。．.!?！？;；\n])")
_MAX_PROJECTION_CACHE_ENTRIES = 4_096


class CurrentSessionSourceAdapter:
    """把完整已提交 Turn 对确定性投影为同构的检索单元。"""

    source_type = SourceType.CURRENT_SESSION

    def __init__(
        self,
        *,
        chunking_profile: ChunkingProfile,
        store: SessionStoreReadPort,
    ) -> None:
        self._profile = chunking_profile
        self._store = store
        self._cache: OrderedDict[
            tuple[str, str, str, str, str],
            tuple[IndexableSourceUnit, ...],
        ] = OrderedDict()
        self._cache_lock = RLock()

    def availability(self, source_filter: SourceFilter) -> SourceAvailability:
        return self.open_retrieval_access(source_filter).availability

    def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess:
        session_id = _scope_value(source_filter, "session_id")
        if not session_id or not _current_session_filter_is_exact(source_filter):
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code=(
                    "current_session_id_missing"
                    if not session_id
                    else "current_session_scope_invalid"
                ),
            )
        try:
            availability = self._session_availability(session_id)
            if availability is not SourceAvailability.READY:
                return SourceAccess(
                    self.source_type,
                    source_filter,
                    availability,
                    reason_code=f"current_session_{availability.value}",
                )
            bindings = self._bindings_for_pairs(
                source_filter,
                self._store.list_committed_turn_pairs(session_id),
            )
            digest, revision_map = _binding_manifest(bindings)
            if not bindings:
                return SourceAccess(
                    self.source_type,
                    source_filter,
                    SourceAvailability.EMPTY,
                    source_snapshot_id=f"current-session:{digest}",
                    source_revision_map=revision_map,
                )
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.READY,
                source_snapshot_id=f"current-session:{digest}",
                source_revision_map=revision_map,
            )
        except Exception:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.UNAVAILABLE,
                reason_code="current_session_authority_unavailable",
            )

    def revalidate_retrieval_access(self, access: SourceAccess) -> SourceAccess:
        if access.source_type is not self.source_type:
            raise ValueError("CurrentSession revalidation requires matching access")
        current = self.open_retrieval_access(access.source_filter)
        if current.availability is not SourceAvailability.READY:
            return current
        return SourceAccess(
            self.source_type,
            access.source_filter,
            SourceAvailability.READY,
            source_snapshot_id=access.source_snapshot_id,
            source_revision_map=current.source_revision_map,
        )

    def fetch_units(
        self,
        access: SourceAccess,
        refs: Sequence[SourceUnitRef],
    ) -> tuple[SourceUnit, ...]:
        if access.source_type is not self.source_type:
            return ()
        session_id = _scope_value(access.source_filter, "session_id")
        if (
            not session_id
            or self._session_availability(session_id)
            is not SourceAvailability.READY
        ):
            return ()
        units: list[SourceUnit] = []
        for ref in refs:
            identity = parse_current_session_source_unit_id(ref.source_unit_id)
            if ref.source_type is not self.source_type or identity is None:
                continue
            if identity.session_id != session_id:
                continue
            pair = self._store.get_committed_turn_pair(session_id, identity.run_id)
            if pair is None or not _filter_selects_pair(access.source_filter, pair):
                continue
            projected = next(
                (item for item in self._project_pair(pair) if item.ref == ref),
                None,
            )
            if projected is None:
                continue
            units.append(
                SourceUnit(
                    ref=projected.ref,
                    content=projected.content,
                    citation=_pair_citation(pair),
                )
            )
        return tuple(units)

    def read_for_index(self, event: RetrievalUpdateEvent) -> IndexableSourceUnit | None:
        if event.ref.source_type is not self.source_type:
            return None
        unit = self.read_current_for_reconcile(event.ref)
        return unit if unit is not None and unit.ref == event.ref else None

    def read_current_for_reconcile(
        self,
        ref: SourceUnitRef,
    ) -> IndexableSourceUnit | None:
        if ref.source_type is not self.source_type:
            return None
        identity = parse_current_session_source_unit_id(ref.source_unit_id)
        if identity is None:
            return None
        if (
            self._session_availability(identity.session_id)
            is not SourceAvailability.READY
        ):
            return None
        pair = self._store.get_committed_turn_pair(
            identity.session_id,
            identity.run_id,
        )
        if pair is None:
            return None
        return next((item for item in self._project_pair(pair) if item.ref == ref), None)

    def get_current_index_binding_snapshot(
        self,
        access: SourceAccess,
        *,
        maximum_bindings: int,
    ) -> SourceIndexBindingSnapshot:
        if access.source_type is not self.source_type:
            raise ValueError("CurrentSession snapshots require matching access")
        if access.availability is not SourceAvailability.READY:
            raise ValueError("CurrentSession snapshots require ready access")
        session_id = _scope_value(access.source_filter, "session_id")
        if (
            not session_id
            or not _current_session_filter_is_exact(access.source_filter)
            or self._session_availability(session_id) is not SourceAvailability.READY
        ):
            return SourceIndexBindingSnapshot(source_snapshot_is_current=False)
        bindings = self._bindings_for_pairs(
            access.source_filter,
            self._store.list_committed_turn_pairs(session_id),
        )
        _validate_maximum_bindings(maximum_bindings)
        return SourceIndexBindingSnapshot(
            source_snapshot_is_current=(
                dict(access.source_revision_map) == _binding_manifest(bindings)[1]
            ),
            bindings=bindings[:maximum_bindings],
            binding_enumeration_complete=len(bindings) <= maximum_bindings,
        )

    def list_indexable_units(
        self,
        source_filter: SourceFilter,
    ) -> tuple[IndexableSourceUnit, ...]:
        if source_filter.source_type is not self.source_type:
            return ()
        session_id = _scope_value(source_filter, "session_id")
        if (
            not session_id
            or not _current_session_filter_is_exact(source_filter)
            or self._session_availability(session_id) is not SourceAvailability.READY
        ):
            return ()
        return self._indexable_units_in_scope(session_id, source_filter)

    def list_indexable_units_for_backfill(
        self,
        source_filter: SourceFilter,
    ) -> tuple[IndexableSourceUnit, ...]:
        # 查询可以把不可用来源投影为状态；发布不能把同一种状态误当成“合法空语料”。
        session_id = _scope_value(source_filter, "session_id")
        if not session_id or not _current_session_filter_is_exact(source_filter):
            raise SessionRetrievalNotReady("session_projection_scope_invalid")
        self.require_index_source_ready(session_id)
        return self._indexable_units_in_scope(session_id, source_filter)

    def _indexable_units_in_scope(
        self, session_id: str, source_filter: SourceFilter,
    ) -> tuple[IndexableSourceUnit, ...]:
        return tuple(
            unit
            for pair in self._store.list_committed_turn_pairs(session_id)
            if _filter_selects_pair(source_filter, pair)
            for unit in self._project_pair(pair)
        )

    def require_index_source_ready(self, session_id: str) -> None:
        """索引发布需要明确可读取的来源，不继承查询侧的空结果表示。"""

        availability = self._session_availability(session_id)
        if availability is not SourceAvailability.READY:
            raise SessionRetrievalNotReady(
                f"session_index_source_{availability.value}"
            )

    def latest_assistant_turn_cutoff(self, session_id: str) -> int | None:
        """Return the latest complete Assistant index without exposing Store ownership."""

        pairs = tuple(self._store.list_committed_turn_pairs(session_id, limit=1))
        return _assistant_turn_index(pairs[-1]) if pairs else None

    def project_committed_pair(
        self,
        pair: Mapping[str, object],
    ) -> tuple[IndexableSourceUnit, ...]:
        """Project one already-authorized committed pair without scanning history."""

        return self._project_pair(pair)

    def _bindings_for_pairs(
        self,
        source_filter: SourceFilter,
        pairs: Sequence[Mapping[str, object]],
    ) -> tuple[SourceIndexBinding, ...]:
        return tuple(
            SourceIndexBinding(
                source_unit_id=unit.ref.source_unit_id,
                source_revision=unit.ref.source_revision,
                indexed_content_hash=unit.ref.indexed_content_hash,
            )
            for pair in pairs
            if _filter_selects_pair(source_filter, pair)
            for unit in self._project_pair(pair)
        )

    def _project_pair(
        self,
        pair: Mapping[str, object],
    ) -> tuple[IndexableSourceUnit, ...]:
        session_id, run_id, created_at = _pair_identity(pair)
        user_content = str(pair.get("user_content") or "").strip()
        assistant_content = str(pair.get("assistant_content") or "").strip()
        cache_key = (
            session_id,
            run_id,
            created_at,
            hashlib.sha256(user_content.encode("utf-8")).hexdigest(),
            hashlib.sha256(assistant_content.encode("utf-8")).hexdigest(),
        )
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        assistant_turn_idx = _assistant_turn_index(pair)
        source_filter = SourceFilter.from_mapping(
            SourceType.CURRENT_SESSION,
            {
                "session_id": session_id,
                CURRENT_SESSION_TURN_INDEX_SCOPE_KEY: str(assistant_turn_idx),
            },
        )
        atomic = f"用户：{user_content}\n\n助手：{assistant_content}"
        projected: list[IndexableSourceUnit] = []
        if atomic.strip() and self._profile.tokens_of(atomic) <= SESSION_MAX_TOKENS:
            projected.append(
                _indexable_session_unit(
                    session_id=session_id,
                    run_id=run_id,
                    created_at=created_at,
                    role="pair",
                    ordinal=0,
                    content=atomic,
                    source_filter=source_filter,
                )
            )
        else:
            for role, label, content in (
                ("user", "用户", user_content),
                ("assistant", "助手", assistant_content),
            ):
                if not content:
                    continue
                for ordinal, unit_content in enumerate(
                    _split_role_content(
                        content,
                        label=label,
                        profile=self._profile,
                    )
                ):
                    projected.append(
                        _indexable_session_unit(
                            session_id=session_id,
                            run_id=run_id,
                            created_at=created_at,
                            role=role,
                            ordinal=ordinal,
                            content=unit_content,
                            source_filter=source_filter,
                        )
                    )
        result = tuple(projected)
        with self._cache_lock:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            while len(self._cache) > _MAX_PROJECTION_CACHE_ENTRIES:
                self._cache.popitem(last=False)
        return result

    def _session_availability(self, session_id: str) -> SourceAvailability:
        try:
            session = self._store.get_session(session_id)
        except Exception:
            return SourceAvailability.UNAVAILABLE
        if session is None:
            return SourceAvailability.UNAVAILABLE
        return (
            SourceAvailability.BLOCKED
            if session.get("status") == "trashed"
            else SourceAvailability.READY
        )


def _split_role_content(
    content: str,
    *,
    label: str,
    profile: ChunkingProfile,
) -> tuple[str, ...]:
    prefix = f"{label}："
    base: list[str] = []
    current = ""
    for sentence in _SESSION_SENTENCE_BOUNDARY.split(content):
        if not sentence:
            continue
        if profile.tokens_of(f"{prefix}{sentence}") > SESSION_MAX_TOKENS:
            if current:
                base.append(current)
                current = ""
            base.extend(_hard_split_role_body(sentence, prefix=prefix, profile=profile))
            continue
        candidate = f"{current}{sentence}" if current else sentence
        if current and profile.tokens_of(f"{prefix}{candidate}") > SESSION_TARGET_TOKENS:
            base.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        base.append(current)

    if not base and content:
        base.extend(_hard_split_role_body(content, prefix=prefix, profile=profile))
    for body in base:
        if profile.tokens_of(f"{prefix}{body}") > SESSION_MAX_TOKENS:
            raise SessionRetrievalNotReady("session_projection_exceeded_hard_limit")

    overlapped: list[str] = []
    for index, body in enumerate(base):
        if index == 0:
            overlapped.append(f"{prefix}{body}")
            continue
        overlap_budget = SESSION_OVERLAP_TOKENS
        tail = _tail_within_tokens(base[index - 1], overlap_budget, profile)
        candidate_body = f"{tail}\n{body}" if tail else body
        while (
            tail
            and profile.tokens_of(f"{prefix}{candidate_body}") > SESSION_MAX_TOKENS
        ):
            overlap_budget -= 1
            tail = _tail_within_tokens(base[index - 1], overlap_budget, profile)
            candidate_body = f"{tail}\n{body}" if tail else body
        overlapped.append(f"{prefix}{candidate_body}")
    return tuple(overlapped)


def _hard_split_role_body(
    text: str,
    *,
    prefix: str,
    profile: ChunkingProfile,
) -> tuple[str, ...]:
    if not text:
        return ()
    boundaries = _token_boundaries(text, profile)
    if len(boundaries) == 2 and profile.tokens_of(f"{prefix}{text}") <= SESSION_MAX_TOKENS:
        return (text,)
    pieces: list[str] = []
    start = 0
    while start < len(boundaries) - 1:
        end = _largest_fitting_boundary(
            text,
            prefix=prefix,
            boundaries=boundaries,
            start=start,
            limit=SESSION_TARGET_TOKENS,
            profile=profile,
        )
        if end is None:
            end = _largest_fitting_boundary(
                text,
                prefix=prefix,
                boundaries=boundaries,
                start=start,
                limit=SESSION_MAX_TOKENS,
                profile=profile,
            )
        if end is None or end <= start:
            raise SessionRetrievalNotReady("session_projection_cannot_fit_token")
        piece = text[boundaries[start]:boundaries[end]]
        if piece:
            pieces.append(piece)
        start = end
    return tuple(pieces)


def _largest_fitting_boundary(
    text: str,
    *,
    prefix: str,
    boundaries: tuple[int, ...],
    start: int,
    limit: int,
    profile: ChunkingProfile,
) -> int | None:
    low, high = start + 1, len(boundaries) - 1
    best: int | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate = text[boundaries[start]:boundaries[middle]]
        if profile.tokens_of(f"{prefix}{candidate}") <= limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _token_boundaries(text: str, profile: ChunkingProfile) -> tuple[int, ...]:
    offsets = _validated_offsets(text, profile)
    if offsets is None:
        return tuple(range(len(text) + 1))
    if not offsets:
        return (0, len(text))
    return tuple([0, *(offsets[index][0] for index in range(1, len(offsets))), len(text)])


def _tail_within_tokens(
    text: str,
    budget: int,
    profile: ChunkingProfile,
) -> str:
    if budget <= 0 or not text:
        return ""
    offsets = _validated_offsets(text, profile)
    if offsets is not None:
        if len(offsets) <= budget:
            return text
        return text[offsets[-budget][0]:]
    low, high = 0, len(text)
    best = len(text)
    while low <= high:
        middle = (low + high) // 2
        if profile.tokens_of(text[middle:]) <= budget:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    return text[best:]


def _validated_offsets(
    text: str,
    profile: ChunkingProfile,
) -> tuple[tuple[int, int], ...] | None:
    if profile.token_offsets is None:
        return None
    offsets = tuple((int(start), int(end)) for start, end in profile.token_offsets(text))
    previous_start = -1
    for start, end in offsets:
        if start < previous_start or start < 0 or end <= start or end > len(text):
            raise SessionRetrievalNotReady("session_token_offsets_invalid")
        previous_start = start
    if len(offsets) != profile.tokens_of(text):
        raise SessionRetrievalNotReady("session_token_offsets_disagree")
    return offsets


def _indexable_session_unit(
    *,
    session_id: str,
    run_id: str,
    created_at: str,
    role: str,
    ordinal: int,
    content: str,
    source_filter: SourceFilter,
) -> IndexableSourceUnit:
    ref = SourceUnitRef(
        source_type=SourceType.CURRENT_SESSION,
        source_unit_id=current_session_source_unit_id(
            session_id=session_id,
            run_id=run_id,
            role=role,
            ordinal=ordinal,
        ),
        source_revision=created_at,
        indexed_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    return IndexableSourceUnit(ref=ref, source_filter=source_filter, content=content)


def _pair_identity(pair: Mapping[str, object]) -> tuple[str, str, str]:
    values = tuple(
        str(pair.get(key) or "")
        for key in ("session_id", "run_id", "created_at")
    )
    if any(not value for value in values):
        raise ValueError("committed Turn pair identity is incomplete")
    return values  # type: ignore[return-value]


def _assistant_turn_index(pair: Mapping[str, object]) -> int:
    value = pair.get("assistant_turn_idx")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("committed Turn pair has an invalid assistant index")
    return value


def _pair_citation(pair: Mapping[str, object]) -> dict[str, str]:
    session_id, _run_id, _created_at = _pair_identity(pair)
    user_turn_idx = pair.get("user_turn_idx")
    assistant_turn_idx = _assistant_turn_index(pair)
    if (
        isinstance(user_turn_idx, bool)
        or not isinstance(user_turn_idx, int)
        or user_turn_idx < 0
    ):
        raise ValueError("committed Turn pair has an invalid user index")
    return {
        "session_id": session_id,
        "user_turn_idx": str(user_turn_idx),
        "assistant_turn_idx": str(assistant_turn_idx),
    }


def _scope_value(source_filter: SourceFilter, key: str) -> str | None:
    return source_filter.as_mapping().get(key)


def _current_session_filter_is_exact(source_filter: SourceFilter) -> bool:
    if source_filter.source_type is not SourceType.CURRENT_SESSION:
        return False
    keys = set(source_filter.as_mapping())
    return keys in (
        {"session_id"},
        {"session_id", CURRENT_SESSION_TURN_INDEX_SCOPE_KEY},
        {"session_id", CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY},
    )


def _filter_selects_pair(
    source_filter: SourceFilter,
    pair: Mapping[str, object],
) -> bool:
    try:
        session_id, _run_id, _created_at = _pair_identity(pair)
        indexed = SourceFilter.from_mapping(
            SourceType.CURRENT_SESSION,
            {
                "session_id": session_id,
                CURRENT_SESSION_TURN_INDEX_SCOPE_KEY: str(
                    _assistant_turn_index(pair)
                ),
            },
        )
        return source_filter.selects(indexed)
    except (TypeError, ValueError):
        return False


def _binding_manifest(
    bindings: Sequence[SourceIndexBinding],
) -> tuple[str, dict[str, str]]:
    digest, revision_map = source_index_binding_manifest(tuple(bindings))
    return digest, dict(revision_map)


def _validate_maximum_bindings(maximum_bindings: int) -> None:
    if (
        isinstance(maximum_bindings, bool)
        or not isinstance(maximum_bindings, int)
        or maximum_bindings <= 0
    ):
        raise ValueError("maximum_bindings must be positive")


__all__ = [
    "CurrentSessionSourceAdapter",
    "SESSION_MAX_TOKENS",
    "SESSION_OVERLAP_TOKENS",
    "SESSION_PROJECTION_CONTRACT",
    "SESSION_TARGET_TOKENS",
]
