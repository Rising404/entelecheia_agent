"""Picture-observation authority adapter and dormant outbox publisher.

Each non-blank observation in a picture's active FIFO window is one immutable
retrieval Source Unit.  This module only projects pointers into the retrieval
subsystem: observation text remains authoritative in ``workspace.pictures`` and is
re-read with Picture, PictureUnit and FileVersion authority on every use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import re
import sqlite3
from typing import Protocol

from ...workspace.pictures.contracts import (
    PictureRecord,
    PictureSourceKind,
    PictureUnitRecord,
)
from ...workspace.pictures.observations.contracts import (
    DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY,
    PictureObservationRecord,
    PictureObservationWindowPolicy,
)
from ...workspace.pictures.observations.repository import (
    PictureObservationRepository,
)
from ...workspace.pictures.storage.repository import (
    get_picture,
    get_picture_unit,
    picture_binding_is_current,
    picture_unit_binding_is_current,
)
from ...workspace.storage.context import connect_current
from ..contracts import (
    SourceAccess,
    SourceAvailability,
    SourceFilter,
    SourceIndexBinding,
    SourceIndexBindingSnapshot,
    SourceType,
    SourceUnit,
    SourceUnitRef,
)
from ..lifecycle.outbox import RetrievalUpdateEvent
from ..lifecycle.sync import IndexableSourceUnit
from .identity import (
    parse_picture_observation_source_unit_id,
    picture_observation_ref_and_content,
)


DEFAULT_MAX_ACTIVE_PICTURE_OBSERVATIONS = (
    DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY.max_active_entries
)
DEFAULT_MAX_PICTURE_ACCESS_BINDINGS = 5_000

_INDEX_SCOPE_KEYS = frozenset({"file_id", "file_version_id", "picture_id"})
_ONLINE_SCOPE_KEYS = _INDEX_SCOPE_KEYS | {"session_id"}
_AUTHORITY_REASON_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_MAX_AUTHORITY_IDENTIFIER_LENGTH = 512

ConnectionFactory = Callable[[], AbstractContextManager[sqlite3.Connection]]


@dataclass(frozen=True, slots=True)
class PictureFileAccessDecision:
    """One exact, immutable Session-to-file authority observation.

    Retrieval owns this neutral contract.  Concrete Session/workspace adapters may
    implement the port, but Picture never imports those domains or infers access
    from project-local rows alone.
    """

    session_id: str
    file_id: str
    file_version_id: str
    picture_id: str | None
    allowed: bool
    authority_snapshot_id: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("session_id", self.session_id),
            ("file_id", self.file_id),
            ("file_version_id", self.file_version_id),
        ):
            _authority_identifier(value, name)
        if self.picture_id is not None:
            _authority_identifier(self.picture_id, "picture_id")
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be a bool")
        if self.allowed:
            _authority_identifier(
                self.authority_snapshot_id,
                "authority_snapshot_id",
            )
            if self.reason_code is not None:
                raise ValueError("an allowed decision must not carry reason_code")
            return
        if self.authority_snapshot_id is not None:
            raise ValueError("a denied decision must not carry authority_snapshot_id")
        if (
            not isinstance(self.reason_code, str)
            or _AUTHORITY_REASON_CODE_PATTERN.fullmatch(self.reason_code) is None
        ):
            raise ValueError("a denied decision requires a bounded reason_code")


class PictureFileAccessAuthorityPort(Protocol):
    """Recheck one exact online Picture file scope without leaking Session types."""

    def authorize(
        self,
        *,
        session_id: str,
        file_id: str,
        file_version_id: str,
        picture_id: str | None,
    ) -> PictureFileAccessDecision: ...


@dataclass(frozen=True, slots=True)
class _ResolvedObservation:
    record: PictureObservationRecord
    picture: PictureRecord
    unit: PictureUnitRecord
    ref: SourceUnitRef
    content: str
    source_filter: SourceFilter
    citation: dict[str, str]


@dataclass(frozen=True, slots=True)
class _ScopeState:
    file_content_sha256: str
    resolved: tuple[_ResolvedObservation, ...]
    snapshot_id: str
    enumeration_complete: bool

    @property
    def revision_map(self) -> dict[str, str]:
        return {
            item.record.observation_id: item.record.payload_sha256
            for item in self.resolved
        }


class PictureObservationSourceAdapter:
    """Read only active, non-blank observation entries from the shared project DB."""

    source_type = SourceType.PICTURE

    def __init__(
        self,
        *,
        connection_factory: ConnectionFactory = connect_current,
        access_authority: PictureFileAccessAuthorityPort | None = None,
        repository: PictureObservationRepository | None = None,
        window_policy: PictureObservationWindowPolicy | None = None,
        maximum_access_bindings: int = DEFAULT_MAX_PICTURE_ACCESS_BINDINGS,
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        if (
            isinstance(maximum_access_bindings, bool)
            or not isinstance(maximum_access_bindings, int)
            or maximum_access_bindings <= 0
        ):
            raise ValueError("maximum_access_bindings must be a positive integer")
        if access_authority is not None and not callable(
            getattr(access_authority, "authorize", None)
        ):
            raise TypeError("access_authority must implement authorize")
        self._connection_factory = connection_factory
        self._access_authority = access_authority
        self._repository = repository or PictureObservationRepository()
        self._policy = window_policy or DEFAULT_PICTURE_OBSERVATION_WINDOW_POLICY
        self._maximum_access_bindings = maximum_access_bindings

    @property
    def window_policy(self) -> PictureObservationWindowPolicy:
        """Expose the composition-frozen FIFO policy without permitting mutation."""

        return self._policy

    def catalog_source_filters(
        self,
        access: SourceAccess,
    ) -> tuple[SourceFilter, ...]:
        if access.source_type is not self.source_type:
            raise ValueError("Picture catalog filters require Picture access")
        if access.availability is not SourceAvailability.READY:
            raise ValueError("Picture catalog filters require ready access")
        scope = _validated_online_scope(access.source_filter)
        if scope is None:
            return ()
        return (_index_filter_for_online_scope(scope),)

    def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess:
        scope = _validated_online_scope(source_filter)
        if scope is None:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="picture_scope_invalid",
            )
        if self._access_authority is None:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="picture_file_access_authority_missing",
            )
        try:
            decision = self._authorize_online_scope(scope)
            if not decision.allowed:
                return SourceAccess(
                    self.source_type,
                    source_filter,
                    SourceAvailability.BLOCKED,
                    reason_code=decision.reason_code,
                )
            with self._connection_factory() as conn:
                state = self._read_scope_state(
                    conn,
                    source_filter=source_filter,
                    authority_snapshot_id=str(decision.authority_snapshot_id),
                    maximum=self._maximum_access_bindings,
                )
        except Exception:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.UNAVAILABLE,
                reason_code="picture_authority_unavailable",
            )
        if state is None:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="picture_file_version_not_current",
            )
        if not state.enumeration_complete:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="picture_access_binding_limit_exceeded",
            )
        if not state.resolved:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.EMPTY,
            )
        return SourceAccess(
            self.source_type,
            source_filter,
            SourceAvailability.READY,
            source_snapshot_id=state.snapshot_id,
            source_revision_map=state.revision_map,
        )

    def revalidate_retrieval_access(self, access: SourceAccess) -> SourceAccess:
        if access.source_type is not self.source_type:
            raise ValueError("Picture revalidation requires Picture access")
        current = self.open_retrieval_access(access.source_filter)
        if access.availability is not SourceAvailability.READY:
            return current
        if (
            current.availability is SourceAvailability.READY
            and current.source_snapshot_id == access.source_snapshot_id
            and dict(current.source_revision_map) == dict(access.source_revision_map)
        ):
            return current
        return SourceAccess(
            self.source_type,
            access.source_filter,
            SourceAvailability.BLOCKED,
            reason_code="picture_source_changed_during_retrieval",
        )

    def fetch_units(
        self,
        access: SourceAccess,
        refs: Sequence[SourceUnitRef],
    ) -> tuple[SourceUnit, ...]:
        if (
            access.source_type is not self.source_type
            or access.availability is not SourceAvailability.READY
        ):
            return ()
        scope = _validated_online_scope(access.source_filter)
        if scope is None or self._access_authority is None:
            return ()
        decision = self._authorize_online_scope(scope)
        if not decision.allowed:
            return ()
        units: list[SourceUnit] = []
        index_filter = _index_filter_for_online_scope(scope)
        with self._connection_factory() as conn:
            state = self._read_scope_state(
                conn,
                source_filter=access.source_filter,
                authority_snapshot_id=str(decision.authority_snapshot_id),
                maximum=self._maximum_access_bindings,
            )
            if (
                state is None
                or not state.enumeration_complete
                or state.snapshot_id != access.source_snapshot_id
                or state.revision_map != dict(access.source_revision_map)
            ):
                return ()
            for ref in refs:
                if ref.source_type is not self.source_type:
                    continue
                identity = parse_picture_observation_source_unit_id(
                    ref.source_unit_id
                )
                if identity is None:
                    continue
                if (
                    access.source_revision_map.get(identity.observation_id)
                    != ref.source_revision
                ):
                    continue
                resolved = self._resolve_observation(
                    conn,
                    observation_id=identity.observation_id,
                )
                if (
                    resolved is None
                    or resolved.ref != ref
                    or not index_filter.selects(resolved.source_filter)
                ):
                    continue
                units.append(
                    SourceUnit(
                        ref=resolved.ref,
                        content=resolved.content,
                        citation=resolved.citation,
                    )
                )
        return tuple(units)

    def get_current_index_binding_snapshot(
        self,
        access: SourceAccess,
        *,
        maximum_bindings: int,
    ) -> SourceIndexBindingSnapshot:
        if access.source_type is not self.source_type:
            raise ValueError("Picture binding snapshots require Picture access")
        if access.availability is not SourceAvailability.READY:
            raise ValueError("Picture binding snapshots require ready access")
        if (
            isinstance(maximum_bindings, bool)
            or not isinstance(maximum_bindings, int)
            or maximum_bindings <= 0
        ):
            raise ValueError("maximum_bindings must be a positive integer")
        scope = _validated_online_scope(access.source_filter)
        if scope is None or self._access_authority is None:
            return SourceIndexBindingSnapshot(source_snapshot_is_current=False)
        decision = self._authorize_online_scope(scope)
        if not decision.allowed:
            return SourceIndexBindingSnapshot(source_snapshot_is_current=False)
        with self._connection_factory() as conn:
            state = self._read_scope_state(
                conn,
                source_filter=access.source_filter,
                authority_snapshot_id=str(decision.authority_snapshot_id),
                maximum=self._maximum_access_bindings,
            )
        if state is None or not state.enumeration_complete:
            return SourceIndexBindingSnapshot(source_snapshot_is_current=False)
        snapshot_is_current = (
            state.snapshot_id == access.source_snapshot_id
            and state.revision_map == dict(access.source_revision_map)
        )
        if not snapshot_is_current:
            return SourceIndexBindingSnapshot(source_snapshot_is_current=False)
        complete = len(state.resolved) <= maximum_bindings
        return SourceIndexBindingSnapshot(
            source_snapshot_is_current=True,
            bindings=tuple(
                SourceIndexBinding(
                    source_unit_id=item.ref.source_unit_id,
                    source_revision=item.ref.source_revision,
                    indexed_content_hash=item.ref.indexed_content_hash,
                )
                for item in state.resolved[:maximum_bindings]
            ),
            binding_enumeration_complete=complete,
        )

    def read_for_index(
        self,
        event: RetrievalUpdateEvent,
    ) -> IndexableSourceUnit | None:
        if event.ref.source_type is not self.source_type:
            return None
        current = self.read_current_for_reconcile(event.ref)
        if current is None or current.ref != event.ref:
            return None
        return current

    def read_current_for_reconcile(
        self,
        ref: SourceUnitRef,
    ) -> IndexableSourceUnit | None:
        if ref.source_type is not self.source_type:
            return None
        identity = parse_picture_observation_source_unit_id(ref.source_unit_id)
        if identity is None:
            return None
        with self._connection_factory() as conn:
            resolved = self._resolve_observation(
                conn,
                observation_id=identity.observation_id,
            )
        if resolved is None or resolved.ref != ref:
            return None
        return IndexableSourceUnit(
            ref=resolved.ref,
            source_filter=resolved.source_filter,
            content=resolved.content,
        )

    def list_indexable_units_for_backfill(
        self,
        source_filter: SourceFilter,
    ) -> tuple[IndexableSourceUnit, ...]:
        with self._connection_factory() as conn:
            return self._list_indexable_units_in_connection(
                conn,
                source_filter=source_filter,
            )

    def list_indexable_units_for_backfill_in_transaction(
        self,
        conn: sqlite3.Connection,
        source_filter: SourceFilter,
    ) -> tuple[IndexableSourceUnit, ...]:
        """Read the same maintenance manifest under a publisher-owned fence."""

        if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction:
            raise ValueError(
                "Picture backfill transaction read requires an active transaction"
            )
        return self._list_indexable_units_in_connection(
            conn,
            source_filter=source_filter,
        )

    def _list_indexable_units_in_connection(
        self,
        conn: sqlite3.Connection,
        *,
        source_filter: SourceFilter,
    ) -> tuple[IndexableSourceUnit, ...]:
        scope = _validated_maintenance_scope(source_filter)
        if scope is None:
            return ()
        records = self._repository.list_current_active_for_scope(
            conn,
            max_active_entries=self._policy.max_active_entries,
            file_id=scope.get("file_id"),
            file_version_id=scope.get("file_version_id"),
            picture_id=scope.get("picture_id"),
            nonblank_only=True,
        )
        units: list[IndexableSourceUnit] = []
        for record in records:
            resolved = self._resolve_observation(
                conn,
                observation_id=record.observation_id,
                record=record,
            )
            if resolved is None or not source_filter.selects(
                resolved.source_filter
            ):
                continue
            units.append(
                IndexableSourceUnit(
                    ref=resolved.ref,
                    source_filter=resolved.source_filter,
                    content=resolved.content,
                )
            )
        return tuple(units)

    def _authorize_online_scope(
        self,
        scope: dict[str, str],
    ) -> PictureFileAccessDecision:
        authority = self._access_authority
        if authority is None:
            raise RuntimeError("picture file access authority is not configured")
        decision = authority.authorize(
            session_id=scope["session_id"],
            file_id=scope["file_id"],
            file_version_id=scope["file_version_id"],
            picture_id=scope.get("picture_id"),
        )
        if not isinstance(decision, PictureFileAccessDecision):
            raise TypeError(
                "picture file access authority returned an invalid decision"
            )
        if (
            decision.session_id != scope["session_id"]
            or decision.file_id != scope["file_id"]
            or decision.file_version_id != scope["file_version_id"]
            or decision.picture_id != scope.get("picture_id")
        ):
            raise ValueError(
                "picture file access authority crossed the requested scope"
            )
        return decision

    def _read_scope_state(
        self,
        conn: sqlite3.Connection,
        *,
        source_filter: SourceFilter,
        authority_snapshot_id: str,
        maximum: int,
    ) -> _ScopeState | None:
        scope = _validated_online_scope(source_filter)
        if scope is None:
            return None
        index_filter = _index_filter_for_online_scope(scope)
        file_id = scope["file_id"]
        file_version_id = scope["file_version_id"]
        file_content_sha256 = _current_file_version_content_sha256(
            conn,
            file_id=file_id,
            file_version_id=file_version_id,
        )
        if file_content_sha256 is None:
            return None
        records = self._repository.list_current_active_for_scope(
            conn,
            max_active_entries=self._policy.max_active_entries,
            file_id=file_id,
            file_version_id=file_version_id,
            picture_id=scope.get("picture_id"),
            maximum=maximum + 1,
            nonblank_only=True,
        )
        enumeration_complete = len(records) <= maximum
        resolved: list[_ResolvedObservation] = []
        for record in records[:maximum]:
            item = self._resolve_observation(
                conn,
                observation_id=record.observation_id,
                record=record,
            )
            if item is None or not index_filter.selects(item.source_filter):
                raise ValueError("picture authority changed during scope enumeration")
            resolved.append(item)
        snapshot_id = _source_snapshot_id(
            source_filter=source_filter,
            authority_snapshot_id=authority_snapshot_id,
            file_content_sha256=file_content_sha256,
            resolved=resolved,
            enumeration_complete=enumeration_complete,
        )
        return _ScopeState(
            file_content_sha256=file_content_sha256,
            resolved=tuple(resolved),
            snapshot_id=snapshot_id,
            enumeration_complete=enumeration_complete,
        )

    def _resolve_observation(
        self,
        conn: sqlite3.Connection,
        *,
        observation_id: str,
        record: PictureObservationRecord | None = None,
    ) -> _ResolvedObservation | None:
        current_record = record or self._repository.get_by_id(
            conn,
            observation_id=observation_id,
        )
        if current_record is None or current_record.observation_id != observation_id:
            return None
        if not current_record.draft.text.strip():
            return None
        if not self._repository.is_in_current_active_window(
            conn,
            observation_id=observation_id,
            max_active_entries=self._policy.max_active_entries,
        ):
            return None
        picture = get_picture(conn, current_record.picture_id)
        unit = get_picture_unit(conn, current_record.picture_unit_id)
        if picture is None or unit is None or unit.picture_id != picture.picture_id:
            return None
        file_content_sha256 = _current_file_version_content_sha256(
            conn,
            file_id=picture.file_id,
            file_version_id=picture.file_version_id,
        )
        if file_content_sha256 is None:
            return None
        if (
            picture.source_locator.kind is PictureSourceKind.WHOLE_FILE
            and picture.source_content_sha256 != file_content_sha256
        ):
            return None
        if not picture_binding_is_current(
            conn,
            picture_id=picture.picture_id,
            file_id=picture.file_id,
            file_version_id=picture.file_version_id,
            source_locator=picture.source_locator,
            source_content_sha256=picture.source_content_sha256,
        ):
            return None
        if not picture_unit_binding_is_current(
            conn,
            picture_unit_id=unit.picture_unit_id,
            picture_id=picture.picture_id,
            locator=unit.locator,
            parent_picture_unit_id=unit.parent_picture_unit_id,
            producer_fingerprint=unit.producer_fingerprint,
            pixel_sha256=unit.pixel_sha256,
        ):
            return None
        ref, content = picture_observation_ref_and_content(
            observation_id=current_record.observation_id,
            payload_sha256=current_record.payload_sha256,
            text=current_record.draft.text,
            question=current_record.draft.question,
        )
        source_filter = SourceFilter.from_mapping(
            self.source_type,
            {
                "file_id": picture.file_id,
                "file_version_id": picture.file_version_id,
                "picture_id": picture.picture_id,
            },
        )
        return _ResolvedObservation(
            record=current_record,
            picture=picture,
            unit=unit,
            ref=ref,
            content=content,
            source_filter=source_filter,
            citation=_bounded_citation(
                observation=current_record,
                picture=picture,
                unit=unit,
            ),
        )


def _validated_online_scope(source_filter: SourceFilter) -> dict[str, str] | None:
    if source_filter.source_type is not SourceType.PICTURE:
        return None
    scope = source_filter.as_mapping()
    if not set(scope).issubset(_ONLINE_SCOPE_KEYS):
        return None
    if set(scope) not in (
        {"session_id", "file_id", "file_version_id"},
        {"session_id", "file_id", "file_version_id", "picture_id"},
    ):
        return None
    return scope


def _validated_maintenance_scope(
    source_filter: SourceFilter,
) -> dict[str, str] | None:
    if source_filter.source_type is not SourceType.PICTURE:
        return None
    scope = source_filter.as_mapping()
    if not scope:
        return scope
    if not set(scope).issubset(_INDEX_SCOPE_KEYS):
        return None
    if set(scope) not in (
        {"file_id", "file_version_id"},
        {"file_id", "file_version_id", "picture_id"},
    ):
        return None
    return scope


def _index_filter_for_online_scope(scope: dict[str, str]) -> SourceFilter:
    return SourceFilter.from_mapping(
        SourceType.PICTURE,
        {
            key: value
            for key, value in scope.items()
            if key in _INDEX_SCOPE_KEYS
        },
    )


def _current_file_version_content_sha256(
    conn: sqlite3.Connection,
    *,
    file_id: str,
    file_version_id: str,
) -> str | None:
    row = conn.execute(
        "SELECT version.content_sha256 FROM file_versions AS version "
        "JOIN files AS file ON file.id = version.file_id "
        "AND file.current_version_id = version.id "
        "WHERE version.id = ? AND version.file_id = ?",
        (file_version_id, file_id),
    ).fetchone()
    if row is None:
        return None
    value = str(row[0])
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("file version authority contains an invalid content hash")
    return value


def _source_snapshot_id(
    *,
    source_filter: SourceFilter,
    authority_snapshot_id: str,
    file_content_sha256: str,
    resolved: Sequence[_ResolvedObservation],
    enumeration_complete: bool,
) -> str:
    material = {
        "contract": "picture-observation-source-snapshot-v1",
        "authority_snapshot_id": authority_snapshot_id,
        "enumeration_complete": enumeration_complete,
        "file_content_sha256": file_content_sha256,
        "scope": source_filter.as_mapping(),
        "units": sorted(
            (
                item.ref.source_unit_id,
                item.ref.source_revision,
                item.ref.indexed_content_hash,
            )
            for item in resolved
        ),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "picture-observation:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _authority_identifier(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_AUTHORITY_IDENTIFIER_LENGTH
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a bounded non-empty identifier")


def _bounded_citation(
    *,
    observation: PictureObservationRecord,
    picture: PictureRecord,
    unit: PictureUnitRecord,
) -> dict[str, str]:
    citation = {
        "file_id": picture.file_id,
        "file_version_id": picture.file_version_id,
        "picture_id": picture.picture_id,
        "picture_unit_id": unit.picture_unit_id,
        "observation_id": observation.observation_id,
        "source_kind": picture.source_locator.kind.value,
        "source_locator_sha256": picture.source_locator.sha256,
        "unit_kind": unit.locator.kind.value,
        "unit_locator_sha256": unit.locator.sha256,
        "modality": observation.draft.modality.value,
        "observation_kind": observation.draft.kind,
        "purpose": observation.draft.purpose,
        "processor_fingerprint": observation.draft.processor_fingerprint,
        "created_at": observation.created_at,
    }
    if picture.source_locator.kind is PictureSourceKind.DOCUMENT_SURFACE:
        payload = picture.source_locator.payload
        citation["surface_kind"] = str(payload["surface_kind"])
        citation["surface_ordinal"] = str(payload["ordinal"])
    if observation.draft.question is not None:
        citation["question"] = observation.draft.question
    return citation


__all__ = [
    "DEFAULT_MAX_ACTIVE_PICTURE_OBSERVATIONS",
    "DEFAULT_MAX_PICTURE_ACCESS_BINDINGS",
    "PictureFileAccessAuthorityPort",
    "PictureFileAccessDecision",
    "PictureObservationSourceAdapter",
]
