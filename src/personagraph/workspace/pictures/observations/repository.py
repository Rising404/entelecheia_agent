"""SQLite persistence for the append-only picture observation ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3

from .contracts import (
    PictureObservationAppendResult,
    PictureObservationDraft,
    PictureObservationModality,
    PictureObservationRecord,
    PictureObservationStructuredPayload,
    canonical_timestamp,
)


_OBSERVATION_COLUMNS = (
    "sequence, observation_id, picture_id, picture_unit_id, "
    "logical_invocation_id, request_ordinal, modality, purpose, question, kind, text, "
    "structured_payload_json, uncertainty, processor_fingerprint, "
    "prompt_fingerprint, request_sha256, output_sha256, payload_sha256, created_at"
)
_OBSERVATION_COLUMNS_ALIASED = (
    "observation.sequence, observation.observation_id, observation.picture_id, "
    "observation.picture_unit_id, observation.logical_invocation_id, "
    "observation.request_ordinal, observation.modality, observation.purpose, "
    "observation.question, observation.kind, observation.text, "
    "observation.structured_payload_json, "
    "observation.uncertainty, observation.processor_fingerprint, "
    "observation.prompt_fingerprint, observation.request_sha256, "
    "observation.output_sha256, observation.payload_sha256, observation.created_at"
)
_APPEND_SAVEPOINT = "picture_observation_append"


class PictureObservationRepositoryError(RuntimeError):
    """Base class for observation persistence failures."""


class PictureObservationTransactionRequired(PictureObservationRepositoryError):
    """A write was attempted outside a caller-owned transaction."""


class PictureObservationForeignKeysRequired(PictureObservationRepositoryError):
    """The supplied connection does not enforce SQLite foreign keys."""


class PictureObservationUnitNotFound(PictureObservationRepositoryError):
    """A proposed evidence unit is absent or belongs to another picture."""


class PictureObservationIdempotencyConflict(PictureObservationRepositoryError):
    """An idempotency key was replayed for different request semantics."""


class PictureObservationPersistenceConflict(PictureObservationRepositoryError):
    """Persisted rows violate the observation domain's invariants."""


class PictureObservationRepository:
    """Stateless repository operating only on an explicit SQLite connection."""

    def append_in_transaction(
        self,
        conn: sqlite3.Connection,
        draft: PictureObservationDraft,
        *,
        created_at: str,
    ) -> PictureObservationAppendResult:
        """Atomically append within a caller-owned transaction.

        Callers should use ``BEGIN IMMEDIATE`` for predictable writer serialization.
        A deferred transaction may surface ``SQLITE_BUSY`` under competing writers and
        should be retried by its outer transaction owner.
        """

        _require_write_transaction(conn)
        if not isinstance(draft, PictureObservationDraft):
            raise TypeError("draft must be PictureObservationDraft")
        stored_at = canonical_timestamp(created_at, name="created_at")

        conn.execute(f"SAVEPOINT {_APPEND_SAVEPOINT}")
        try:
            result = self._append_under_savepoint(
                conn,
                draft=draft,
                stored_at=stored_at,
            )
        except BaseException:
            conn.execute(f"ROLLBACK TO SAVEPOINT {_APPEND_SAVEPOINT}")
            conn.execute(f"RELEASE SAVEPOINT {_APPEND_SAVEPOINT}")
            raise
        conn.execute(f"RELEASE SAVEPOINT {_APPEND_SAVEPOINT}")
        return result

    def _append_under_savepoint(
        self,
        conn: sqlite3.Connection,
        *,
        draft: PictureObservationDraft,
        stored_at: str,
    ) -> PictureObservationAppendResult:
        existing = self.get_by_idempotency_key(
            conn,
            picture_id=draft.picture_id,
            logical_invocation_id=draft.logical_invocation_id,
            request_ordinal=draft.request_ordinal,
        )
        if existing is not None:
            return _resolve_replay(existing=existing, draft=draft)

        self._require_evidence_unit(conn, draft=draft)
        observation_id = _observation_id(draft)
        structured_payload_json = (
            None
            if draft.structured_payload is None
            else draft.structured_payload.canonical_json
        )
        try:
            cursor = conn.execute(
                "INSERT INTO picture_observations "
                "(observation_id, picture_id, picture_unit_id, logical_invocation_id, "
                "request_ordinal, modality, purpose, question, kind, text, "
                "structured_payload_json, uncertainty, processor_fingerprint, "
                "prompt_fingerprint, request_sha256, output_sha256, payload_sha256, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(picture_id, logical_invocation_id, request_ordinal) DO NOTHING",
                (
                    observation_id,
                    draft.picture_id,
                    draft.picture_unit_id,
                    draft.logical_invocation_id,
                    draft.request_ordinal,
                    draft.modality.value,
                    draft.purpose,
                    draft.question,
                    draft.kind,
                    draft.text,
                    structured_payload_json,
                    draft.uncertainty,
                    draft.processor_fingerprint,
                    draft.prompt_fingerprint,
                    draft.request_sha256,
                    draft.output_sha256,
                    draft.payload_sha256,
                    stored_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PictureObservationPersistenceConflict(
                "observation insert violated an immutable storage constraint"
            ) from exc
        if cursor.rowcount == 0:
            replayed = self.get_by_idempotency_key(
                conn,
                picture_id=draft.picture_id,
                logical_invocation_id=draft.logical_invocation_id,
                request_ordinal=draft.request_ordinal,
            )
            if replayed is None:
                raise PictureObservationPersistenceConflict(
                    "observation insert was ignored without an idempotent row"
                )
            return _resolve_replay(existing=replayed, draft=draft)

        sequence = cursor.lastrowid
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise PictureObservationPersistenceConflict(
                "observation insert did not allocate a valid global sequence"
            )
        return PictureObservationAppendResult(
            observation=PictureObservationRecord(
                observation_id=observation_id,
                sequence=sequence,
                draft=draft,
                request_sha256=draft.request_sha256,
                output_sha256=draft.output_sha256,
                payload_sha256=draft.payload_sha256,
                created_at=stored_at,
            ),
            inserted=True,
        )

    def get_by_id(
        self,
        conn: sqlite3.Connection,
        *,
        observation_id: str,
    ) -> PictureObservationRecord | None:
        row = conn.execute(
            f"SELECT {_OBSERVATION_COLUMNS} FROM picture_observations "
            "WHERE observation_id=?",
            (observation_id,),
        ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def list_current_active_for_scope(
        self,
        conn: sqlite3.Connection,
        *,
        max_active_entries: int,
        file_id: str | None = None,
        file_version_id: str | None = None,
        picture_id: str | None = None,
        maximum: int | None = None,
        nonblank_only: bool = False,
    ) -> tuple[PictureObservationRecord, ...]:
        """Enumerate each current picture's active FIFO window in global FIFO order.

        Blank observations deliberately occupy a FIFO slot; downstream projections
        decide whether an active ledger entry has indexable content.
        ``maximum`` exists so online freshness/coverage checks can prove that their
        bounded enumeration was complete.  Offline backfill may pass ``None``.
        """

        if (
            isinstance(max_active_entries, bool)
            or not isinstance(max_active_entries, int)
            or not 1 <= max_active_entries <= 256
        ):
            raise ValueError("max_active_entries must be between 1 and 256")
        for name, value in (
            ("file_id", file_id),
            ("file_version_id", file_version_id),
            ("picture_id", picture_id),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be non-empty when provided")
        if file_version_id is not None and file_id is None:
            raise ValueError("file_version_id requires file_id")
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("maximum must be a positive integer when provided")
        if not isinstance(nonblank_only, bool):
            raise ValueError("nonblank_only must be a bool")

        clauses: list[str] = []
        parameters: list[object] = []
        if file_id is not None:
            clauses.append("picture.file_id = ?")
            parameters.append(file_id)
        if file_version_id is not None:
            clauses.append("picture.file_version_id = ?")
            parameters.append(file_version_id)
        if picture_id is not None:
            clauses.append("picture.picture_id = ?")
            parameters.append(picture_id)
        clauses.append(
            "(SELECT COUNT(*) FROM picture_observations AS newer "
            "WHERE newer.picture_id = observation.picture_id "
            "AND newer.sequence > observation.sequence) < ?"
        )
        parameters.append(max_active_entries)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        limit = ""
        # SQLite ``trim`` does not implement Python's/Unicode's whitespace rule.
        # The retrieval identity normalizes with ``str.strip()``, so non-blank
        # filtering must use the same rule instead of a subtly different SQL one.
        if maximum is not None and not nonblank_only:
            limit = " LIMIT ?"
            parameters.append(maximum)
        cursor = conn.execute(
            f"SELECT {_OBSERVATION_COLUMNS_ALIASED} "
            "FROM picture_observations AS observation "
            "JOIN pictures AS picture ON picture.picture_id = observation.picture_id "
            "JOIN file_versions AS version ON version.id = picture.file_version_id "
            "AND version.file_id = picture.file_id "
            "JOIN files AS file ON file.id = picture.file_id "
            "AND file.current_version_id = picture.file_version_id"
            + where
            + " ORDER BY observation.sequence, observation.observation_id"
            + limit,
            tuple(parameters),
        )
        if not nonblank_only:
            return tuple(self._record_from_row(row) for row in cursor.fetchall())
        records: list[PictureObservationRecord] = []
        for row in cursor:
            record = self._record_from_row(row)
            if not record.draft.text.strip():
                continue
            records.append(record)
            if maximum is not None and len(records) >= maximum:
                break
        return tuple(records)

    def get_immediately_before(
        self,
        conn: sqlite3.Connection,
        *,
        picture_id: str,
        sequence: int,
    ) -> PictureObservationRecord | None:
        """Read the nearest historical FIFO entry before ``sequence``."""

        if not isinstance(picture_id, str) or not picture_id.strip():
            raise ValueError("picture_id must be a non-empty string")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ValueError("sequence must be a positive integer")
        row = conn.execute(
            f"SELECT {_OBSERVATION_COLUMNS} FROM picture_observations "
            "WHERE picture_id = ? AND sequence < ? "
            "ORDER BY sequence DESC LIMIT 1",
            (picture_id, sequence),
        ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def is_in_current_active_window(
        self,
        conn: sqlite3.Connection,
        *,
        observation_id: str,
        max_active_entries: int,
    ) -> bool:
        """Check current file-version authority and per-picture FIFO membership."""

        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("observation_id must be a non-empty string")
        if (
            isinstance(max_active_entries, bool)
            or not isinstance(max_active_entries, int)
            or not 1 <= max_active_entries <= 256
        ):
            raise ValueError("max_active_entries must be between 1 and 256")
        row = conn.execute(
            "SELECT EXISTS ("
            "SELECT 1 FROM picture_observations AS observation "
            "JOIN pictures AS picture ON picture.picture_id = observation.picture_id "
            "JOIN file_versions AS version ON version.id = picture.file_version_id "
            "AND version.file_id = picture.file_id "
            "JOIN files AS file ON file.id = picture.file_id "
            "AND file.current_version_id = picture.file_version_id "
            "WHERE observation.observation_id = ? "
            "AND (SELECT COUNT(*) FROM picture_observations AS newer "
            "WHERE newer.picture_id = observation.picture_id "
            "AND newer.sequence > observation.sequence) < ?)"
            ,
            (observation_id, max_active_entries),
        ).fetchone()
        return bool(row[0])

    def list_recent_fifo(
        self,
        conn: sqlite3.Connection,
        *,
        picture_id: str,
        limit: int,
    ) -> tuple[PictureObservationRecord, ...]:
        """Return the latest per-picture entries in oldest-to-newest FIFO order."""

        if not isinstance(picture_id, str) or not picture_id.strip():
            raise ValueError("picture_id must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 256:
            raise ValueError("limit must be between 1 and 256")
        rows = conn.execute(
            f"SELECT {_OBSERVATION_COLUMNS} FROM picture_observations "
            "WHERE picture_id=? ORDER BY sequence DESC LIMIT ?",
            (picture_id, limit),
        ).fetchall()
        records = [self._record_from_row(row) for row in rows]
        records.reverse()
        return tuple(records)

    def get_by_idempotency_key(
        self,
        conn: sqlite3.Connection,
        *,
        picture_id: str,
        logical_invocation_id: str,
        request_ordinal: int,
    ) -> PictureObservationRecord | None:
        """Read an already committed logical request before provider execution."""

        if not isinstance(picture_id, str) or not picture_id.strip():
            raise ValueError("picture_id must be a non-empty string")
        if (
            not isinstance(logical_invocation_id, str)
            or not logical_invocation_id.strip()
        ):
            raise ValueError("logical_invocation_id must be a non-empty string")
        if (
            isinstance(request_ordinal, bool)
            or not isinstance(request_ordinal, int)
            or request_ordinal < 0
        ):
            raise ValueError("request_ordinal must be a non-negative integer")
        row = conn.execute(
            f"SELECT {_OBSERVATION_COLUMNS} FROM picture_observations "
            "WHERE picture_id=? AND logical_invocation_id=? AND request_ordinal=?",
            (
                picture_id,
                logical_invocation_id,
                request_ordinal,
            ),
        ).fetchone()
        return self._record_from_row(row) if row is not None else None

    @staticmethod
    def _require_evidence_unit(
        conn: sqlite3.Connection,
        *,
        draft: PictureObservationDraft,
    ) -> None:
        row = conn.execute(
            "SELECT 1 FROM picture_units WHERE picture_unit_id=? AND picture_id=?",
            (draft.picture_unit_id, draft.picture_id),
        ).fetchone()
        if row is None:
            raise PictureObservationUnitNotFound(
                "picture unit is absent or belongs to another picture: "
                f"{draft.picture_unit_id}"
            )

    def _record_from_row(
        self,
        row: sqlite3.Row | tuple[object, ...],
    ) -> PictureObservationRecord:
        observation_id = str(row[1])
        picture_id = str(row[2])
        try:
            structured_payload = (
                None
                if row[11] is None
                else PictureObservationStructuredPayload.from_canonical_json(
                    str(row[11])
                )
            )
            draft = PictureObservationDraft(
                picture_id=picture_id,
                picture_unit_id=str(row[3]),
                logical_invocation_id=str(row[4]),
                request_ordinal=int(row[5]),
                modality=PictureObservationModality(str(row[6])),
                purpose=str(row[7]),
                question=None if row[8] is None else str(row[8]),
                kind=str(row[9]),
                text=str(row[10]),
                structured_payload=structured_payload,
                uncertainty=float(row[12]) if row[12] is not None else None,
                processor_fingerprint=str(row[13]),
                prompt_fingerprint=str(row[14]) if row[14] is not None else None,
            )
            expected_observation_id = _observation_id(draft)
            if observation_id != expected_observation_id:
                raise ValueError("stored observation_id does not match idempotency key")
            return PictureObservationRecord(
                observation_id=observation_id,
                sequence=int(row[0]),
                draft=draft,
                request_sha256=str(row[15]),
                output_sha256=str(row[16]),
                payload_sha256=str(row[17]),
                created_at=str(row[18]),
            )
        except (TypeError, ValueError) as exc:
            raise PictureObservationPersistenceConflict(
                "stored observation violates its canonical contract"
            ) from exc


def _resolve_replay(
    *,
    existing: PictureObservationRecord,
    draft: PictureObservationDraft,
) -> PictureObservationAppendResult:
    if existing.request_sha256 != draft.request_sha256:
        raise PictureObservationIdempotencyConflict(
            "logical_invocation_id and request_ordinal were reused for another request"
        )
    return PictureObservationAppendResult(observation=existing, inserted=False)


def _observation_id(draft: PictureObservationDraft) -> str:
    material = json.dumps(
        {
            "logical_invocation_id": draft.logical_invocation_id,
            "picture_id": draft.picture_id,
            "request_ordinal": draft.request_ordinal,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "pobs_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _require_write_transaction(conn: sqlite3.Connection) -> None:
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be sqlite3.Connection")
    if not conn.in_transaction:
        raise PictureObservationTransactionRequired(
            "picture observation writes require a caller-owned transaction"
        )
    foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()
    if foreign_keys is None or int(foreign_keys[0]) != 1:
        raise PictureObservationForeignKeysRequired(
            "picture observation writes require PRAGMA foreign_keys=ON"
        )


__all__ = [
    "PictureObservationForeignKeysRequired",
    "PictureObservationIdempotencyConflict",
    "PictureObservationPersistenceConflict",
    "PictureObservationRepository",
    "PictureObservationRepositoryError",
    "PictureObservationTransactionRequired",
    "PictureObservationUnitNotFound",
]
