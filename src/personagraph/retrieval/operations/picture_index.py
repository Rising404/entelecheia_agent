"""把已提交的图片观察同步到冻结 FILE 索引，并安全发布首次 generation。

工具应用层只传入冻结组合和 Observation 身份。这里只消费权威 Outbox；不调用
视觉模型、不解析文件，也不把尚未完成的空索引标记为可查询。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3
import time

from ...workspace.pictures.observations import PictureObservationRepository
from ..contracts import RetrievalStatus, SourceType
from ..lifecycle.corpus import FILE_CORPUS
from ..lifecycle.generation import document_index_methods
from ..lifecycle.outbox import OutboxStatus, RetrievalUpdateEvent, SqliteRetrievalOutbox
from ..lifecycle.sync import RetrievalOutboxConsumer
from ..sources.events import build_picture_observation_upsert_event
from ..sqlite_store import (
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    UnitIndexState,
)
from .document_generation import (
    DocumentGenerationAuthority,
    DocumentGenerationRetryableFailure,
    RetrievalPublicationRefused,
)
from .document_maintenance import DocumentRetrievalComposition


class PictureObservationIndexError(RuntimeError):
    """图片观察尚未被当前精确 FILE generation 安全接纳。"""


class PictureObservationIndex:
    """一次构造绑定一个 FILE 配方；跨配置的 ACTIVE 绝不自动切换。"""

    def __init__(
        self,
        *,
        composition: DocumentRetrievalComposition,
        connect_documents: Callable[[], sqlite3.Connection],
        advance_outbox: Callable[[int], int] | None = None,
    ) -> None:
        if not isinstance(composition, DocumentRetrievalComposition):
            raise TypeError("composition must be DocumentRetrievalComposition")
        if not callable(connect_documents):
            raise TypeError("connect_documents must be callable")
        if advance_outbox is not None and not callable(advance_outbox):
            raise TypeError("advance_outbox must be callable")
        foundation = composition.foundation
        spec = composition.generation_spec
        if foundation.generation_spec != spec or (
            spec.source_types != FILE_CORPUS.generation_source_types
        ):
            raise ValueError("picture indexing requires one exact FILE composition")
        self._composition = composition
        self._connect_documents = connect_documents
        self._advance_outbox = advance_outbox
        self._outbox = SqliteRetrievalOutbox()
        self._repository = PictureObservationRepository()
        self._authority = DocumentGenerationAuthority(
            catalog=foundation.catalog,
            generation_spec=spec,
            method_store=foundation.method_store,
            connect_documents=connect_documents,
            picture_source_reader=foundation.source_adapters[SourceType.PICTURE],
        )
        self._consumer = RetrievalOutboxConsumer(
            outbox=self._outbox,
            sync_service=foundation.sync_service,
            allowed_source_types=FILE_CORPUS.source_types,
            data_version_id=spec.version_id,
        )
        self._target: RetrievalDataVersion | None = None

    def prepare_target(self) -> str:
        """仅选择/建立待建索引；构造与此步骤都不编码内容。"""

        with self._connection():
            pass  # Verify the explicit connection belongs to the frozen catalog.
        self._target = self._authority._select_source_commit_target()
        return self._target.id

    def resolve_target_in_transaction(self, conn: sqlite3.Connection) -> str:
        """供 Picture Outbox publisher 在同一来源提交事务内重验目标。"""

        if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction:
            raise ValueError(
                "picture indexing target requires a caller-owned transaction"
            )
        self._require_database(conn)
        if self._target is None:
            raise PictureObservationIndexError("picture_retrieval_target_not_prepared")
        return self._authority._require_writable_source_commit_target_in_connection(
            conn,
            self._target,
        ).id

    def synchronize(
        self,
        observation_ids: Sequence[str],
        *,
        worker_id: str,
        now: str,
        batch_limit: int = 20,
        max_batches: int = 20,
        deadline_monotonic: float | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> RetrievalDataVersion:
        """先证明原始发布事件完成，再使冷项目中的观察真正可被检索。

        重放不依赖本次 publish 返回的“新事件”列表：从持久 Observation 重建原始
        UPSERT 身份并要求该事件确实存在。失败留下原事件/原观察，后续只重试索引。
        """

        ids = _observation_ids(observation_ids)
        if (
            not worker_id.strip()
            or not now.strip()
            or batch_limit <= 0
            or max_batches <= 0
        ):
            raise ValueError(
                "picture synchronization execution parameters must be valid"
            )
        if deadline_monotonic is not None and (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("deadline_monotonic must be finite or None")

        def check_execution() -> None:
            if checkpoint is not None:
                checkpoint()
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise PictureObservationIndexError("picture_retrieval_wait_timeout")

        check_execution()
        if self._target is None:
            raise PictureObservationIndexError("picture_retrieval_target_not_prepared")
        self._authority._refresh_source_commit_target(self._target)
        events = self._observation_events(ids, now=now)
        self._require_original_events(events)
        current_time = now
        while True:
            self._drain_target(
                worker_id=worker_id, now=current_time,
                batch_limit=batch_limit, max_batches=max_batches, checkpoint=check_execution,
            )
            check_execution()
            ready = self._confirm_events_ready(events)
            if ready is not None:
                return ready
            if deadline_monotonic is None:
                raise PictureObservationIndexError("picture_retrieval_outbox_incomplete")
            # 合法 lease 或 bootstrap 的其他来源仍在处理。共用原 deadline；
            # 不偷取 lease，不持有连接等待，也不另开 backfill/encoder。
            time.sleep(min(0.25, max(0.0, deadline_monotonic - time.monotonic())))
            check_execution()
            current_time = datetime.now(timezone.utc).isoformat()

    def confirm_ready(
        self, observation_ids: Sequence[str], *, now: str,
    ) -> RetrievalDataVersion | None:
        """后台只确认已完成的原事件；不等待、不领取工作、不调用 encoder。"""

        events = self._observation_events(_observation_ids(observation_ids), now=now)
        return self._confirm_events_ready(events)

    def _confirm_events_ready(
        self, events: tuple[RetrievalUpdateEvent, ...],
    ) -> RetrievalDataVersion | None:
        if self._target is None:
            raise PictureObservationIndexError("picture_retrieval_target_not_prepared")
        if not self._original_events_applied(events):
            return None
        target = self._authority._refresh_source_commit_target(self._target)
        self._require_observation_coverage(events)
        if target.role is not RetrievalDataVersionRole.ACTIVE:
            try:
                self._authority.publish_bootstrap_generation()
            except RetrievalPublicationRefused as exc:
                if isinstance(exc.__cause__, DocumentGenerationRetryableFailure):
                    return None
                raise
        active = self._composition.foundation.catalog.active_data_version()
        if active is None:
            return None
        self._target = self._authority._require_exact_active(active)
        return self._target

    def _observation_events(
        self, observation_ids: tuple[str, ...], *, now: str
    ) -> tuple[RetrievalUpdateEvent, ...]:
        events = []
        with self._connection() as conn:
            for observation_id in observation_ids:
                observation = self._repository.get_by_id(
                    conn, observation_id=observation_id
                )
                if observation is None:
                    raise PictureObservationIndexError(
                        "picture_retrieval_observation_missing"
                    )
                event = build_picture_observation_upsert_event(
                    observation=observation,
                    retrieval_data_version=self._composition.generation_spec.version_id,
                    occurred_at=now,
                )
                if event is None:
                    raise PictureObservationIndexError(
                        "picture_retrieval_observation_empty"
                    )
                events.append(event)
        return tuple(events)

    def _require_original_events(
        self, events: tuple[RetrievalUpdateEvent, ...], *, applied: bool = False
    ) -> None:
        complete = self._original_events_applied(events)
        if applied and not complete:
            raise PictureObservationIndexError("picture_retrieval_outbox_incomplete")

    def _original_events_applied(self, events: tuple[RetrievalUpdateEvent, ...]) -> bool:
        complete = True
        with self._connection() as conn:
            for event in events:
                outcome = self._outbox.get_outcome_in_transaction(conn, event.event_id)
                if outcome is None:
                    raise PictureObservationIndexError(
                        "picture_retrieval_outbox_missing"
                    )
                if outcome[0] is OutboxStatus.TERMINAL_FAILED:
                    raise PictureObservationIndexError(
                        "picture_retrieval_outbox_terminal_failure"
                    )
                complete = complete and outcome[0] is OutboxStatus.APPLIED
        return complete

    def _drain_target(
        self, *, worker_id: str, now: str, batch_limit: int, max_batches: int,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        for _ in range(max_batches):
            if checkpoint is not None:
                checkpoint()
            if self._advance_outbox is not None:
                if not self._advance_outbox(batch_limit):
                    break
                continue
            with self._connection() as conn:
                results = self._consumer.consume_due(
                    conn,
                    worker_id=worker_id,
                    now=now,
                    limit=batch_limit,
                )
            if not results:
                break

    def _require_observation_coverage(
        self, events: tuple[RetrievalUpdateEvent, ...]
    ) -> None:
        foundation = self._composition.foundation
        required = frozenset(
            document_index_methods(self._composition.generation_spec.index_recipe)
        )
        for event in events:
            source = foundation.source_adapters[SourceType.PICTURE].read_for_index(
                event
            )
            stored = foundation.catalog.get_unit(
                event.ref, event.retrieval_data_version
            )
            if source is None and (
                stored is None
                or stored.unit.retrieval_status is RetrievalStatus.TRASHED
            ):
                with self._connection() as conn:
                    if self._has_applied_removal(conn, event):
                        # A READY receipt may lose its session acknowledgement,
                        # then leave the FIFO before recovery. Prove its later
                        # removal instead of resurrecting an expired observation.
                        continue
            if (
                source is None
                or stored is None
                or stored.unit.ref != source.ref
                or stored.unit.source_filter != source.source_filter
                or stored.unit.retrieval_status is not RetrievalStatus.ACTIVE
                or stored.index_state is not UnitIndexState.READY
            ):
                raise PictureObservationIndexError(
                    "picture_retrieval_coverage_incomplete"
                )
            health = {
                item.method: item
                for item in foundation.method_store.method_index_health(stored.unit_id)
            }
            if any(
                method not in health
                or health[method].expected_state != "ready"
                or not health[method].representation_present
                for method in required
            ):
                raise PictureObservationIndexError(
                    "picture_retrieval_method_coverage_incomplete"
                )

    @staticmethod
    def _has_applied_removal(
        conn: sqlite3.Connection, event: RetrievalUpdateEvent
    ) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM retrieval_update_outbox original "
                "JOIN retrieval_update_outbox cleanup "
                "ON cleanup.data_version_id=original.data_version_id "
                "AND cleanup.source_type=original.source_type "
                "AND cleanup.source_unit_id=original.source_unit_id "
                "AND cleanup.source_revision=original.source_revision "
                "AND cleanup.indexed_content_hash=original.indexed_content_hash "
                "WHERE original.event_id=? AND cleanup.authority_sequence>original.authority_sequence "
                "AND cleanup.kind IN ('trash','purge') AND cleanup.status='applied' LIMIT 1",
                (event.event_id,),
            ).fetchone()
            is not None
        )

    def _require_database(self, conn: sqlite3.Connection) -> None:
        databases = conn.execute("PRAGMA database_list").fetchall()
        main = next((str(row[2]) for row in databases if row[1] == "main"), "")
        if (
            not main
            or Path(main).resolve()
            != self._composition.foundation.catalog.db_path.resolve()
        ):
            raise PictureObservationIndexError("picture_retrieval_database_mismatch")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect_documents()
        if not isinstance(conn, sqlite3.Connection):
            raise TypeError("connect_documents must return sqlite3.Connection")
        try:
            conn.row_factory = sqlite3.Row
            self._require_database(conn)
            yield conn
        finally:
            conn.close()


def _observation_ids(values: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(values, (str, bytes)) or not values
        or any(not isinstance(item, str) or not item.strip() for item in values)
    ):
        raise ValueError("observation_ids must be a non-empty sequence of identifiers")
    return tuple(dict.fromkeys(values))


__all__ = ["PictureObservationIndex", "PictureObservationIndexError"]
