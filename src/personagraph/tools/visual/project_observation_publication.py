"""视觉工具的共享 Project 发布应用边界：回执转命令、原子发布确认和无源恢复。

文件 ID 与路径入口复用这里，不在 handler 中写数据库或各自实现恢复。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import TYPE_CHECKING

from ...input_processing.vision.contracts import (
    VisionDetail,
    VisionObservation,
    VisionPurpose,
    VisionRegion,
    VisionStatus,
)
from ...input_processing.vision.providers.http import PROMPT_CONTRACT_VERSION
from ...input_processing.vision.imaging import PreparedVisualArtifactReceipt
from ...workspace.files.access import AuthorizedFileSource
from ...workspace.pictures.contracts import PictureSourceLocator, PictureUnitLocator
from ...workspace.pictures.observations import PictureObservationStructuredPayload
from ...workspace.pictures.project_publication import (
    ProjectPictureBinding,
    ProjectPictureBindingStale,
    ProjectPictureObservationSpec,
    ProjectPicturePublicationCommand,
    ProjectPicturePublicationResult,
    ProjectPicturePublicationService,
    ProjectPictureUnitSpec,
)
from ...workspace.storage.context import current as current_workspace_database
from ...retrieval.sources.picture_publication import PictureObservationOutboxPublisher
from ...retrieval.operations.picture_index import PictureObservationIndexError
from ...runtime.model_calls.vision import (
    MountedVisualCallReceipt,
    MountedVisualCallLedgerError,
    MountedVisualPictureLocator,
    MountedVisualProjectPublicationEnvelope,
    MountedVisualProjectPublicationTarget,
    MountedVisualPublicationState,
    SqliteMountedVisualCallLedger,
)
from .file_visual_tools import MAX_OBSERVATION_CHARS
from .question_contract import visual_call_identity
from .visual_observation_service import (
    HostVisualObservationExecution,
    VisualObservationRequest,
    VisualObservationResult,
    VisualObservationService,
)
from .visual_tool_boundary import FrozenVisualToolBoundary, VisualUnitRef
from ..execution_context import ToolInvocationCancelled, current_tool_execution
from ..retrieval.execution_bridge import file_retrieval_execution

if TYPE_CHECKING:
    from ...retrieval.operations.picture_index import PictureObservationIndex

_VLM_PICTURE_OBSERVATION_CONTRACT = "vlm-picture-observation-v1"
_READY_PUBLICATION_RECOVERY_LIMIT = 100
_MAX_STRUCTURED_METADATA_ITEMS = 12
_MAX_STRUCTURED_METADATA_TEXT_BYTES = 96


@dataclass(frozen=True, slots=True)
class _BoundedMetadataText:
    value: str
    sha256: str
    truncated: bool


class VisualPublicationIndexFailure(MountedVisualCallLedgerError):
    """视觉回执已保存；精确索引失败，但不表示后台必然能够自动恢复。"""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class VisualObservationPublisher:
    """拥有一次视觉结果的发布闭环；provider 仍由共同观察服务调用。"""

    def __init__(
        self,
        *,
        session_id: str,
        call_ledger: SqliteMountedVisualCallLedger,
        picture_publication_service: ProjectPicturePublicationService | None = None,
        picture_index: PictureObservationIndex | None = None,
    ) -> None:
        self._session_id = session_id
        self._call_ledger = call_ledger
        self._picture_publication_service = picture_publication_service
        self._owns_publication_service = picture_publication_service is None
        self._picture_index = picture_index

    def observe(
        self,
        *,
        service: VisualObservationService,
        boundary: FrozenVisualToolBoundary,
        request: VisualObservationRequest,
        binding: AuthorizedFileSource,
    ) -> tuple[VisualObservationResult, ProjectPicturePublicationResult | None]:
        control = current_tool_execution()
        if control is not None:
            control.checkpoint()
        self._ensure_publication_ready()
        if boundary.session_id != self._session_id:
            raise MountedVisualCallLedgerError(
                "visual publication crossed Session authority"
            )
        if self._picture_publication_service.project_id != binding.project_id:
            raise MountedVisualCallLedgerError(
                "visual publication crossed Project authority"
            )
        if len(boundary.units) != 1 or boundary.units[0].unit_id != request.unit_id:
            raise MountedVisualCallLedgerError(
                "visual publication requires one exact unit"
            )
        execution = service.observe_with_durable_receipt(
            boundary,
            request,
            publication_envelope_factory=lambda artifact: (
                build_picture_publication_envelope(
                    binding=binding,
                    unit=boundary.units[0],
                    purpose=request.purpose,
                    detail=request.detail,
                    region=request.region,
                    question=request.question,
                    logical_tool_call_id=visual_call_identity(request.purpose),
                    prepared_artifact=artifact,
                )
            ),
        )
        return execution.projection, self.finish(execution)

    def finish(
        self,
        execution: HostVisualObservationExecution,
    ) -> ProjectPicturePublicationResult | None:
        """Publish/ack a ready result before it is projected as tool success."""

        receipt = execution.durable_receipt
        if receipt is None:
            # Disclosure/preparation refusals have no provider result and no
            # Project semantics to persist.
            return
        state = receipt.publication_state
        if state is MountedVisualPublicationState.PUBLISHED:
            return self._publish_ready_picture_receipt(receipt)
        if state is MountedVisualPublicationState.SKIPPED:
            if receipt.result.status in {
                VisionStatus.COMPLETED,
                VisionStatus.PARTIAL,
            }:
                raise MountedVisualCallLedgerError(
                    "successful visual semantics were terminally skipped"
                )
            return
        if state is not MountedVisualPublicationState.READY:
            raise MountedVisualCallLedgerError(
                "visual publication did not reach a publishable state"
            )
        try:
            published = self._publish_ready_picture_receipt(receipt)
            self._call_ledger.mark_publication_published(receipt)
            return published
        except ProjectPictureBindingStale as exc:
            try:
                self._call_ledger.mark_publication_skipped(
                    receipt,
                    reason_code="project_picture_binding_stale",
                )
            except Exception as ack_exc:
                raise MountedVisualCallLedgerError(
                    "stale Project picture publication could not reach terminal state"
                ) from ack_exc
            raise MountedVisualCallLedgerError(
                "Project picture publication binding is stale"
            ) from exc
        except (MountedVisualCallLedgerError, ToolInvocationCancelled):
            raise
        except Exception as exc:
            raise MountedVisualCallLedgerError(
                "Project picture publication remains ready for recovery"
            ) from exc

    def _publish_ready_picture_receipt(
        self,
        receipt: MountedVisualCallReceipt,
    ) -> ProjectPicturePublicationResult:
        published = self._commit_picture_receipt(receipt)
        if (
            self._picture_index is not None
            and receipt.publication_state is MountedVisualPublicationState.READY
        ):
            # replay 的 outbox_publications 可能为空；观察身份仍然稳定，索引 owner
            # 从权威记录定位待处理事件。索引成功之前不确认物理调用发布完成。
            # PUBLISHED 已完成过索引；旧观察可能退出 FIFO，重放不得将它重新激活。
            control = current_tool_execution()
            execution_options = (
                {"deadline_monotonic": control.deadline_monotonic,
                 "checkpoint": control.checkpoint}
                if control is not None else {}
            )
            try:
                with file_retrieval_execution():
                    self._picture_index.synchronize(
                        tuple(
                            commit.observation.observation_id
                            for commit in published.observation_commits
                        ),
                        worker_id="visual-publication-" + _sha256_text(self._session_id)[:24],
                        now=datetime.now(timezone.utc).isoformat(),
                        **execution_options,
                    )
            except PictureObservationIndexError as exc:
                # 仅允许已知索引结果降为页级失败；身份、数据库、原事件缺失
                # 等完整性错误仍沿原失败链抛出，不继续处理批次剩余页面。
                if str(exc) not in {
                    "picture_retrieval_outbox_incomplete",
                    "picture_retrieval_wait_timeout",
                    "picture_retrieval_outbox_terminal_failure",
                    "picture_retrieval_coverage_incomplete",
                    "picture_retrieval_method_coverage_incomplete",
                }:
                    raise
                raise VisualPublicationIndexFailure(str(exc)) from exc
        return published

    def _commit_picture_receipt(
        self, receipt: MountedVisualCallReceipt,
    ) -> ProjectPicturePublicationResult:
        command = build_picture_publication_command(receipt)
        self._ensure_publication_ready()
        return self._picture_publication_service.publish(command)

    def _ensure_publication_ready(self) -> None:
        try:
            if self._owns_publication_service and self._picture_index is None:
                self._picture_index = _default_picture_index()
                self._picture_publication_service = None
            if self._picture_index is not None:
                self._picture_index.prepare_target()
            if self._picture_publication_service is None:
                self._picture_publication_service = (
                    _default_picture_publication_service(self._picture_index)
                )
            if self._picture_publication_service is None:
                raise MountedVisualCallLedgerError(
                    "Project picture publication service is unavailable"
                )
        except MountedVisualCallLedgerError:
            raise
        except Exception as exc:
            raise MountedVisualCallLedgerError(
                "Project picture publication index is unavailable"
            ) from exc

    def recover_ready(self) -> int:
        """Non-blocking ack recovery; never wait on this maintenance owner's work."""

        try:
            receipts = self._call_ledger.list_ready_publications(
                session_id=self._session_id,
                limit=_READY_PUBLICATION_RECOVERY_LIMIT,
            )
        except Exception:
            return 0
        recovered = 0
        for receipt in receipts:
            try:
                published = self._commit_picture_receipt(receipt)
                if self._picture_index is not None and self._picture_index.confirm_ready(
                    tuple(commit.observation.observation_id
                          for commit in published.observation_commits),
                    now=datetime.now(timezone.utc).isoformat(),
                ) is None:
                    continue
                self._call_ledger.mark_publication_published(receipt)
            except ProjectPictureBindingStale:
                try:
                    self._call_ledger.mark_publication_skipped(
                        receipt,
                        reason_code="project_picture_binding_stale",
                    )
                except Exception:
                    # Until the terminal ack commits, READY is still the only
                    # crash-safe authority and will be revisited later.
                    continue
                continue
            except Exception:
                # READY remains authoritative.  A later bounded scan may retry
                # Project publication, but provider dispatch is never repeated.
                continue
            recovered += 1
        return recovered


def _default_picture_index() -> PictureObservationIndex:
    """仅在首次语义观察时装配现行 FILE 配方，不在建工具表时加载模型。"""
    from ...retrieval.operations.document_maintenance import (
        build_document_retrieval_composition,
    )
    from ...retrieval.operations.picture_index import PictureObservationIndex
    from ...workspace.ingestion.composition import resolve_document_ingest_owner

    database = current_workspace_database()
    if database is None:
        raise MountedVisualCallLedgerError(
            "Project picture index requires a bound database"
        )
    composition = build_document_retrieval_composition(retrieval_db_path=database.db_path)
    owner = resolve_document_ingest_owner()
    if owner.generation_identity.version_id != composition.generation_spec.version_id:
        raise MountedVisualCallLedgerError("Project picture index generation differs from execution owner")
    return PictureObservationIndex(
        composition=composition,
        connect_documents=database.open_connection,
        advance_outbox=owner.advance_outbox,
    )


def _default_picture_publication_service(
    picture_index: PictureObservationIndex | None = None,
) -> ProjectPicturePublicationService | None:
    """Compose the cross-domain adapter only at the tools/runtime boundary."""

    database = current_workspace_database()
    if database is None:
        # Some unit-level and legacy hosts intentionally run without a Project
        # database.  Their public visual behavior remains compatible, while the
        # bound production host takes the durable Project path.
        return None
    return ProjectPicturePublicationService(
        database=database,
        publication_port=PictureObservationOutboxPublisher(
            retrieval_data_version_resolver=(
                picture_index.resolve_target_in_transaction
                if picture_index is not None
                else _active_ready_retrieval_data_version_id
            ),
        ),
    )


def _active_ready_retrieval_data_version_id(conn) -> str | None:
    row = conn.execute(
        "SELECT id FROM retrieval_data_versions WHERE role='active' AND state='ready'"
    ).fetchone()
    return None if row is None else str(row[0])


def build_picture_publication_envelope(
    *,
    binding: AuthorizedFileSource,
    unit: VisualUnitRef,
    purpose: VisionPurpose,
    detail: VisionDetail,
    region: VisionRegion,
    prepared_artifact: PreparedVisualArtifactReceipt,
    question: str | None = None,
    logical_tool_call_id: str | None = None,
) -> MountedVisualProjectPublicationEnvelope:
    """Bind one final sent raster to a path-free Project picture locator.

    PDF observations use a parentless ``render`` unit.  The current mounted
    visual authority has the exact final region pixels but no independently
    materialized full-page parent; inventing a ``crop`` parent would therefore
    be false authority.
    """

    if binding.media_type in {"image/png", "image/jpeg"}:
        source_kind = "whole_file"
        source_locator = MountedVisualPictureLocator.from_payload(
            kind=source_kind,
            payload={},
        )
        unit_kind = "full"
        unit_locator = MountedVisualPictureLocator.from_payload(
            kind=unit_kind,
            payload={},
        )
    elif binding.media_type == "application/pdf":
        page = unit.locator.page
        if page is None:
            raise ValueError("PDF visual publication requires an exact page")
        source_kind = "document_surface"
        source_locator = MountedVisualPictureLocator.from_payload(
            kind=source_kind,
            payload={"surface_kind": "pdf_page", "ordinal": page},
        )
        unit_kind = "render"
        unit_locator = MountedVisualPictureLocator.from_payload(
            kind=unit_kind,
            payload={
                "bbox": (
                    None if unit.locator.bbox is None else list(unit.locator.bbox)
                ),
                "detail": detail.value,
                "page": page,
                "region": region.value,
            },
        )
    else:
        raise ValueError("visual publication file media type is unsupported")

    target = MountedVisualProjectPublicationTarget(
        project_id=binding.project_id,
        file_id=binding.file_id,
        file_version_id=binding.file_version_id,
        file_content_sha256=binding.fingerprint.sha256,
        file_media_type=binding.media_type,
        purpose=purpose,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        picture_source_kind=source_kind,
        picture_source_locator=source_locator,
        picture_unit_kind=unit_kind,
        picture_unit_locator=unit_locator,
        prepared_artifact=prepared_artifact,
        question=question,
        logical_tool_call_id=logical_tool_call_id,
    )
    return MountedVisualProjectPublicationEnvelope.from_target(target)


def build_picture_publication_command(
    receipt: MountedVisualCallReceipt,
) -> ProjectPicturePublicationCommand:
    envelope = receipt.publication_envelope
    if envelope is None or receipt.publication_state not in {
        MountedVisualPublicationState.READY,
        MountedVisualPublicationState.PUBLISHED,
    }:
        raise MountedVisualCallLedgerError(
            "only a ready or published receipt can enter idempotent Project publication"
        )
    target = envelope.target
    _require_supported_picture_publication_locator(target)
    result = receipt.result
    if result.status not in {VisionStatus.COMPLETED, VisionStatus.PARTIAL} or (
        not result.observations
    ):
        raise MountedVisualCallLedgerError(
            "ready visual publication has no publishable observations"
        )

    source_locator_payload = target.picture_source_locator.as_payload()["payload"]
    unit_locator_payload = target.picture_unit_locator.as_payload()["payload"]
    if not isinstance(source_locator_payload, Mapping) or not isinstance(
        unit_locator_payload,
        Mapping,
    ):
        raise MountedVisualCallLedgerError(
            "visual publication locator payload is malformed"
        )
    source_locator = PictureSourceLocator.from_payload(
        target.picture_source_kind,
        source_locator_payload,
    )
    unit_locator = PictureUnitLocator.from_payload(
        target.picture_unit_kind,
        unit_locator_payload,
    )
    artifact = target.prepared_artifact
    primary = result.observations[0]
    endpoint = _bounded_metadata_text(result.endpoint_identity)
    model = _bounded_metadata_text(result.model)
    provider = _bounded_metadata_text(result.provider)
    provider_observations = _bounded_provider_observation_metadata(result.observations)
    warnings = _bounded_metadata_sequence(result.warnings)
    unresolved_gaps = _bounded_metadata_sequence(result.unresolved_gap_refs)
    primary_id = _bounded_metadata_text(primary.observation_id)
    structured_common: dict[str, object] = {
        "endpoint_identity": endpoint.value,
        "endpoint_identity_sha256": endpoint.sha256,
        "endpoint_identity_truncated": endpoint.truncated,
        "model": model.value,
        "model_sha256": model.sha256,
        "model_truncated": model.truncated,
        "output_sha256": result.output_sha256,
        "provider": provider.value,
        "provider_observation_count": provider_observations["count"],
        "provider_observations": provider_observations["items"],
        "provider_observations_sha256": provider_observations["sha256"],
        "provider_observations_truncated": provider_observations["truncated"],
        "provider_sha256": provider.sha256,
        "provider_truncated": provider.truncated,
        "result_sha256": receipt.result_sha256,
        "unresolved_gap_ref_count": unresolved_gaps["count"],
        "unresolved_gap_refs": unresolved_gaps["items"],
        "unresolved_gap_refs_sha256": unresolved_gaps["sha256"],
        "unresolved_gap_refs_truncated": unresolved_gaps["truncated"],
        "warning_count": warnings["count"],
        "warnings": warnings["items"],
        "warnings_sha256": warnings["sha256"],
        "warnings_truncated": warnings["truncated"],
    }
    observations = (
        ProjectPictureObservationSpec(
            request_ordinal=0,
            purpose=target.purpose.value,
            question=target.question,
            kind=_picture_label(primary.kind, label="observation-kind"),
            text=_picture_observation_text(primary.text),
            uncertainty=primary.uncertainty,
            processor_fingerprint=_picture_label(
                result.processor_fingerprint,
                label="processor",
            ),
            prompt_fingerprint=target.prompt_contract_version,
            structured_payload=PictureObservationStructuredPayload.from_payload(
                contract=_VLM_PICTURE_OBSERVATION_CONTRACT,
                payload={
                    **structured_common,
                    "provider_observation_id": primary_id.value,
                    "provider_observation_id_sha256": primary_id.sha256,
                    "provider_observation_id_truncated": primary_id.truncated,
                },
            ),
        ),
    )
    return ProjectPicturePublicationCommand(
        binding=ProjectPictureBinding(
            project_id=target.project_id,
            file_id=target.file_id,
            file_version_id=target.file_version_id,
            file_content_sha256=target.file_content_sha256,
            file_media_type=target.file_media_type,
        ),
        unit=ProjectPictureUnitSpec(
            source_locator=source_locator,
            source_content_sha256=target.file_content_sha256,
            source_media_type=target.file_media_type,
            unit_locator=unit_locator,
            producer_fingerprint=_picture_unit_producer_fingerprint(artifact),
            parent_picture_unit_id=None,
            pixel_sha256=artifact.pixel_sha256,
            media_type=artifact.media_type,
            width=artifact.width,
            height=artifact.height,
        ),
        logical_invocation_id=receipt.call_key,
        observations=observations,
        occurred_at=datetime.now(timezone.utc).isoformat(),
    )


def _require_supported_picture_publication_locator(
    target: MountedVisualProjectPublicationTarget,
) -> None:
    """Cross-check the closed locator with this runtime's three source formats."""

    if target.file_media_type in {"image/png", "image/jpeg"}:
        if (
            target.picture_source_kind != "whole_file"
            or target.picture_unit_kind != "full"
        ):
            raise MountedVisualCallLedgerError(
                "raster-file publication locator is inconsistent"
            )
        return
    if target.file_media_type != "application/pdf":
        raise MountedVisualCallLedgerError(
            "visual publication media type is unsupported"
        )
    if (
        target.picture_source_kind != "document_surface"
        or target.picture_unit_kind != "render"
    ):
        raise MountedVisualCallLedgerError("PDF publication locator is inconsistent")
    source = target.picture_source_locator.as_payload()["payload"]
    unit = target.picture_unit_locator.as_payload()["payload"]
    if not isinstance(source, Mapping) or not isinstance(unit, Mapping):
        raise MountedVisualCallLedgerError(
            "PDF publication locator payload is malformed"
        )
    if set(unit) != {"bbox", "detail", "page", "region"} or (
        unit["page"] != source.get("ordinal")
        or unit["detail"] not in {item.value for item in VisionDetail}
        or unit["region"] not in {item.value for item in VisionRegion}
    ):
        raise MountedVisualCallLedgerError(
            "PDF render locator does not bind its document surface"
        )
    bbox = unit["bbox"]
    if bbox is not None and (
        not isinstance(bbox, Sequence)
        or isinstance(bbox, (str, bytes))
        or len(bbox) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in bbox
        )
    ):
        raise MountedVisualCallLedgerError("PDF render locator bbox is malformed")


def _bounded_metadata_text(value: str) -> _BoundedMetadataText:
    """Project arbitrary provider text into a fixed UTF-8 display budget + hash."""

    if not isinstance(value, str):
        raise TypeError("provider metadata text must be a string")
    digest = _sha256_text(value)
    printable = "".join(
        character if character.isprintable() else "\ufffd" for character in value
    )
    encoded = printable.encode("utf-8")
    if len(encoded) <= _MAX_STRUCTURED_METADATA_TEXT_BYTES:
        projected = printable
    else:
        projected = encoded[:_MAX_STRUCTURED_METADATA_TEXT_BYTES].decode(
            "utf-8", errors="ignore"
        )
    return _BoundedMetadataText(
        value=projected,
        sha256=digest,
        truncated=(projected != value),
    )


def _bounded_metadata_sequence(values: Sequence[str]) -> dict[str, object]:
    frozen = tuple(values)
    projected = tuple(
        _bounded_metadata_text(value)
        for value in frozen[:_MAX_STRUCTURED_METADATA_ITEMS]
    )
    return {
        "count": len(frozen),
        "items": [
            {
                "sha256": item.sha256,
                "truncated": item.truncated,
                "value": item.value,
            }
            for item in projected
        ],
        "sha256": _canonical_sha256(list(frozen)),
        "truncated": (
            len(frozen) > _MAX_STRUCTURED_METADATA_ITEMS
            or any(item.truncated for item in projected)
        ),
    }


def _bounded_provider_observation_metadata(
    observations: Sequence[VisionObservation],
) -> dict[str, object]:
    frozen = tuple(observations)
    exact_metadata = [
        {
            "kind": item.kind,
            "observation_id": item.observation_id,
            "text_sha256": _sha256_text(item.text),
            "uncertainty": item.uncertainty,
        }
        for item in frozen
    ]
    items: list[dict[str, object]] = []
    any_truncated = False
    for item in frozen[:_MAX_STRUCTURED_METADATA_ITEMS]:
        observation_id = _bounded_metadata_text(item.observation_id)
        kind = _bounded_metadata_text(item.kind)
        any_truncated = any_truncated or observation_id.truncated or kind.truncated
        items.append(
            {
                "kind": kind.value,
                "kind_sha256": kind.sha256,
                "kind_truncated": kind.truncated,
                "observation_id": observation_id.value,
                "observation_id_sha256": observation_id.sha256,
                "observation_id_truncated": observation_id.truncated,
                "text_sha256": _sha256_text(item.text),
                "uncertainty": item.uncertainty,
            }
        )
    return {
        "count": len(frozen),
        "items": items,
        "sha256": _canonical_sha256(exact_metadata),
        "truncated": (len(frozen) > _MAX_STRUCTURED_METADATA_ITEMS or any_truncated),
    }


def _picture_label(value: str, *, label: str) -> str:
    if (
        isinstance(value, str)
        and value
        and value == value.strip()
        and len(value) <= 1024
        and all(character.isprintable() for character in value)
    ):
        return value
    return f"{label}-sha256-{_sha256_text(value)}"


def _picture_observation_text(value: str) -> str:
    """Match the public projection ceiling and remove Picture-domain NULs."""

    return value.replace("\x00", "\ufffd")[:MAX_OBSERVATION_CHARS]


def _picture_unit_producer_fingerprint(
    artifact: PreparedVisualArtifactReceipt,
) -> str:
    """Bind implementation plus effective recipe for a locator's pixel unit."""

    return _canonical_sha256(
        {
            "contract_version": "prepared-picture-unit-producer-v1",
            "implementation_fingerprint": artifact.preparation_fingerprint,
            "render_recipe": json.loads(artifact.canonical_render_recipe),
        }
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
