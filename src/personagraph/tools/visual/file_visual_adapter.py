"""File-version visual tools using shared source, egress policy and Picture owners.

Construction performs no source I/O. Calls resolve exact files lazily, validate
the complete batch before rendering, and keep physical-call recovery independent
of mutable file lookup. Public identity is never tied to a Turn or Session token.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import re
from typing import Any, Protocol, TYPE_CHECKING

from ...input_processing.vision.providers import vision_adapter_transmits_externally
from ...input_processing.vision.contracts import (
    VisionCapabilitySnapshot, VisionDetail, VisionPurpose,
    VisionRegion, VisionStatus,
)
from .egress_policy import auto_visual_egress_receipt
from ...workspace.files.access import AuthorizedFileSource, FileAccessError
from ...workspace.pictures.project_publication import ProjectPicturePublicationService, ProjectPicturePublicationResult
from ...runtime.model_calls.vision import (
    DurableMountedVisionAdapter, MountedVisualCallLedgerError,
    MountedVisualCallWaitingExternal, SqliteMountedVisualCallLedger,
)
from ..effects import EffectAction, EffectResource, EffectScopeKind
from ..execution import ToolBusinessFailure
from ..policy import (
    AuthorityFacts, ProtectedToolExecutionAuthority, ScopeGrant, ToolInvocationAuthority,
)
from .file_visual_catalog import (
    ExternalFileVisualReadBindingFacts, FileVisualBindingFacts,
    build_file_visual_tool_bindings, derive_file_visual_source_fingerprint,
)
from .file_visual_tools import (
    DEFAULT_VISUAL_PAGE_SIZE, LIST_FILE_VISUALS_CONTRACT_VERSION,
    LIST_FILE_VISUALS_TOOL_ID, MAX_LIST_PAGE_FILTERS, MAX_OBSERVATION_CHARS,
    MAX_VISUALS_PER_PAGE, MAX_VISUALS_PER_READ, READ_FILE_VISUALS_CONTRACT_VERSION,
    build_list_file_visuals_registration, build_read_file_visuals_registration,
)
from .file_visual_source_authority import (
    FileVisualAuthorityError, FileVisualUnit, FrozenFileVisualSource,
    freeze_file_visual_source, visual_page_number,
)
from .visual_observation_service import (
    VisualObservationRequest, VisualObservationService,
    default_vision_adapter,
)
from .visual_tool_boundary import FrozenVisualToolBoundary
from .question_contract import parse_visual_question, visual_call_identity
from .project_observation_publication import VisualObservationPublisher

if TYPE_CHECKING:
    from ...retrieval.operations.picture_index import PictureObservationIndex


_SAFE_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")


class VisionAdapterPort(Protocol):
    transmits_externally: bool
    def capabilities(self) -> VisionCapabilitySnapshot: ...
    def analyze(self, request: object) -> object: ...


class _CapabilityBoundVisionAdapter:
    def __init__(self, delegate, capabilities, *, sends_externally: bool) -> None:
        self._delegate = delegate
        self._capabilities = capabilities
        self.transmits_externally = sends_externally

    def capabilities(self) -> VisionCapabilitySnapshot:
        return self._capabilities

    def analyze(self, request: object) -> object:
        return self._delegate.analyze(request)


@dataclass(frozen=True, slots=True)
class _VisualRead:
    source: FrozenFileVisualSource
    public: FileVisualUnit
    boundary: FrozenVisualToolBoundary
    purpose: VisionPurpose
    detail: VisionDetail
    region: VisionRegion
    question: str | None = None


class FileVisualRuntime:
    """Session-scoped wiring; File/Document/Picture remain canonical identities."""

    def __init__(
        self, *, session_id: str,
        resolve_file: Callable[..., AuthorizedFileSource],
        revalidate_source: Callable[[AuthorizedFileSource], bool],
        adapter: VisionAdapterPort | None = None,
        call_ledger: SqliteMountedVisualCallLedger | None = None,
        picture_publication_service: ProjectPicturePublicationService | None = None,
        picture_index: PictureObservationIndex | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty")
        if not callable(resolve_file) or not callable(revalidate_source):
            raise TypeError("file resolution and revalidation must be callable")
        self._session_id = session_id
        self._resolve_file = resolve_file
        self._revalidate_source = revalidate_source
        resolved_adapter = adapter or default_vision_adapter()
        capabilities = resolved_adapter.capabilities()
        if not isinstance(capabilities, VisionCapabilitySnapshot):
            raise TypeError("vision adapter returned an invalid capability snapshot")
        self.sends_externally = vision_adapter_transmits_externally(resolved_adapter)
        bound_adapter = _CapabilityBoundVisionAdapter(
            resolved_adapter, capabilities, sends_externally=self.sends_externally,
        )
        self._call_ledger = call_ledger if call_ledger is not None else SqliteMountedVisualCallLedger()
        self._capabilities = capabilities
        self._visual_publisher = VisualObservationPublisher(
            session_id=session_id, call_ledger=self._call_ledger,
            picture_publication_service=picture_publication_service,
            picture_index=picture_index,
        )
        self._observation_service = VisualObservationService(
            adapter=(
                DurableMountedVisionAdapter(
                    bound_adapter, session_id=session_id, ledger=self._call_ledger,
                ) if self.sends_externally else bound_adapter
            ),
        )
        self.effect_scope = "file_visuals:" + _sha256_text(session_id)[:32]
        facts = _file_visual_binding_facts(
            session_id=session_id, capabilities=capabilities,
            call_ledger=self._call_ledger,
            external_read_available=self.sends_externally and capabilities.available,
        )
        registrations = (
            build_list_file_visuals_registration(handler=self.list_file_visuals, effect_scope=self.effect_scope),
            build_read_file_visuals_registration(
                handler=self.read_file_visuals, effect_scope=self.effect_scope,
                sends_externally=self.sends_externally,
            ),
        )
        self.registrations = tuple(
            replace(registration, source=replace(
                registration.source,
                fingerprint=derive_file_visual_source_fingerprint(facts, tool_id=registration.tool_id),
            ))
            if registration.tool_id == LIST_FILE_VISUALS_TOOL_ID or facts.external_read is not None
            else registration
            for registration in registrations
        )
        self.file_visual_bindings = build_file_visual_tool_bindings(
            self.registrations if facts.external_read is not None else self.registrations[:1],
            facts=facts,
        )
        # 表级只有本地读取许可。外发和持久发布的批准在具体文件、用途确定后生成。
        self.authority = _authority_facts(effect_scope=self.effect_scope, sends_externally=False)
        self.backend_identity_sha256 = _canonical_sha256({
            "session_id": session_id,
            "sources": [registration.source.fingerprint for registration in self.registrations],
            "sends_externally": self.sends_externally,
        })

    def _source(self, request: Mapping[str, object], *, pages=None) -> FrozenFileVisualSource | None:
        file_id = _identifier(request.get("file_id"), "file_id")
        version_id = _identifier(request.get("file_version_id"), "file_version_id")
        try:
            source = self._resolve_file(file_id=file_id, file_version_id=version_id)
        except FileAccessError as exc:
            raise ToolBusinessFailure(exc.reason_code, "The requested file is unavailable.") from exc
        if (
            not isinstance(source, AuthorizedFileSource)
            or source.file_id != file_id or source.file_version_id != version_id
        ):
            raise ToolBusinessFailure("file_resolution_mismatch", "File resolution crossed its exact identity.")
        try:
            return freeze_file_visual_source(
                session_id=self._session_id, source=source,
                revalidate_source=self._revalidate_source, source_pages=pages,
            )
        except FileVisualAuthorityError as exc:
            raise ToolBusinessFailure(exc.reason_code, "File visual authority is no longer available.") from exc

    def list_file_visuals(self, payload: dict[str, Any]) -> dict[str, Any]:
        pages = _list_pages(payload.get("pages"))
        limit = _list_limit(payload.get("limit", DEFAULT_VISUAL_PAGE_SIZE))
        cursor = payload.get("cursor")
        if pages is not None and cursor is not None:
            raise ToolBusinessFailure("invalid_visual_cursor", "Page filters cannot be combined with a cursor.")
        # An unfiltered PDF list stays cold; exact page selection is explicit.
        prepared = self._source(payload, pages=pages if pages is not None else ())
        result = {
            "contract_version": LIST_FILE_VISUALS_CONTRACT_VERSION,
            "file_id": payload["file_id"], "file_version_id": payload["file_version_id"],
            "status": "not_established", "visuals": [], "next_cursor": None,
            "reason_code": "visual_inspection_not_prepared",
        }
        if prepared is None:
            if cursor is not None:
                raise ToolBusinessFailure("invalid_visual_cursor", "Visual authority is unavailable.")
            return result
        offset = _cursor_offset(cursor, prepared)
        selected = prepared.catalog[offset:offset + limit]
        next_offset = offset + len(selected)
        result.update({
            "status": "ready", "visuals": [unit.to_model_dict() for unit in selected],
            "next_cursor": (
                _cursor(prepared, next_offset)
                if pages is None and next_offset < len(prepared.catalog) else None
            ),
            "reason_code": (
                "visual_pages_required" if pages is None and prepared.source.media_type == "application/pdf"
                else "visual_page_limit_reached" if next_offset < len(prepared.catalog) else None
            ),
        })
        return result

    def read_file_visuals(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Crash recovery is source-free and never repeats a physical provider call.
        if self.sends_externally:
            self._visual_publisher.recover_ready()
        validated = self._validated_reads(payload)
        # Recheck the complete batch before the first render/provider side effect.
        for selected in validated:
            visual_call_identity(selected.purpose)
            _validate_current(selected.source)
        results = []
        for selected in validated:
            _validate_current(selected.source)
            results.append(self._observe(selected))
        return {"contract_version": READ_FILE_VISUALS_CONTRACT_VERSION, "results": results}

    def _validated_reads(self, payload: Mapping[str, Any]) -> tuple[_VisualRead, ...]:
        requests = _read_requests(payload)
        validated: list[_VisualRead] = []
        seen = set()
        for request in requests:
            identifier = _identifier(request.get("visual_unit_id"), "visual_unit_id")
            key = (request.get("file_id"), request.get("file_version_id"), identifier)
            if key in seen:
                raise ToolBusinessFailure("duplicate_visual_unit", "A visual unit may be read only once per call.")
            seen.add(key)
            page = visual_page_number(identifier)
            prepared = self._source(request, pages=(page,) if page is not None else None)
            if prepared is None:
                raise ToolBusinessFailure("visual_inspection_not_prepared", "Prepare the file before reading its visuals.")
            try:
                public, boundary = prepared.resolve(identifier)
            except FileVisualAuthorityError as exc:
                raise ToolBusinessFailure(exc.reason_code, "Visual unit does not belong to current file authority.") from exc
            purpose = _enum_value(VisionPurpose, request.get("purpose"), "purpose")
            detail = _enum_value(VisionDetail, request.get("detail"), "detail")
            region = _enum_value(VisionRegion, request.get("region"), "region")
            question = parse_visual_question(purpose, request.get("question"))
            if purpose.value not in public.allowed_purposes:
                raise ToolBusinessFailure("visual_purpose_not_allowed", "Purpose is unavailable for this visual.")
            validated.append(_VisualRead(prepared, public, boundary, purpose, detail, region, question))
        return tuple(validated)

    def prepare_read_authority(self, payload: Mapping[str, Any]) -> ToolInvocationAuthority:
        """准入前核验整批来源，绑定默认外发策略及实际来源供持久派发复查。"""

        selected = self._validated_reads(payload)

        def receipts() -> tuple[str, ...]:
            values: list[str] = []
            for item in selected:
                _validate_current(item.source)
                unit = item.boundary.units[0]
                receipt_id = auto_visual_egress_receipt(
                    session_id=self._session_id,
                    source_sha256=unit.source_sha256,
                    endpoint_identity=self._capabilities.endpoint_identity,
                    model=self._capabilities.model,
                    purpose=item.purpose,
                )
                if receipt_id not in values:
                    values.append(receipt_id)
            return tuple(values)

        expected = receipts()

        def revalidate() -> bool:
            try:
                return receipts() == expected
            except (OSError, RuntimeError, TypeError, ValueError, ToolBusinessFailure):
                return False

        return ToolInvocationAuthority(
            authority=_authority_facts(effect_scope=self.effect_scope, sends_externally=True),
            protected_authority=ProtectedToolExecutionAuthority(
                approval_receipt_ids=expected,
                execution_backend_identity_sha256=self.backend_identity_sha256,
                revalidate=revalidate,
            ),
        )

    def _observe(self, selected: _VisualRead) -> dict[str, object]:
        publication = None
        try:
            request = VisualObservationRequest(
                unit_id=selected.boundary.units[0].unit_id,
                purpose=selected.purpose, detail=selected.detail, region=selected.region,
                question=selected.question,
            )
            if self.sends_externally:
                projection, publication = self._visual_publisher.observe(
                    service=self._observation_service, boundary=selected.boundary,
                    request=request, binding=selected.source.source,
                )
                raw = projection.to_dict()
            else:
                raw = self._observation_service.observe(selected.boundary, (request,)).results[0].to_dict()
            return _project_visual_result(selected, raw=raw, publication=publication)
        except MountedVisualCallWaitingExternal as exc:
            result = _blocked_visual_result(selected, _safe_reason_code(exc.reason_code))
            result["background_wait_active"] = False
            if exc.failure_diagnostics is not None:
                result["failure_diagnostics"] = exc.failure_diagnostics.to_dict()
            return result
        except MountedVisualCallLedgerError:
            return _blocked_visual_result(selected, "visual_dispatch_authority_unavailable")


def _list_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolBusinessFailure("invalid_request", "limit must be an integer.")
    if not 1 <= value <= MAX_VISUALS_PER_PAGE:
        raise ToolBusinessFailure(
            "invalid_request",
            "limit is outside the visual catalog bound.",
        )
    return value


def _list_pages(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or len(value) > MAX_LIST_PAGE_FILTERS
        or len(value) != len(set(value))
        or any(
            isinstance(page, bool) or not isinstance(page, int) or page < 1
            for page in value
        )
    ):
        raise ToolBusinessFailure(
            "invalid_request",
            "pages must contain one to eight unique positive page numbers.",
        )
    return tuple(sorted(value))


def _read_requests(payload: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(payload, Mapping):
        raise ToolBusinessFailure("invalid_request", "Tool input must be an object.")
    raw = payload.get("requests")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or not raw
        or len(raw) > MAX_VISUALS_PER_READ
        or any(not isinstance(item, Mapping) for item in raw)
    ):
        raise ToolBusinessFailure(
            "invalid_request",
            "requests must be a bounded non-empty list.",
        )
    return tuple(raw)


def _enum_value(enum_type, value: object, field_name: str):
    try:
        return enum_type(str(value or ""))
    except ValueError as exc:
        raise ToolBusinessFailure(
            "invalid_request",
            f"{field_name} is unsupported.",
        ) from exc


def _authority_facts(
    *,
    effect_scope: str,
    sends_externally: bool,
) -> AuthorityFacts:
    return AuthorityFacts(
        grants=(
            ScopeGrant(
                EffectResource.FILESYSTEM,
                EffectAction.READ,
                EffectScopeKind.SESSION,
                effect_scope,
            ),
        ) + (
            (
                ScopeGrant(
                    EffectResource.NETWORK,
                    EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION,
                    effect_scope,
                ),
                # Default egress includes derived observation and ledger state,
                # not permission to modify workspace files.
                ScopeGrant(
                    EffectResource.RUNTIME_STATE,
                    EffectAction.UPDATE,
                    EffectScopeKind.SESSION,
                    effect_scope,
                ),
            )
            if sends_externally
            else ()
        ),
    )


def build_file_visual_runtime(
    *,
    session_id: str,
    resolve_file: Callable[..., AuthorizedFileSource],
    revalidate_source: Callable[[AuthorizedFileSource], bool],
    adapter: VisionAdapterPort | None = None,
    call_ledger: SqliteMountedVisualCallLedger | None = None,
    picture_publication_service: ProjectPicturePublicationService | None = None,
    picture_index: PictureObservationIndex | None = None,
) -> FileVisualRuntime:
    return FileVisualRuntime(
        session_id=session_id,
        resolve_file=resolve_file,
        revalidate_source=revalidate_source,
        adapter=adapter,
        call_ledger=call_ledger,
        picture_publication_service=picture_publication_service,
        picture_index=picture_index,
    )


def _validate_current(source: FrozenFileVisualSource) -> None:
    try:
        source.validate_current()
    except FileVisualAuthorityError as exc:
        raise ToolBusinessFailure(exc.reason_code, "File visual authority is no longer current.") from exc


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ToolBusinessFailure("invalid_request", f"{name} must be a bounded identity.")
    return value


def _cursor(source: FrozenFileVisualSource, offset: int) -> str:
    return f"visualcursor_{offset}_{source.snapshot_id}"


def _cursor_offset(value: object, source: FrozenFileVisualSource) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        parts = value.split("_")
        if len(parts) == 3 and parts[0] == "visualcursor" and parts[1].isdigit():
            offset = int(parts[1])
            if 0 <= offset <= len(source.catalog) and hmac.compare_digest(value, _cursor(source, offset)):
                return offset
    raise ToolBusinessFailure("invalid_visual_cursor", "The cursor does not match this File version and manifest.")


def _result_identity(selected: _VisualRead) -> dict[str, object]:
    return {
        "file_id": selected.source.source.file_id,
        "file_version_id": selected.source.source.file_version_id,
        "visual_unit_id": selected.public.visual_unit_id,
        "pages": list(selected.public.source_pages), "kind": selected.public.kind,
        "purpose": selected.purpose.value, "detail": selected.detail.value,
        "region": selected.region.value,
        **({"question": selected.question} if selected.question is not None else {}),
    }


def _project_visual_result(
    selected: _VisualRead, *, raw: Mapping[str, object],
    publication: ProjectPicturePublicationResult | None,
) -> dict[str, object]:
    try:
        status = VisionStatus(str(raw.get("status") or ""))
    except ValueError as exc:
        raise ToolBusinessFailure("visual_provider_result_invalid", "The provider returned an invalid status.") from exc
    observation = raw.get("observation")
    reason = None
    if status in {VisionStatus.UNAVAILABLE, VisionStatus.FAILED}:
        observation = None
        reason = _safe_reason_code(raw.get("failure_code"))
    elif not isinstance(observation, str) or not observation:
        raise ToolBusinessFailure("visual_provider_result_invalid", "The provider omitted its observation.")
    elif status is VisionStatus.PARTIAL:
        reason = "visual_observation_partial"
    if isinstance(observation, str) and len(observation) > MAX_OBSERVATION_CHARS:
        observation = observation[:MAX_OBSERVATION_CHARS]
        status = VisionStatus.PARTIAL
        reason = "visual_observation_truncated"
    result = _result_identity(selected)
    result.update(status=status.value, observation=observation, reason_code=reason)
    if raw.get("failure_diagnostics") is not None:
        result["failure_diagnostics"] = raw["failure_diagnostics"]
    if publication is not None:
        result.update(
            picture_id=publication.picture_id,
            picture_unit_id=publication.picture_unit_id,
            observation_id=publication.observation_commits[0].observation.observation_id,
        )
    uncertainty = raw.get("uncertainty")
    if isinstance(uncertainty, (float, int)) and not isinstance(uncertainty, bool) and 0 <= uncertainty <= 1:
        result["uncertainty"] = float(uncertainty)
    return result


def _blocked_visual_result(selected: _VisualRead, reason_code: str) -> dict[str, object]:
    result = _result_identity(selected)
    result.update(status="blocked", observation=None, reason_code=reason_code)
    return result


def _safe_reason_code(value: object, *, fallback: str = "visual_observation_unavailable") -> str:
    return value if isinstance(value, str) and _SAFE_REASON_CODE.fullmatch(value) else fallback


def _file_visual_binding_facts(
    *, session_id: str, capabilities: VisionCapabilitySnapshot,
    call_ledger: SqliteMountedVisualCallLedger,
    external_read_available: bool,
) -> FileVisualBindingFacts:
    external = None
    if external_read_available:
        external = ExternalFileVisualReadBindingFacts(
            provider_identity_sha256=_canonical_sha256({
                "provider": capabilities.provider, "model": capabilities.model,
                "endpoint_identity": capabilities.endpoint_identity,
                "processor_fingerprint": capabilities.processor_fingerprint,
            }),
            capability_snapshot_sha256=_canonical_sha256({
                "available": capabilities.available, "provider": capabilities.provider,
                "model": capabilities.model, "endpoint_identity": capabilities.endpoint_identity,
                "processor_fingerprint": capabilities.processor_fingerprint,
                "supported_purposes": [purpose.value for purpose in capabilities.supported_purposes],
                "reason_code": capabilities.reason_code,
            }),
            egress_policy_sha256=_canonical_sha256({
                "session_id": session_id, "egress_policy": "default-allow",
            }),
            physical_call_ledger_sha256=_canonical_sha256({
                "session_id": session_id, "path": str(call_ledger.path_for(session_id).resolve()),
            }),
        )
    return FileVisualBindingFacts(
        session_scope_sha256=_sha256_text(session_id),
        file_access_policy_sha256=_sha256_text("shared-file-version-with-current-session-path-authorization"),
        external_read=external,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


__all__ = ["FileVisualRuntime", "build_file_visual_runtime"]
