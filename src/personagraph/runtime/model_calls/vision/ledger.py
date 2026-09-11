"""外部传输挂载视觉读取的崩溃安全权威状态。

Host 原语账本能证明已选择视觉资源读取，但无法证明视觉 provider 是否已收到像素。
普通 Runtime ToolCall 账本无法弥补这一缺口：它有意外键关联到已物化
WorkRun/Attempt/ToolCall，而挂载视觉读取在 Host 原语之下执行，没有这些标识。

因此本模块持有一个狭窄、独立的物理发送账本。披露 gate 仍位于其外部。只有 gate
允许调用后，:class:`DurableMountedVisionAdapter` 才会进入此处，持久地预留精确的
provider 绑定请求并调用真实 adapter。已完成有类型结果会逐字节重放；pending 或
uncertain 调用绝不会自动发送第二次。

自然语言问答进一步绑定 Host 已预留的逻辑 ToolCall：新调用独立发送，同一调用内的
请求漂移拒绝。既有四种固定观察用途仍按原合同恢复，不将其旧记录改写成问答请求。

数据库中有意不存储私有路径和图像字节。请求绑定只包含不可变来源/像素/locator
事实。默认情况下，每个 Session 将账本保存在其受管 Session 目录下，因此普通
Session 清除会随附件一并移除 observation。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from personagraph.workspace.files import attachments as attachment_storage
from personagraph.input_processing.vision.providers import (
    vision_adapter_transmits_externally,
)
from personagraph.input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionFailureDiagnostics,
    VisionObservation,
    VisionPurpose,
    VisionRequest,
    VisionResult,
    VisionStatus,
    normalize_vision_question,
    validate_vision_call_identity,
)
from personagraph.input_processing.vision.imaging import (
    PreparedVisualArtifactReceipt,
    validate_prepared_visual_artifact_receipt,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CALL_KEY = re.compile(r"^mvc_[0-9a-f]{64}$")
_STABLE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_MAX_RESULT_BYTES = 1_000_000
_MAX_PUBLICATION_ENVELOPE_BYTES = 64 * 1024
_MAX_PUBLICATION_LOCATOR_BYTES = 16 * 1024
_MAX_PUBLICATION_IDENTIFIER_CHARS = 512
_PUBLICATION_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_UNCERTAIN_FAILURE_CODES = frozenset({"vision_provider_unreachable"})
_PICTURE_SOURCE_KINDS = frozenset(
    {"whole_file", "embedded_asset", "document_surface"}
)
_PICTURE_UNIT_KINDS = frozenset({"full", "render", "crop", "tile"})
_DOCUMENT_SURFACE_KINDS = frozenset({"pdf_page", "pptx_slide"})
_LOCATOR_FORBIDDEN_KEYS = frozenset(
    {
        "absolute_path",
        "authorization",
        "base64",
        "bytes",
        "credential",
        "filesystem_path",
        "image_path",
        "observation_text",
        "observations",
        "password",
        "provider_result",
        "raw_bytes",
        "result_text",
        "secret",
        "source_path",
    }
)

MOUNTED_VISUAL_PROJECT_PUBLICATION_CONTRACT = (
    "mounted-visual-project-publication-v1"
)
_QUESTION_PUBLICATION_CONTRACT = "mounted-visual-project-publication-v2"
MOUNTED_VISUAL_CALL_RECEIPT_CONTRACT = "mounted-visual-call-receipt-v1"


class MountedVisualCallLedgerError(RuntimeError):
    """物理视觉发送权威状态不可用或格式错误。"""


class MountedVisualCallWaitingExternal(RuntimeError):
    """provider 可能已收到像素；绝不盲目重发。"""

    def __init__(
        self, *, reason_code: str = "visual_completion_unconfirmed",
        failure_diagnostics: VisionFailureDiagnostics | None = None,
        call_key: str | None = None,
    ) -> None:
        if _STABLE_CODE.fullmatch(reason_code) is None:
            raise ValueError("visual waiting reason must be a stable code")
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.failure_diagnostics = failure_diagnostics
        self.call_key = call_key

    def diagnostic_details(self) -> dict[str, object]:
        # 不确定账本不是后台等待任务；安全诊断沿 ToolResult 进入 trajectory。
        details: dict[str, object] = {
            "reason_code": self.reason_code,
            "background_wait_active": False,
            "automatic_retries": 0,
        }
        if self.call_key is not None:
            details["call_key"] = self.call_key
        if self.failure_diagnostics is not None:
            details["failure_diagnostics"] = self.failure_diagnostics.to_dict()
        return details


class MountedVisualPublicationState(StrEnum):
    """Session durable ledger 内的单向 Project publication 状态。"""

    AWAITING_RESULT = "awaiting_result"
    READY = "ready"
    PUBLISHED = "published"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class MountedVisualPictureLocator:
    """Canonical logical locator; never a host filesystem path."""

    kind: str
    canonical_json: str

    def __post_init__(self) -> None:
        if self.kind not in _PICTURE_SOURCE_KINDS | _PICTURE_UNIT_KINDS:
            raise ValueError("mounted visual picture locator kind is unsupported")
        if not isinstance(self.canonical_json, str):
            raise TypeError("mounted visual picture locator JSON must be text")
        encoded = self.canonical_json.encode("utf-8")
        if not encoded or len(encoded) > _MAX_PUBLICATION_LOCATOR_BYTES:
            raise ValueError("mounted visual picture locator is empty or too large")
        try:
            value = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("mounted visual picture locator is invalid JSON") from exc
        if _canonical_json(value) != self.canonical_json:
            raise ValueError("mounted visual picture locator must be canonical JSON")
        if not isinstance(value, dict) or set(value) != {"kind", "payload"}:
            raise ValueError("mounted visual picture locator has unsupported fields")
        if value["kind"] != self.kind or not isinstance(value["payload"], dict):
            raise ValueError("mounted visual picture locator kind does not match")
        _normalize_locator_value(value["payload"])
        _validate_locator_contract(self.kind, value["payload"])

    @classmethod
    def from_payload(
        cls,
        *,
        kind: str,
        payload: Mapping[str, object],
    ) -> "MountedVisualPictureLocator":
        if not isinstance(payload, Mapping):
            raise TypeError("mounted visual picture locator payload must be a mapping")
        normalized = _normalize_locator_value(payload)
        assert isinstance(normalized, dict)
        canonical_json = _canonical_json({"kind": kind, "payload": normalized})
        return cls(kind=kind, canonical_json=canonical_json)

    @classmethod
    def from_canonical_object(
        cls,
        value: Mapping[str, object],
    ) -> "MountedVisualPictureLocator":
        if not isinstance(value, Mapping) or set(value) != {"kind", "payload"}:
            raise ValueError("mounted visual picture locator has unsupported fields")
        kind = value["kind"]
        payload = value["payload"]
        if not isinstance(kind, str) or not isinstance(payload, Mapping):
            raise ValueError("mounted visual picture locator is malformed")
        return cls.from_payload(kind=kind, payload=payload)

    def as_payload(self) -> dict[str, object]:
        return dict(json.loads(self.canonical_json))


@dataclass(frozen=True, slots=True)
class MountedVisualProjectPublicationTarget:
    """图像指针与有界问答元数据，绑定一次精确的 Project 观察发布。"""

    project_id: str
    file_id: str
    file_version_id: str
    file_content_sha256: str
    file_media_type: str
    purpose: VisionPurpose
    prompt_contract_version: str
    picture_source_kind: str
    picture_source_locator: MountedVisualPictureLocator
    picture_unit_kind: str
    picture_unit_locator: MountedVisualPictureLocator
    prepared_artifact: PreparedVisualArtifactReceipt
    question: str | None = None
    logical_tool_call_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("project_id", "file_id", "file_version_id"):
            _validate_publication_identifier(getattr(self, name), field=name)
        _validate_sha256(self.file_content_sha256, field="file_content_sha256")
        _validate_media_type(self.file_media_type, field="file_media_type")
        if not isinstance(self.purpose, VisionPurpose):
            raise TypeError("mounted visual publication purpose is unsupported")
        object.__setattr__(
            self, "question", normalize_vision_question(self.purpose, self.question)
        )
        validate_vision_call_identity(self.purpose, self.logical_tool_call_id)
        _validate_publication_version(
            self.prompt_contract_version,
            field="prompt_contract_version",
        )
        if self.picture_source_kind not in _PICTURE_SOURCE_KINDS:
            raise ValueError("mounted visual picture source kind is unsupported")
        if self.picture_unit_kind not in _PICTURE_UNIT_KINDS:
            raise ValueError("mounted visual picture unit kind is unsupported")
        if not isinstance(self.picture_source_locator, MountedVisualPictureLocator):
            raise TypeError("picture_source_locator has an unsupported type")
        if not isinstance(self.picture_unit_locator, MountedVisualPictureLocator):
            raise TypeError("picture_unit_locator has an unsupported type")
        if self.picture_source_locator.kind != self.picture_source_kind:
            raise ValueError("picture source locator kind conflicts with its target")
        if self.picture_unit_locator.kind != self.picture_unit_kind:
            raise ValueError("picture unit locator kind conflicts with its target")
        if not isinstance(self.prepared_artifact, PreparedVisualArtifactReceipt):
            raise TypeError("prepared_artifact has an unsupported type")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "MountedVisualProjectPublicationTarget":
        expected_fields = {
            "file_content_sha256",
            "file_id",
            "file_media_type",
            "file_version_id",
            "prompt_contract_version",
            "purpose",
            "picture_source_kind",
            "picture_source_locator",
            "picture_unit_kind",
            "picture_unit_locator",
            "prepared_artifact",
            "project_id",
        }
        if isinstance(payload, Mapping) and payload.get("purpose") == "question":
            expected_fields.update({"question", "logical_tool_call_id"})
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise ValueError("mounted visual publication target has unsupported fields")
        source_locator = payload["picture_source_locator"]
        unit_locator = payload["picture_unit_locator"]
        prepared_artifact = payload["prepared_artifact"]
        if (
            not isinstance(source_locator, Mapping)
            or not isinstance(unit_locator, Mapping)
            or not isinstance(prepared_artifact, Mapping)
        ):
            raise ValueError("mounted visual publication target is malformed")
        purpose = payload["purpose"]
        if not isinstance(purpose, str):
            raise ValueError("mounted visual publication purpose is malformed")
        return cls(
            project_id=payload["project_id"],
            file_id=payload["file_id"],
            file_version_id=payload["file_version_id"],
            file_content_sha256=payload["file_content_sha256"],
            file_media_type=payload["file_media_type"],
            purpose=VisionPurpose(purpose),
            prompt_contract_version=payload["prompt_contract_version"],
            picture_source_kind=payload["picture_source_kind"],
            picture_source_locator=(
                MountedVisualPictureLocator.from_canonical_object(source_locator)
            ),
            picture_unit_kind=payload["picture_unit_kind"],
            picture_unit_locator=(
                MountedVisualPictureLocator.from_canonical_object(unit_locator)
            ),
            prepared_artifact=(
                PreparedVisualArtifactReceipt.from_payload(prepared_artifact)
            ),
            question=payload.get("question"),
            logical_tool_call_id=payload.get("logical_tool_call_id"),
        )

    def as_payload(self) -> dict[str, object]:
        payload = {
            "file_content_sha256": self.file_content_sha256,
            "file_id": self.file_id,
            "file_media_type": self.file_media_type,
            "file_version_id": self.file_version_id,
            "prompt_contract_version": self.prompt_contract_version,
            "purpose": self.purpose.value,
            "picture_source_kind": self.picture_source_kind,
            "picture_source_locator": self.picture_source_locator.as_payload(),
            "picture_unit_kind": self.picture_unit_kind,
            "picture_unit_locator": self.picture_unit_locator.as_payload(),
            "prepared_artifact": self.prepared_artifact.as_payload(),
            "project_id": self.project_id,
        }
        if self.purpose is VisionPurpose.QUESTION:
            payload.update(
                question=self.question,
                logical_tool_call_id=self.logical_tool_call_id,
            )
        return payload


@dataclass(frozen=True, slots=True)
class MountedVisualProjectPublicationEnvelope:
    """Host 私有、无字节/路径/密钥的版本化 Project publication 输入。"""

    contract_version: str
    canonical_json: str
    envelope_sha256: str

    def __post_init__(self) -> None:
        if self.contract_version not in {
            MOUNTED_VISUAL_PROJECT_PUBLICATION_CONTRACT,
            _QUESTION_PUBLICATION_CONTRACT,
        }:
            raise ValueError("mounted visual publication contract is unsupported")
        if not isinstance(self.canonical_json, str):
            raise TypeError("publication envelope JSON must be text")
        encoded = self.canonical_json.encode("utf-8")
        if not encoded or len(encoded) > _MAX_PUBLICATION_ENVELOPE_BYTES:
            raise ValueError("publication envelope JSON is empty or too large")
        if not isinstance(self.envelope_sha256, str) or (
            _SHA256.fullmatch(self.envelope_sha256) is None
        ):
            raise ValueError("publication envelope hash must be lowercase sha256")
        if _sha256_text(self.canonical_json) != self.envelope_sha256:
            raise ValueError("publication envelope hash does not match its JSON")
        try:
            envelope = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("publication envelope is invalid JSON") from exc
        if _canonical_json(envelope) != self.canonical_json:
            raise ValueError("publication envelope JSON must be canonical")
        if not isinstance(envelope, dict) or set(envelope) != {
            "contract_version",
            "target",
        }:
            raise ValueError("publication envelope has unsupported fields")
        if envelope["contract_version"] != self.contract_version:
            raise ValueError("publication envelope contract does not match")
        if not isinstance(envelope["target"], dict):
            raise ValueError("publication envelope target must be an object")
        target = MountedVisualProjectPublicationTarget.from_payload(envelope["target"])
        expected_contract = (
            _QUESTION_PUBLICATION_CONTRACT
            if target.purpose is VisionPurpose.QUESTION
            else MOUNTED_VISUAL_PROJECT_PUBLICATION_CONTRACT
        )
        if self.contract_version != expected_contract:
            raise ValueError("publication envelope contract does not match its purpose")
        if target.as_payload() != envelope["target"]:
            raise ValueError("publication envelope target must be normalized")

    @classmethod
    def from_target(
        cls,
        target: MountedVisualProjectPublicationTarget,
        *,
        contract_version: str | None = None,
    ) -> "MountedVisualProjectPublicationEnvelope":
        if not isinstance(target, MountedVisualProjectPublicationTarget):
            raise TypeError("publication target has an unsupported type")
        if contract_version is None:
            contract_version = (
                _QUESTION_PUBLICATION_CONTRACT
                if target.purpose is VisionPurpose.QUESTION
                else MOUNTED_VISUAL_PROJECT_PUBLICATION_CONTRACT
            )
        canonical_json = _canonical_json(
            {
                "contract_version": contract_version,
                "target": target.as_payload(),
            }
        )
        return cls(
            contract_version=contract_version,
            canonical_json=canonical_json,
            envelope_sha256=_sha256_text(canonical_json),
        )

    @classmethod
    def from_canonical_json(
        cls,
        *,
        contract_version: str,
        canonical_json: str,
        envelope_sha256: str,
    ) -> "MountedVisualProjectPublicationEnvelope":
        return cls(
            contract_version=contract_version,
            canonical_json=canonical_json,
            envelope_sha256=envelope_sha256,
        )

    @property
    def target(self) -> MountedVisualProjectPublicationTarget:
        return MountedVisualProjectPublicationTarget.from_payload(
            json.loads(self.canonical_json)["target"]
        )

    @property
    def payload(self) -> dict[str, object]:
        """Compatibility projection; construction remains typed-only."""

        return self.target.as_payload()


@dataclass(frozen=True, slots=True)
class MountedVisualCallReceipt:
    """一个已持久结算 provider 结果及其可选 Project publication 状态。"""

    call_key: str
    session_id: str
    provider_identity_sha256: str
    request_binding_sha256: str
    result_sha256: str
    result: VisionResult
    publication_envelope: MountedVisualProjectPublicationEnvelope | None
    publication_state: MountedVisualPublicationState | None
    replayed: bool

    @property
    def contract_version(self) -> str:
        return MOUNTED_VISUAL_CALL_RECEIPT_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.call_key, str) or _CALL_KEY.fullmatch(self.call_key) is None:
            raise ValueError("mounted visual receipt call_key is invalid")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("mounted visual receipt session_id must not be empty")
        for name in (
            "provider_identity_sha256",
            "request_binding_sha256",
            "result_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase sha256 digest")
        if not isinstance(self.result, VisionResult):
            raise TypeError("mounted visual receipt result must be a VisionResult")
        expected_result_sha256 = _sha256_text(
            _canonical_json(_result_payload(self.result))
        )
        if expected_result_sha256 != self.result_sha256:
            raise ValueError("mounted visual receipt result hash does not match")
        if (self.publication_envelope is None) != (self.publication_state is None):
            raise ValueError(
                "publication envelope and state must either both exist or both be absent"
            )
        if self.publication_envelope is not None and not isinstance(
            self.publication_envelope,
            MountedVisualProjectPublicationEnvelope,
        ):
            raise TypeError("publication_envelope has an unsupported type")
        if self.publication_state is not None and not isinstance(
            self.publication_state,
            MountedVisualPublicationState,
        ):
            raise TypeError("publication_state has an unsupported type")
        if not isinstance(self.replayed, bool):
            raise TypeError("mounted visual receipt replayed must be boolean")


class _VisionAdapter(Protocol):
    transmits_externally: bool

    def capabilities(self) -> VisionCapabilitySnapshot: ...

    def analyze(self, request: VisionRequest) -> VisionResult: ...


@dataclass(frozen=True, slots=True)
class _BoundVisualCall:
    call_key: str
    session_id: str
    provider_identity_sha256: str
    request_binding_sha256: str
    request_json: str
    question_call_key: str | None = None


class SqliteMountedVisualCallLedger:
    """持久化一个 provider 绑定挂载视觉发送的精确状态。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self._explicit_path = Path(path) if path is not None else None

    def path_for(self, session_id: str) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        return (
            attachment_storage.session_root(session_id)
            / "runtime"
            / "mounted_visual_calls.sqlite"
        )

    def dispatch(
        self,
        *,
        session_id: str,
        adapter: _VisionAdapter,
        request: VisionRequest,
        capabilities: VisionCapabilitySnapshot | None = None,
    ) -> VisionResult:
        """兼容旧调用方：返回结果，并将内部 ledger 错误投影为等待外部。"""

        if not isinstance(request, VisionRequest):
            raise TypeError("request must be a VisionRequest")
        try:
            return self.dispatch_with_receipt(
                session_id=session_id,
                adapter=adapter,
                request=request,
                capabilities=capabilities,
            ).result
        except MountedVisualCallWaitingExternal:
            raise
        except Exception as exc:
            raise MountedVisualCallWaitingExternal(
                reason_code="visual_dispatch_authority_unavailable"
            ) from exc

    def dispatch_with_receipt(
        self,
        *,
        session_id: str,
        adapter: _VisionAdapter,
        request: VisionRequest,
        capabilities: VisionCapabilitySnapshot | None = None,
        publication_envelope: MountedVisualProjectPublicationEnvelope | None = None,
    ) -> MountedVisualCallReceipt:
        """预留、发送并结算，返回可恢复的 provider call receipt。

        有意不捕获 ``BaseException``。它代表硬进程丢失窗口；已提交的 ``pending`` 与
        ``awaiting_result`` 正是阻止下一进程再次外发像素的依据。
        """

        if not isinstance(request, VisionRequest):
            raise TypeError("request must be a VisionRequest")
        if publication_envelope is not None and not isinstance(
            publication_envelope,
            MountedVisualProjectPublicationEnvelope,
        ):
            raise TypeError(
                "publication_envelope must be MountedVisualProjectPublicationEnvelope"
            )
        try:
            if publication_envelope is not None:
                _validate_publication_target_for_request(
                    publication_envelope.target,
                    request,
                )
            capabilities = capabilities or adapter.capabilities()
            bound = _bind_call(
                session_id=session_id,
                capabilities=capabilities,
                request=request,
            )
            replay = self._reserve_or_replay(
                bound,
                publication_envelope=publication_envelope,
            )
        except (MountedVisualCallWaitingExternal, MountedVisualCallLedgerError):
            raise
        except Exception as exc:
            raise MountedVisualCallWaitingExternal(
                reason_code="visual_dispatch_authority_unavailable"
            ) from exc
        if replay is not None:
            return replay

        try:
            result = adapter.analyze(request)
        except Exception as exc:
            self._best_effort_mark_uncertain(
                bound,
                error_code="vision_adapter_exception",
            )
            raise MountedVisualCallWaitingExternal(
                reason_code="vision_adapter_exception", call_key=bound.call_key,
            ) from exc

        if not isinstance(result, VisionResult) or not _same_provider(capabilities, result):
            self._best_effort_mark_uncertain(
                bound,
                error_code="vision_provider_identity_mismatch",
            )
            raise MountedVisualCallWaitingExternal(
                reason_code="vision_provider_identity_mismatch", call_key=bound.call_key,
            )
        prepared = request.prepared_payload
        if prepared is None or result.input_sha256 != prepared.sent_sha256:
            self._best_effort_mark_uncertain(
                bound,
                error_code="vision_input_identity_mismatch",
            )
            raise MountedVisualCallWaitingExternal(
                reason_code="vision_input_identity_mismatch", call_key=bound.call_key,
            )
        if (
            result.status is VisionStatus.FAILED
            and (
                result.failure_code in _UNCERTAIN_FAILURE_CODES
                or (result.failure_diagnostics is not None
                    and result.failure_diagnostics.completion_uncertain)
            )
        ):
            self._best_effort_mark_uncertain(
                bound,
                error_code=str(result.failure_code),
            )
            raise MountedVisualCallWaitingExternal(
                reason_code=str(result.failure_code),
                failure_diagnostics=result.failure_diagnostics,
                call_key=bound.call_key,
            )
        try:
            return self._settle_succeeded(
                bound,
                result,
                expected_publication_envelope=publication_envelope,
            )
        except Exception as exc:
            # Provider I/O 已发生。响应持久化失败不能成为再次发送的权威依据。
            raise MountedVisualCallWaitingExternal() from exc

    def _reserve_or_replay(
        self,
        bound: _BoundVisualCall,
        *,
        publication_envelope: MountedVisualProjectPublicationEnvelope | None,
    ) -> MountedVisualCallReceipt | None:
        waiting = False
        replay: MountedVisualCallReceipt | None = None
        with self._connect(bound.session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                bound.question_call_key is not None
                and bound.question_call_key != bound.call_key
                and connection.execute(
                    "SELECT 1 FROM mounted_visual_provider_calls WHERE call_key=?",
                    (bound.question_call_key,),
                ).fetchone() is not None
            ):
                # 已预留的问答不能改用旧模式绕开精确请求校验；其他旧模式仍用原内容键。
                raise MountedVisualCallLedgerError(
                    "mounted visual call key crossed immutable request authority"
                )
            row = connection.execute(
                "SELECT * FROM mounted_visual_provider_calls WHERE call_key=?",
                (bound.call_key,),
            ).fetchone()
            if row is not None:
                _require_same_bound_call(row, bound)
                publication_row = _publication_row(connection, bound.call_key)
                if publication_envelope is not None:
                    if publication_row is None:
                        _insert_publication_row(
                            connection,
                            bound=bound,
                            envelope=publication_envelope,
                        )
                        publication_row = _publication_row(
                            connection,
                            bound.call_key,
                        )
                    else:
                        _require_same_publication_envelope(
                            publication_row,
                            publication_envelope,
                        )
                status = str(row["status"])
                if status == "succeeded":
                    result = _result_from_row(row)
                    if publication_row is not None and str(
                        publication_row["publication_state"]
                    ) == MountedVisualPublicationState.AWAITING_RESULT.value:
                        _advance_awaiting_publication(
                            connection,
                            call_key=bound.call_key,
                            result=result,
                        )
                        publication_row = _publication_row(
                            connection,
                            bound.call_key,
                        )
                    replay = _receipt_from_rows(
                        row,
                        publication_row,
                        replayed=True,
                    )
                elif status in {"pending", "uncertain"}:
                    # 先提交可能刚补上的 awaiting_result，再在事务外投影等待。若在此抛出，
                    # sqlite context manager 会回滚 legacy attachment。
                    waiting = True
                else:
                    raise MountedVisualCallLedgerError(
                        "mounted visual call has an unsupported durable status"
                    )
            else:
                now = _now()
                connection.execute(
                    """
                    INSERT INTO mounted_visual_provider_calls (
                        call_key,
                        session_id,
                        provider_identity_sha256,
                        request_binding_sha256,
                        request_json,
                        status,
                        result_json,
                        result_sha256,
                        error_code,
                        created_at,
                        settled_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, NULL)
                    """,
                    (
                        bound.call_key,
                        bound.session_id,
                        bound.provider_identity_sha256,
                        bound.request_binding_sha256,
                        bound.request_json,
                        now,
                    ),
                )
                if publication_envelope is not None:
                    _insert_publication_row(
                        connection,
                        bound=bound,
                        envelope=publication_envelope,
                        created_at=now,
                    )
        # 上下文管理器会在本方法返回及 ``adapter.analyze`` 运行前提交 pending 记录。
        if waiting:
            raise MountedVisualCallWaitingExternal(
                reason_code=str(row["error_code"] or "visual_completion_unconfirmed"),
                call_key=bound.call_key,
            )
        return replay

    def _settle_succeeded(
        self,
        bound: _BoundVisualCall,
        result: VisionResult,
        *,
        expected_publication_envelope: (
            MountedVisualProjectPublicationEnvelope | None
        ),
    ) -> MountedVisualCallReceipt:
        result_json = _canonical_json(_result_payload(result))
        if len(result_json.encode("utf-8")) > _MAX_RESULT_BYTES:
            self._best_effort_mark_uncertain(
                bound,
                error_code="vision_result_too_large",
            )
            raise MountedVisualCallLedgerError("vision result exceeds durable limit")
        result_sha256 = _sha256_text(result_json)
        with self._connect(bound.session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE mounted_visual_provider_calls
                   SET status='succeeded',
                       result_json=?,
                       result_sha256=?,
                       settled_at=?
                 WHERE call_key=?
                   AND session_id=?
                   AND provider_identity_sha256=?
                   AND request_binding_sha256=?
                   AND status='pending'
                """,
                (
                    result_json,
                    result_sha256,
                    _now(),
                    bound.call_key,
                    bound.session_id,
                    bound.provider_identity_sha256,
                    bound.request_binding_sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise MountedVisualCallLedgerError(
                    "mounted visual success did not settle its exact pending call"
                )
            publication_row = _publication_row(connection, bound.call_key)
            if expected_publication_envelope is not None:
                if publication_row is None:
                    raise MountedVisualCallLedgerError(
                        "mounted visual publication envelope disappeared before settlement"
                    )
                _require_same_publication_envelope(
                    publication_row,
                    expected_publication_envelope,
                )
            if publication_row is not None:
                if str(publication_row["publication_state"]) != (
                    MountedVisualPublicationState.AWAITING_RESULT.value
                ):
                    raise MountedVisualCallLedgerError(
                        "pending provider call crossed publication terminal state"
                    )
                _advance_awaiting_publication(
                    connection,
                    call_key=bound.call_key,
                    result=result,
                )
                publication_row = _publication_row(connection, bound.call_key)
            settled = connection.execute(
                "SELECT * FROM mounted_visual_provider_calls WHERE call_key=?",
                (bound.call_key,),
            ).fetchone()
            if settled is None:
                raise MountedVisualCallLedgerError(
                    "settled mounted visual call disappeared"
                )
            return _receipt_from_rows(
                settled,
                publication_row,
                replayed=False,
            )

    def list_ready_publications(
        self,
        *,
        session_id: str,
        limit: int = 100,
    ) -> tuple[MountedVisualCallReceipt, ...]:
        """仅从 durable ledger 扫描 ready 项；不读取或重新准备任何源文件。"""

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("ready publication limit must be between 1 and 1000")
        with self._connect(session_id) as connection:
            rows = connection.execute(
                """
                SELECT calls.*,
                       publications.call_key AS publication_call_key,
                       publications.session_id AS publication_session_id,
                       publications.envelope_version,
                       publications.envelope_json,
                       publications.envelope_sha256,
                       publications.state AS publication_state,
                       publications.skip_reason,
                       publications.completed_at
                  FROM mounted_visual_provider_calls AS calls
                  JOIN mounted_visual_project_publications AS publications
                    ON publications.call_key = calls.call_key
                 WHERE calls.session_id=?
                   AND calls.status='succeeded'
                   AND publications.state='ready'
                 ORDER BY publications.created_at, publications.call_key
                 LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return tuple(
            _receipt_from_rows(
                row,
                row,
                replayed=True,
                publication_call_key_column="publication_call_key",
                publication_session_id_column="publication_session_id",
            )
            for row in rows
        )

    def mark_publication_published(
        self,
        receipt: MountedVisualCallReceipt,
    ) -> MountedVisualCallReceipt:
        """在 Project 原子提交完成后，将同一 ready envelope 幂等确认。"""

        return self._finish_publication(
            receipt,
            target=MountedVisualPublicationState.PUBLISHED,
            skip_reason=None,
        )

    def mark_publication_skipped(
        self,
        receipt: MountedVisualCallReceipt,
        *,
        reason_code: str,
    ) -> MountedVisualCallReceipt:
        """显式终止一个无法发布的 ready 项，不允许回到可发布状态。"""

        if (
            not isinstance(reason_code, str)
            or _STABLE_CODE.fullmatch(reason_code) is None
        ):
            raise ValueError("publication skip reason must be a stable code")
        return self._finish_publication(
            receipt,
            target=MountedVisualPublicationState.SKIPPED,
            skip_reason=reason_code,
        )

    def _finish_publication(
        self,
        receipt: MountedVisualCallReceipt,
        *,
        target: MountedVisualPublicationState,
        skip_reason: str | None,
    ) -> MountedVisualCallReceipt:
        if not isinstance(receipt, MountedVisualCallReceipt):
            raise TypeError("receipt must be a MountedVisualCallReceipt")
        if receipt.publication_envelope is None or receipt.publication_state is None:
            raise MountedVisualCallLedgerError(
                "mounted visual call has no Project publication envelope"
            )
        with self._connect(receipt.session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            call_row = connection.execute(
                "SELECT * FROM mounted_visual_provider_calls WHERE call_key=?",
                (receipt.call_key,),
            ).fetchone()
            publication_row = _publication_row(connection, receipt.call_key)
            if call_row is None or publication_row is None:
                raise MountedVisualCallLedgerError(
                    "mounted visual publication receipt no longer exists"
                )
            _require_same_call_receipt(call_row, publication_row, receipt)
            current = MountedVisualPublicationState(
                str(publication_row["publication_state"])
            )
            if current is MountedVisualPublicationState.READY:
                cursor = connection.execute(
                    """
                    UPDATE mounted_visual_project_publications
                       SET state=?, skip_reason=?, completed_at=?
                     WHERE call_key=? AND state='ready'
                    """,
                    (target.value, skip_reason, _now(), receipt.call_key),
                )
                if cursor.rowcount != 1:
                    raise MountedVisualCallLedgerError(
                        "mounted visual publication did not finish its exact ready row"
                    )
            elif current is not target:
                raise MountedVisualCallLedgerError(
                    "mounted visual publication cannot cross a terminal state"
                )
            elif (
                current is MountedVisualPublicationState.SKIPPED
                and str(publication_row["skip_reason"]) != skip_reason
            ):
                raise MountedVisualCallLedgerError(
                    "mounted visual skipped publication reason changed"
                )
            updated = _publication_row(connection, receipt.call_key)
            if updated is None:
                raise MountedVisualCallLedgerError(
                    "finished mounted visual publication disappeared"
                )
            return _receipt_from_rows(call_row, updated, replayed=True)

    def _best_effort_mark_uncertain(
        self,
        bound: _BoundVisualCall,
        *,
        error_code: str,
    ) -> None:
        try:
            with self._connect(bound.session_id) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE mounted_visual_provider_calls
                       SET status='uncertain', error_code=?, settled_at=?
                     WHERE call_key=?
                       AND session_id=?
                       AND provider_identity_sha256=?
                       AND request_binding_sha256=?
                       AND status='pending'
                    """,
                    (
                        error_code,
                        _now(),
                        bound.call_key,
                        bound.session_id,
                        bound.provider_identity_sha256,
                        bound.request_binding_sha256,
                    ),
                )
        except Exception:
            # 保持 ``pending`` 同样属于失败关闭。调用方仍投影 completion_unconfirmed，
            # 且绝不自动重试。
            return

    def _connect(self, session_id: str) -> sqlite3.Connection:
        path = self.path_for(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA synchronous=FULL")
        self._ensure_schema(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mounted_visual_provider_calls (
                    call_key                    TEXT PRIMARY KEY,
                    session_id                  TEXT NOT NULL,
                    provider_identity_sha256    TEXT NOT NULL,
                    request_binding_sha256      TEXT NOT NULL,
                    request_json                TEXT NOT NULL,
                    status                      TEXT NOT NULL CHECK (
                        status IN ('pending', 'succeeded', 'uncertain')
                    ),
                    result_json                 TEXT,
                    result_sha256               TEXT,
                    error_code                  TEXT,
                    created_at                  TEXT NOT NULL,
                    settled_at                  TEXT,
                    CHECK (
                        (status='pending'
                            AND result_json IS NULL
                            AND result_sha256 IS NULL
                            AND error_code IS NULL
                            AND settled_at IS NULL)
                        OR
                        (status='succeeded'
                            AND result_json IS NOT NULL
                            AND result_sha256 IS NOT NULL
                            AND error_code IS NULL
                            AND settled_at IS NOT NULL)
                        OR
                        (status='uncertain'
                            AND result_json IS NULL
                            AND result_sha256 IS NULL
                            AND error_code IS NOT NULL
                            AND settled_at IS NOT NULL)
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mounted_visual_calls_session
                    ON mounted_visual_provider_calls(session_id, created_at, call_key)
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mounted_visual_calls_authority
                    ON mounted_visual_provider_calls(call_key, session_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mounted_visual_project_publications (
                    call_key          TEXT PRIMARY KEY,
                    session_id        TEXT NOT NULL,
                    envelope_version  TEXT NOT NULL,
                    envelope_json     TEXT NOT NULL,
                    envelope_sha256   TEXT NOT NULL,
                    state             TEXT NOT NULL CHECK (
                        state IN ('awaiting_result', 'ready', 'published', 'skipped')
                    ),
                    skip_reason       TEXT,
                    created_at        TEXT NOT NULL,
                    completed_at      TEXT,
                    FOREIGN KEY(call_key, session_id)
                        REFERENCES mounted_visual_provider_calls(call_key, session_id)
                        ON DELETE RESTRICT,
                    CHECK (
                        (state IN ('awaiting_result', 'ready')
                            AND skip_reason IS NULL
                            AND completed_at IS NULL)
                        OR
                        (state='published'
                            AND skip_reason IS NULL
                            AND completed_at IS NOT NULL)
                        OR
                        (state='skipped'
                            AND skip_reason IS NOT NULL
                            AND completed_at IS NOT NULL)
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mounted_visual_publications_ready
                    ON mounted_visual_project_publications(
                        session_id, state, created_at, call_key
                    )
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_mounted_visual_publication_authority_insert
                BEFORE INSERT ON mounted_visual_project_publications
                WHEN NOT EXISTS (
                    SELECT 1
                      FROM mounted_visual_provider_calls AS provider_call
                     WHERE provider_call.call_key = NEW.call_key
                       AND provider_call.session_id = NEW.session_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'mounted visual publication authority mismatch');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_mounted_visual_publication_authority_update
                BEFORE UPDATE OF call_key, session_id
                    ON mounted_visual_project_publications
                WHEN NOT EXISTS (
                    SELECT 1
                      FROM mounted_visual_provider_calls AS provider_call
                     WHERE provider_call.call_key = NEW.call_key
                       AND provider_call.session_id = NEW.session_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'mounted visual publication authority mismatch');
                END
                """
            )


class DurableMountedVisionAdapter:
    """仅在披露 gate 允许 I/O 后应用的 adapter 包装器。"""

    transmits_externally = True

    def __init__(
        self,
        delegate: _VisionAdapter,
        *,
        session_id: str,
        ledger: SqliteMountedVisualCallLedger,
    ) -> None:
        if not vision_adapter_transmits_externally(delegate):
            raise ValueError("durable visual wrapper requires an external adapter")
        self._delegate = delegate
        self._session_id = session_id
        self._ledger = ledger
        # 上游披露决策与持久化 provider 绑定使用完全相同的快照。否则在两步之间重读
        # 可变 capability 对象，可能会授权 endpoint A 却预留 endpoint B。
        self._capabilities = delegate.capabilities()

    def capabilities(self) -> VisionCapabilitySnapshot:
        return self._capabilities

    def analyze(self, request: VisionRequest) -> VisionResult:
        return self._ledger.dispatch(
            session_id=self._session_id,
            adapter=self._delegate,
            request=request,
            capabilities=self._capabilities,
        )

    def analyze_with_receipt(
        self,
        request: VisionRequest,
        *,
        publication_envelope: MountedVisualProjectPublicationEnvelope | None = None,
    ) -> MountedVisualCallReceipt:
        """供 Host application service 使用；旧 ``analyze`` 契约保持不变。"""

        return self._ledger.dispatch_with_receipt(
            session_id=self._session_id,
            adapter=self._delegate,
            request=request,
            capabilities=self._capabilities,
            publication_envelope=publication_envelope,
        )


def _insert_publication_row(
    connection: sqlite3.Connection,
    *,
    bound: _BoundVisualCall,
    envelope: MountedVisualProjectPublicationEnvelope,
    created_at: str | None = None,
) -> None:
    try:
        connection.execute(
            """
            INSERT INTO mounted_visual_project_publications (
                call_key,
                session_id,
                envelope_version,
                envelope_json,
                envelope_sha256,
                state,
                skip_reason,
                created_at,
                completed_at
            ) VALUES (?, ?, ?, ?, ?, 'awaiting_result', NULL, ?, NULL)
            """,
            (
                bound.call_key,
                bound.session_id,
                envelope.contract_version,
                envelope.canonical_json,
                envelope.envelope_sha256,
                created_at or _now(),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise MountedVisualCallLedgerError(
            "mounted visual publication could not bind its provider call"
        ) from exc


def _publication_row(
    connection: sqlite3.Connection,
    call_key: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT call_key,
               session_id,
               envelope_version,
               envelope_json,
               envelope_sha256,
               state AS publication_state,
               skip_reason,
               created_at,
               completed_at
          FROM mounted_visual_project_publications
         WHERE call_key=?
        """,
        (call_key,),
    ).fetchone()


def _require_same_publication_envelope(
    row: sqlite3.Row,
    envelope: MountedVisualProjectPublicationEnvelope,
) -> None:
    if (
        str(row["envelope_version"]) != envelope.contract_version
        or str(row["envelope_sha256"]) != envelope.envelope_sha256
        or str(row["envelope_json"]) != envelope.canonical_json
    ):
        raise MountedVisualCallLedgerError(
            "mounted visual call key crossed immutable publication authority"
        )


def _advance_awaiting_publication(
    connection: sqlite3.Connection,
    *,
    call_key: str,
    result: VisionResult,
) -> None:
    if result.status in {VisionStatus.COMPLETED, VisionStatus.PARTIAL} and (
        result.observations
    ):
        state = MountedVisualPublicationState.READY
        skip_reason = None
        completed_at = None
    else:
        state = MountedVisualPublicationState.SKIPPED
        skip_reason = f"visual_result_{result.status.value}"
        completed_at = _now()
    cursor = connection.execute(
        """
        UPDATE mounted_visual_project_publications
           SET state=?, skip_reason=?, completed_at=?
         WHERE call_key=? AND state='awaiting_result'
        """,
        (state.value, skip_reason, completed_at, call_key),
    )
    if cursor.rowcount != 1:
        raise MountedVisualCallLedgerError(
            "mounted visual result did not advance its exact publication envelope"
        )


def _receipt_from_rows(
    call_row: sqlite3.Row,
    publication_row: sqlite3.Row | None,
    *,
    replayed: bool,
    publication_call_key_column: str = "call_key",
    publication_session_id_column: str = "session_id",
) -> MountedVisualCallReceipt:
    _validate_durable_call_row(call_row)
    result = _result_from_row(call_row)
    envelope: MountedVisualProjectPublicationEnvelope | None = None
    publication_state: MountedVisualPublicationState | None = None
    if publication_row is not None:
        try:
            envelope = MountedVisualProjectPublicationEnvelope.from_canonical_json(
                contract_version=str(publication_row["envelope_version"]),
                canonical_json=str(publication_row["envelope_json"]),
                envelope_sha256=str(publication_row["envelope_sha256"]),
            )
            publication_state = MountedVisualPublicationState(
                str(publication_row["publication_state"])
            )
        except (TypeError, ValueError) as exc:
            raise MountedVisualCallLedgerError(
                "mounted visual publication envelope is malformed"
            ) from exc
        if str(publication_row[publication_call_key_column]) != str(
            call_row["call_key"]
        ) or str(
            publication_row[publication_session_id_column]
        ) != str(call_row["session_id"]):
            raise MountedVisualCallLedgerError(
                "mounted visual publication crossed provider call authority"
            )
    return MountedVisualCallReceipt(
        call_key=str(call_row["call_key"]),
        session_id=str(call_row["session_id"]),
        provider_identity_sha256=str(call_row["provider_identity_sha256"]),
        request_binding_sha256=str(call_row["request_binding_sha256"]),
        result_sha256=str(call_row["result_sha256"]),
        result=result,
        publication_envelope=envelope,
        publication_state=publication_state,
        replayed=replayed,
    )


def _validate_durable_call_row(row: sqlite3.Row) -> None:
    call_key = row["call_key"]
    session_id = row["session_id"]
    provider_identity_sha256 = row["provider_identity_sha256"]
    request_binding_sha256 = row["request_binding_sha256"]
    request_json = row["request_json"]
    if (
        not isinstance(call_key, str)
        or _CALL_KEY.fullmatch(call_key) is None
        or not isinstance(session_id, str)
        or not session_id.strip()
        or not isinstance(provider_identity_sha256, str)
        or _SHA256.fullmatch(provider_identity_sha256) is None
        or not isinstance(request_binding_sha256, str)
        or _SHA256.fullmatch(request_binding_sha256) is None
        or not isinstance(request_json, str)
        or _sha256_text(request_json) != request_binding_sha256
        or str(row["status"]) != "succeeded"
    ):
        raise MountedVisualCallLedgerError(
            "mounted visual provider call failed its durable binding"
        )
    try:
        request_payload = json.loads(request_json)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise MountedVisualCallLedgerError(
            "mounted visual provider request is malformed"
        ) from exc
    if _canonical_json(request_payload) != request_json:
        raise MountedVisualCallLedgerError(
            "mounted visual provider request JSON is not canonical"
        )
    expected_call_key = _call_key(
        session_id=session_id,
        provider_identity_sha256=provider_identity_sha256,
        request_binding_sha256=request_binding_sha256,
        request_payload=request_payload,
    )
    if call_key != expected_call_key:
        raise MountedVisualCallLedgerError(
            "mounted visual provider call key failed its durable hash"
        )


def _require_same_call_receipt(
    call_row: sqlite3.Row,
    publication_row: sqlite3.Row,
    receipt: MountedVisualCallReceipt,
) -> None:
    durable = _receipt_from_rows(call_row, publication_row, replayed=True)
    if (
        durable.call_key != receipt.call_key
        or durable.session_id != receipt.session_id
        or durable.provider_identity_sha256 != receipt.provider_identity_sha256
        or durable.request_binding_sha256 != receipt.request_binding_sha256
        or durable.result_sha256 != receipt.result_sha256
        or durable.result != receipt.result
        or durable.publication_envelope != receipt.publication_envelope
    ):
        raise MountedVisualCallLedgerError(
            "mounted visual publication receipt crossed durable authority"
        )


def _bind_call(
    *,
    session_id: str,
    capabilities: VisionCapabilitySnapshot,
    request: VisionRequest,
) -> _BoundVisualCall:
    if not isinstance(capabilities, VisionCapabilitySnapshot):
        raise MountedVisualCallLedgerError(
            "vision adapter returned an invalid capability snapshot"
        )
    provider_identity_sha256 = _sha256_value(
        {
            "endpoint_identity": capabilities.endpoint_identity,
            "model": capabilities.model,
            "processor_fingerprint": capabilities.processor_fingerprint,
            "provider": capabilities.provider,
            "schema_version": "mounted-visual-provider-identity-v1",
        }
    )
    request_payload = _request_payload(request)
    request_json = _canonical_json(request_payload)
    request_binding_sha256 = _sha256_text(request_json)
    call_key = _call_key(
        session_id=session_id,
        provider_identity_sha256=provider_identity_sha256,
        request_binding_sha256=request_binding_sha256,
        request_payload=request_payload,
    )
    return _BoundVisualCall(
        call_key=call_key,
        session_id=session_id,
        provider_identity_sha256=provider_identity_sha256,
        request_binding_sha256=request_binding_sha256,
        request_json=request_json,
        question_call_key=(
            _question_call_key(
                session_id=session_id,
                logical_tool_call_id=request.logical_tool_call_id,
                source_unit_id=request.source_unit_id,
            )
            if request.logical_tool_call_id is not None
            else None
        ),
    )


def _call_key(
    *,
    session_id: str,
    provider_identity_sha256: str,
    request_binding_sha256: str,
    request_payload: Mapping[str, object],
) -> str:
    """问答按 Host 调用预留；内容摘要只校验恢复一致性，不形成内容缓存。"""

    if not isinstance(request_payload, Mapping):
        raise MountedVisualCallLedgerError("mounted visual request must be an object")
    if request_payload.get("schema_version") == "mounted-visual-bound-request-v3":
        try:
            purpose = VisionPurpose(request_payload.get("purpose"))
            if purpose is not VisionPurpose.QUESTION:
                raise ValueError("question request version requires question purpose")
            question = request_payload.get("question")
            if normalize_vision_question(purpose, question) != question:
                raise ValueError("question is not normalized")
            call_id = request_payload.get("logical_tool_call_id")
            validate_vision_call_identity(purpose, call_id)
            unit_id = request_payload.get("source_unit_id")
            if not isinstance(unit_id, str) or not unit_id.strip():
                raise ValueError("source_unit_id is empty")
        except (TypeError, ValueError) as exc:
            raise MountedVisualCallLedgerError(
                "mounted visual question request has invalid identity"
            ) from exc
        # 不把问题、像素或 provider 放进 key；同一调用内改变它们应命中原预留并拒绝漂移。
        return _question_call_key(
            session_id=session_id,
            logical_tool_call_id=call_id,
            source_unit_id=unit_id,
        )
    elif (
        request_payload.get("schema_version") == "mounted-visual-bound-request-v2"
        and request_payload.get("purpose") in {"caption", "chart", "formula", "general"}
    ):
        # 历史四种固定用途的既有调用身份保持不变；版本适配只位于持久合同边界。
        binding = {
            "provider_identity_sha256": provider_identity_sha256,
            "request_binding_sha256": request_binding_sha256,
            "schema_version": "mounted-visual-call-key-v2",
            "session_id": session_id,
        }
    else:
        raise MountedVisualCallLedgerError("mounted visual request contract is unsupported")
    return "mvc_" + _sha256_value(binding)


def _question_call_key(
    *, session_id: str, logical_tool_call_id: str, source_unit_id: str
) -> str:
    return "mvc_" + _sha256_value(
        {
            "schema_version": "mounted-visual-call-key-v3",
            "session_id": session_id,
            "logical_tool_call_id": logical_tool_call_id,
            "source_unit_id": source_unit_id,
        }
    )


def _request_payload(request: VisionRequest) -> dict[str, object]:
    prepared = request.prepared_payload
    if prepared is None:
        raise MountedVisualCallLedgerError(
            "external visual dispatch requires a verified prepared payload"
        )
    locator = request.locator
    payload = {
        "byte_count": request.byte_count,
        "detail": request.detail.value,
        "image_sha256": request.image_sha256,
        "locator": {
            "bbox": list(locator.bbox) if locator.bbox is not None else None,
            "char_range": (
                list(locator.char_range) if locator.char_range is not None else None
            ),
            "ordinal": locator.ordinal,
            "page": locator.page,
            "section_path": list(locator.section_path),
        },
        "mime_type": request.mime_type,
        "pixel_size": {
            "height": request.pixel_size.height,
            "width": request.pixel_size.width,
        },
        "prepared_payload": {
            "byte_count": len(prepared.data),
            "mime_type": prepared.mime_type,
            "pixel_size": {
                "height": prepared.pixel_size.height,
                "width": prepared.pixel_size.width,
            },
            "resampled": prepared.resampled,
            "sent_sha256": prepared.sent_sha256,
            "source_sha256": prepared.source_sha256,
        },
        "prompt_contract_version": request.prompt_contract_version,
        "purpose": request.purpose.value,
        "region": request.region.value,
        "schema_version": "mounted-visual-bound-request-v2",
        "source_sha256": request.source_sha256,
        "source_unit_id": request.source_unit_id,
    }
    if request.purpose is VisionPurpose.QUESTION:
        payload.update(
            schema_version="mounted-visual-bound-request-v3",
            question=request.question,
            logical_tool_call_id=request.logical_tool_call_id,
        )
    return payload


def _result_payload(result: VisionResult) -> dict[str, object]:
    return {
        "endpoint_identity": result.endpoint_identity,
        "failure_code": result.failure_code,
        **({"failure_diagnostics": result.failure_diagnostics.to_dict()}
           if result.failure_diagnostics is not None else {}),
        "input_sha256": result.input_sha256,
        "model": result.model,
        "observations": [
            {
                "kind": item.kind,
                "observation_id": item.observation_id,
                "text": item.text,
                "uncertainty": item.uncertainty,
            }
            for item in result.observations
        ],
        "output_sha256": result.output_sha256,
        "processor_fingerprint": result.processor_fingerprint,
        "provider": result.provider,
        "schema_version": "mounted-visual-result-v1",
        "status": result.status.value,
        "unresolved_gap_refs": list(result.unresolved_gap_refs),
        "warnings": list(result.warnings),
    }


def _result_from_row(row: sqlite3.Row) -> VisionResult:
    result_json = row["result_json"]
    result_sha256 = row["result_sha256"]
    if (
        not isinstance(result_json, str)
        or not isinstance(result_sha256, str)
        or _SHA256.fullmatch(result_sha256) is None
        or _sha256_text(result_json) != result_sha256
    ):
        raise MountedVisualCallLedgerError(
            "mounted visual replay result failed its durable hash"
        )
    try:
        payload = json.loads(result_json)
        if _canonical_json(payload) != result_json:
            raise ValueError("result JSON is not canonical")
        if payload.pop("schema_version") != "mounted-visual-result-v1":
            raise ValueError("result schema is unsupported")
        observations = tuple(
            VisionObservation(
                observation_id=str(item["observation_id"]),
                kind=str(item["kind"]),
                text=str(item["text"]),
                uncertainty=item["uncertainty"],
            )
            for item in payload.pop("observations")
        )
        result = VisionResult(
            status=VisionStatus(str(payload.pop("status"))),
            provider=str(payload.pop("provider")),
            model=str(payload.pop("model")),
            endpoint_identity=str(payload.pop("endpoint_identity")),
            processor_fingerprint=str(payload.pop("processor_fingerprint")),
            input_sha256=str(payload.pop("input_sha256")),
            output_sha256=payload.pop("output_sha256"),
            observations=observations,
            unresolved_gap_refs=tuple(payload.pop("unresolved_gap_refs")),
            warnings=tuple(payload.pop("warnings")),
            failure_code=payload.pop("failure_code"),
            failure_diagnostics=(
                VisionFailureDiagnostics(**payload.pop("failure_diagnostics"))
                if "failure_diagnostics" in payload else None
            ),
        )
        if payload:
            raise ValueError("result JSON contains unsupported fields")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise MountedVisualCallLedgerError(
            "mounted visual replay result is malformed"
        ) from exc


def _require_same_bound_call(
    row: sqlite3.Row,
    bound: _BoundVisualCall,
) -> None:
    if (
        str(row["session_id"]) != bound.session_id
        or str(row["provider_identity_sha256"])
        != bound.provider_identity_sha256
        or str(row["request_binding_sha256"]) != bound.request_binding_sha256
        or str(row["request_json"]) != bound.request_json
    ):
        raise MountedVisualCallLedgerError(
            "mounted visual call key crossed immutable request authority"
        )


def _same_provider(
    capabilities: VisionCapabilitySnapshot,
    result: VisionResult,
) -> bool:
    # HTTP adapter 会将精确的已准备像素描述符追加到其 capability 指纹（例如传输尺寸及
    # 是否重采样）。该后缀是 observation 来源，而非 provider 标识变化。接受基础
    # capability 指纹或其显式分隔的载荷专用后代之一；仅有字符串前缀并不足够。
    processor_matches = (
        result.processor_fingerprint == capabilities.processor_fingerprint
        or result.processor_fingerprint.startswith(
            f"{capabilities.processor_fingerprint}+"
        )
    )
    return (
        result.provider == capabilities.provider
        and result.model == capabilities.model
        and result.endpoint_identity == capabilities.endpoint_identity
        and processor_matches
    )


def _validate_publication_target_for_request(
    target: MountedVisualProjectPublicationTarget,
    request: VisionRequest,
) -> None:
    try:
        if target.file_content_sha256 != request.source_sha256:
            raise ValueError("publication file content does not match the request")
        if target.purpose is not request.purpose:
            raise ValueError("publication purpose does not match the request")
        if request.purpose is VisionPurpose.QUESTION and (
            target.question != request.question
            or target.logical_tool_call_id != request.logical_tool_call_id
        ):
            raise ValueError("publication question or call identity does not match the request")
        if target.prompt_contract_version != request.prompt_contract_version:
            raise ValueError(
                "publication prompt contract does not match the request"
            )
        validate_prepared_visual_artifact_receipt(
            request,
            target.prepared_artifact,
        )
    except (TypeError, ValueError) as exc:
        raise MountedVisualCallLedgerError(
            "mounted visual publication target does not bind the prepared request"
        ) from exc


def _normalize_locator_value(value: object, *, depth: int = 0) -> object:
    """Normalize logical locator JSON; path-like package parts remain legitimate."""

    if depth > 32:
        raise ValueError("mounted visual picture locator nesting is too deep")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("mounted visual picture locator numbers must be finite")
        return value
    if isinstance(value, str):
        if any(ord(character) < 32 for character in value):
            raise ValueError("mounted visual picture locator text is not printable")
        if (
            value.startswith(("/", "\\", "~/"))
            or re.match(r"^[A-Za-z]:[\\/]", value) is not None
            or value.casefold().startswith("file://")
            or "\\" in value
            or ".." in value.split("/")
        ):
            raise ValueError("mounted visual picture locator contains a filesystem path")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("mounted visual picture locator keys must be text")
            if key.casefold() in _LOCATOR_FORBIDDEN_KEYS:
                raise ValueError("mounted visual picture locator field is not locative")
            normalized[key] = _normalize_locator_value(item, depth=depth + 1)
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _normalize_locator_value(item, depth=depth + 1)
            for item in value
        ]
    raise ValueError("mounted visual picture locator contains a non-JSON value")


def _validate_locator_contract(kind: str, payload: Mapping[str, object]) -> None:
    if kind in {"whole_file", "full"} and payload:
        raise ValueError(f"{kind} picture locator payload must be empty")
    if kind == "document_surface":
        if set(payload) != {"surface_kind", "ordinal"}:
            raise ValueError("document surface locator has unsupported fields")
        if payload["surface_kind"] not in _DOCUMENT_SURFACE_KINDS:
            raise ValueError("document surface locator kind is unsupported")
        ordinal = payload["ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise ValueError("document surface locator ordinal must be positive")


def _validate_publication_identifier(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_PUBLICATION_IDENTIFIER_CHARS
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field} must be a bounded non-empty identifier")


def _validate_publication_version(value: object, *, field: str) -> None:
    if not isinstance(value, str) or _PUBLICATION_VERSION.fullmatch(value) is None:
        raise ValueError(f"{field} must be a bounded version string")


def _validate_sha256(value: object, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase sha256 digest")


def _validate_media_type(value: object, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip().lower()
        or ";" in value
        or value.count("/") != 1
        or any(not part for part in value.split("/"))
        or any(ord(character) < 33 for character in value)
    ):
        raise ValueError(f"{field} must be a canonical type/subtype media type")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    'DurableMountedVisionAdapter',
    "MOUNTED_VISUAL_CALL_RECEIPT_CONTRACT",
    "MOUNTED_VISUAL_PROJECT_PUBLICATION_CONTRACT",
    "MountedVisualCallReceipt",
    "MountedVisualCallLedgerError",
    "MountedVisualCallWaitingExternal",
    "MountedVisualPictureLocator",
    "MountedVisualProjectPublicationEnvelope",
    "MountedVisualProjectPublicationTarget",
    "MountedVisualPublicationState",
    'SqliteMountedVisualCallLedger',
]
