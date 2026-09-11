"""当前 Turn 执行期间可修订的 Findings 工作笔记。

Findings 保存模型认为值得跨 Attempt 保留的发现、决定和未解缺口。它不是最终答案、
长期记忆、AcceptanceProgress 或“已读完”的机械证明；自然语言记录在后续语义验证或
证据提升前仍是不可信候选项。模型通过 :mod:`personagraph.tools.findings` 请求修改，
Host 在 :mod:`personagraph.session.persistence` 中负责标识、配额、来源准入和物理存储。
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

EXECUTION_FINDINGS_CONTRACT_VERSION = "execution-findings-v1"
EXECUTION_FINDING_CLAIM_MAX_CHARACTERS = 4_096
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def l1_execution_note_writer_id(attempt_id: str) -> str:
    """一次已提交 L1 决定对应一个幂等 Host 台账调用，不占模型工具名额。"""
    return "l1notes:" + hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ExecutionFindingsOwnerKind(StrEnum):
    L1_TURN_RUN = "l1_turn_run"
    WORK_RUN = "work_run"


class ExecutionFindingsLedgerStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class ExecutionFindingKind(StrEnum):
    FINDING = "finding"
    DECISION = "decision"
    GAP = "gap"


class ExecutionFindingStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class ExecutionFindingRevisionOperation(StrEnum):
    RECORD = "record"
    SUPERSEDE = "supersede"
    RETRACT = "retract"


class ExecutionFindingsPersistenceError(RuntimeError):
    """所请求账本操作没有有效持久化权威。"""


class ExecutionFindingsOwnerClosed(ExecutionFindingsPersistenceError):
    """执行所有者或其 findings 账本已处于终态。"""


class ExecutionFindingsRevisionConflict(ExecutionFindingsPersistenceError):
    """某项 findings 变更基于过期账本 revision。"""

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"execution findings revision changed: expected {expected}, got {actual}"
        )


class ExecutionFindingsMutationIdentityCollision(
    ExecutionFindingsPersistenceError
):
    """同一稳定变更标识被复用于不同事实。"""


class ExecutionFindingsQuotaExceeded(ExecutionFindingsPersistenceError):
    """某项变更会超出账本的持久化配额或逐写入配额。"""


class ExecutionFindingsSourceReferenceInvalid(
    ExecutionFindingsPersistenceError
):
    """某个模型撰写的来源指针没有持久化工具输出支持。"""


class ExecutionFindingsScopeInvalid(ExecutionFindingsPersistenceError):
    """所有者当前义务权威状态中缺少某个 finding scope。"""


class ExecutionFindingsStoredAuthorityCorrupt(
    ExecutionFindingsPersistenceError
):
    """已存储 findings 或所引用执行权威状态不规范。"""


class ExecutionFindingsQuota(_Contract):
    schema_version: Literal["execution-findings-quota-v1"] = (
        "execution-findings-quota-v1"
    )
    max_mutation_items: int = Field(default=4, ge=1, le=4)
    max_claim_characters: int = Field(
        default=EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
        ge=80,
        le=EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    )
    max_source_refs_per_entry: int = Field(default=16, ge=1, le=64)
    max_scope_keys_per_entry: int = Field(default=24, ge=1, le=64)
    max_active_entries: int = Field(default=64, ge=1, le=128)
    max_active_projection_utf8_bytes: int = Field(default=32_000, ge=1_024)
    max_durable_entry_revisions: int = Field(default=256, ge=1, le=2_048)
    max_durable_utf8_bytes: int = Field(default=256_000, ge=8_192)

    @model_validator(mode="after")
    def _require_coherent_limits(self) -> 'ExecutionFindingsQuota':
        if self.max_active_entries > self.max_durable_entry_revisions:
            raise ValueError("active entry limit cannot exceed durable revision limit")
        if self.max_active_projection_utf8_bytes > self.max_durable_utf8_bytes:
            raise ValueError("active byte limit cannot exceed durable byte limit")
        return self


class ExecutionFindingSourceRef(_Contract):
    """引用当前运行中已返回的 tool_result_id；可选 chunk_id 必须属于该结果。"""

    tool_result_id: str = Field(pattern=_ID_PATTERN)
    chunk_id: str | None = Field(default=None, pattern=_ID_PATTERN)


class RecordExecutionFinding(_Contract):
    operation: Literal["record"] = "record"
    kind: ExecutionFindingKind
    claim: str = Field(
        min_length=1,
        max_length=EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    )
    source_refs: tuple[ExecutionFindingSourceRef, ...] = Field(
        default=(),
        max_length=64,
    )
    scope_keys: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def _validate_record(self) -> 'RecordExecutionFinding':
        validate_execution_finding_text(self.claim)
        validate_execution_finding_scope_keys(self.scope_keys)
        validate_execution_finding_source_refs(self.source_refs)
        return self


class SupersedeExecutionFinding(_Contract):
    operation: Literal["supersede"] = "supersede"
    entry_id: str = Field(pattern=_ID_PATTERN)
    kind: ExecutionFindingKind
    claim: str = Field(
        min_length=1,
        max_length=EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    )
    source_refs: tuple[ExecutionFindingSourceRef, ...] = Field(
        default=(),
        max_length=64,
    )
    scope_keys: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def _validate_supersede(self) -> 'SupersedeExecutionFinding':
        validate_execution_finding_text(self.claim)
        validate_execution_finding_scope_keys(self.scope_keys)
        validate_execution_finding_source_refs(self.source_refs)
        return self


class RetractExecutionFinding(_Contract):
    operation: Literal["retract"] = "retract"
    entry_id: str = Field(pattern=_ID_PATTERN)
    reason: str = Field(min_length=1, max_length=400)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        validate_execution_finding_text(value)
        return value


ExecutionFindingMutationItem = Annotated[
    RecordExecutionFinding
    | SupersedeExecutionFinding
    | RetractExecutionFinding,
    Field(discriminator="operation"),
]


class ExecutionFindingsMutationCommand(_Contract):
    schema_version: Literal["execution-findings-mutation-command-v1"] = (
        "execution-findings-mutation-command-v1"
    )
    ledger_id: str = Field(pattern=_ID_PATTERN)
    mutation_id: str = Field(pattern=_ID_PATTERN)
    expected_ledger_revision: int = Field(ge=0)
    writer_unit_id: str = Field(pattern=_ID_PATTERN)
    writer_tool_call_id: str = Field(pattern=_ID_PATTERN)
    items: tuple[ExecutionFindingMutationItem, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def _reject_ambiguous_targets(self) -> 'ExecutionFindingsMutationCommand':
        targets = [
            item.entry_id
            for item in self.items
            if isinstance(
                item,
                (SupersedeExecutionFinding, RetractExecutionFinding),
            )
        ]
        if len(targets) != len(set(targets)):
            raise ValueError("one mutation cannot revise the same entry twice")
        return self


class ExecutionFindingEntry(_Contract):
    schema_version: Literal["execution-finding-entry-v1"] = (
        "execution-finding-entry-v1"
    )
    ledger_id: str = Field(pattern=_ID_PATTERN)
    entry_id: str = Field(pattern=_ID_PATTERN)
    entry_revision_id: str = Field(pattern=_ID_PATTERN)
    entry_revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    operation: ExecutionFindingRevisionOperation
    kind: ExecutionFindingKind
    claim: str = Field(
        min_length=1,
        max_length=EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    )
    source_refs: tuple[ExecutionFindingSourceRef, ...] = Field(
        default=(),
        max_length=64,
    )
    scope_keys: tuple[str, ...] = Field(default=(), max_length=64)
    status: ExecutionFindingStatus
    writer_unit_id: str = Field(pattern=_ID_PATTERN)
    writer_tool_call_id: str = Field(pattern=_ID_PATTERN)
    mutation_id: str = Field(pattern=_ID_PATTERN)
    supersedes_entry_revision_id: str | None = Field(
        default=None,
        pattern=_ID_PATTERN,
    )
    revision_reason: str | None = Field(default=None, min_length=1, max_length=400)
    created_at: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _validate_entry(self) -> 'ExecutionFindingEntry':
        validate_execution_finding_text(self.claim)
        validate_execution_finding_scope_keys(self.scope_keys)
        validate_execution_finding_source_refs(self.source_refs)
        if self.revision_reason is not None:
            validate_execution_finding_text(self.revision_reason)
        if self.operation is ExecutionFindingRevisionOperation.RECORD:
            if (
                self.entry_revision != 1
                or self.supersedes_entry_revision_id is not None
                or self.revision_reason is not None
            ):
                raise ValueError("record must be the first unlinked entry revision")
        elif self.entry_revision <= 1 or self.supersedes_entry_revision_id is None:
            raise ValueError("a revision must bind its immediate predecessor")
        if self.operation is ExecutionFindingRevisionOperation.RETRACT:
            if (
                self.status is not ExecutionFindingStatus.RETRACTED
                or self.revision_reason is None
            ):
                raise ValueError("retract requires a reason and retracted status")
        elif self.status is ExecutionFindingStatus.RETRACTED:
            raise ValueError("only a retract revision may be retracted")
        return self


def validate_persisted_execution_findings_quota_json(
    value: str,
) -> ExecutionFindingsQuota:
    """Decode a frozen quota without replacing its recorded limits."""

    return ExecutionFindingsQuota.model_validate_json(value)


def validate_persisted_execution_finding_entry(
    value: object,
) -> ExecutionFindingEntry:
    """Decode a frozen entry using the same claim contract as current writes."""

    return ExecutionFindingEntry.model_validate(value)


def reduce_execution_findings_active_queue(
    entry_revisions: tuple[ExecutionFindingEntry, ...],
    *,
    capacity: int,
    max_projection_utf8_bytes: int | None = None,
    projection_utf8_bytes: Callable[
        [
            tuple[ExecutionFindingEntry, ...],
            tuple[ExecutionFindingEntry, ...],
            tuple[ExecutionFindingEntry, ...],
            int,
        ],
        int,
    ]
    | None = None,
) -> tuple[ExecutionFindingEntry, ...]:
    """Replay revisions into a bounded FIFO working set.

    Durable revisions are never deleted.  A record enters at the tail, a
    supersede refreshes its logical entry at the tail, and a retract removes it.
    Once the head is evicted for capacity it does not reappear merely because a
    newer entry is later retracted.  Callers that enforce a serialized projection
    budget provide its exact deterministic measurer; byte eviction then happens
    during the same replay and is equally monotonic.
    """

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("findings queue capacity must be a positive integer")
    if (max_projection_utf8_bytes is None) != (projection_utf8_bytes is None):
        raise ValueError(
            "findings queue byte limit and projection measurer must be paired"
        )
    if max_projection_utf8_bytes is not None and (
        isinstance(max_projection_utf8_bytes, bool)
        or not isinstance(max_projection_utf8_bytes, int)
        or max_projection_utf8_bytes < 1
    ):
        raise ValueError("findings queue byte limit must be a positive integer")
    sequences = tuple(item.sequence for item in entry_revisions)
    if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
        raise ValueError("findings revisions must have ordered unique sequences")

    queue: list[str] = []
    queued: set[str] = set()
    latest: dict[str, ExecutionFindingEntry] = {}
    normalized_revisions: list[ExecutionFindingEntry] = []
    latest_revision_index: dict[str, int] = {}
    seen_mutation_ids: set[str] = set()
    revision_index = 0
    ledger_revision = 0
    while revision_index < len(entry_revisions):
        mutation_id = entry_revisions[revision_index].mutation_id
        if mutation_id in seen_mutation_ids:
            raise ValueError("findings mutation revisions must be contiguous")
        seen_mutation_ids.add(mutation_id)
        ledger_revision += 1

        while (
            revision_index < len(entry_revisions)
            and entry_revisions[revision_index].mutation_id == mutation_id
        ):
            revision = entry_revisions[revision_index]
            revision_index += 1
            entry_id = revision.entry_id
            current = latest.get(entry_id)
            if revision.operation is ExecutionFindingRevisionOperation.RECORD:
                if current is not None:
                    raise ValueError("record cannot reuse an existing finding entry")
            else:
                if current is None:
                    raise ValueError("finding revision has no recorded predecessor")
                if current.operation is ExecutionFindingRevisionOperation.RETRACT:
                    raise ValueError("retracted finding cannot receive another revision")
                if (
                    revision.entry_revision != current.entry_revision + 1
                    or revision.supersedes_entry_revision_id
                    != current.entry_revision_id
                ):
                    raise ValueError(
                        "finding revision does not bind its immediate predecessor"
                    )
                prior_index = latest_revision_index[entry_id]
                normalized_revisions[prior_index] = normalized_revisions[
                    prior_index
                ].model_copy(update={"status": ExecutionFindingStatus.SUPERSEDED})

            normalized_status = (
                ExecutionFindingStatus.RETRACTED
                if revision.operation is ExecutionFindingRevisionOperation.RETRACT
                else ExecutionFindingStatus.ACTIVE
            )
            normalized = revision.model_copy(update={"status": normalized_status})
            latest[entry_id] = normalized
            latest_revision_index[entry_id] = len(normalized_revisions)
            normalized_revisions.append(normalized)

            if entry_id in queued:
                queue.remove(entry_id)
                queued.remove(entry_id)
            if revision.operation is ExecutionFindingRevisionOperation.RETRACT:
                continue
            queue.append(entry_id)
            queued.add(entry_id)

        while len(queue) > capacity:
            queued.remove(queue.pop(0))
        if projection_utf8_bytes is not None:
            assert max_projection_utf8_bytes is not None
            while True:
                active_entries = tuple(
                    entry
                    for entry in latest.values()
                    if entry.status is ExecutionFindingStatus.ACTIVE
                )
                queue_entries = tuple(latest[entry_id] for entry_id in queue)
                measured = projection_utf8_bytes(
                    queue_entries,
                    active_entries,
                    tuple(normalized_revisions),
                    ledger_revision,
                )
                if isinstance(measured, bool) or not isinstance(measured, int):
                    raise ValueError(
                        "findings projection measurer must return an integer"
                    )
                if measured <= max_projection_utf8_bytes:
                    break
                if not queue:
                    raise ValueError(
                        "findings projection byte limit cannot hold its envelope"
                    )
                queued.remove(queue.pop(0))

    active = tuple(latest[entry_id] for entry_id in queue)
    if any(item.status is not ExecutionFindingStatus.ACTIVE for item in active):
        raise ValueError("findings queue contains a non-active latest revision")
    return active


class ExecutionFindingsLedger(_Contract):
    schema_version: Literal["execution-findings-ledger-v1"] = (
        "execution-findings-ledger-v1"
    )
    ledger_id: str = Field(pattern=_ID_PATTERN)
    owner_kind: ExecutionFindingsOwnerKind
    execution_owner_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    originating_turn_id: str = Field(pattern=_ID_PATTERN)
    status: ExecutionFindingsLedgerStatus
    revision: int = Field(ge=0)
    quota: ExecutionFindingsQuota
    entry_revisions: tuple[ExecutionFindingEntry, ...] = ()
    mutation_count: int = Field(ge=0)
    created_at: str = Field(min_length=1, max_length=100)
    updated_at: str = Field(min_length=1, max_length=100)
    closed_at: str | None = Field(default=None, min_length=1, max_length=100)
    ledger_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_ledger(self) -> 'ExecutionFindingsLedger':
        if (self.status is ExecutionFindingsLedgerStatus.CLOSED) != (
            self.closed_at is not None
        ):
            raise ValueError("ledger close timestamp conflicts with status")
        sequences = tuple(item.sequence for item in self.entry_revisions)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(
            set(sequences)
        ):
            raise ValueError("ledger entry sequences must be ordered and unique")
        if any(item.ledger_id != self.ledger_id for item in self.entry_revisions):
            raise ValueError("ledger contains an entry from another owner")
        if any(
            len(item.claim) > self.quota.max_claim_characters
            for item in self.entry_revisions
        ):
            raise ValueError("ledger entry exceeds its configured claim quota")
        if self.ledger_sha256 != execution_findings_ledger_sha256(self):
            raise ValueError("ledger hash does not match its canonical payload")
        return self


class ExecutionFindingsActiveProjection(_Contract):
    schema_version: Literal["execution-findings-active-projection-v1"] = (
        "execution-findings-active-projection-v1"
    )
    ledger_id: str = Field(pattern=_ID_PATTERN)
    ledger_revision: int = Field(ge=0)
    active_entries: tuple[ExecutionFindingEntry, ...]
    omitted_active_count: int = Field(ge=0)
    omitted_entry_ids_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    remaining_durable_revisions: int = Field(ge=0)
    remaining_durable_utf8_bytes: int = Field(ge=0)
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_projection(self) -> 'ExecutionFindingsActiveProjection':
        if (self.omitted_active_count == 0) != (
            self.omitted_entry_ids_sha256 is None
        ):
            raise ValueError("omitted-entry hash conflicts with omitted count")
        if self.projection_sha256 != execution_findings_projection_sha256(self):
            raise ValueError("active projection hash does not match its payload")
        return self


class ExecutionFindingsMutationReceipt(_Contract):
    schema_version: Literal["execution-findings-mutation-receipt-v1"] = (
        "execution-findings-mutation-receipt-v1"
    )
    ledger_id: str = Field(pattern=_ID_PATTERN)
    mutation_id: str = Field(pattern=_ID_PATTERN)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    previous_ledger_revision: int = Field(ge=0)
    applied_ledger_revision: int = Field(ge=1)
    affected_entry_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    active_projection_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_at: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _validate_receipt(self) -> 'ExecutionFindingsMutationReceipt':
        if self.applied_ledger_revision != self.previous_ledger_revision + 1:
            raise ValueError("a findings mutation must advance the ledger once")
        _require_unique_nonblank(
            self.affected_entry_ids,
            label="affected entry IDs",
        )
        return self


class ExecutionFindingsMutationResult(_Contract):
    receipt: ExecutionFindingsMutationReceipt
    ledger: ExecutionFindingsLedger
    active_projection: ExecutionFindingsActiveProjection
    replayed: bool

    @model_validator(mode="after")
    def _validate_result(self) -> 'ExecutionFindingsMutationResult':
        if (
            self.receipt.ledger_id != self.ledger.ledger_id
            or self.active_projection.ledger_id != self.ledger.ledger_id
            or self.receipt.applied_ledger_revision != self.ledger.revision
            or self.active_projection.ledger_revision != self.ledger.revision
            or self.receipt.active_projection_sha256
            != self.active_projection.projection_sha256
        ):
            raise ValueError("mutation result components disagree")
        return self


def validate_persisted_execution_findings_mutation_result_json(
    value: str,
) -> ExecutionFindingsMutationResult:
    """Decode a frozen mutation result using the shared findings contract."""

    return ExecutionFindingsMutationResult.model_validate_json(value)


def validate_persisted_execution_findings_mutation_result(
    value: object,
) -> ExecutionFindingsMutationResult:
    """Validate a materialized frozen mutation result without changing its facts."""

    return ExecutionFindingsMutationResult.model_validate(value)


class ExecutionFindingsLedgerCreateResult(_Contract):
    ledger: ExecutionFindingsLedger
    active_projection: ExecutionFindingsActiveProjection
    replayed: bool


class ExecutionFindingsSnapshot(_Contract):
    ledger: ExecutionFindingsLedger
    active_projection: ExecutionFindingsActiveProjection

    @model_validator(mode="after")
    def _validate_snapshot(self) -> 'ExecutionFindingsSnapshot':
        if (
            self.ledger.ledger_id != self.active_projection.ledger_id
            or self.ledger.revision
            != self.active_projection.ledger_revision
        ):
            raise ValueError("findings snapshot components disagree")
        return self


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_execution_findings_ledger_id(
    *,
    owner_kind: ExecutionFindingsOwnerKind,
    execution_owner_id: str,
) -> str:
    digest = sha256_json(
        {
            "contract": EXECUTION_FINDINGS_CONTRACT_VERSION,
            "owner_kind": owner_kind.value,
            "execution_owner_id": execution_owner_id,
        }
    )
    return f"efledger_{digest}"


def derive_execution_findings_mutation_id(*, writer_tool_call_id: str) -> str:
    digest = sha256_json(
        {
            "contract": EXECUTION_FINDINGS_CONTRACT_VERSION,
            "writer_tool_call_id": writer_tool_call_id,
        }
    )
    return f"efmutation_{digest}"


def derive_execution_finding_entry_id(
    *, ledger_id: str, mutation_id: str, item_ordinal: int,
) -> str:
    """派生一次 findings mutation 内第 N 个记录的唯一持久身份。"""

    return "efentry_" + sha256_json(
        {
            "ledger_id": ledger_id,
            "mutation_id": mutation_id,
            "item_ordinal": item_ordinal,
        }
    )


def execution_findings_ledger_sha256(ledger: ExecutionFindingsLedger) -> str:
    return sha256_json(
        ledger.model_dump(mode="json", exclude={"ledger_sha256"})
    )


def execution_findings_projection_sha256(
    projection: ExecutionFindingsActiveProjection,
) -> str:
    return sha256_json(
        projection.model_dump(mode="json", exclude={"projection_sha256"})
    )


def validate_execution_finding_text(value: str) -> str:
    """校验公开笔记的规范文字；不修改已持久化的原始内容。"""

    if value != value.strip() or "\x00" in value:
        raise ValueError("finding text must be trimmed non-NUL text")
    return value


def validate_execution_finding_scope_keys(values: tuple[str, ...]) -> tuple[str, ...]:
    """提案与持久条目共用范围标识规则，不自动改写或去重标识。"""

    _require_unique_nonblank(values, label="scope keys")
    return values


def _require_unique_nonblank(values: tuple[str, ...], *, label: str) -> None:
    if any(not value.strip() or len(value) > 200 for value in values):
        raise ValueError(f"{label} must be nonblank bounded identifiers")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def validate_execution_finding_source_refs(
    values: tuple[ExecutionFindingSourceRef, ...],
) -> tuple[ExecutionFindingSourceRef, ...]:
    """提案与持久条目共用来源重复检查，不拼接或修正引用。"""

    encoded = tuple(canonical_json(value) for value in values)
    if len(encoded) != len(set(encoded)):
        raise ValueError("source references must be unique")
    return values


__all__ = [
    "validate_execution_finding_text",
    "validate_execution_finding_scope_keys",
    "validate_execution_finding_source_refs",
    "EXECUTION_FINDING_CLAIM_MAX_CHARACTERS",
    "EXECUTION_FINDINGS_CONTRACT_VERSION",
    'ExecutionFindingEntry',
    'ExecutionFindingKind',
    'ExecutionFindingMutationItem',
    'ExecutionFindingRevisionOperation',
    'ExecutionFindingSourceRef',
    'ExecutionFindingStatus',
    'ExecutionFindingsActiveProjection',
    'ExecutionFindingsLedgerCreateResult',
    'ExecutionFindingsLedgerStatus',
    'ExecutionFindingsLedger',
    'ExecutionFindingsMutationCommand',
    "ExecutionFindingsMutationIdentityCollision",
    'ExecutionFindingsMutationReceipt',
    'ExecutionFindingsMutationResult',
    'ExecutionFindingsOwnerKind',
    "ExecutionFindingsOwnerClosed",
    "ExecutionFindingsPersistenceError",
    'ExecutionFindingsQuota',
    "ExecutionFindingsQuotaExceeded",
    "ExecutionFindingsRevisionConflict",
    "ExecutionFindingsScopeInvalid",
    'ExecutionFindingsSnapshot',
    "ExecutionFindingsSourceReferenceInvalid",
    "ExecutionFindingsStoredAuthorityCorrupt",
    'RecordExecutionFinding',
    'RetractExecutionFinding',
    'SupersedeExecutionFinding',
    "canonical_json",
    "derive_execution_finding_entry_id",
    "derive_execution_findings_ledger_id",
    "derive_execution_findings_mutation_id",
    "execution_findings_ledger_sha256",
    "execution_findings_projection_sha256",
    "reduce_execution_findings_active_queue",
    "sha256_json",
    "validate_persisted_execution_finding_entry",
    "validate_persisted_execution_findings_mutation_result",
    "validate_persisted_execution_findings_mutation_result_json",
    "validate_persisted_execution_findings_quota_json",
]
