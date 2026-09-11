"""一个 Runtime ToolCall 的崩溃安全、provider 中立权威状态。

账本会区分模型可见的逻辑调用与每一次实际分派。本模块负责 Host 侧的状态转换
规则；它有意不执行工具，也不定义持久化 schema。

adapter 必须将本地、MCP 或远程 provider 响应封装为
``RuntimeToolDispatchObservation``。该 observation 会先把响应绑定到精确的
provider、WorkRun/Attempt 主体、逻辑调用和物理分派，之后它才能成为持久化结算。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError

from .contracts import (
    RuntimeToolCallAuthorityError,
    RuntimeToolDispatchObservation,
    RuntimeToolLedgerStore,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
    RuntimeToolPhysicalAttemptSettlement,
    RuntimeToolPhysicalOutcome,
    RuntimeToolRetryAuthority,
    RuntimeToolTypedResult,
)


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SHA256 = re.compile(_SHA256_PATTERN)


class RuntimeToolCallStateGuardRejected(RuntimeToolCallAuthorityError):
    """catalog、主体、策略或资源状态在分派前发生了变化。"""


class RuntimeToolCallWaitingExternal(RuntimeToolCallAuthorityError):
    """处于 pending 或 uncertain 状态的分派需要显式对账。"""


class RuntimeToolCallTerminalState(RuntimeToolCallAuthorityError):
    """逻辑 ToolCall 已有终态结果，或没有重试授权。"""


@dataclass(frozen=True, slots=True)
class RuntimeToolCallReplay:
    """无需再次分派工具即可重建的精确 typed success。"""

    typed_result: RuntimeToolTypedResult
    physical_attempt_id: str
    physical_ordinal: int

    def __post_init__(self) -> None:
        validated = _revalidate_contract(
            self.typed_result,
            RuntimeToolTypedResult,
            label="typed tool replay",
        )
        object.__setattr__(self, "typed_result", validated)
        if not self.physical_attempt_id:
            raise ValueError("tool replay requires its physical attempt identity")
        if self.physical_ordinal < 1:
            raise ValueError("physical_ordinal must be positive")

    @property
    def value(self) -> Any:
        return self.typed_result.parsed()


@dataclass(frozen=True, slots=True)
class _ValidatedStoredPhysical:
    request: RuntimeToolPhysicalAttemptRequest
    settlement: RuntimeToolPhysicalAttemptSettlement | None = None


@dataclass(frozen=True, slots=True)
class _ValidatedStoredLogical:
    request: RuntimeToolLogicalRequest
    physical_attempts: tuple[_ValidatedStoredPhysical, ...] = ()


StateGuardSha256 = Callable[[], str]


@dataclass(frozen=True, slots=True)
class RuntimeLogicalToolCallAuthority:
    """一个精确逻辑 ToolCall 及其物理尝试的 Host 权威状态。"""

    logical_request: RuntimeToolLogicalRequest
    state_guard_sha256: StateGuardSha256 = field(repr=False, compare=False)
    store: RuntimeToolLedgerStore = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        validated = _revalidate_contract(
            self.logical_request,
            RuntimeToolLogicalRequest,
            label="logical tool request",
        )
        object.__setattr__(self, "logical_request", validated)
        if not callable(self.state_guard_sha256):
            raise TypeError("state_guard_sha256 must be callable")

    @property
    def semantic_call_id(self) -> str:
        return self.logical_request.logical_tool_call_id

    def require_current_state(self) -> None:
        try:
            current = self.state_guard_sha256()
        except RuntimeToolCallStateGuardRejected:
            raise
        except Exception as exc:
            raise RuntimeToolCallStateGuardRejected(
                "tool-call state guard could not rederive current authority"
            ) from exc
        if (
            not isinstance(current, str)
            or not _SHA256.fullmatch(current)
            or current != self.logical_request.state_guard_sha256
        ):
            raise RuntimeToolCallStateGuardRejected(
                "tool-call state authority no longer matches its frozen request"
            )

    def reserve(self, *, turn_id: str) -> object:
        _require_id(turn_id, "turn_id")
        self.require_current_state()
        result = self.store.reserve_runtime_tool_logical_call(
            request=self.logical_request
        )
        self._stored()
        return result

    def replay_succeeded_result(self) -> RuntimeToolCallReplay | None:
        stored = self._stored()
        attempts = tuple(stored.physical_attempts)
        if not attempts:
            return None
        last = attempts[-1]
        settlement = last.settlement
        if settlement is None:
            raise RuntimeToolCallWaitingExternal(
                "pending tool dispatch requires external reconciliation"
            )
        outcome = settlement.outcome
        if outcome is RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE:
            if _retry_is_authorized(self.logical_request.retry_authority):
                return None
            raise RuntimeToolCallTerminalState(
                "logical tool call has a retryable failure but no retry authority"
            )
        if outcome is RuntimeToolPhysicalOutcome.UNCERTAIN:
            raise RuntimeToolCallWaitingExternal(
                "uncertain tool outcome requires external reconciliation"
            )
        if outcome is RuntimeToolPhysicalOutcome.TERMINAL_FAILURE:
            raise RuntimeToolCallTerminalState(
                "logical tool call already has a terminal failure"
            )
        typed = settlement.typed_result
        if typed is None:
            raise RuntimeToolCallAuthorityError(
                "successful tool receipt has no replayable typed result"
            )
        return RuntimeToolCallReplay(
            typed_result=typed,
            physical_attempt_id=last.request.physical_attempt_id,
            physical_ordinal=last.request.physical_ordinal,
        )

    def begin_physical_attempt(
        self,
        *,
        turn_id: str,
        max_physical_attempts: int,
    ) -> RuntimeToolPhysicalAttemptRequest:
        _require_id(turn_id, "turn_id")
        self.require_current_state()
        if (
            isinstance(max_physical_attempts, bool)
            or not 1 <= max_physical_attempts <= self.logical_request.max_physical_attempts
        ):
            raise ValueError(
                "caller physical-attempt bound exceeds logical tool authority"
            )
        stored = self._stored()
        attempts = tuple(stored.physical_attempts)
        if attempts:
            last_settlement = attempts[-1].settlement
            if last_settlement is None:
                raise RuntimeToolCallWaitingExternal(
                    "pending tool dispatch requires external reconciliation"
                )
            if (
                last_settlement.outcome
                is not RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE
            ):
                if (
                    last_settlement.outcome
                    is RuntimeToolPhysicalOutcome.UNCERTAIN
                ):
                    raise RuntimeToolCallWaitingExternal(
                        "uncertain tool outcome requires external reconciliation"
                    )
                raise RuntimeToolCallTerminalState(
                    "logical tool call already has a terminal physical outcome"
                )
            if not _retry_is_authorized(self.logical_request.retry_authority):
                raise RuntimeToolCallTerminalState(
                    "logical tool call retry is not authorized by its frozen policy"
                )
        ordinal = len(attempts) + 1
        if ordinal > max_physical_attempts:
            raise RuntimeToolCallTerminalState(
                "logical tool call physical-attempt limit is exhausted"
            )
        physical = self._make_physical(turn_id=turn_id, ordinal=ordinal)
        self.store.append_runtime_tool_physical_attempt(request=physical)
        refreshed = self._stored()
        refreshed_attempts = tuple(refreshed.physical_attempts)
        if not refreshed_attempts or refreshed_attempts[-1].request != physical:
            raise RuntimeToolCallAuthorityError(
                "tool Store returned a cross-bound physical attempt"
            )
        return physical

    def settle_physical_attempt(
        self,
        *,
        turn_id: str,
        physical: RuntimeToolPhysicalAttemptRequest,
        observation: RuntimeToolDispatchObservation,
    ) -> RuntimeToolPhysicalAttemptSettlement:
        """持久化精确的分派事实，包括精确的幂等重放。

        此处有意不重新检查状态 guard：Provider I/O 已经发生，因此即使另一状态
        版本与响应形成竞态，也必须持久地闭合这一物理事实。负责该流程的 driver
        可以在调用此方法之前，将这种竞态映射为终态 observation。
        """

        _require_id(turn_id, "turn_id")
        physical = _revalidate_contract(
            physical,
            RuntimeToolPhysicalAttemptRequest,
            label="physical tool request",
        )
        observation = _revalidate_contract(
            observation,
            RuntimeToolDispatchObservation,
            label="tool dispatch observation",
        )
        self._require_observation_identity(
            physical=physical,
            observation=observation,
        )
        stored = self._stored()
        attempts = tuple(stored.physical_attempts)
        if not attempts or attempts[-1].request != physical:
            raise RuntimeToolCallAuthorityError(
                "physical tool settlement is not the current logical dispatch"
            )
        settlement = self._make_settlement(
            turn_id=turn_id,
            physical=physical,
            observation=observation,
        )
        existing = attempts[-1].settlement
        if existing is not None:
            if existing == settlement:
                return existing
            raise RuntimeToolCallAuthorityError(
                "physical tool dispatch already has a different settlement"
            )
        self.store.settle_runtime_tool_physical_attempt(settlement=settlement)
        refreshed = self._stored()
        persisted = tuple(refreshed.physical_attempts)[-1].settlement
        if persisted != settlement:
            raise RuntimeToolCallAuthorityError(
                "tool Store returned a cross-bound physical settlement"
            )
        return settlement

    def _make_physical(
        self,
        *,
        turn_id: str,
        ordinal: int,
    ) -> RuntimeToolPhysicalAttemptRequest:
        identity_hash = _physical_identity_hash(
            logical_request=self.logical_request,
            ordinal=ordinal,
        )
        provider_key = _provider_idempotency_key(self.logical_request)
        dispatch_sha256 = _dispatch_authority_sha256(
            logical_request=self.logical_request,
            ordinal=ordinal,
            turn_id=turn_id,
            provider_idempotency_key=provider_key,
        )
        return RuntimeToolPhysicalAttemptRequest.create(
            physical_attempt_id=f"rtpa_v1_{identity_hash}",
            physical_attempt_key=f"rtcall_v1:{identity_hash}:{ordinal}",
            logical_tool_call_id=self.semantic_call_id,
            logical_request_binding_sha256=self.logical_request.binding_sha256,
            physical_ordinal=ordinal,
            started_turn_id=turn_id,
            retry_authority=self.logical_request.retry_authority,
            provider_idempotency_key=provider_key,
            dispatch_authority_sha256=dispatch_sha256,
        )

    def _make_settlement(
        self,
        *,
        turn_id: str,
        physical: RuntimeToolPhysicalAttemptRequest,
        observation: RuntimeToolDispatchObservation,
    ) -> RuntimeToolPhysicalAttemptSettlement:
        typed_result = None
        if observation.outcome is RuntimeToolPhysicalOutcome.SUCCEEDED:
            typed_result = RuntimeToolTypedResult.create(
                result_contract=self.logical_request.result_contract,
                result=observation.parsed_result(),
            )
        settlement_identity = _fingerprint(
            {
                "contract": "runtime-tool-settlement-id-v1",
                "physical_request_binding_sha256": physical.binding_sha256,
                "outcome_fingerprint": observation.binding_sha256,
            }
        )
        return RuntimeToolPhysicalAttemptSettlement.create(
            settlement_id=f"rtps_v1_{settlement_identity}",
            settle_apply_id=(
                "rtps_apply_v1_"
                + _fingerprint(
                    {
                        "contract": "runtime-tool-settle-apply-id-v1",
                        "physical_request_binding_sha256": physical.binding_sha256,
                    }
                )
            ),
            physical_attempt_id=physical.physical_attempt_id,
            physical_request_binding_sha256=physical.binding_sha256,
            logical_tool_call_id=self.semantic_call_id,
            physical_ordinal=physical.physical_ordinal,
            settled_turn_id=turn_id,
            outcome=observation.outcome,
            provider_request_id=observation.provider_request_id,
            error_code=observation.error_code,
            duration_ms=observation.duration_ms,
            typed_result=typed_result,
            outcome_fingerprint=observation.binding_sha256,
        )

    def _require_observation_identity(
        self,
        *,
        physical: RuntimeToolPhysicalAttemptRequest,
        observation: RuntimeToolDispatchObservation,
    ) -> None:
        logical = self.logical_request
        expected = {
            "session_id": logical.session_id,
            "work_run_id": logical.work_run_id,
            "attempt_id": logical.attempt_id,
            "logical_tool_call_id": logical.logical_tool_call_id,
            "logical_request_binding_sha256": logical.binding_sha256,
            "physical_attempt_id": physical.physical_attempt_id,
            "physical_request_binding_sha256": physical.binding_sha256,
            "physical_ordinal": physical.physical_ordinal,
            "provider_identity_sha256": logical.provider_identity_sha256,
            "tool_id": logical.tool_id,
            "contract_version": logical.contract_version,
            "implementation_version": logical.implementation_version,
        }
        actual = observation.model_dump(mode="json", include=set(expected))
        if actual != expected:
            raise RuntimeToolCallAuthorityError(
                "dispatch observation crossed provider, subject, or call authority"
            )

    def _stored(self) -> _ValidatedStoredLogical:
        stored = self.store.get_runtime_tool_logical_call(
            session_id=self.logical_request.session_id,
            logical_tool_call_id=self.semantic_call_id,
        )
        if stored is None:
            raise RuntimeToolCallAuthorityError(
                "logical tool call has not been durably reserved"
            )
        try:
            request_value = stored.request
            attempts = tuple(stored.physical_attempts)
        except (AttributeError, TypeError) as exc:
            raise RuntimeToolCallAuthorityError(
                "tool Store returned an invalid logical-call projection"
            ) from exc
        request = _revalidate_contract(
            request_value,
            RuntimeToolLogicalRequest,
            label="stored logical tool request",
        )
        if request != self.logical_request:
            raise RuntimeToolCallAuthorityError(
                "stored logical tool request crossed immutable authority"
            )
        if len(attempts) > request.max_physical_attempts:
            raise RuntimeToolCallAuthorityError(
                "stored tool attempts exceed logical request authority"
            )
        physical_ids: set[str] = set()
        physical_keys: set[str] = set()
        validated_attempts: list[_ValidatedStoredPhysical] = []
        for ordinal, stored_attempt in enumerate(attempts, start=1):
            try:
                physical_value = stored_attempt.request
                settlement_value = stored_attempt.settlement
            except AttributeError as exc:
                raise RuntimeToolCallAuthorityError(
                    "tool Store returned an invalid physical-attempt projection"
                ) from exc
            physical = _revalidate_contract(
                physical_value,
                RuntimeToolPhysicalAttemptRequest,
                label="stored physical tool request",
            )
            self._validate_stored_physical(physical=physical, ordinal=ordinal)
            if (
                physical.physical_attempt_id in physical_ids
                or physical.physical_attempt_key in physical_keys
            ):
                raise RuntimeToolCallAuthorityError(
                    "stored tool ledger repeats a physical identity"
                )
            physical_ids.add(physical.physical_attempt_id)
            physical_keys.add(physical.physical_attempt_key)
            settlement = None
            if settlement_value is not None:
                settlement = _revalidate_contract(
                    settlement_value,
                    RuntimeToolPhysicalAttemptSettlement,
                    label="stored physical tool settlement",
                )
                self._validate_stored_settlement(
                    physical=physical,
                    settlement=settlement,
                )
            if ordinal < len(attempts):
                if settlement is None:
                    raise RuntimeToolCallAuthorityError(
                        "a pending tool dispatch cannot have a successor"
                    )
                if (
                    settlement.outcome
                    is not RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE
                    or not _retry_is_authorized(request.retry_authority)
                ):
                    raise RuntimeToolCallAuthorityError(
                        "a terminal or unauthorized tool outcome has a successor"
                    )
            validated_attempts.append(
                _ValidatedStoredPhysical(
                    request=physical,
                    settlement=settlement,
                )
            )
        return _ValidatedStoredLogical(
            request=request,
            physical_attempts=tuple(validated_attempts),
        )

    def _validate_stored_physical(
        self,
        *,
        physical: RuntimeToolPhysicalAttemptRequest,
        ordinal: int,
    ) -> None:
        logical = self.logical_request
        identity_hash = _physical_identity_hash(
            logical_request=logical,
            ordinal=ordinal,
        )
        provider_key = _provider_idempotency_key(logical)
        if (
            physical.physical_attempt_id != f"rtpa_v1_{identity_hash}"
            or physical.physical_attempt_key
            != f"rtcall_v1:{identity_hash}:{ordinal}"
            or physical.logical_tool_call_id != logical.logical_tool_call_id
            or physical.logical_request_binding_sha256 != logical.binding_sha256
            or physical.physical_ordinal != ordinal
            or physical.retry_authority is not logical.retry_authority
            or physical.provider_idempotency_key != provider_key
            or physical.dispatch_authority_sha256
            != _dispatch_authority_sha256(
                logical_request=logical,
                ordinal=ordinal,
                turn_id=physical.started_turn_id,
                provider_idempotency_key=provider_key,
            )
        ):
            raise RuntimeToolCallAuthorityError(
                "stored physical tool request crossed immutable authority"
            )

    def _validate_stored_settlement(
        self,
        *,
        physical: RuntimeToolPhysicalAttemptRequest,
        settlement: RuntimeToolPhysicalAttemptSettlement,
    ) -> None:
        settlement_identity = _fingerprint(
            {
                "contract": "runtime-tool-settlement-id-v1",
                "physical_request_binding_sha256": physical.binding_sha256,
                "outcome_fingerprint": settlement.outcome_fingerprint,
            }
        )
        expected_apply_id = "rtps_apply_v1_" + _fingerprint(
            {
                "contract": "runtime-tool-settle-apply-id-v1",
                "physical_request_binding_sha256": physical.binding_sha256,
            }
        )
        typed = settlement.typed_result
        if (
            settlement.settlement_id != f"rtps_v1_{settlement_identity}"
            or settlement.settle_apply_id != expected_apply_id
            or settlement.physical_attempt_id != physical.physical_attempt_id
            or settlement.physical_request_binding_sha256 != physical.binding_sha256
            or settlement.logical_tool_call_id != self.semantic_call_id
            or settlement.physical_ordinal != physical.physical_ordinal
            or (
                typed is not None
                and typed.result_contract != self.logical_request.result_contract
            )
        ):
            raise RuntimeToolCallAuthorityError(
                "stored physical tool settlement crossed immutable authority"
            )


def _retry_is_authorized(authority: RuntimeToolRetryAuthority) -> bool:
    return authority in {
        RuntimeToolRetryAuthority.READ_ONLY_REPLAY,
        RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY,
    }


def _provider_idempotency_key(
    logical_request: RuntimeToolLogicalRequest,
) -> str | None:
    if (
        logical_request.retry_authority
        is not RuntimeToolRetryAuthority.PROVIDER_IDEMPOTENCY
    ):
        return None
    return "rtidem_v1_" + _fingerprint(
        {
            "contract": "runtime-tool-provider-idempotency-key-v1",
            "logical_request_binding_sha256": logical_request.binding_sha256,
            "provider_identity_sha256": logical_request.provider_identity_sha256,
        }
    )


def _physical_identity_hash(
    *,
    logical_request: RuntimeToolLogicalRequest,
    ordinal: int,
) -> str:
    return _fingerprint(
        {
            "contract": "runtime-tool-physical-attempt-id-v1",
            "logical_tool_call_id": logical_request.logical_tool_call_id,
            "logical_request_binding_sha256": logical_request.binding_sha256,
            "physical_ordinal": ordinal,
        }
    )


def _dispatch_authority_sha256(
    *,
    logical_request: RuntimeToolLogicalRequest,
    ordinal: int,
    turn_id: str,
    provider_idempotency_key: str | None,
) -> str:
    return _fingerprint(
        {
            "contract": "runtime-tool-dispatch-authority-v1",
            "logical_request_binding_sha256": logical_request.binding_sha256,
            "physical_ordinal": ordinal,
            "started_turn_id": turn_id,
            "retry_authority": logical_request.retry_authority.value,
            "provider_idempotency_key": provider_idempotency_key,
            "state_guard_sha256": logical_request.state_guard_sha256,
        }
    )


ContractT = TypeVar("ContractT", bound=BaseModel)


def _revalidate_contract(
    value: object,
    contract: type[ContractT],
    *,
    label: str,
) -> ContractT:
    if not isinstance(value, contract):
        raise RuntimeToolCallAuthorityError(f"{label} has the wrong contract type")
    try:
        payload = value.model_dump(mode="json", warnings="error")
        return contract.model_validate(payload)
    except (
        PydanticSerializationError,
        ValidationError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeToolCallAuthorityError(
            f"{label} failed Host self-hash validation"
        ) from exc


def _require_id(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(_ID_PATTERN, value) is None:
        raise ValueError(f"{label} must be a valid durable identifier")


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


def _fingerprint(value: object) -> str:
    return _sha256_text(_canonical_json(value))


__all__ = [
    'RuntimeLogicalToolCallAuthority',
    'RuntimeToolCallReplay',
    "RuntimeToolCallStateGuardRejected",
    "RuntimeToolCallTerminalState",
    "RuntimeToolCallWaitingExternal",
]
