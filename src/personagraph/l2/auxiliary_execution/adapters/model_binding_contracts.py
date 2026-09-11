"""AuxiliaryGraph WorkRuns 的规范不可变模型调用绑定。

这里的值冻结了提供者中立的提示和权威事实。它们不读取 前沿、选择能力、访问模型账本、调用提供者或结算 WorkRun。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.auxiliary_graph import AuxiliaryNodeExecutorKind
from personagraph.l2.work_run import AuxiliaryNodeSubject
from personagraph.runtime.model_calls.policy import MAX_MODEL_ATTEMPTS


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
AUXILIARY_WORK_RUN_ATTEMPT_RESULT_CONTRACT = (
    "auxiliary-v2-attempt-decision-model-envelope-v1"
)
AUXILIARY_WORK_RUN_VERIFICATION_RESULT_CONTRACT = (
    "auxiliary-v2-node-verification-model-envelope-v1"
)


class _BindingContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class AuxiliaryBoundModelCall(_BindingContract):
    """一个完整的 Host 绑定，用于持久化的 WorkRun 模型调用。

    逻辑模型账本拥有 Provider 分发/重放，但它无法推断出哪个 目标、执行主题、WorkRun 游标或确切的提示授权了一个调用。此合同在工厂看到之前冻结了所有这些权威状态。``request_json`` 是完整的提供者中立请求，两个哈希确保了复制/更新篡改的即时验证。
    """

    schema_version: Literal["auxiliary-v2-bound-model-call-v1"] = (
        "auxiliary-v2-bound-model-call-v1"
    )
    call_kind: Literal["attempt_decision", "node_verification"]
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    semantic_unit_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    execution_subject_id: str = Field(pattern=_ID_PATTERN)
    subject: AuxiliaryNodeSubject
    executor_kind: Literal[
        AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
        AuxiliaryNodeExecutorKind.USER_GATE,
        AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
    ]
    request_turn_id: str = Field(pattern=_ID_PATTERN)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    work_run_id: str = Field(pattern=_ID_PATTERN)
    work_run_revision: int = Field(ge=1)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    attempt_ordinal: int = Field(ge=1)
    verification_request_id: str | None = Field(
        default=None,
        pattern=_ID_PATTERN,
    )
    verification_request_revision: int | None = Field(default=None, ge=1)
    purpose: str = Field(min_length=1, max_length=200)
    system_prompt: str = Field(min_length=1)
    user_content: str = Field(min_length=1)
    request_contract: str = Field(pattern=_ID_PATTERN)
    request_json: str = Field(min_length=1)
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    typed_result_contract: str = Field(pattern=_ID_PATTERN)
    max_physical_attempts: Literal[MAX_MODEL_ATTEMPTS] = MAX_MODEL_ATTEMPTS
    state_guard_sha256: str = Field(pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        call_kind: Literal["attempt_decision", "node_verification"],
        logical_call_id: str,
        session_id: str,
        goal_id: str,
        subject: AuxiliaryNodeSubject,
        executor_kind: Literal[
            AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
            AuxiliaryNodeExecutorKind.USER_GATE,
            AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
        ],
        request_turn_id: str,
        invocation_turn_id: str,
        work_run_id: str,
        work_run_revision: int,
        attempt_id: str,
        attempt_ordinal: int,
        verification_request_id: str | None,
        verification_request_revision: int | None,
        system_prompt: str,
        user_content: str,
        state_guard_sha256: str,
    ) -> "AuxiliaryBoundModelCall":
        purpose, request_contract, typed_result_contract = (
            _work_run_model_call_contracts(call_kind)
        )
        semantic_unit_id = (
            attempt_id
            if call_kind == "attempt_decision"
            else verification_request_id
        )
        if semantic_unit_id is None:
            raise ValueError("verification binding requires its request identity")
        execution_subject_id = _auxiliary_execution_subject_id(
            session_id=session_id,
            goal_id=goal_id,
            subject=subject,
        )
        user_payload = _require_canonical_json_object(
            user_content,
            label="WorkRun model user content",
        )
        values: dict[str, object] = {
            "call_kind": call_kind,
            "logical_call_id": logical_call_id,
            "semantic_unit_id": semantic_unit_id,
            "session_id": session_id,
            "task_id": subject.task_id,
            "auxiliary_graph_id": subject.auxiliary_graph_id,
            "goal_id": goal_id,
            "auxiliary_graph_revision": subject.auxiliary_graph_revision,
            "execution_subject_id": execution_subject_id,
            "subject": subject,
            "executor_kind": executor_kind,
            "request_turn_id": request_turn_id,
            "invocation_turn_id": invocation_turn_id,
            "work_run_id": work_run_id,
            "work_run_revision": work_run_revision,
            "attempt_id": attempt_id,
            "attempt_ordinal": attempt_ordinal,
            "verification_request_id": verification_request_id,
            "verification_request_revision": verification_request_revision,
            "purpose": purpose,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "request_contract": request_contract,
            "request_json": "pending",
            "request_sha256": "0" * 64,
            "typed_result_contract": typed_result_contract,
            "max_physical_attempts": MAX_MODEL_ATTEMPTS,
            "state_guard_sha256": state_guard_sha256,
            "binding_sha256": "0" * 64,
        }
        provisional = cls.model_construct(**values)
        request_json = _canonical_json_text(
            _bound_model_request_payload(
                provisional,
                user_payload=user_payload,
            )
        )
        values["request_json"] = request_json
        values["request_sha256"] = hashlib.sha256(
            request_json.encode("utf-8")
        ).hexdigest()
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _bound_model_call_binding_sha256(provisional)
        return cls.model_validate(values)

    @model_validator(mode="after")
    def _validate_complete_binding(self) -> "AuxiliaryBoundModelCall":
        purpose, request_contract, typed_result_contract = (
            _work_run_model_call_contracts(self.call_kind)
        )
        if (
            self.purpose != purpose
            or self.request_contract != request_contract
            or self.typed_result_contract != typed_result_contract
        ):
            raise ValueError("WorkRun model-call contracts crossed call kind")
        if (
            self.task_id != self.subject.task_id
            or self.auxiliary_graph_id != self.subject.auxiliary_graph_id
            or self.auxiliary_graph_revision
            != self.subject.auxiliary_graph_revision
        ):
            raise ValueError("WorkRun model binding crossed its subject")
        if self.execution_subject_id != _auxiliary_execution_subject_id(
            session_id=self.session_id,
            goal_id=self.goal_id,
            subject=self.subject,
        ):
            raise ValueError("WorkRun execution-subject identity is invalid")
        verification = self.call_kind == "node_verification"
        if verification != (
            self.verification_request_id is not None
            and self.verification_request_revision is not None
        ):
            raise ValueError("only verification calls bind a verification request")
        expected_semantic_unit = (
            self.verification_request_id if verification else self.attempt_id
        )
        if self.semantic_unit_id != expected_semantic_unit:
            raise ValueError("WorkRun semantic unit crossed call kind")

        user_payload = _require_canonical_json_object(
            self.user_content,
            label="WorkRun model user content",
        )
        _require_bound_prompt_identity(self, user_payload=user_payload)
        expected_request_json = _canonical_json_text(
            _bound_model_request_payload(self, user_payload=user_payload)
        )
        if (
            self.request_json != expected_request_json
            or self.request_sha256
            != hashlib.sha256(expected_request_json.encode("utf-8")).hexdigest()
        ):
            raise ValueError("WorkRun model request is not exact and canonical")
        if self.binding_sha256 != _bound_model_call_binding_sha256(self):
            raise ValueError("WorkRun model binding hash does not match")
        return self


def _work_run_model_call_contracts(
    call_kind: Literal["attempt_decision", "node_verification"],
) -> tuple[str, str, str]:
    if call_kind == "attempt_decision":
        return (
            "runtime_auxiliary_v2_attempt_decision",
            "auxiliary-v2-attempt-decision-model-request-v1",
            AUXILIARY_WORK_RUN_ATTEMPT_RESULT_CONTRACT,
        )
    if call_kind == "node_verification":
        return (
            "runtime_auxiliary_v2_node_verification",
            "auxiliary-v2-node-verification-model-request-v1",
            AUXILIARY_WORK_RUN_VERIFICATION_RESULT_CONTRACT,
        )
    raise ValueError("unsupported WorkRun model-call kind")


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_text(payload).encode("utf-8")).hexdigest()


def _require_canonical_json_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict) or _canonical_json_text(parsed) != value:
        raise ValueError(f"{label} must be one canonical JSON object")
    return parsed


def _auxiliary_execution_subject_id(
    *,
    session_id: str,
    goal_id: str,
    subject: AuxiliaryNodeSubject,
) -> str:
    """镜像不可变的 注册表身份，不读取私有行。"""

    payload = _canonical_json_text(
        [
            "execsubject",
            "auxiliary_node_v2",
            session_id,
            subject.task_id,
            subject.auxiliary_graph_id,
            goal_id,
            subject.auxiliary_graph_revision,
            subject.node_id,
            subject.node_revision,
        ]
    )
    return f"execsubject-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _bound_model_call_binding_sha256(
    binding: AuxiliaryBoundModelCall,
) -> str:
    return _canonical_sha256(
        binding.model_dump(mode="json", exclude={"binding_sha256"})
    )


def _bound_model_request_payload(
    binding: AuxiliaryBoundModelCall,
    *,
    user_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": binding.request_contract,
        "authority": {
            "authority_contract": binding.schema_version,
            "call_kind": binding.call_kind,
            "logical_call_id": binding.logical_call_id,
            "semantic_unit_id": binding.semantic_unit_id,
            "session_id": binding.session_id,
            "task_id": binding.task_id,
            "auxiliary_graph_id": binding.auxiliary_graph_id,
            "goal_id": binding.goal_id,
            "auxiliary_graph_revision": binding.auxiliary_graph_revision,
            "execution_subject_id": binding.execution_subject_id,
            "subject": binding.subject.model_dump(mode="json"),
            "executor_kind": binding.executor_kind.value,
            "request_turn_id": binding.request_turn_id,
            "invocation_turn_id": binding.invocation_turn_id,
            "work_run_id": binding.work_run_id,
            "work_run_revision": binding.work_run_revision,
            "attempt_id": binding.attempt_id,
            "attempt_ordinal": binding.attempt_ordinal,
            "verification_request_id": binding.verification_request_id,
            "verification_request_revision": (
                binding.verification_request_revision
            ),
            "state_guard_sha256": binding.state_guard_sha256,
        },
        "system_prompt": binding.system_prompt,
        "user_content": user_payload,
    }


def _require_bound_prompt_identity(
    binding: AuxiliaryBoundModelCall,
    *,
    user_payload: dict[str, Any],
) -> None:
    raw = user_payload.get("bindings")
    if not isinstance(raw, dict):
        raise ValueError("WorkRun model prompt omitted its bindings")
    common = {
        "session_id": binding.session_id,
        "work_run_id": binding.work_run_id,
        "subject": binding.subject.model_dump(mode="json"),
    }
    if any(raw.get(key) != value for key, value in common.items()):
        raise ValueError("WorkRun model prompt crossed its common authority")
    if binding.call_kind == "attempt_decision":
        expected = {
            "turn_id": binding.invocation_turn_id,
            "work_run_revision": binding.work_run_revision,
            "attempt_id": binding.attempt_id,
            "attempt_ordinal": binding.attempt_ordinal,
        }
    else:
        expected = {
            "request_turn_id": binding.request_turn_id,
            "verification_request_id": binding.verification_request_id,
            "locked_work_run_revision": binding.work_run_revision,
            "submitted_attempt_id": binding.attempt_id,
        }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("WorkRun model prompt crossed its call authority")


__all__ = [
    "AUXILIARY_WORK_RUN_ATTEMPT_RESULT_CONTRACT",
    "AUXILIARY_WORK_RUN_VERIFICATION_RESULT_CONTRACT",
    "AuxiliaryBoundModelCall",
]
