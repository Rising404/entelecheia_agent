"""普通 TaskNode 模型调用的规范不可变绑定。

此处的值会冻结一个提供商中立的语义请求，并验证其提示词权威。它们不选择端点、不查询模型账本、
不构造持久权威、不调用提供商，也不访问会话状态。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.work_run import TaskNodeSubject
from personagraph.runtime.model_calls.policy import MAX_MODEL_ATTEMPTS


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
TASK_NODE_ATTEMPT_RESULT_CONTRACT = (
    "task-node-attempt-decision-model-envelope-v1"
)
TASK_NODE_VERIFICATION_RESULT_CONTRACT = (
    "task-node-verification-model-envelope-v1"
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskNodeBoundModelCall(_Contract):
    """一次普通 TaskNode 语义模型调用的完整 Host 绑定。"""

    schema_version: Literal["task-node-bound-model-call-v1"] = (
        "task-node-bound-model-call-v1"
    )
    call_kind: Literal["attempt_decision", "node_verification"]
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    semantic_unit_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    execution_subject_id: str = Field(pattern=_ID_PATTERN)
    subject: TaskNodeSubject
    request_turn_id: str = Field(pattern=_ID_PATTERN)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    work_run_id: str = Field(pattern=_ID_PATTERN)
    dispatch_work_run_revision: int = Field(ge=1)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    attempt_ordinal: int = Field(ge=1)
    verification_request_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    verification_request_revision: int | None = Field(default=None, ge=1)
    locked_work_run_revision: int | None = Field(default=None, ge=1)
    purpose: str = Field(pattern=_ID_PATTERN)
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
        subject: TaskNodeSubject,
        request_turn_id: str,
        invocation_turn_id: str,
        work_run_id: str,
        dispatch_work_run_revision: int,
        attempt_id: str,
        attempt_ordinal: int,
        verification_request_id: str | None,
        verification_request_revision: int | None,
        locked_work_run_revision: int | None,
        system_prompt: str,
        user_content: str,
        state_guard_sha256: str,
    ) -> 'TaskNodeBoundModelCall':
        purpose, request_contract, typed_result_contract = _call_contracts(
            call_kind
        )
        semantic_unit_id = (
            attempt_id
            if call_kind == "attempt_decision"
            else verification_request_id
        )
        if semantic_unit_id is None:
            raise ValueError("verification binding requires its request identity")
        user_payload = _require_canonical_json_object(
            user_content,
            label="TaskNode model user content",
        )
        values: dict[str, object] = {
            "call_kind": call_kind,
            "logical_call_id": logical_call_id,
            "semantic_unit_id": semantic_unit_id,
            "session_id": session_id,
            "task_id": subject.task_id,
            "execution_subject_id": _task_node_execution_subject_id(
                session_id=session_id,
                subject=subject,
            ),
            "subject": subject,
            "request_turn_id": request_turn_id,
            "invocation_turn_id": invocation_turn_id,
            "work_run_id": work_run_id,
            "dispatch_work_run_revision": dispatch_work_run_revision,
            "attempt_id": attempt_id,
            "attempt_ordinal": attempt_ordinal,
            "verification_request_id": verification_request_id,
            "verification_request_revision": verification_request_revision,
            "locked_work_run_revision": locked_work_run_revision,
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
        request_json = _canonical_json(
            _bound_request_payload(provisional, user_payload=user_payload)
        )
        values["request_json"] = request_json
        values["request_sha256"] = _sha256_text(request_json)
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _binding_sha256(provisional)
        return cls.model_validate(values)

    @model_validator(mode="after")
    def _validate_complete_binding(self) -> 'TaskNodeBoundModelCall':
        purpose, request_contract, typed_result_contract = _call_contracts(
            self.call_kind
        )
        if (
            self.purpose != purpose
            or self.request_contract != request_contract
            or self.typed_result_contract != typed_result_contract
        ):
            raise ValueError("TaskNode model-call contracts crossed call kind")
        if self.task_id != self.subject.task_id:
            raise ValueError("TaskNode model binding crossed its Task owner")
        if self.execution_subject_id != _task_node_execution_subject_id(
            session_id=self.session_id,
            subject=self.subject,
        ):
            raise ValueError("TaskNode execution-subject identity is invalid")
        verification = self.call_kind == "node_verification"
        if verification != (
            self.verification_request_id is not None
            and self.verification_request_revision is not None
            and self.locked_work_run_revision is not None
        ):
            raise ValueError(
                "only verification calls bind verification and locked revisions"
            )
        expected_semantic_unit = (
            self.verification_request_id if verification else self.attempt_id
        )
        if self.semantic_unit_id != expected_semantic_unit:
            raise ValueError("TaskNode semantic unit crossed call kind")
        user_payload = _require_canonical_json_object(
            self.user_content,
            label="TaskNode model user content",
        )
        _require_prompt_identity(self, user_payload=user_payload)
        expected_request_json = _canonical_json(
            _bound_request_payload(self, user_payload=user_payload)
        )
        if (
            self.request_json != expected_request_json
            or self.request_sha256 != _sha256_text(expected_request_json)
        ):
            raise ValueError("TaskNode model request is not exact and canonical")
        if self.binding_sha256 != _binding_sha256(self):
            raise ValueError("TaskNode model binding hash does not match")
        return self


def task_node_model_state_guard_sha256(payload: object) -> str:
    """控制器绑定与重推导共享的规范守卫摘要。"""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _call_contracts(
    call_kind: Literal["attempt_decision", "node_verification"],
) -> tuple[str, str, str]:
    if call_kind == "attempt_decision":
        return (
            "runtime_work_run_attempt_decision",
            "task-node-attempt-decision-model-request-v1",
            TASK_NODE_ATTEMPT_RESULT_CONTRACT,
        )
    if call_kind == "node_verification":
        return (
            "runtime_task_node_semantic_verification",
            "task-node-verification-model-request-v1",
            TASK_NODE_VERIFICATION_RESULT_CONTRACT,
        )
    raise ValueError("unsupported TaskNode model-call kind")


def _task_node_execution_subject_id(
    *,
    session_id: str,
    subject: TaskNodeSubject,
) -> str:
    payload = _canonical_json(
        [
            "execsubject",
            "task_node_v1",
            session_id,
            subject.task_id,
            subject.graph_revision,
            subject.node_id,
            subject.node_revision,
        ]
    )
    return f"execsubject-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _bound_request_payload(
    binding: TaskNodeBoundModelCall,
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
            "execution_subject_id": binding.execution_subject_id,
            "subject": binding.subject.model_dump(mode="json"),
            "request_turn_id": binding.request_turn_id,
            "invocation_turn_id": binding.invocation_turn_id,
            "work_run_id": binding.work_run_id,
            "dispatch_work_run_revision": binding.dispatch_work_run_revision,
            "attempt_id": binding.attempt_id,
            "attempt_ordinal": binding.attempt_ordinal,
            "verification_request_id": binding.verification_request_id,
            "verification_request_revision": (
                binding.verification_request_revision
            ),
            "locked_work_run_revision": binding.locked_work_run_revision,
            "state_guard_sha256": binding.state_guard_sha256,
        },
        "system_prompt": binding.system_prompt,
        "user_content": user_payload,
    }


def _require_prompt_identity(
    binding: TaskNodeBoundModelCall,
    *,
    user_payload: dict[str, Any],
) -> None:
    raw = user_payload.get("bindings")
    if not isinstance(raw, dict):
        raise ValueError("TaskNode model prompt omitted its bindings")
    common = {
        "session_id": binding.session_id,
        "work_run_id": binding.work_run_id,
    }
    if any(raw.get(key) != value for key, value in common.items()):
        raise ValueError("TaskNode model prompt crossed its common authority")
    if binding.call_kind == "attempt_decision":
        expected = {
            "turn_id": binding.invocation_turn_id,
            "work_run_revision": binding.dispatch_work_run_revision,
            "attempt_id": binding.attempt_id,
            "attempt_ordinal": binding.attempt_ordinal,
        }
        node = user_payload.get("node")
        prompt_subject = node.get("subject") if isinstance(node, dict) else None
    else:
        expected = {
            "request_turn_id": binding.request_turn_id,
            "verification_request_id": binding.verification_request_id,
            "locked_work_run_revision": binding.locked_work_run_revision,
            "submitted_attempt_id": binding.attempt_id,
        }
        prompt_subject = raw.get("subject")
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("TaskNode model prompt crossed its call authority")
    if prompt_subject != binding.subject.model_dump(mode="json"):
        raise ValueError("TaskNode model prompt crossed its subject")


def _binding_sha256(binding: TaskNodeBoundModelCall) -> str:
    return task_node_model_state_guard_sha256(
        binding.model_dump(mode="json", exclude={"binding_sha256"})
    )


def _require_canonical_json_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be one JSON object")
    return parsed


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


__all__ = [
    "TASK_NODE_ATTEMPT_RESULT_CONTRACT",
    "TASK_NODE_VERIFICATION_RESULT_CONTRACT",
    'TaskNodeBoundModelCall',
    "task_node_model_state_guard_sha256",
]
