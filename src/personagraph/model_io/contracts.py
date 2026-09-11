from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssistantText(_Contract):
    kind: Literal["assistant_text"] = "assistant_text"
    text: str


class ToolCall(_Contract):
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    transport: Literal["native", "prompt_json"]


class ToolCallBatch(_Contract):
    kind: Literal["tool_call_batch"] = "tool_call_batch"
    calls: list[ToolCall] = Field(min_length=1, max_length=3)
    assistant_text: str | None = None


class ControlProposal(_Contract):
    kind: Literal["control_proposal"] = "control_proposal"
    proposal: dict[str, Any]


class StructuredArtifact(_Contract):
    kind: Literal["structured_artifact"] = "structured_artifact"
    artifact: dict[str, Any]


class Refusal(_Contract):
    kind: Literal["refusal"] = "refusal"
    reason: str
    user_message: str


class ProtocolError(_Contract):
    kind: Literal["protocol_error"] = "protocol_error"
    code: str
    message: str
    transport: Literal["native", "prompt_json"]
    retryable: bool = True


ModelTurnOutput = Annotated[
    AssistantText | ToolCallBatch | ControlProposal | StructuredArtifact | Refusal | ProtocolError,
    Field(discriminator="kind"),
]


@dataclass
class ModelResult:
    """一次 provider 调用的中立结果合同。"""

    reply: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    finish_reason: str | None = None
    output: ModelTurnOutput | None = None
    model_call_id: str | None = None
    purpose: str | None = None
    control_transport: str | None = None


_OUTPUT_ADAPTER = TypeAdapter(ModelTurnOutput)


def model_output_to_dict(output: ModelTurnOutput) -> dict[str, Any]:
    return output.model_dump(mode="json")


def model_output_from_dict(value: object) -> ModelTurnOutput | None:
    if not isinstance(value, dict):
        return None
    try:
        return _OUTPUT_ADAPTER.validate_python(value)
    except Exception:
        return None


def output_tool_calls(output: ModelTurnOutput | None) -> list[dict[str, Any]]:
    if not isinstance(output, ToolCallBatch):
        return []
    return [
        {
            "tool": call.tool_name,
            "args": dict(call.arguments),
            "call_id": call.call_id,
            "transport": call.transport,
        }
        for call in output.calls
    ]
