"""严格且无副作用的 SessionContext 候选抽取。

模型可以提出带类型观察，但本模块只负责派生 ID 和版本、校验证据所有权并返回候选。
它绝不会写入视图、长期记忆、对话或 Graph 状态。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...model_io.gateway import complete_json
from ...model_io.output_language import OUTPUT_LANGUAGE_CLAUSE
from .catalog import CATALOG_VERSION, all_policies, policy_for
from .models import (
    CandidateRejection,
    EvidenceKind,
    EvidenceRecord,
    ObservationCandidate,
    Operation,
    SessionDomain,
    SessionExtractionResult,
    SourceKind,
)


EXTRACTOR_VERSION = f"{CATALOG_VERSION}:extractor-v1"


class _RawCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: SessionDomain
    state_type: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: Any = None
    operation: Operation
    source_kind: SourceKind
    evidence_refs: list[str] = Field(min_length=1)
    confidence_hint: float = Field(default=1.0, ge=0.0, le=1.0)


class _RawOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[_RawCandidate] = Field(default_factory=list)
    should_store: bool | None = None  # 接受现有模拟网关的空操作结构


def _catalog_lines() -> str:
    lines = []
    for policy in all_policies():
        operations = "/".join(sorted(operation.value for operation in policy.allowed_operations))
        sources = "/".join(sorted(source.value for source in policy.allowed_sources))
        lines.append(f"- {policy.domain.value}.{policy.state_type}: ops={operations}; sources={sources}")
    return "\n".join(lines)


EXTRACTION_SYSTEM = """你是 SessionContext 候选抽取器。只抽取当前会话内用于协作适配、任务连续性和交互连续性的状态。

只输出 JSON，不要 markdown 或解释：
{"items":[{"domain":"user|task|interaction","state_type":"类型","key":"稳定维度或事件键","value":"JSON 值","operation":"set|append|resolve|retract|touch","source_kind":"explicit|inferred|tool|assistant","evidence_refs":["当前输入给出的 evidence id"],"confidence_hint":0到1}]}

允许目录：
«CATALOG»

边界：
- 一句话可以抽多个 facet，不要揉成一个自由文本袋。
- explicit 只表示用户明确陈述；inferred 必须保守并降低 confidence_hint。
- assistant 只用于 assistant_commitment，不能冒充用户证据。
- tool 只能引用 tool_result/artifact evidence。
- 不抽助手身份、系统实现、外部作品设定、密码等敏感信息、无依据心理诊断或外部世界真伪结论。
- 不直接生成长期用户记忆或长期任务记忆；长期提升由其他治理流程决定。
- 不确定、缺少 evidence 或目录中无合适类型时不要输出该项。
""".replace("«CATALOG»", _catalog_lines()) + "\n\n" + OUTPUT_LANGUAGE_CLAUSE


def _json_object(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _candidate_id(session_id: str, item: _RawCandidate) -> str:
    value = json.dumps(item.value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    parts = (
        session_id,
        item.domain,
        item.state_type,
        item.key,
        item.operation,
        item.source_kind,
        "\x1e".join(item.evidence_refs),
        value,
        EXTRACTOR_VERSION,
    )
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"candidate:{digest}"


def _source_matches(item: _RawCandidate, evidence: dict[str, EvidenceRecord]) -> bool:
    kinds = {evidence[ref].kind for ref in item.evidence_refs}
    if item.source_kind in {SourceKind.EXPLICIT, SourceKind.INFERRED}:
        return EvidenceKind.USER_TURN in kinds
    if item.source_kind == SourceKind.ASSISTANT:
        return kinds == {EvidenceKind.ASSISTANT_TURN}
    return bool(kinds) and kinds <= {EvidenceKind.TOOL_RESULT, EvidenceKind.ARTIFACT}


def _validate_item(
    item: _RawCandidate,
    session_id: str,
    evidence: dict[str, EvidenceRecord],
) -> str | None:
    policy = policy_for(item.domain, item.state_type)
    if not item.key.strip():
        return "invalid_key"
    if policy is None:
        return "invalid_type"
    if item.operation not in policy.allowed_operations:
        return "invalid_operation"
    if item.source_kind not in policy.allowed_sources:
        return "source_not_allowed"
    if any(ref not in evidence for ref in item.evidence_refs):
        return "unknown_evidence_ref"
    if any(evidence[ref].session_id != session_id for ref in item.evidence_refs):
        return "evidence_scope_mismatch"
    if not _source_matches(item, evidence):
        return "source_evidence_mismatch"
    return None


def parse_candidates(
    text: str,
    session_id: str,
    evidence_records: Sequence[EvidenceRecord],
) -> SessionExtractionResult:
    """解析模型 JSON，只返回目录与证据均有效的候选。"""
    data = _json_object(text)
    if data is None:
        return SessionExtractionResult((), error="parse_failed")
    try:
        output = _RawOutput.model_validate(data)
    except ValidationError:
        return SessionExtractionResult((), error="schema_validation_failed")

    evidence = {record.id: record for record in evidence_records}
    candidates: list[ObservationCandidate] = []
    rejected: list[CandidateRejection] = []
    seen: set[str] = set()
    for index, item in enumerate(output.items):
        reason = _validate_item(item, session_id, evidence)
        if reason:
            rejected.append(CandidateRejection(index, reason))
            continue
        candidate_id = _candidate_id(session_id, item)
        if candidate_id in seen:
            rejected.append(CandidateRejection(index, "duplicate_candidate"))
            continue
        seen.add(candidate_id)
        valid_from = max(evidence[ref].created_at for ref in item.evidence_refs)
        candidates.append(ObservationCandidate(
            candidate_id=candidate_id,
            session_id=session_id,
            domain=item.domain,
            state_type=item.state_type,
            key=item.key.strip(),
            proposed_value=item.value,
            operation=item.operation,
            source_kind=item.source_kind,
            derived_from=tuple(dict.fromkeys(item.evidence_refs)),
            extractor_version=EXTRACTOR_VERSION,
            confidence_hint=item.confidence_hint,
            valid_from=valid_from,
        ))
    return SessionExtractionResult(tuple(candidates), tuple(rejected))


def extract_candidates(
    session_id: str,
    user_input: str,
    assistant_reply: str,
    evidence_records: Sequence[EvidenceRecord],
    *,
    complete: Callable[[str, str], str] | None = None,
) -> SessionExtractionResult:
    """调用一次可注入模型；失败时软降级为空候选集。"""
    evidence_lines = "\n".join(
        f"- {record.id} [{record.kind.value}]: {record.content_excerpt}"
        for record in evidence_records
    )
    user_content = (
        f"session_id: {session_id}\n"
        f"evidence:\n{evidence_lines or '（无）'}\n\n"
        f"user_input:\n{user_input}\n\nassistant_reply:\n{assistant_reply}"
    )
    try:
        raw = (complete or complete_json)(EXTRACTION_SYSTEM, user_content)
    except Exception as exc:
        return SessionExtractionResult((), error=f"model_error:{type(exc).__name__}")
    return parse_candidates(raw, session_id, evidence_records)
