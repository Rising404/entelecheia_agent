"""当前 Session 历史检索与共享证据信封契约。

本模块特意不了解 Session、Task、文件系统或检索数据库标识。Runtime 将每项注册绑定到
已冻结的 Host 范围并提供处理器。因此，模型可以构造查询并缩小该范围，但没有任何模式字段
允许它虚构路径、Session/Task 标识符、Source 筛选条件或检索世代。

文件内容检索由候选优先工具族负责；本模块只注册 ``retrieve_history``。共享证据信封
仍服务于底层检索数据面，其中只含公共别名和不透明指纹；权威标识符留在 Runtime
可注入端口之后。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..registration import ToolExecutionProfile, ToolRegistration


RETRIEVE_HISTORY_TOOL_ID = "retrieve_history"
RETRIEVE_HISTORY_CONTRACT_VERSION = "history-retrieval-v1"
RETRIEVAL_TOOL_IMPLEMENTATION_VERSION = "1"
RETRIEVAL_EVIDENCE_CONTRACT_VERSION = "retrieval-evidence-v1"
HISTORY_RETRIEVAL_SOURCE_ID = "personagraph.retrieval.history"

MAX_RETRIEVAL_QUERY_CHARS = 2_000
MAX_RETRIEVAL_ITEMS = 20
MAX_EVIDENCE_TEXT_CHARS = 8_000
MAX_GAPS = 64

_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RetrievalCorpus(StrEnum):
    FILES = "files"
    HISTORY = "history"


class HistoryRetrievalScope(StrEnum):
    CURRENT_SESSION = "current_session"
    LONG_TERM_USER = "long_term_user"
    CURRENT_TASK = "current_task"


class RetrievalEvidenceSource(StrEnum):
    DOCUMENT = "document"
    CURRENT_SESSION = "current_session"
    LONG_TERM_USER = "long_term_user"
    LONG_TERM_TASK = "long_term_task"


class RetrievalEvidenceStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class RetrievalEvidenceOutcome(StrEnum):
    MATCHED = "matched"
    NO_MATCH = "no_match"
    NOT_ESTABLISHED = "not_established"


class RetrievalEvidenceOrigin(StrEnum):
    WORKSPACE = "workspace"
    USER_UPLOAD = "user_upload"
    AGENT_OUTPUT = "agent_output"
    MOUNTED_DOCUMENT = "mounted_document"
    CURRENT_SESSION = "current_session"
    LONG_TERM_USER = "long_term_user"
    CURRENT_TASK = "current_task"


@dataclass(frozen=True, slots=True)
class RetrievalEvidenceLocator:
    """有界公共定位器；无法表示路径及权威 ID。"""

    location: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_fingerprint: str | None = None
    user_turn_index: int | None = None
    assistant_turn_index: int | None = None
    memory_type: str | None = None

    def __post_init__(self) -> None:
        if self.location is not None:
            _bounded_public_text(self.location, "location", maximum=512)
            if _looks_like_private_path(self.location):
                raise ValueError("location must not expose a filesystem path")
        for name, value in (
            ("page_start", self.page_start),
            ("page_end", self.page_end),
            ("user_turn_index", self.user_turn_index),
            ("assistant_turn_index", self.assistant_turn_index),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page range must not be reversed")
        if self.chunk_fingerprint is not None:
            _sha256(self.chunk_fingerprint, "chunk_fingerprint")
        if self.memory_type is not None:
            _safe_code(self.memory_type, "memory_type")

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name in (
            "location",
            "page_start",
            "page_end",
            "chunk_fingerprint",
            "user_turn_index",
            "assistant_turn_index",
            "memory_type",
        ):
            value = getattr(self, name)
            if value is not None:
                output[name] = value
        return output


@dataclass(frozen=True, slots=True)
class RetrievalExactReadRoute:
    """当前冻结 Tool Catalog 中可直接调用的后续路由。

    ``source_alias`` 明确保留证据与路由的关系，而 ``arguments`` 携带目标工具所需、
    经 Host 冻结的精确输入。消费者不得从别名推断文件系统路径、文档标识符或参数名。
    """

    tool_id: str
    source_alias: str
    arguments: Mapping[str, str | int | bool]

    def __post_init__(self) -> None:
        _safe_code(self.tool_id, "exact-read tool_id")
        _public_alias(self.source_alias)
        if not isinstance(self.arguments, Mapping) or not self.arguments:
            raise ValueError("exact-read arguments must be a non-empty mapping")
        normalized: dict[str, str | int | bool] = {}
        for key, value in self.arguments.items():
            _safe_code(key, "exact-read argument name")
            if isinstance(value, str):
                _bounded_public_text(
                    value,
                    "exact-read argument value",
                    maximum=512,
                )
            elif isinstance(value, bool):
                pass
            elif not isinstance(value, int):
                raise ValueError(
                    "exact-read argument values must be string, integer, or bool"
                )
            normalized[key] = value
        if len(normalized) > 8:
            raise ValueError("exact-read route has too many fixed arguments")
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "source_alias": self.source_alias,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    handle: str
    source_type: RetrievalEvidenceSource
    origin: RetrievalEvidenceOrigin
    text: str
    content_sha256: str
    source_revision_fingerprint: str
    rank: int
    estimated_tokens: int
    locator: RetrievalEvidenceLocator
    source_alias: str | None = None
    exact_read: RetrievalExactReadRoute | None = None

    def __post_init__(self) -> None:
        _opaque_public_id(self.handle, "handle")
        if not isinstance(self.source_type, RetrievalEvidenceSource):
            raise ValueError("source_type must be a RetrievalEvidenceSource")
        if not isinstance(self.origin, RetrievalEvidenceOrigin):
            raise ValueError("origin must be a RetrievalEvidenceOrigin")
        _bounded_public_text(self.text, "text", maximum=MAX_EVIDENCE_TEXT_CHARS)
        _sha256(self.content_sha256, "content_sha256")
        _sha256(
            self.source_revision_fingerprint,
            "source_revision_fingerprint",
        )
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if (
            isinstance(self.estimated_tokens, bool)
            or not isinstance(self.estimated_tokens, int)
            or self.estimated_tokens <= 0
        ):
            raise ValueError("estimated_tokens must be a positive integer")
        if not isinstance(self.locator, RetrievalEvidenceLocator):
            raise ValueError("locator must be a RetrievalEvidenceLocator")
        if self.source_alias is not None:
            _public_alias(self.source_alias)
        if self.exact_read is not None:
            if not isinstance(self.exact_read, RetrievalExactReadRoute):
                raise ValueError("exact_read must be a RetrievalExactReadRoute")
            if self.source_alias is None or self.exact_read.source_alias != self.source_alias:
                raise ValueError("exact_read must use this evidence source alias")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "handle": self.handle,
            "source_type": self.source_type.value,
            "origin": self.origin.value,
            "text": self.text,
            "content_sha256": self.content_sha256,
            "source_revision_fingerprint": self.source_revision_fingerprint,
            "rank": self.rank,
            "estimated_tokens": self.estimated_tokens,
            "locator": self.locator.to_dict(),
        }
        if self.source_alias is not None:
            result["source_alias"] = self.source_alias
        if self.exact_read is not None:
            result["exact_read"] = self.exact_read.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class RetrievalEvidenceGap:
    code: str
    blocking: bool
    source_type: RetrievalEvidenceSource | None = None
    source_alias: str | None = None
    known_count: int | None = None

    def __post_init__(self) -> None:
        _safe_code(self.code, "gap code")
        if not isinstance(self.blocking, bool):
            raise ValueError("blocking must be a bool")
        if self.source_type is not None and not isinstance(
            self.source_type,
            RetrievalEvidenceSource,
        ):
            raise ValueError("source_type must be a RetrievalEvidenceSource")
        if self.source_alias is not None:
            _public_alias(self.source_alias)
        if self.known_count is not None and (
            isinstance(self.known_count, bool)
            or not isinstance(self.known_count, int)
            or self.known_count < 0
        ):
            raise ValueError("known_count must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "code": self.code,
            "blocking": self.blocking,
        }
        if self.source_type is not None:
            output["source_type"] = self.source_type.value
        if self.source_alias is not None:
            output["source_alias"] = self.source_alias
        if self.known_count is not None:
            output["known_count"] = self.known_count
        return output


@dataclass(frozen=True, slots=True)
class RetrievalEvidenceCoverage:
    scope_fingerprint: str
    retrieval_generation_fingerprint: str
    requested_aliases: tuple[str, ...] = ()
    requested_scopes: tuple[HistoryRetrievalScope, ...] = ()
    returned_items: int = 0
    configured_token_limit: int = 1
    packed_tokens: int = 0
    encoder_fingerprint: str | None = None
    reranker_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _sha256(self.scope_fingerprint, "scope_fingerprint")
        _sha256(
            self.retrieval_generation_fingerprint,
            "retrieval_generation_fingerprint",
        )
        if any(not isinstance(alias, str) for alias in self.requested_aliases):
            raise ValueError("requested_aliases must contain strings")
        for alias in self.requested_aliases:
            _public_alias(alias)
        if len(set(self.requested_aliases)) != len(self.requested_aliases):
            raise ValueError("requested_aliases must be unique")
        if any(
            not isinstance(scope, HistoryRetrievalScope)
            for scope in self.requested_scopes
        ):
            raise ValueError("requested_scopes contains an invalid scope")
        if len(set(self.requested_scopes)) != len(self.requested_scopes):
            raise ValueError("requested_scopes must be unique")
        if (
            isinstance(self.returned_items, bool)
            or not isinstance(self.returned_items, int)
            or self.returned_items < 0
        ):
            raise ValueError("returned_items must be a non-negative integer")
        if (
            isinstance(self.configured_token_limit, bool)
            or not isinstance(self.configured_token_limit, int)
            or self.configured_token_limit <= 0
        ):
            raise ValueError("configured_token_limit must be positive")
        if (
            isinstance(self.packed_tokens, bool)
            or not isinstance(self.packed_tokens, int)
            or self.packed_tokens < 0
            or self.packed_tokens > self.configured_token_limit
        ):
            raise ValueError("packed_tokens must fit configured_token_limit")
        for name, value in (
            ("encoder_fingerprint", self.encoder_fingerprint),
            ("reranker_fingerprint", self.reranker_fingerprint),
        ):
            if value is not None:
                _sha256(value, name)

    def to_dict(self) -> dict[str, Any]:
        output = {
            "scope_fingerprint": self.scope_fingerprint,
            "retrieval_generation_fingerprint": (
                self.retrieval_generation_fingerprint
            ),
            "requested_aliases": list(self.requested_aliases),
            "requested_scopes": [scope.value for scope in self.requested_scopes],
            "returned_items": self.returned_items,
            "configured_token_limit": self.configured_token_limit,
            "packed_tokens": self.packed_tokens,
        }
        if self.encoder_fingerprint is not None:
            output["encoder_fingerprint"] = self.encoder_fingerprint
        if self.reranker_fingerprint is not None:
            output["reranker_fingerprint"] = self.reranker_fingerprint
        return output


@dataclass(frozen=True, slots=True)
class RetrievalEvidenceEnvelope:
    corpus: RetrievalCorpus
    status: RetrievalEvidenceStatus
    outcome: RetrievalEvidenceOutcome
    query: str
    evidence: tuple[RetrievalEvidence, ...]
    gaps: tuple[RetrievalEvidenceGap, ...]
    coverage: RetrievalEvidenceCoverage
    truncated: bool = False
    contract_version: str = RETRIEVAL_EVIDENCE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != RETRIEVAL_EVIDENCE_CONTRACT_VERSION:
            raise ValueError("unsupported retrieval evidence contract version")
        if not isinstance(self.corpus, RetrievalCorpus):
            raise ValueError("corpus must be a RetrievalCorpus")
        if not isinstance(self.status, RetrievalEvidenceStatus):
            raise ValueError("status must be a RetrievalEvidenceStatus")
        if not isinstance(self.outcome, RetrievalEvidenceOutcome):
            raise ValueError("outcome must be a RetrievalEvidenceOutcome")
        _bounded_public_text(
            self.query,
            "query",
            maximum=MAX_RETRIEVAL_QUERY_CHARS,
        )
        if any(not isinstance(item, RetrievalEvidence) for item in self.evidence):
            raise ValueError("evidence contains an invalid item")
        if len(self.evidence) > MAX_RETRIEVAL_ITEMS:
            raise ValueError("evidence exceeds the public item limit")
        if any(not isinstance(gap, RetrievalEvidenceGap) for gap in self.gaps):
            raise ValueError("gaps contains an invalid item")
        if len(self.gaps) > MAX_GAPS:
            raise ValueError("gaps exceeds the public gap limit")
        if not isinstance(self.coverage, RetrievalEvidenceCoverage):
            raise ValueError("coverage must be RetrievalEvidenceCoverage")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a bool")
        if self.coverage.returned_items != len(self.evidence):
            raise ValueError("coverage returned_items must match evidence")
        if self.status is RetrievalEvidenceStatus.BLOCKED and self.evidence:
            raise ValueError("blocked status cannot publish evidence")
        if self.status is RetrievalEvidenceStatus.PARTIAL and not (
            self.gaps or self.truncated
        ):
            raise ValueError("partial status requires a gap or truncation")
        if self.status is RetrievalEvidenceStatus.COMPLETE and (
            self.gaps or self.truncated
        ):
            raise ValueError("complete status cannot carry gaps or truncation")
        if self.outcome is RetrievalEvidenceOutcome.MATCHED and not self.evidence:
            raise ValueError("matched outcome requires evidence")
        if self.outcome is RetrievalEvidenceOutcome.NO_MATCH and (
            self.evidence or self.status is not RetrievalEvidenceStatus.COMPLETE
        ):
            raise ValueError("no_match requires complete coverage and no evidence")
        if self.outcome is RetrievalEvidenceOutcome.NOT_ESTABLISHED and self.evidence:
            raise ValueError("not_established cannot accompany published evidence")
        if self.status is RetrievalEvidenceStatus.BLOCKED and (
            self.outcome is not RetrievalEvidenceOutcome.NOT_ESTABLISHED
        ):
            raise ValueError("blocked status cannot establish a match outcome")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "corpus": self.corpus.value,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "query": self.query,
            "evidence": [item.to_dict() for item in self.evidence],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "coverage": self.coverage.to_dict(),
            "truncated": self.truncated,
        }


RetrievalToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


def build_retrieve_history_registration(
    *,
    handler: RetrievalToolHandler,
    effect_scope: str,
    source_fingerprint: str | None = None,
) -> ToolRegistration:
    """围绕一个已冻结 Runtime 处理器创建历史语料库工具。"""

    if source_fingerprint is not None:
        _sha256(source_fingerprint, "source_fingerprint")
    return ToolRegistration(
        spec=build_retrieve_history_tool_spec(),
        implementation_version=RETRIEVAL_TOOL_IMPLEMENTATION_VERSION,
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            HISTORY_RETRIEVAL_SOURCE_ID,
            fingerprint=source_fingerprint,
        ),
        handler=handler,
        effect_profile=build_retrieve_history_effect_profile(
            default_scope=effect_scope,
        ),
        execution_profile=build_retrieve_history_execution_profile(),
    )


def build_retrieve_history_tool_spec() -> ToolSpec:
    """Build the process-stable model contract for History retrieval."""

    return ToolSpec(
        tool_id=RETRIEVE_HISTORY_TOOL_ID,
        contract_version=RETRIEVE_HISTORY_CONTRACT_VERSION,
        name="Retrieve conversation and memory evidence",
        description=(
            "Search the conversation and memory scopes authorized for this "
            "model call. Optionally narrow to current-session, long-term-user, "
            "or current-task evidence. This tool cannot accept Session, user, "
            "or Task identifiers and cannot widen the Host-frozen scope."
        ),
        input_schema=_history_input_schema(),
        output_schema=_evidence_output_schema(RetrievalCorpus.HISTORY),
        catalog_tags=("memory", "search", "read"),
    )


def build_retrieve_history_effect_profile(
    *,
    default_scope: str,
) -> ToolEffectProfile:
    """Build the stable effect shape narrowed to one Host-owned scope."""

    _bounded_public_text(default_scope, "default_scope", maximum=256)
    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.MEMORY,
                action=EffectAction.SEARCH,
                scope_kind=EffectScopeKind.SESSION,
                default_scope=default_scope,
                data_egress=DataEgress.CONTENT,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )


def build_retrieve_history_execution_profile() -> ToolExecutionProfile:
    """Build the process-stable execution contract for History retrieval."""

    return ToolExecutionProfile(
        default_timeout_s=30.0,
        hard_timeout_s=90.0,
        max_output_bytes=96_000,
        max_transparent_retries=1,
        concurrency_class="history_retrieval",
    )


def opaque_fingerprint(value: str) -> str:
    """对私有范围、世代、修订或权威标识执行哈希。"""

    if not isinstance(value, str) or not value:
        raise ValueError("fingerprint input must be a non-empty string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _history_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_RETRIEVAL_QUERY_CHARS,
            },
            "scopes": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(HistoryRetrievalScope),
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": [scope.value for scope in HistoryRetrievalScope],
                },
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_RETRIEVAL_ITEMS,
                "default": 8,
            },
        },
    }


def _evidence_output_schema(corpus: RetrievalCorpus) -> dict[str, Any]:
    source_values = [source.value for source in RetrievalEvidenceSource]
    status_values = [status.value for status in RetrievalEvidenceStatus]
    outcome_values = [outcome.value for outcome in RetrievalEvidenceOutcome]
    origin_values = [origin.value for origin in RetrievalEvidenceOrigin]
    scope_values = [scope.value for scope in HistoryRetrievalScope]
    locator_properties = {
        "location": {"type": "string", "minLength": 1, "maxLength": 512},
        "page_start": {"type": "integer", "minimum": 0},
        "page_end": {"type": "integer", "minimum": 0},
        "chunk_fingerprint": {"type": "string", "pattern": _SHA256.pattern},
        "user_turn_index": {"type": "integer", "minimum": 0},
        "assistant_turn_index": {"type": "integer", "minimum": 0},
        "memory_type": {"type": "string", "pattern": _SAFE_CODE.pattern},
    }
    evidence_properties = {
        "handle": {"type": "string", "minLength": 1, "maxLength": 128},
        "source_type": {"type": "string", "enum": source_values},
        "origin": {"type": "string", "enum": origin_values},
        "source_alias": {"type": "string", "minLength": 1, "maxLength": 512},
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_EVIDENCE_TEXT_CHARS,
        },
        "content_sha256": {"type": "string", "pattern": _SHA256.pattern},
        "source_revision_fingerprint": {
            "type": "string",
            "pattern": _SHA256.pattern,
        },
        "rank": {"type": "integer", "minimum": 1},
        "estimated_tokens": {"type": "integer", "minimum": 1},
        "locator": {
            "type": "object",
            "additionalProperties": False,
            "properties": locator_properties,
        },
        "exact_read": {
            "type": "object",
            "additionalProperties": False,
            "required": ["tool_id", "source_alias", "arguments"],
            "properties": {
                "tool_id": {"type": "string", "pattern": _SAFE_CODE.pattern},
                "source_alias": {"type": "string", "minLength": 1, "maxLength": 512},
                "arguments": {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 8,
                    "propertyNames": {"pattern": _SAFE_CODE.pattern},
                    "additionalProperties": {
                        "oneOf": [
                            {"type": "string", "minLength": 1, "maxLength": 512},
                            {"type": "integer"},
                            {"type": "boolean"},
                        ]
                    },
                },
            },
        },
    }
    gap_properties = {
        "code": {"type": "string", "pattern": _SAFE_CODE.pattern},
        "blocking": {"type": "boolean"},
        "source_type": {"type": "string", "enum": source_values},
        "source_alias": {"type": "string", "minLength": 1, "maxLength": 512},
        "known_count": {"type": "integer", "minimum": 0},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "contract_version",
            "corpus",
            "status",
            "outcome",
            "query",
            "evidence",
            "gaps",
            "coverage",
            "truncated",
        ],
        "properties": {
            "contract_version": {
                "type": "string",
                "const": RETRIEVAL_EVIDENCE_CONTRACT_VERSION,
            },
            "corpus": {"type": "string", "const": corpus.value},
            "status": {"type": "string", "enum": status_values},
            "outcome": {"type": "string", "enum": outcome_values},
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_RETRIEVAL_QUERY_CHARS,
            },
            "evidence": {
                "type": "array",
                "maxItems": MAX_RETRIEVAL_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "handle",
                        "source_type",
                        "origin",
                        "text",
                        "content_sha256",
                        "source_revision_fingerprint",
                        "rank",
                        "estimated_tokens",
                        "locator",
                    ],
                    "properties": evidence_properties,
                },
            },
            "gaps": {
                "type": "array",
                "maxItems": MAX_GAPS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "blocking"],
                    "properties": gap_properties,
                },
            },
            "coverage": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "scope_fingerprint",
                    "retrieval_generation_fingerprint",
                    "requested_aliases",
                    "requested_scopes",
                    "returned_items",
                    "configured_token_limit",
                    "packed_tokens",
                ],
                "properties": {
                    "scope_fingerprint": {
                        "type": "string",
                        "pattern": _SHA256.pattern,
                    },
                    "retrieval_generation_fingerprint": {
                        "type": "string",
                        "pattern": _SHA256.pattern,
                    },
                    "requested_aliases": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 512,
                        },
                    },
                    "requested_scopes": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": scope_values},
                    },
                    "returned_items": {"type": "integer", "minimum": 0},
                    "configured_token_limit": {"type": "integer", "minimum": 1},
                    "packed_tokens": {"type": "integer", "minimum": 0},
                    "encoder_fingerprint": {
                        "type": "string",
                        "pattern": _SHA256.pattern,
                    },
                    "reranker_fingerprint": {
                        "type": "string",
                        "pattern": _SHA256.pattern,
                    },
                },
            },
            "truncated": {"type": "boolean"},
        },
    }


def _bounded_public_text(value: str, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded non-empty string")
    if any(ord(character) < 32 and character not in "\n\t\r" for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _public_alias(value: str) -> str:
    _bounded_public_text(value, "source alias", maximum=512)
    normalized = value.replace("\\", "/")
    if normalized.startswith(("/", "../", "~/", "file://")):
        raise ValueError("source alias must not be an absolute or parent path")
    if "/../" in normalized or normalized.endswith("/.."):
        raise ValueError("source alias must not escape its frozen scope")
    if re.match(r"^[a-zA-Z]:/", normalized):
        raise ValueError("source alias must not be an absolute path")
    return value


def _looks_like_private_path(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    return (
        normalized.startswith("/")
        or normalized.startswith("../")
        or normalized.startswith("~/")
        or normalized.startswith("file://")
        or "/../" in normalized
        or bool(re.match(r"^[a-zA-Z]:/", normalized))
    )


def _safe_code(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_CODE.fullmatch(value):
        raise ValueError(f"{name} must be a safe code")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _opaque_public_id(value: str, name: str) -> str:
    _bounded_public_text(value, name, maximum=128)
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", value):
        raise ValueError(f"{name} must be an opaque public identifier")
    return value


__all__ = (
    "HISTORY_RETRIEVAL_SOURCE_ID",
    'HistoryRetrievalScope',
    "MAX_RETRIEVAL_ITEMS",
    "MAX_RETRIEVAL_QUERY_CHARS",
    "RETRIEVAL_EVIDENCE_CONTRACT_VERSION",
    "RETRIEVE_HISTORY_CONTRACT_VERSION",
    "RETRIEVE_HISTORY_TOOL_ID",
    'RetrievalCorpus',
    'RetrievalEvidenceCoverage',
    'RetrievalEvidenceEnvelope',
    'RetrievalEvidenceGap',
    'RetrievalEvidenceLocator',
    'RetrievalEvidenceOrigin',
    'RetrievalEvidenceOutcome',
    'RetrievalEvidenceSource',
    'RetrievalEvidenceStatus',
    'RetrievalEvidence',
    'RetrievalExactReadRoute',
    "build_retrieve_history_effect_profile",
    "build_retrieve_history_execution_profile",
    "build_retrieve_history_registration",
    "build_retrieve_history_tool_spec",
    "opaque_fingerprint",
)
