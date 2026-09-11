"""模型 query 提案进入 Retrieval 前的确定性准入门。

Guard 检查条数、长度、重复、控制字符，以及试图把 SQL 或 Host 私有 scope 参数塞进自然语言
query 的低层语法。它只接受或拒绝完整提案，不调用模型做修复，也不以规则重写用户语义。

通过 Guard 只表示 query 的结构可安全交给检索器，不表示它语义相关、证据充分或来源已获授权；
来源集合、``SourceFilter``、generation 与预算仍由 Host 冻结的 ``RetrievalRequest`` 决定。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import re

from .contracts import QueryProposal


DEFAULT_MAX_RETRIEVAL_QUERIES = 8
DEFAULT_MAX_RETRIEVAL_QUERY_CHARACTERS = 2_000
DEFAULT_MAX_RETRIEVAL_QUERY_TOKENS = 256


class QueryGuardErrorCode(StrEnum):
    """语义 Query 提案被拒绝的安全稳定原因。"""

    UNKNOWN = "unknown"
    EMPTY_PROPOSAL = "empty_proposal"
    TOO_MANY_QUERIES = "too_many_queries"
    DUPLICATE_QUERY = "duplicate_query"
    EMPTY_QUERY = "empty_query"
    QUERY_TOO_LONG = "query_too_long"
    QUERY_TOO_MANY_TOKENS = "query_too_many_tokens"
    QUERY_TOKEN_COUNT_UNAVAILABLE = "query_token_count_unavailable"
    CONTROL_CHARACTER = "control_character"
    FORBIDDEN_LOW_LEVEL_SYNTAX = "forbidden_low_level_syntax"


class QueryGuardError(ValueError):
    """某项提案尝试越过检索公开边界。

    对现有调用方而言，``message`` 仍是普通 ``ValueError`` 文本；``code`` 是非敏感诊断
    字段，供未来 Runtime 事件和特定用途错误处理使用。
    """

    def __init__(
        self,
        message: str,
        *,
        code: QueryGuardErrorCode = QueryGuardErrorCode.UNKNOWN,
    ) -> None:
        if not isinstance(code, QueryGuardErrorCode):
            raise TypeError("QueryGuardError.code must be a QueryGuardErrorCode")
        super().__init__(message)
        self.code: QueryGuardErrorCode = code


@dataclass(frozen=True, slots=True)
class QueryGuardConfig:
    max_queries: int = DEFAULT_MAX_RETRIEVAL_QUERIES
    max_query_characters: int = DEFAULT_MAX_RETRIEVAL_QUERY_CHARACTERS
    max_query_tokens: int = DEFAULT_MAX_RETRIEVAL_QUERY_TOKENS

    def __post_init__(self) -> None:
        values = (
            self.max_queries,
            self.max_query_characters,
            self.max_query_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError("QueryGuard limits must be greater than zero")


_SQL_PREFIX = re.compile(
    r"^\s*(?:select|insert|update|delete|pragma|attach|detach|drop|alter|create)\b",
    re.IGNORECASE,
)
_LOW_LEVEL_FILTER = re.compile(
    r"\b(?:"
    r"owner_id|user_id|session_id|task_id|workspace_id|"
    r"source_type|doc_id|document_id|attachment_id|persona_id|"
    r"resource_id|source_unit_id|source_revision|"
    r"working_dir|path|"
    r"data_version|retrieval_data_version|retrieval_method|"
    r"top_k|candidate_limit|context_token_limit|fallback"
    r")\s*=",
    re.IGNORECASE,
)


class QueryGuard:
    """对一个完整 ``QueryProposal`` 执行无副作用、fail-closed 校验。"""

    def __init__(
        self,
        config: QueryGuardConfig | None = None,
        *,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        self._config = config or QueryGuardConfig()
        if count_tokens is not None and not callable(count_tokens):
            raise TypeError("count_tokens must be callable when provided")
        self._count_tokens = count_tokens

    @property
    def max_queries(self) -> int:
        """调用方可以预先声明、而不是事后强制执行的边界。

        拒绝超限提案是正确响应，但若提案方从未获知限制，用拒绝来传达限制并不理想。
        """

        return self._config.max_queries

    @property
    def max_query_tokens(self) -> int:
        return self._config.max_query_tokens

    def validate(self, proposal: QueryProposal) -> tuple[str, ...]:
        queries = tuple(query.strip() for query in proposal.queries)
        if not queries:
            raise QueryGuardError(
                "query proposal must include at least one query",
                code=QueryGuardErrorCode.EMPTY_PROPOSAL,
            )
        if len(queries) > self._config.max_queries:
            raise QueryGuardError(
                "query proposal exceeds resource query limit",
                code=QueryGuardErrorCode.TOO_MANY_QUERIES,
            )
        # 经过唯一允许的规范化步骤后仍有重复项，会使整个提案无效。不要静默去重或凭空
        # 创造回退方案。
        if len(queries) != len(set(queries)):
            raise QueryGuardError(
                "query proposal contains duplicate queries",
                code=QueryGuardErrorCode.DUPLICATE_QUERY,
            )
        for query in queries:
            if not query:
                raise QueryGuardError(
                    "query proposal contains an empty query",
                    code=QueryGuardErrorCode.EMPTY_QUERY,
                )
            if len(query) > self._config.max_query_characters:
                raise QueryGuardError(
                    "query proposal exceeds character limit",
                    code=QueryGuardErrorCode.QUERY_TOO_LONG,
                )
            if self._count_tokens is not None:
                try:
                    token_count = self._count_tokens(query)
                except Exception as exc:
                    raise QueryGuardError(
                        "query token count is unavailable",
                        code=QueryGuardErrorCode.QUERY_TOKEN_COUNT_UNAVAILABLE,
                    ) from exc
                if (
                    isinstance(token_count, bool)
                    or not isinstance(token_count, int)
                    or token_count <= 0
                ):
                    raise QueryGuardError(
                        "query token count is unavailable",
                        code=QueryGuardErrorCode.QUERY_TOKEN_COUNT_UNAVAILABLE,
                    )
                if token_count > self._config.max_query_tokens:
                    raise QueryGuardError(
                        "query proposal exceeds retrieval-token limit",
                        code=QueryGuardErrorCode.QUERY_TOO_MANY_TOKENS,
                    )
            if any(ord(character) < 32 and character not in "\n\t" for character in query):
                raise QueryGuardError(
                    "query proposal contains control characters",
                    code=QueryGuardErrorCode.CONTROL_CHARACTER,
                )
            if _SQL_PREFIX.match(query) or _LOW_LEVEL_FILTER.search(query):
                raise QueryGuardError(
                    "query proposal contains forbidden low-level retrieval syntax",
                    code=QueryGuardErrorCode.FORBIDDEN_LOW_LEVEL_SYNTAX,
                )
        return queries
