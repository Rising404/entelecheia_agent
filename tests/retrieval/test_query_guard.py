from __future__ import annotations

import pytest

from personagraph.retrieval.contracts import QueryProposal
from personagraph.retrieval.query_guard import (
    QueryGuard,
    QueryGuardConfig,
    QueryGuardError,
    QueryGuardErrorCode,
)


@pytest.mark.parametrize(
    ("proposal", "config", "expected_code"),
    [
        (QueryProposal(()), None, QueryGuardErrorCode.EMPTY_PROPOSAL),
        (
            QueryProposal(("first", "second")),
            QueryGuardConfig(max_queries=1),
            QueryGuardErrorCode.TOO_MANY_QUERIES,
        ),
        (QueryProposal(("   ",)), None, QueryGuardErrorCode.EMPTY_QUERY),
        (
            QueryProposal(("too long",)),
            QueryGuardConfig(max_query_characters=3),
            QueryGuardErrorCode.QUERY_TOO_LONG,
        ),
        (QueryProposal(("unsafe\x01",)), None, QueryGuardErrorCode.CONTROL_CHARACTER),
        (
            QueryProposal(("session_id = 'untrusted'",)),
            None,
            QueryGuardErrorCode.FORBIDDEN_LOW_LEVEL_SYNTAX,
        ),
    ],
)
def test_query_guard_rejections_expose_typed_stable_codes(
    proposal: QueryProposal,
    config: QueryGuardConfig | None,
    expected_code: QueryGuardErrorCode,
):
    with pytest.raises(QueryGuardError) as raised:
        QueryGuard(config).validate(proposal)

    error = raised.value
    assert type(error.code) is QueryGuardErrorCode
    assert isinstance(error.code, str)
    assert error.code is expected_code
    assert error.code.value == expected_code.value


def test_query_guard_rejects_trim_normalized_exact_duplicate_as_a_whole_proposal():
    proposal = QueryProposal(("北桥项目风险", "  北桥项目风险\t"))

    with pytest.raises(QueryGuardError) as raised:
        QueryGuard().validate(proposal)

    assert raised.value.code is QueryGuardErrorCode.DUPLICATE_QUERY


def test_query_guard_error_rejects_untyped_code_values():
    with pytest.raises(TypeError, match="QueryGuardErrorCode"):
        QueryGuardError("query proposal rejected", code="untrusted-value")  # type: ignore[arg-type]


def test_query_guard_uses_injected_retrieval_tokenizer_for_query_limit():
    observed: list[str] = []

    def count_tokens(value: str) -> int:
        observed.append(value)
        return len(value.split())

    guard = QueryGuard(
        QueryGuardConfig(max_query_tokens=3),
        count_tokens=count_tokens,
    )

    assert guard.validate(QueryProposal(("one two three",))) == ("one two three",)
    with pytest.raises(QueryGuardError) as raised:
        guard.validate(QueryProposal(("one two three four",)))

    assert raised.value.code is QueryGuardErrorCode.QUERY_TOO_MANY_TOKENS
    assert observed == ["one two three", "one two three four"]
