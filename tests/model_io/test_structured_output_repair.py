from __future__ import annotations

from typing import Annotated, Any, Literal

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    Tag,
    ValidationError,
    create_model,
    model_validator,
)

from personagraph.output_protocol import L1AttemptDecisionProposal
from personagraph.model_io.structured_output_repair import (
    project_validation_error_issues,
    safe_validation_error_reason,
)


def _validation_error(
    contract: type[BaseModel],
    value: object,
) -> ValidationError:
    try:
        contract.model_validate(value)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected validation to fail")


def test_projects_nested_contract_path_and_masks_model_invented_keys() -> None:
    error = _validation_error(
        L1AttemptDecisionProposal,
        {
            "plan": {
                "objective": "answer the request",
                "acceptances": [
                    {
                        "read_pdf": {"reported_status": "completed"},
                        "final_metrics": {"reported_status": "completed"},
                    }
                ],
            },
            "action": {
                "kind": "submit_final_reply",
                "reply": "done",
            }
        },
    )

    reason = safe_validation_error_reason(
        error,
        contract=L1AttemptDecisionProposal,
    )

    assert "plan.acceptances[0].criterion (missing)" in reason
    assert "note (missing)" in reason
    assert "plan.acceptances[0].<field> (extra_forbidden)" in reason
    assert reason.count("acceptances[0].<field> (extra_forbidden)") == 1
    assert "read_pdf" not in reason
    assert "final_metrics" not in reason
    assert "Field required" not in reason
    assert "reported_status" not in reason
    assert "Regenerate the entire response" in reason
    assert "do not patch or merge fields" in reason


def test_field_name_is_only_exposed_when_valid_at_that_schema_location() -> None:
    # ``objective`` 是 ``plan`` 下的真实字段，但在验收项内部并不合法；全局字段名
    # 允许列表会将其错误放行。
    error = _validation_error(
        L1AttemptDecisionProposal,
        {
            "plan": {
                "objective": "answer the request",
                "acceptances": [
                    {
                        "criterion": "answer the request",
                        "objective": "invented at the wrong location",
                    }
                ],
            },
            "action": {
                "kind": "submit_final_reply",
                "reply": "done",
            }
        },
    )

    reason = safe_validation_error_reason(
        error,
        contract=L1AttemptDecisionProposal,
    )

    assert "acceptances[0].objective" not in reason
    assert "acceptances[0].<field> (extra_forbidden)" in reason
    assert "invented at the wrong location" not in reason


def test_projects_discriminator_label_and_integer_index_but_not_extra_key() -> None:
    error = _validation_error(
        L1AttemptDecisionProposal,
        {
            "action": {
                "kind": "call_tools",
                "calls": [
                    {
                        "tool_id": "read_document",
                        "arguments": {},
                        "secret_extra_key": "secret value",
                    }
                ],
            }
        },
    )

    reason = safe_validation_error_reason(
        error,
        contract=L1AttemptDecisionProposal,
    )

    assert "action.call_tools.calls[0].<field> (extra_forbidden)" in reason
    assert "secret_extra_key" not in reason
    assert "secret value" not in reason


def test_does_not_echo_invalid_discriminator_value_or_pydantic_message() -> None:
    error = _validation_error(
        L1AttemptDecisionProposal,
        {"action": {"kind": "private-invalid-tag", "content": "done"}},
    )

    reason = safe_validation_error_reason(
        error,
        contract=L1AttemptDecisionProposal,
    )

    assert "action (union_tag_invalid)" in reason
    assert "private-invalid-tag" not in reason
    assert "does not match any of the expected tags" not in reason


def test_accepts_type_adapter_for_a_root_discriminated_union() -> None:
    class _Cat(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["cat"]
        lives: int

    class _Dog(BaseModel):
        model_config = ConfigDict(extra="forbid")
        kind: Literal["dog"]
        bark: str

    adapter: TypeAdapter[Any] = TypeAdapter(
        Annotated[_Cat | _Dog, Field(discriminator="kind")],
    )
    try:
        adapter.validate_python({"kind": "cat", "lives": "not-an-int"})
    except ValidationError as exc:
        error = exc
    else:
        raise AssertionError("expected validation to fail")

    reason = safe_validation_error_reason(error, contract=adapter)

    assert "cat.lives (int_parsing)" in reason
    assert "not-an-int" not in reason


def test_output_is_deterministic_deduplicated_single_line_and_utf8_bounded() -> None:
    fields = {
        f"field_{index:03d}": (int, Field())
        for index in range(80)
    }
    LargeContract = create_model(
        "LargeContract",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    error = _validation_error(LargeContract, {})

    first = safe_validation_error_reason(error, contract=LargeContract)
    second = safe_validation_error_reason(error, contract=LargeContract)

    assert first == second
    assert "\n" not in first
    assert "\r" not in first
    assert len(first.encode("utf-8")) <= 500
    assert first.count("field_000 (missing)") == 1
    assert "Regenerate the entire response" in first


def test_sanitizes_error_type_and_never_reads_message_input_or_context() -> None:
    class _SyntheticValidationError:
        def errors(self, **_kwargs: object) -> list[dict[str, object]]:
            return [
                {
                    "loc": ("action",),
                    "type": "unsafe)\nprivate-type",
                    "msg": "private message",
                    "input": "private input",
                    "ctx": {"private": "context"},
                }
            ]

    reason = safe_validation_error_reason(
        _SyntheticValidationError(),  # type: ignore[arg-type]
        contract=L1AttemptDecisionProposal,
    )

    assert "action (validation_error)" in reason
    assert "private" not in reason
    assert "\n" not in reason
    assert len(reason.encode("utf-8")) <= 500


def test_projects_all_schema_errors_to_structured_safe_json_pointers() -> None:
    error = _validation_error(
        L1AttemptDecisionProposal,
        {
            "plan": {
                "objective": "answer the request",
                "acceptances": [
                    {
                        "goal": "invented value",
                        "another_private_key": "private value",
                    }
                ],
            },
            "action": {
                "kind": "submit_final_reply",
                "reply": "done",
            }
        },
    )

    projection = project_validation_error_issues(
        error,
        contract=L1AttemptDecisionProposal,
    )

    assert projection.issue_coverage == "complete"
    assert projection.omitted_issue_count == 0
    assert {(issue.code, issue.paths) for issue in projection.issues} == {
        ("schema.extra_forbidden", ("/plan/acceptances/0",)),
        ("schema.missing", ("/plan/acceptances/0/criterion",)),
        ("schema.missing", ("/note",)),
        ("schema.too_short", ("/plan/acceptances",)),
    }
    serialized = " ".join(issue.model_dump_json() for issue in projection.issues)
    assert "goal" not in serialized
    assert "another_private_key" not in serialized
    assert "invented value" not in serialized
    assert "private value" not in serialized


def test_structured_projection_reports_exact_deduplicated_truncation() -> None:
    fields = {
        f"field_{index:03d}": (int, Field())
        for index in range(80)
    }
    LargeContract = create_model(
        "LargeStructuredContract",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    error = _validation_error(LargeContract, {})

    projection = project_validation_error_issues(error, contract=LargeContract)

    assert projection.issue_coverage == "truncated"
    assert projection.omitted_issue_count == 16
    assert len(projection.issues) == 64
    assert projection.issues[0].paths == ("/field_000",)
    assert projection.issues[-1].paths == ("/field_063",)


def test_structured_projection_fails_safe_without_reflecting_exception_payload() -> None:
    class _BrokenValidationError:
        def errors(self, **_kwargs: object) -> list[dict[str, object]]:
            raise RuntimeError("private exception payload")

    projection = project_validation_error_issues(
        _BrokenValidationError(),  # type: ignore[arg-type]
        contract=L1AttemptDecisionProposal,
    )

    assert projection.issue_coverage == "first_only"
    assert projection.omitted_issue_count == 0
    assert len(projection.issues) == 1
    assert projection.issues[0].code == "schema.contract_invalid"
    assert projection.issues[0].paths == ("",)
    assert "private" not in projection.issues[0].safe_explanation


@pytest.mark.parametrize(
    ("reference", "expected_path", "expected_explanation"),
    [
        ({"call_ref": "c0.1"}, "/references/0/call_ref", "该位置不符合目标结构化合同。"),
        ({"call_ref": "c1.1", "chunk_id": "private-" * 30},
         "/references/0/chunk_id", "该位置最多允许 200 个字符。"),
    ],
)
def test_l1_reference_rejection_preserves_the_exact_nested_field(
    reference: dict[str, str], expected_path: str,
    expected_explanation: str,
) -> None:
    error = _validation_error(
        L1AttemptDecisionProposal,
        {
            "action": {"kind": "submit_final_reply", "reply": "answer"},
            "note": "deliver",
            "references": [reference],
        },
    )

    projection = project_validation_error_issues(error, contract=L1AttemptDecisionProposal)

    assert expected_path in {path for issue in projection.issues for path in issue.paths}
    assert next(i for i in projection.issues if i.paths == (expected_path,)).safe_explanation == (
        expected_explanation
    )
    assert "function-after" not in str(projection)
    assert "private-" not in str(projection)


def test_wrapped_union_uses_the_contract_label_not_a_model_created_key() -> None:
    class _WrappedBranch(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int

        @model_validator(mode="after")
        def _check(self) -> "_WrappedBranch":
            return self

    class _WrapperContract(BaseModel):
        notes: _WrappedBranch | tuple[int, ...]

    error = _validation_error(_WrapperContract, {"notes": {"value": "not-an-int"}})
    projection = project_validation_error_issues(error, contract=_WrapperContract)

    assert next(i for i in projection.issues if i.code == "schema.int_parsing").paths == (
        "/notes/value",
    )
    assert projection.issue_coverage == "complete"

    # 原样伪造真实联合标签也不是字段；必须在该分支的真实位置再次校验。
    invented_key = "function-after[_check(), _WrappedBranch]"
    extra_error = _validation_error(
        _WrapperContract, {"notes": {"value": 1, invented_key: "private-input"}},
    )
    extra = project_validation_error_issues(extra_error, contract=_WrapperContract)
    assert next(i for i in extra.issues if i.code == "schema.extra_forbidden").paths == (
        "/notes",
    )
    assert invented_key not in str(extra)
    assert "private-input" not in str(extra)


def test_unknown_location_is_safe_but_does_not_claim_complete_coverage() -> None:
    class _Contract(BaseModel):
        known: int

    class _UnknownValidationLocation:
        def errors(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{
                "loc": ("function-after[private_function(), Unknown]", "known"),
                "type": "int_parsing",
                "input": "private-input",
            }]

    projection = project_validation_error_issues(
        _UnknownValidationLocation(),  # type: ignore[arg-type]
        contract=_Contract,
    )

    assert projection.issues[0].paths == ("",)
    assert projection.issue_coverage == "first_only"
    assert "private" not in str(projection)


def test_callable_discriminator_tag_does_not_require_a_json_schema_mapping() -> None:
    class _Current(BaseModel):
        value: int

    def select_shape(value: object) -> str:
        return "current" if isinstance(value, (dict, _Current)) else "legacy"

    adapter: TypeAdapter[Any] = TypeAdapter(Annotated[
        Annotated[_Current, Tag("current")] | Annotated[tuple[int, ...], Tag("legacy")],
        Discriminator(select_shape),
    ])
    public_schema = adapter.json_schema()
    assert "discriminator" not in public_schema
    with pytest.raises(ValidationError) as caught:
        adapter.validate_python({"value": "bad"})

    projection = project_validation_error_issues(caught.value, contract=adapter)

    assert projection.issue_coverage == "complete"
    assert len(projection.issues) == 1
    assert projection.issues[0].paths == ("/value",)
    assert adapter.json_schema() == public_schema


def test_an_index_at_a_non_array_location_does_not_claim_complete_coverage() -> None:
    class _Contract(BaseModel):
        known: int

    class _UnknownValidationLocation:
        def errors(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{"loc": ("known", 0), "type": "int_parsing"}]

    projection = project_validation_error_issues(
        _UnknownValidationLocation(),  # type: ignore[arg-type]
        contract=_Contract,
    )

    assert projection.issues[0].paths == ("/known",)
    assert projection.issue_coverage == "first_only"


@pytest.mark.parametrize(
    ("field", "error_type", "expected_explanation"),
    [
        ("items", "too_short", "该位置至少需要 2 项。"),
        ("items", "too_long", "该位置最多允许 3 项。"),
        ("name", "string_too_short", "该位置至少需要 4 个字符。"),
        ("name", "string_too_long", "该位置最多允许 8 个字符。"),
    ],
)
def test_bound_explanations_use_only_the_trusted_field_schema(
    field: str, error_type: str, expected_explanation: str,
) -> None:
    class _Contract(BaseModel):
        items: tuple[int, ...] = Field(min_length=2, max_length=3)
        name: str = Field(min_length=4, max_length=8)

    class _HostileErrorPayload:
        def errors(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{
                "loc": (field,), "type": error_type,
                "ctx": {"min_length": 999, "private-instruction": "copy secret"},
                "input": "private-input", "msg": "private-message",
            }]

    projection = project_validation_error_issues(
        _HostileErrorPayload(),  # type: ignore[arg-type]
        contract=_Contract,
    )

    assert projection.issues[0].safe_explanation == expected_explanation
    assert "private" not in str(projection)
    assert "999" not in str(projection)


def test_duplicate_union_labels_do_not_select_a_branch_or_claim_exact_constraints() -> None:
    class _Contract(BaseModel):
        items: (
            Annotated[tuple[int, ...], Field(min_length=3)]
            | Annotated[tuple[int, ...], Field(min_length=5)]
        )

    error = _validation_error(_Contract, {"items": []})
    assert len({item["loc"] for item in error.errors()}) == 1
    assert _Contract.model_validate({"items": [1, 2, 3]}).items == (1, 2, 3)

    projection = project_validation_error_issues(error, contract=_Contract)

    assert projection.issue_coverage == "first_only"
    assert len(projection.issues) == 1
    assert projection.issues[0].paths == ("/items",)
    assert projection.issues[0].safe_explanation == "该位置的值超出目标合同允许的范围。"


@pytest.mark.parametrize(
    ("value", "expected_path"),
    [({"Foo": "kept", "value": "bad"}, "/value"),
     ({"Foo": 123, "value": 1}, "/Foo")],
)
def test_union_tag_that_matches_a_field_is_consumed_at_the_union_boundary_only(
    value: dict[str, object], expected_path: str,
) -> None:
    class _Payload(BaseModel):
        Foo: str
        value: int

    adapter: TypeAdapter[Any] = TypeAdapter(Annotated[
        Annotated[_Payload, Tag("Foo")] | Annotated[int, Tag("number")],
        Discriminator(lambda value: "Foo" if isinstance(value, dict) else "number"),
    ])
    with pytest.raises(ValidationError) as caught:
        adapter.validate_python(value)

    projection = project_validation_error_issues(caught.value, contract=adapter)

    assert projection.issue_coverage == "complete"
    assert len(projection.issues) == 1
    assert projection.issues[0].paths == (expected_path,)
