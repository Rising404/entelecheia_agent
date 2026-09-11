"""类型化模型输出验证失败的安全有界修复反馈。

投影特意排除 Pydantic 消息、上下文及被拒输入值。只有目标契约的 JSON Schema 在精确位置
声明位置组件时，才显示该组件。这样可防止仅因同一键在契约其他位置有效，就回显模型创建的额外键。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.json_schema import GenerateJsonSchema, PydanticInvalidForJsonSchema
from pydantic_core import PydanticOmit, SchemaError, SchemaValidator, core_schema

from .output_repair_contracts import (
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCoverage,
    runtime_model_output_repair_issue_sort_key,
)


_MAX_REASON_BYTES = 500
_MAX_VALIDATION_ERRORS = 64
_MAX_LOCATION_PARTS = 32
_DEFAULT_FALLBACK = "The response violates the required output contract."
_REGENERATE_INSTRUCTION = (
    "Regenerate the entire response from the contract; "
    "do not patch or merge fields."
)
_SAFE_SCHEMA_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}").fullmatch
_SAFE_ERROR_TYPE = re.compile(r"[a-z0-9_.-]{1,80}").fullmatch
_UNION_LOCATION_BRANCHES = "x-repair-location-branches"

JsonSchema = Mapping[str, Any]
StructuredOutputContract = type[BaseModel] | TypeAdapter[Any]


class _RepairLocationSchema(GenerateJsonSchema):
    """只供错误定位使用的合同视图，不改变发给模型的公开 JSON Schema。

    JSON Schema 不保留 Pydantic 的 after-validator 包装标签和 callable
    discriminator 标签。标签从可信 core schema 取得，不从异常输入或键名猜测。
    """

    def generate(self, schema: Any, mode: Any = "validation") -> dict[str, Any]:
        self._location_definitions = (
            schema.get("definitions", []) if schema.get("type") == "definitions" else []
        )
        return super().generate(schema, mode=mode)

    def union_schema(self, schema: Any) -> dict[str, Any]:
        rendered = super().union_schema(schema)
        branches: dict[str, Any] = {}
        for choice in schema["choices"]:
            branch, label = choice if isinstance(choice, tuple) else (choice, None)
            try:
                if label is None:
                    label = SchemaValidator(core_schema.definitions_schema(
                        branch, self._location_definitions,
                    )).title
                if label in branches:
                    # 不同分支可有相同 Pydantic 标签，loc 无法证明选中了哪一支。
                    branches[label] = None
                    continue
                branches[label] = self.generate_inner(branch)
            except (PydanticOmit, PydanticInvalidForJsonSchema, SchemaError):
                # 无法安全获取标签时保留普通 schema；定位会显式降为非完整覆盖。
                continue
        return {**rendered, _UNION_LOCATION_BRANCHES: branches}

    def tagged_union_schema(self, schema: Any) -> dict[str, Any]:
        rendered = super().tagged_union_schema(schema)
        branches: dict[str, Any] = {}
        for label, branch in schema["choices"].items():
            try:
                branches[str(label)] = self.generate_inner(branch)
            except (PydanticOmit, PydanticInvalidForJsonSchema):
                continue
        return {**rendered, _UNION_LOCATION_BRANCHES: branches}


@dataclass(frozen=True, slots=True)
class StructuredOutputRepairIssueProjection:
    """将一个 Pydantic 失败投影为规范修复问题的有界结果。"""

    issues: tuple[RuntimeModelOutputRepairIssue, ...]
    issue_coverage: RuntimeModelOutputRepairIssueCoverage
    omitted_issue_count: int


def project_validation_error_issues(
    error: ValidationError,
    *,
    contract: StructuredOutputContract,
) -> StructuredOutputRepairIssueProjection:
    """将所有独立可见 Pydantic 错误投影为安全的规范修复问题。

    绝不复制 Pydantic 消息、上下文及被拒输入值。省略不是真实 JSON 成员的联合分支标签，
    而模型创建的额外键会缩减为最近的已知父路径。
    """

    try:
        root_schema = _contract_json_schema(contract)
        raw_errors = _validation_errors_without_payload(error)
        projected: dict[
            tuple[object, ...], RuntimeModelOutputRepairIssue
        ] = {}
        all_locations_known = True
        for item in raw_errors:
            error_type = _safe_error_type(item.get("type"))
            pointer, location_known, field_schema = _safe_json_pointer(
                item.get("loc"),
                root_schema=root_schema,
                allow_unknown_leaf=error_type == "extra_forbidden",
            )
            all_locations_known = all_locations_known and location_known
            issue = RuntimeModelOutputRepairIssue(
                category="schema",
                code=f"schema.{error_type}",
                paths=(pointer,),
                safe_explanation=_schema_issue_explanation(
                    error_type,
                    field_schema=(
                        _resolve_schema(field_schema, root_schema=root_schema)
                        if location_known else {}
                    ),
                ),
            )
            projected.setdefault(
                runtime_model_output_repair_issue_sort_key(issue),
                issue,
            )
    except Exception:
        return _fallback_issue_projection()

    if not projected:
        return _fallback_issue_projection()

    ordered = tuple(projected[key] for key in sorted(projected))
    accepted = ordered[:_MAX_VALIDATION_ERRORS]
    omitted = len(ordered) - len(accepted)
    coverage = (
        RuntimeModelOutputRepairIssueCoverage.TRUNCATED
        if omitted
        else (
            RuntimeModelOutputRepairIssueCoverage.COMPLETE
            if all_locations_known
            else RuntimeModelOutputRepairIssueCoverage.FIRST_ONLY
        )
    )
    return StructuredOutputRepairIssueProjection(
        issues=accepted,
        issue_coverage=coverage,
        omitted_issue_count=omitted,
    )


def _fallback_issue_projection() -> StructuredOutputRepairIssueProjection:
    return StructuredOutputRepairIssueProjection(
        issues=(
            RuntimeModelOutputRepairIssue(
                category="schema",
                code="schema.contract_invalid",
                paths=("",),
                safe_explanation="输出不符合目标结构化合同。",
            ),
        ),
        issue_coverage=RuntimeModelOutputRepairIssueCoverage.FIRST_ONLY,
        omitted_issue_count=0,
    )


def _schema_issue_explanation(error_type: str, *, field_schema: JsonSchema) -> str:
    constraint = _schema_constraint_explanation(error_type, field_schema=field_schema)
    if constraint is not None:
        return constraint
    if error_type == "missing":
        return "目标合同要求此位置必须存在。"
    if error_type == "extra_forbidden":
        return "该位置含有目标合同未声明的额外字段。"
    if error_type in {"literal_error", "enum"}:
        return "该位置的值不在目标合同允许的范围内。"
    if error_type in {"union_tag_invalid", "union_tag_not_found"}:
        return "该位置缺少有效的联合类型标记，或标记值不受合同支持。"
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "该位置的值类型不符合目标合同。"
    if error_type in {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "multiple_of",
        "string_too_long",
        "string_too_short",
        "too_long",
        "too_short",
    }:
        return "该位置的值超出目标合同允许的范围。"
    return "该位置不符合目标结构化合同。"


def _schema_constraint_explanation(
    error_type: str, *, field_schema: JsonSchema,
) -> str | None:
    """仅把字段 schema 的有界数值及受限模式译为说明，不读取异常 ctx/input。

    不直接拼接任意 pattern/description；无法确定的约束继续使用通用拒绝原因。
    """

    # Nullable 字段的范围约束位于唯一非 null 分支；不猜测多业务分支的约束。
    branches = field_schema.get("anyOf")
    if isinstance(branches, (list, tuple)) and len(branches) == 2:
        non_null = [branch for branch in branches
                    if isinstance(branch, Mapping) and branch.get("type") != "null"]
        if len(non_null) == 1 and any(
            isinstance(branch, Mapping) and branch.get("type") == "null" for branch in branches
        ):
            return _schema_constraint_explanation(error_type, field_schema=non_null[0])
    length_rule = {
        "too_short": ("minItems", "至少需要", "项"),
        "too_long": ("maxItems", "最多允许", "项"),
        "string_too_short": ("minLength", "至少需要", "个字符"),
        "string_too_long": ("maxLength", "最多允许", "个字符"),
    }.get(error_type)
    if length_rule is not None:
        key, relation, unit = length_rule
        bound = field_schema.get(key)
        if type(bound) is int and 0 <= bound <= 1_000_000_000:
            return f"该位置{relation} {bound} {unit}。"
    if error_type == "string_pattern_mismatch":
        pattern = field_schema.get("pattern")
        match = (
            re.fullmatch(r"\^\[0-9a-f\]\{([1-9][0-9]{0,3})\}\$", pattern)
            if isinstance(pattern, str) else None
        )
        if match is not None:
            return f"该位置必须是 {int(match[1])} 位小写十六进制字符（0–9、a–f）。"
    return None


def safe_validation_error_reason(
    error: ValidationError,
    *,
    contract: StructuredOutputContract,
    fallback: str = _DEFAULT_FALLBACK,
) -> str:
    """返回一个 Pydantic 验证失败的安全修复反馈。

    ``fallback`` 必须是由开发者编写的静态句子。绝不能通过它传递被拒模型输入和异常文本。
    每个返回原因都会要求模型重新生成一个完整响应；被拒响应中单独有效的字段不会保留或合并。
    """

    try:
        root_schema = _contract_json_schema(contract)
        raw_errors = _validation_errors_without_payload(error)
    except Exception:
        return _fallback_reason(fallback)

    findings: set[tuple[str, str]] = set()
    for item in raw_errors[:_MAX_VALIDATION_ERRORS]:
        if not isinstance(item, Mapping):
            continue
        location = _safe_location(item.get("loc"), root_schema=root_schema)
        error_type = _safe_error_type(item.get("type"))
        findings.add((location, error_type))

    if not findings:
        return _fallback_reason(fallback)

    ordered_findings = [
        f"{location} ({error_type})"
        for location, error_type in sorted(findings)
    ]
    prefix = "Contract rejection at "
    suffix = f". {_REGENERATE_INSTRUCTION}"
    accepted: list[str] = []
    for finding in ordered_findings:
        candidate = prefix + "; ".join((*accepted, finding)) + suffix
        if len(candidate.encode("utf-8")) > _MAX_REASON_BYTES:
            break
        accepted.append(finding)

    if not accepted:
        return _fallback_reason(fallback)
    return prefix + "; ".join(accepted) + suffix


def _contract_json_schema(contract: StructuredOutputContract) -> JsonSchema:
    if isinstance(contract, TypeAdapter):
        schema = contract.json_schema(schema_generator=_RepairLocationSchema)
    elif isinstance(contract, type) and issubclass(contract, BaseModel):
        schema = contract.model_json_schema(schema_generator=_RepairLocationSchema)
    else:
        raise TypeError("contract must be a BaseModel type or TypeAdapter")
    if not isinstance(schema, Mapping):
        raise TypeError("contract JSON Schema must be an object")
    return schema


def _validation_errors_without_payload(
    error: ValidationError,
) -> list[Mapping[str, Any]]:
    try:
        errors = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    except TypeError:  # pragma: no cover - Pydantic 兼容边界
        errors = error.errors()
    return [item for item in errors if isinstance(item, Mapping)]


def _safe_location(raw_location: object, *, root_schema: JsonSchema) -> str:
    if not isinstance(raw_location, (tuple, list)):
        return "<root>"

    current_schema: JsonSchema = root_schema
    rendered: list[str] = []
    for part in raw_location[:_MAX_LOCATION_PARTS]:
        if isinstance(part, bool):
            rendered.append(_join_location_part(rendered, "<field>"))
            current_schema = {}
            continue
        if isinstance(part, int):
            rendered.append(f"[{part}]")
            current_schema = _schema_after_index(
                current_schema,
                index=part,
                root_schema=root_schema,
            ) or {}
            continue
        if not isinstance(part, str):
            rendered.append(_join_location_part(rendered, "<field>"))
            current_schema = {}
            continue

        declared, next_schema = _schema_after_string(
            current_schema,
            part=part,
            root_schema=root_schema,
        )
        safe_part = part if declared and _SAFE_SCHEMA_LABEL(part) else "<field>"
        rendered.append(_join_location_part(rendered, safe_part))
        current_schema = next_schema

    if len(raw_location) > _MAX_LOCATION_PARTS:
        rendered.append(_join_location_part(rendered, "<field>"))
    return "".join(rendered) or "<root>"


def _safe_json_pointer(
    raw_location: object, *, root_schema: JsonSchema, allow_unknown_leaf: bool = False,
) -> tuple[str, bool, JsonSchema]:
    """返回安全路径、位置可解释性及已到达的合同 schema；不另行猜测字段。"""

    if not isinstance(raw_location, (tuple, list)):
        return "", False, {}

    current_schema: JsonSchema = root_schema
    segments: list[str] = []
    location_known = len(raw_location) <= _MAX_LOCATION_PARTS
    for index, part in enumerate(raw_location[:_MAX_LOCATION_PARTS]):
        if isinstance(part, bool):
            location_known = False
            break
        if isinstance(part, int):
            next_schema = _schema_after_index(
                current_schema,
                index=part,
                root_schema=root_schema,
            )
            if next_schema is None:
                location_known = False
                break
            segments.append(str(part))
            current_schema = next_schema
            continue
        if not isinstance(part, str):
            location_known = False
            break

        step_kind, next_schema = _schema_after_json_pointer_string(
            current_schema,
            part=part,
            root_schema=root_schema,
        )
        if step_kind == "property":
            if not _SAFE_SCHEMA_LABEL(part):
                location_known = False
                break
            segments.append(_escape_json_pointer_part(part))
            current_schema = next_schema
            continue
        if step_kind == "branch":
            # 联合分支标签会出现在 Pydantic loc 中，但不是 JSON 文档的成员。
            current_schema = next_schema
            continue
        if step_kind == "ambiguous":
            location_known = False
            break
        # 剩余位置以模型创建且未声明的键开头。指向已知父级，而不反映该键。
        location_known = allow_unknown_leaf and index == len(raw_location) - 1
        break

    return (
        "" if not segments else "/" + "/".join(segments),
        location_known,
        current_schema,
    )


def _escape_json_pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _schema_after_json_pointer_string(
    schema: JsonSchema,
    *,
    part: str,
    root_schema: JsonSchema,
) -> tuple[str, JsonSchema]:
    candidates = _schema_candidates(schema, root_schema=root_schema)

    # 在联合边界先消费其可信标签；进入分支后才按该分支的实际字段导航。
    for candidate in candidates:
        branches = candidate.get(_UNION_LOCATION_BRANCHES)
        if isinstance(branches, Mapping) and part in branches:
            branch = branches[part]
            if isinstance(branch, Mapping):
                return "branch", branch
            return "ambiguous", {}

    for candidate in candidates:
        discriminator = candidate.get("discriminator")
        if not isinstance(discriminator, Mapping):
            continue
        mapping = discriminator.get("mapping")
        if not isinstance(mapping, Mapping) or part not in mapping:
            continue
        child = mapping[part]
        if isinstance(child, str):
            return "branch", _resolve_ref(child, root_schema=root_schema)
        if isinstance(child, Mapping):
            return "branch", child
        return "branch", {}

    for candidate in _union_branches(schema, root_schema=root_schema):
        if part in _branch_identifiers(candidate, root_schema=root_schema):
            return "branch", candidate

    for candidate in candidates:
        properties = candidate.get("properties")
        if isinstance(properties, Mapping) and part in properties:
            child = properties[part]
            return "property", child if isinstance(child, Mapping) else {}

    return "unknown", {}


def _join_location_part(rendered: list[str], part: str) -> str:
    return part if not rendered else "." + part


def _schema_after_string(
    schema: JsonSchema,
    *,
    part: str,
    root_schema: JsonSchema,
) -> tuple[bool, JsonSchema]:
    candidates = _schema_candidates(schema, root_schema=root_schema)

    for candidate in candidates:
        properties = candidate.get("properties")
        if isinstance(properties, Mapping) and part in properties:
            child = properties[part]
            return True, child if isinstance(child, Mapping) else {}

    for candidate in candidates:
        discriminator = candidate.get("discriminator")
        if not isinstance(discriminator, Mapping):
            continue
        mapping = discriminator.get("mapping")
        if not isinstance(mapping, Mapping) or part not in mapping:
            continue
        child = mapping[part]
        if isinstance(child, str):
            return True, _resolve_ref(child, root_schema=root_schema)
        if isinstance(child, Mapping):
            return True, child
        return True, {}

    for candidate in candidates:
        branches = candidate.get(_UNION_LOCATION_BRANCHES)
        if isinstance(branches, Mapping) and part in branches:
            branch = branches[part]
            if isinstance(branch, Mapping):
                return False, branch
            return False, {}

    # Pydantic 可能向 ``loc`` 添加非判别联合分支标识符。它有助于导航到之后声明的字段，
    # 但并非契约判别标签，因此必须保持脱敏。
    for candidate in _union_branches(schema, root_schema=root_schema):
        if part in _branch_identifiers(candidate, root_schema=root_schema):
            return False, candidate

    return False, {}


def _schema_after_index(
    schema: JsonSchema,
    *,
    index: int,
    root_schema: JsonSchema,
) -> JsonSchema | None:
    if index < 0:
        return None
    for candidate in _schema_candidates(schema, root_schema=root_schema):
        prefix_items = candidate.get("prefixItems")
        if (
            isinstance(prefix_items, list)
            and 0 <= index < len(prefix_items)
            and isinstance(prefix_items[index], Mapping)
        ):
            return prefix_items[index]
        items = candidate.get("items")
        if isinstance(items, Mapping):
            return items
    return None


def _schema_candidates(
    schema: JsonSchema,
    *,
    root_schema: JsonSchema,
) -> tuple[JsonSchema, ...]:
    resolved = _resolve_schema(schema, root_schema=root_schema)
    candidates: list[JsonSchema] = [resolved]
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = resolved.get(keyword)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, Mapping):
                candidates.append(_resolve_schema(branch, root_schema=root_schema))
    return tuple(candidates)


def _union_branches(
    schema: JsonSchema,
    *,
    root_schema: JsonSchema,
) -> tuple[JsonSchema, ...]:
    resolved = _resolve_schema(schema, root_schema=root_schema)
    branches: list[JsonSchema] = []
    for keyword in ("anyOf", "oneOf"):
        raw_branches = resolved.get(keyword)
        if not isinstance(raw_branches, list):
            continue
        for branch in raw_branches:
            if isinstance(branch, Mapping):
                branches.append(_resolve_schema(branch, root_schema=root_schema))
    return tuple(branches)


def _branch_identifiers(
    schema: JsonSchema,
    *,
    root_schema: JsonSchema,
) -> frozenset[str]:
    identifiers: set[str] = set()
    reference = schema.get("$ref")
    if isinstance(reference, str):
        identifiers.add(_decode_json_pointer_part(reference.rsplit("/", 1)[-1]))
    resolved = _resolve_schema(schema, root_schema=root_schema)
    title = resolved.get("title")
    if isinstance(title, str):
        identifiers.add(title)
    return frozenset(identifiers)


def _resolve_schema(
    schema: JsonSchema,
    *,
    root_schema: JsonSchema,
) -> JsonSchema:
    current = schema
    seen: set[str] = set()
    while isinstance(current.get("$ref"), str):
        reference = current["$ref"]
        if reference in seen:
            return {}
        seen.add(reference)
        current = _resolve_ref(reference, root_schema=root_schema)
        if not current:
            return {}
    return current


def _resolve_ref(reference: str, *, root_schema: JsonSchema) -> JsonSchema:
    if not reference.startswith("#/"):
        return {}
    current: object = root_schema
    for raw_part in reference[2:].split("/"):
        if not isinstance(current, Mapping):
            return {}
        current = current.get(_decode_json_pointer_part(raw_part))
    return current if isinstance(current, Mapping) else {}


def _decode_json_pointer_part(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _safe_error_type(raw_type: object) -> str:
    if isinstance(raw_type, str) and _SAFE_ERROR_TYPE(raw_type):
        return raw_type
    return "validation_error"


def _fallback_reason(fallback: str) -> str:
    normalized = " ".join(fallback.split()) if isinstance(fallback, str) else ""
    if not normalized:
        normalized = _DEFAULT_FALLBACK
    result = f"{normalized} {_REGENERATE_INSTRUCTION}"
    if len(result.encode("utf-8")) <= _MAX_REASON_BYTES:
        return result
    return f"{_DEFAULT_FALLBACK} {_REGENERATE_INSTRUCTION}"
