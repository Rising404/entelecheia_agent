"""Tool Platform 的 Draft 2020-12 JSON Schema 编译。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker, SchemaError

from .contracts import ToolError, thaw_json


class SchemaCompilationError(ValueError):
    pass


@dataclass(frozen=True)
class SchemaViolation:
    path: tuple[str | int, ...]
    validator: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": list(self.path), "validator": self.validator, "message": self.message}


class SchemaValidationError(ValueError):
    def __init__(self, role: str, violations: tuple[SchemaViolation, ...]) -> None:
        self.role = role
        self.violations = violations
        super().__init__(f"{role} schema validation failed")

    def to_tool_error(self, *, code: str) -> ToolError:
        return ToolError(
            code,
            f"Tool {self.role} does not satisfy its JSON Schema.",
            {"violations": [violation.to_dict() for violation in self.violations]},
        )


@dataclass(frozen=True)
class CompiledJsonSchema:
    role: str
    schema: Mapping[str, Any]
    validator: Draft202012Validator

    def validate(self, instance: Any) -> None:
        violations = tuple(
            SchemaViolation(tuple(error.absolute_path), error.validator, error.message)
            for error in sorted(self.validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
        )
        if violations:
            raise SchemaValidationError(self.role, violations)


def normalize_json(value: Any, *, role: str = "input") -> Any:
    """复制 JSON 形态的值，并在验证前拒绝仅适用于 Python 的载荷。"""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(
            role,
            (SchemaViolation((), "json", "Tool values must be JSON serializable."),),
        ) from exc


class ToolSchemaCompiler:
    """具有确定性缓存键的应用级编译器。"""

    def compile(self, schema: Mapping[str, Any], *, role: str) -> CompiledJsonSchema:
        plain_schema = thaw_json(schema)
        if not isinstance(plain_schema, dict):
            raise SchemaCompilationError(f"{role} schema must be a JSON object")
        if plain_schema.get("type") not in (None, "object"):
            raise SchemaCompilationError(f"{role} schema root must declare type 'object'")
        canonical = json.dumps(plain_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return _compile_cached(canonical, role)

    def validate_input(self, schema: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_json(dict(payload), role="input")
        if not isinstance(normalized, dict):  # 防御性检查：上方的 dict(payload) 本应保证这一点。
            raise SchemaValidationError("input", (SchemaViolation((), "type", "Tool input must be an object."),))
        self.compile(schema, role="input").validate(normalized)
        return normalized

    def validate_output(self, schema: Mapping[str, Any], result: Any) -> dict[str, Any]:
        # 处理器结果到达此边界前，会临时包装在不可变 ExecutionOutcome 中。
        # 此处恢复为普通 JSON 形态以便序列化/模式验证；最终结果会再次将其冻结。
        normalized = normalize_json(thaw_json(result), role="output")
        if not isinstance(normalized, dict):
            raise SchemaValidationError("output", (SchemaViolation((), "type", "Tool output must be an object."),))
        self.compile(schema, role="output").validate(normalized)
        return normalized


@lru_cache(maxsize=512)
def _compile_cached(canonical_schema: str, role: str) -> CompiledJsonSchema:
    schema = json.loads(canonical_schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise SchemaCompilationError(f"invalid {role} JSON Schema: {exc.message}") from exc
    return CompiledJsonSchema(
        role=role,
        schema=schema,
        validator=Draft202012Validator(schema, format_checker=FormatChecker()),
    )
