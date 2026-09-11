"""严格派生选集的形状、基础文件定位及有序排除关系；不加载题面或运行评测。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


DERIVED_SCHEMA_VERSION = "docbench-derived-selection-v1"
DERIVED_SELECTION_ALGORITHM = "ordered-base-case-exclusion-v1"
_TOP_LEVEL_KEYS = {
    "schema_version",
    "selection_algorithm",
    "base_manifest",
    "excluded_case_ids",
    "case_count",
    "domain_counts",
    "cases",
}


class DerivedSelectionError(ValueError):
    """仅在选集入口转换为现有 SelectionValidationError，不扩散至 runner。"""


@dataclass(frozen=True, slots=True)
class DerivedSelectionSpec:
    base_path: Path
    base_sha256: str
    excluded_case_ids: tuple[str, ...]
    case_count: int
    domain_counts: Mapping[str, int]
    cases: tuple[dict[str, Any], ...]


def parse_derived_selection(
    raw: Mapping[str, Any], *, path: Path
) -> DerivedSelectionSpec:
    """基础 manifest 只能是同目录文件，不允许路径逃逸或隐含重选题。"""

    if set(raw) != _TOP_LEVEL_KEYS:
        raise DerivedSelectionError("derived manifest keys differ from its contract")
    if raw["schema_version"] != DERIVED_SCHEMA_VERSION:
        raise DerivedSelectionError("unsupported derived selection schema")
    if raw["selection_algorithm"] != DERIVED_SELECTION_ALGORITHM:
        raise DerivedSelectionError("unsupported derived selection algorithm")
    base = raw["base_manifest"]
    if not isinstance(base, dict) or set(base) != {"path", "sha256"}:
        raise DerivedSelectionError(
            "derived base_manifest must contain path and sha256"
        )
    filename = base["path"]
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or "\\" in filename
        or Path(filename).suffix != ".json"
    ):
        raise DerivedSelectionError(
            "derived base path must be a same-directory JSON filename"
        )
    directory = path.resolve().parent
    base_path = (directory / filename).resolve()
    if base_path.parent != directory or base_path == path.resolve():
        raise DerivedSelectionError(
            "derived base path escapes its directory or refers to itself"
        )
    digest = base["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise DerivedSelectionError(
            "derived base sha256 must be a lowercase SHA256 digest"
        )
    excluded = raw["excluded_case_ids"]
    if (
        not isinstance(excluded, list)
        or not excluded
        or any(not isinstance(value, str) or not value.strip() for value in excluded)
    ):
        raise DerivedSelectionError(
            "derived excluded_case_ids must be a non-empty string list"
        )
    if len(set(excluded)) != len(excluded):
        raise DerivedSelectionError(
            "derived excluded_case_ids must not contain duplicates"
        )
    cases = raw["cases"]
    if (
        not isinstance(cases, list)
        or not cases
        or any(not isinstance(case, dict) for case in cases)
    ):
        raise DerivedSelectionError("derived cases must be a non-empty object list")
    count = raw["case_count"]
    if type(count) is not int or count != len(cases):
        raise DerivedSelectionError("derived case_count does not match cases")
    domains = raw["domain_counts"]
    if (
        not isinstance(domains, dict)
        or not domains
        or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in domains.items()
        )
    ):
        raise DerivedSelectionError(
            "derived domain_counts must contain nonnegative integer counts"
        )
    return DerivedSelectionSpec(
        base_path, digest, tuple(excluded), count, domains, tuple(cases)
    )


def validate_derived_case_binding(
    spec: DerivedSelectionSpec, *, verified_base: Mapping[str, Any]
) -> None:
    """基础清单已走原校验；子清单只能等于原序排除结果，连文件哈希也不得改写。"""

    base_cases = verified_base["cases"]
    available_ids = {case["case_id"] for case in base_cases}
    excluded_ids = set(spec.excluded_case_ids)
    if not excluded_ids <= available_ids:
        raise DerivedSelectionError("derived exclusion contains unknown base case IDs")
    expected = tuple(case for case in base_cases if case["case_id"] not in excluded_ids)
    # Python 容器相等会把 0.0/False 视作 0；文件身份必须连 JSON 字段类型一起保持。
    try:
        actual_identity = json.dumps(spec.cases, sort_keys=True, allow_nan=False)
        expected_identity = json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DerivedSelectionError(
            "derived cases must contain finite JSON identities"
        ) from exc
    if actual_identity != expected_identity:
        raise DerivedSelectionError(
            "derived case identities or order differ from the base exclusion"
        )
    if spec.case_count != len(expected):
        raise DerivedSelectionError(
            "derived case_count differs from the base exclusion"
        )
    domains = {domain: 0 for domain in verified_base["domain_counts"]}
    domains.update(Counter(case["domain"] for case in expected))
    if dict(spec.domain_counts) != domains:
        raise DerivedSelectionError(
            "derived domain_counts differ from the base exclusion"
        )
