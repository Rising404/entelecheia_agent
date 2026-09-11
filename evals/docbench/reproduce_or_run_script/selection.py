"""生成并加载正式 DocBench 运行使用的确定性无正文选集。

上游布局为 ``data/<doc_id>/<doc_id>_qa.jsonl``，且每个文档目录恰有一个 PDF。
提交到仓库的 selection manifest 只包含标识符、文件哈希和所选问题类型；问题、答案
与证据正文会在加载时从本地授权副本中关联。
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .derived_selection import (
    DERIVED_SCHEMA_VERSION,
    DerivedSelectionError,
    parse_derived_selection,
    validate_derived_case_binding,
)


SCHEMA_VERSION = "docbench-formal-selection-v1"
SELECTION_ALGORITHM = "sha256-null-delimited-rank-v1"
DEFAULT_SEED = "docbench-formal-100-v1"

BALANCED_SCHEMA_VERSION = "docbench-formal-balanced-selection-v2"
BALANCED_SELECTION_ALGORITHM = "sha256-ranked-cell-question-document-maxflow-v1"
TYPE_NORMALIZATION_VERSION = "docbench-question-type-normalization-v1"
DEFAULT_BALANCED_SEED = "docbench-formal-balanced-125-v1"
DEFAULT_MAX_QUESTIONS_PER_DOCUMENT = 1

DOMAIN_RANGES: dict[str, range] = {
    "academia": range(0, 49),
    "finance": range(49, 89),
    "government": range(89, 133),
    "laws": range(133, 179),
    "news": range(179, 229),
}
DOMAIN_ORDER = tuple(DOMAIN_RANGES)
NORMALIZED_TYPE_ORDER = ("text", "multimodal", "metadata", "unanswerable")
DEFAULT_BALANCED_DOMAIN_COUNTS = {domain: 25 for domain in DOMAIN_ORDER}
DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS = {
    "text": 50,
    "multimodal": 25,
    "metadata": 25,
    "unanswerable": 25,
}
DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS = {
    "academia": {
        "text": 5,
        "multimodal": 12,
        "metadata": 3,
        "unanswerable": 5,
    },
    "finance": {
        "text": 7,
        "multimodal": 12,
        "metadata": 3,
        "unanswerable": 3,
    },
    "government": {
        "text": 13,
        "multimodal": 0,
        "metadata": 7,
        "unanswerable": 5,
    },
    "laws": {
        "text": 12,
        "multimodal": 0,
        "metadata": 7,
        "unanswerable": 6,
    },
    "news": {
        "text": 13,
        "multimodal": 1,
        "metadata": 5,
        "unanswerable": 6,
    },
}
QUESTION_TYPES = frozenset(
    {
        "text-only",
        "multimodal-f",
        "multimodal-t",
        "multimodal",
        "meta-data",
        "una",
        "una-web",
        "unanswerable",
    }
)

_NORMALIZED_QUESTION_TYPES = {
    "text-only": "text",
    "multimodal-f": "multimodal",
    "multimodal-t": "multimodal",
    "multimodal": "multimodal",
    "meta-data": "metadata",
    "una": "unanswerable",
    "una-web": "unanswerable",
    "unanswerable": "unanswerable",
}

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "selection_algorithm",
        "seed",
        "case_count",
        "domain_counts",
        "cases",
    }
)
_CASE_KEYS = frozenset(
    {
        "case_id",
        "doc_id",
        "question_index",
        "domain",
        "question_type",
        "pdf_filename",
        "pdf_sha256",
        "qa_sha256",
    }
)
_BALANCED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "selection_algorithm",
        "type_normalization_version",
        "candidate_inventory_sha256",
        "seed",
        "case_count",
        "domain_counts",
        "normalized_type_counts",
        "domain_type_counts",
        "max_questions_per_document",
        "cases",
    }
)
_BALANCED_CASE_KEYS = frozenset(
    {*_CASE_KEYS, "normalized_question_type"}
)
_QA_KEYS = frozenset({"question", "answer", "type", "evidence"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class SelectionValidationError(ValueError):
    """selection manifest 或其绑定的本地数据发生漂移时抛出。"""


@dataclass(frozen=True, slots=True)
class DocBenchRunCase:
    case_id: str
    doc_id: int
    question_index: int
    domain: str
    question_type: str
    pdf_path: Path
    qa_path: Path
    question: str
    answer: str
    evidence: str
    pdf_sha256: str
    qa_sha256: str


@dataclass(frozen=True, slots=True)
class BalancedQuestionSelection:
    """冻结 PDF 前的一项确定性问题级选择。"""

    case_id: str
    doc_id: int
    question_index: int
    domain: str
    question_type: str
    normalized_question_type: str


@dataclass(frozen=True, slots=True)
class LoadedSelection:
    raw: dict[str, Any]
    sha256: str
    cases: tuple[DocBenchRunCase, ...]


@dataclass(slots=True)
class _FlowEdge:
    to: tuple[object, ...]
    reverse_index: int
    capacity: int


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectionValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_loads_strict(text: str, *, source: Path | str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except SelectionValidationError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise SelectionValidationError(f"invalid JSON in {source}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_digest(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _domain_candidates(domain: str) -> tuple[int, ...]:
    candidates = tuple(DOMAIN_RANGES[domain])
    if domain == "academia":
        candidates = tuple(doc_id for doc_id in candidates if doc_id != 0)
    return candidates


def select_document_ids(
    *,
    seed: str = DEFAULT_SEED,
    domain_counts: Mapping[str, int] | None = None,
) -> dict[str, tuple[int, ...]]:
    """返回规范的按哈希排序文档选集。

    结果按 domain 顺序和文档数字顺序序列化。采用哈希排序而非语言运行时 PRNG，
    使选集具备可移植性。``doc_id=0`` 始终排除，因为它属于先前的在线探针。
    """

    normalized_seed = _validated_seed(seed)
    counts = _normalized_domain_counts(domain_counts)
    selected: dict[str, tuple[int, ...]] = {}
    for domain in DOMAIN_ORDER:
        candidates = _domain_candidates(domain)
        count = counts[domain]
        if count > len(candidates):
            raise SelectionValidationError(
                f"domain_counts[{domain!r}] exceeds {len(candidates)} candidates"
            )
        ranked = sorted(
            candidates,
            key=lambda doc_id: (
                _rank_digest(normalized_seed, domain, doc_id),
                doc_id,
            ),
        )
        selected[domain] = tuple(sorted(ranked[:count]))
    return selected


def _validated_seed(seed: object) -> str:
    if not isinstance(seed, str) or not seed.strip() or seed != seed.strip():
        raise SelectionValidationError("seed must be a non-empty trimmed string")
    return seed


def _normalized_domain_counts(
    domain_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    if domain_counts is None:
        return {domain: 20 for domain in DOMAIN_ORDER}
    if not isinstance(domain_counts, Mapping):
        raise SelectionValidationError("domain_counts must be an object")
    if set(domain_counts) != set(DOMAIN_ORDER):
        raise SelectionValidationError(
            "domain_counts must contain exactly: " + ", ".join(DOMAIN_ORDER)
        )
    normalized: dict[str, int] = {}
    for domain in DOMAIN_ORDER:
        value = domain_counts[domain]
        if type(value) is not int or value < 0:
            raise SelectionValidationError(
                f"domain_counts[{domain!r}] must be a non-negative integer"
            )
        normalized[domain] = value
    if not any(normalized.values()):
        raise SelectionValidationError("a selection must contain at least one case")
    return normalized


def _normalized_balanced_domain_counts(
    domain_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    source = (
        DEFAULT_BALANCED_DOMAIN_COUNTS
        if domain_counts is None
        else domain_counts
    )
    if not isinstance(source, Mapping):
        raise SelectionValidationError("domain_counts must be an object")
    if set(source) != set(DOMAIN_ORDER):
        raise SelectionValidationError(
            "domain_counts must contain exactly: " + ", ".join(DOMAIN_ORDER)
        )
    normalized: dict[str, int] = {}
    for domain in DOMAIN_ORDER:
        value = source[domain]
        if type(value) is not int or value < 0:
            raise SelectionValidationError(
                f"domain_counts[{domain!r}] must be a non-negative integer"
            )
        normalized[domain] = value
    if not any(normalized.values()):
        raise SelectionValidationError("a selection must contain at least one case")
    return normalized


def _normalized_balanced_type_counts(
    normalized_type_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    source = (
        DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS
        if normalized_type_counts is None
        else normalized_type_counts
    )
    if not isinstance(source, Mapping):
        raise SelectionValidationError("normalized_type_counts must be an object")
    if set(source) != set(NORMALIZED_TYPE_ORDER):
        raise SelectionValidationError(
            "normalized_type_counts must contain exactly: "
            + ", ".join(NORMALIZED_TYPE_ORDER)
        )
    normalized: dict[str, int] = {}
    for question_type in NORMALIZED_TYPE_ORDER:
        value = source[question_type]
        if type(value) is not int or value < 0:
            raise SelectionValidationError(
                "normalized_type_counts"
                f"[{question_type!r}] must be a non-negative integer"
            )
        normalized[question_type] = value
    if not any(normalized.values()):
        raise SelectionValidationError("a selection must contain at least one case")
    return normalized


def _validated_document_question_cap(value: object) -> int:
    if type(value) is not int or value < 1:
        raise SelectionValidationError(
            "max_questions_per_document must be a positive integer"
        )
    return value


def normalize_question_type(
    raw_type: str,
    *,
    version: str = TYPE_NORMALIZATION_VERSION,
) -> str:
    """将一个冻结的上游标签映射到均衡的四类型分类体系。"""

    if version != TYPE_NORMALIZATION_VERSION:
        raise SelectionValidationError(
            f"unsupported type normalization version: {version!r}"
        )
    if not isinstance(raw_type, str) or raw_type not in _NORMALIZED_QUESTION_TYPES:
        raise SelectionValidationError(
            f"unsupported question type for normalization: {raw_type!r}"
        )
    return _NORMALIZED_QUESTION_TYPES[raw_type]


def _document_paths(data_root: Path, doc_id: int) -> tuple[Path, Path]:
    document_root = data_root / str(doc_id)
    if not document_root.is_dir():
        raise SelectionValidationError(f"missing document directory: {document_root}")
    qa_path = document_root / f"{doc_id}_qa.jsonl"
    if not qa_path.is_file():
        raise SelectionValidationError(f"missing question file: {qa_path}")
    pdf_paths = sorted(
        path for path in document_root.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if len(pdf_paths) != 1:
        raise SelectionValidationError(
            f"document {doc_id} must contain exactly one PDF; found {len(pdf_paths)}"
        )
    return pdf_paths[0], qa_path


def _read_qa_records(qa_path: Path) -> tuple[dict[str, str], ...]:
    try:
        lines = qa_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SelectionValidationError(f"cannot read question file {qa_path}: {exc}") from exc
    if not lines:
        raise SelectionValidationError(f"question file is empty: {qa_path}")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise SelectionValidationError(
                f"blank JSONL record in {qa_path} at line {line_number}"
            )
        value = _json_loads_strict(line, source=f"{qa_path}:{line_number}")
        if not isinstance(value, dict) or set(value) != _QA_KEYS:
            raise SelectionValidationError(
                f"question record {qa_path}:{line_number} must contain exactly "
                f"{sorted(_QA_KEYS)}"
            )
        if any(not isinstance(value[key], str) for key in _QA_KEYS):
            raise SelectionValidationError(
                f"question record {qa_path}:{line_number} fields must be strings"
            )
        if not value["question"].strip():
            raise SelectionValidationError(
                f"question record {qa_path}:{line_number} has an empty question"
            )
        if value["type"] not in QUESTION_TYPES:
            raise SelectionValidationError(
                f"unsupported question type {value['type']!r} in {qa_path}:{line_number}"
            )
        records.append(value)
    return tuple(records)


def _domain_for_document(doc_id: int) -> str | None:
    for domain in DOMAIN_ORDER:
        if doc_id in DOMAIN_RANGES[domain]:
            return domain
    return None


def _eligible_balanced_qa_sources(
    data_root: Path,
) -> tuple[tuple[str, int, Path], ...]:
    root = Path(data_root)
    if not root.is_dir():
        raise SelectionValidationError(f"missing data root: {root}")
    sources: list[tuple[str, int, Path]] = []
    for document_root in root.iterdir():
        if not document_root.is_dir() or not document_root.name.isdigit():
            continue
        doc_id = int(document_root.name)
        if str(doc_id) != document_root.name:
            continue
        domain = _domain_for_document(doc_id)
        if domain is None:
            continue
        qa_path = document_root / f"{doc_id}_qa.jsonl"
        if qa_path.is_file():
            sources.append((domain, doc_id, qa_path))
    domain_position = {
        domain: index for index, domain in enumerate(DOMAIN_ORDER)
    }
    sources.sort(key=lambda item: (domain_position[item[0]], item[1]))
    return tuple(sources)


def _candidate_inventory_sha256(data_root: Path) -> str:
    """为完整规范的 QA 候选来源清单计算哈希。"""

    parts: list[object] = [
        "docbench-balanced-candidate-inventory-v1",
        TYPE_NORMALIZATION_VERSION,
    ]
    for domain, doc_id, qa_path in _eligible_balanced_qa_sources(data_root):
        try:
            qa_sha256 = _sha256_file(qa_path)
        except OSError as exc:
            raise SelectionValidationError(
                f"cannot hash candidate question file {qa_path}: {exc}"
            ) from exc
        parts.extend((domain, doc_id, qa_path.name, qa_sha256))
    return _rank_digest(*parts)


def _discover_balanced_candidates(
    data_root: Path,
) -> tuple[BalancedQuestionSelection, ...]:
    """读取符合条件的 DocBench 范围内全部规范 QA 文件。

    候选发现刻意不检查 PDF 是否存在。只有被选中的问题对会跨过最终冻结边界，
    届时其 PDF 文件名和哈希才成为 manifest 的必要权威信息。
    """

    candidates: list[BalancedQuestionSelection] = []
    for domain, doc_id, qa_path in _eligible_balanced_qa_sources(data_root):
        records = _read_qa_records(qa_path)
        for question_index, record in enumerate(records):
            raw_type = record["type"]
            candidates.append(BalancedQuestionSelection(
                case_id=f"docbench:{doc_id}:{question_index}",
                doc_id=doc_id,
                question_index=question_index,
                domain=domain,
                question_type=raw_type,
                normalized_question_type=normalize_question_type(raw_type),
            ))
    if not candidates:
        raise SelectionValidationError(
            "balanced selection found no readable canonical QA candidates"
        )
    return tuple(candidates)


def _add_flow_edge(
    graph: dict[tuple[object, ...], list[_FlowEdge]],
    source: tuple[object, ...],
    target: tuple[object, ...],
    capacity: int,
) -> _FlowEdge:
    source_edges = graph.setdefault(source, [])
    target_edges = graph.setdefault(target, [])
    forward = _FlowEdge(
        to=target,
        reverse_index=len(target_edges),
        capacity=capacity,
    )
    reverse = _FlowEdge(
        to=source,
        reverse_index=len(source_edges),
        capacity=0,
    )
    source_edges.append(forward)
    target_edges.append(reverse)
    return forward


def _maximum_flow(
    graph: dict[tuple[object, ...], list[_FlowEdge]],
    source: tuple[object, ...],
    sink: tuple[object, ...],
    *,
    target: int,
) -> int:
    """为有界图返回一个确定性的整数 Dinic 流。"""

    total = 0
    while total < target:
        levels: dict[tuple[object, ...], int] = {source: 0}
        pending = deque([source])
        while pending:
            node = pending.popleft()
            for edge in graph.get(node, ()):
                if edge.capacity <= 0 or edge.to in levels:
                    continue
                levels[edge.to] = levels[node] + 1
                pending.append(edge.to)
        if sink not in levels:
            return total
        positions = {node: 0 for node in graph}

        def send(node: tuple[object, ...], available: int) -> int:
            if node == sink:
                return available
            edges = graph.get(node, [])
            while positions[node] < len(edges):
                edge_index = positions[node]
                edge = edges[edge_index]
                if (
                    edge.capacity > 0
                    and levels.get(edge.to) == levels[node] + 1
                ):
                    delivered = send(
                        edge.to,
                        min(available, edge.capacity),
                    )
                    if delivered:
                        edge.capacity -= delivered
                        graph[edge.to][edge.reverse_index].capacity += delivered
                        return delivered
                positions[node] += 1
            return 0

        while total < target:
            delivered = send(source, target - total)
            if delivered == 0:
                break
            total += delivered
    return total


def _balanced_candidate_rank(
    candidates: Sequence[BalancedQuestionSelection],
    *,
    seed: str,
) -> dict[BalancedQuestionSelection, tuple[str, int, int]]:
    return {
        candidate: (
            _rank_digest(
                seed,
                "balanced-question",
                candidate.domain,
                candidate.normalized_question_type,
                candidate.doc_id,
                candidate.question_index,
            ),
            candidate.doc_id,
            candidate.question_index,
        )
        for candidate in candidates
    }


def _canonical_balanced_selection(
    selection_edges: Mapping[BalancedQuestionSelection, _FlowEdge],
    *,
    target: int,
) -> tuple[BalancedQuestionSelection, ...]:
    domain_position = {
        domain: index for index, domain in enumerate(DOMAIN_ORDER)
    }
    selected = [
        candidate
        for candidate, edge in selection_edges.items()
        if edge.capacity == 0
    ]
    selected.sort(
        key=lambda candidate: (
            domain_position[candidate.domain],
            candidate.doc_id,
            candidate.question_index,
        )
    )
    if len(selected) != target or len({item.case_id for item in selected}) != target:
        raise AssertionError("balanced max-flow result lost unique question authority")
    return tuple(selected)


def _derive_domain_type_counts(
    candidates: Sequence[BalancedQuestionSelection],
    *,
    seed: str,
    domain_counts: Mapping[str, int],
    normalized_type_counts: Mapping[str, int],
    max_questions_per_document: int,
) -> dict[str, dict[str, int]]:
    """为旧版边际配额调用派生确定且可行的交叉表。"""

    target = sum(domain_counts.values())
    rank = _balanced_candidate_rank(candidates, seed=seed)
    grouped: dict[tuple[str, int], list[BalancedQuestionSelection]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.domain, candidate.doc_id), []).append(
            candidate
        )

    graph: dict[tuple[object, ...], list[_FlowEdge]] = {}
    source: tuple[object, ...] = ("source",)
    sink: tuple[object, ...] = ("sink",)
    for domain in DOMAIN_ORDER:
        _add_flow_edge(
            graph,
            source,
            ("domain", domain),
            domain_counts[domain],
        )
    for question_type in NORMALIZED_TYPE_ORDER:
        _add_flow_edge(
            graph,
            ("type", question_type),
            sink,
            normalized_type_counts[question_type],
        )

    selection_edges: dict[BalancedQuestionSelection, _FlowEdge] = {}
    for domain in DOMAIN_ORDER:
        document_keys = sorted(
            (key for key in grouped if key[0] == domain),
            key=lambda key: (
                min(rank[candidate] for candidate in grouped[key]),
                key[1],
            ),
        )
        for _domain, doc_id in document_keys:
            document_node: tuple[object, ...] = ("document", doc_id)
            _add_flow_edge(
                graph,
                ("domain", domain),
                document_node,
                max_questions_per_document,
            )
            for candidate in sorted(grouped[(domain, doc_id)], key=rank.__getitem__):
                question_node: tuple[object, ...] = (
                    "question",
                    candidate.doc_id,
                    candidate.question_index,
                )
                _add_flow_edge(graph, document_node, question_node, 1)
                selection_edges[candidate] = _add_flow_edge(
                    graph,
                    question_node,
                    ("type", candidate.normalized_question_type),
                    1,
                )

    achieved = _maximum_flow(graph, source, sink, target=target)
    if achieved != target:
        raise SelectionValidationError(
            "balanced marginal quotas are infeasible under the current "
            "candidate universe and document cap: "
            f"selected {achieved}/{target}"
        )
    selected = _canonical_balanced_selection(
        selection_edges,
        target=target,
    )
    derived = _empty_domain_type_counts()
    for candidate in selected:
        derived[candidate.domain][candidate.normalized_question_type] += 1
    return derived


def _select_balanced_questions_for_matrix(
    candidates: Sequence[BalancedQuestionSelection],
    *,
    seed: str,
    domain_type_counts: Mapping[str, Mapping[str, int]],
    max_questions_per_document: int,
) -> tuple[BalancedQuestionSelection, ...]:
    """求解精确单元格配额，同时对每份文档施加共享容量限制。"""

    target = sum(
        count
        for row in domain_type_counts.values()
        for count in row.values()
    )
    rank = _balanced_candidate_rank(candidates, seed=seed)
    grouped: dict[tuple[str, str], list[BalancedQuestionSelection]] = {}
    document_ids: set[int] = set()
    for candidate in candidates:
        grouped.setdefault(
            (candidate.domain, candidate.normalized_question_type),
            [],
        ).append(candidate)
        document_ids.add(candidate.doc_id)

    graph: dict[tuple[object, ...], list[_FlowEdge]] = {}
    source: tuple[object, ...] = ("source",)
    sink: tuple[object, ...] = ("sink",)
    for doc_id in sorted(document_ids):
        _add_flow_edge(
            graph,
            ("document", doc_id),
            sink,
            max_questions_per_document,
        )

    selection_edges: dict[BalancedQuestionSelection, _FlowEdge] = {}
    for domain in DOMAIN_ORDER:
        for question_type in NORMALIZED_TYPE_ORDER:
            cell_node: tuple[object, ...] = ("cell", domain, question_type)
            _add_flow_edge(
                graph,
                source,
                cell_node,
                domain_type_counts[domain][question_type],
            )
            for candidate in sorted(
                grouped.get((domain, question_type), ()),
                key=rank.__getitem__,
            ):
                question_node: tuple[object, ...] = (
                    "question",
                    candidate.doc_id,
                    candidate.question_index,
                )
                selection_edges[candidate] = _add_flow_edge(
                    graph,
                    cell_node,
                    question_node,
                    1,
                )
                _add_flow_edge(
                    graph,
                    question_node,
                    ("document", candidate.doc_id),
                    1,
                )

    achieved = _maximum_flow(graph, source, sink, target=target)
    if achieved != target:
        raise SelectionValidationError(
            "balanced domain_type_counts are infeasible under the current "
            "candidate universe and document cap: "
            f"selected {achieved}/{target}"
        )
    return _canonical_balanced_selection(
        selection_edges,
        target=target,
    )


def select_balanced_questions(
    data_root: Path,
    *,
    seed: str = DEFAULT_BALANCED_SEED,
    domain_counts: Mapping[str, int] | None = None,
    normalized_type_counts: Mapping[str, int] | None = None,
    domain_type_counts: Mapping[str, Mapping[str, int]] | None = None,
    max_questions_per_document: int = DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
) -> tuple[BalancedQuestionSelection, ...]:
    """在精确单元格与文档配额约束下选择唯一问题对。

    仍支持只提供 domain/type 边际配额的调用：先派生一份确定且可行的交叉表。
    正式默认值则使用 ``DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS`` 作为冻结权威。
    """

    normalized_seed = _validated_seed(seed)
    domains = _normalized_balanced_domain_counts(domain_counts)
    question_types = _normalized_balanced_type_counts(normalized_type_counts)
    document_cap = _validated_document_question_cap(
        max_questions_per_document
    )
    target = sum(domains.values())
    if sum(question_types.values()) != target:
        raise SelectionValidationError(
            "domain_counts and normalized_type_counts must have equal totals"
        )
    candidates = _discover_balanced_candidates(Path(data_root))
    if domain_type_counts is not None:
        resolved_domain_type_counts = _normalized_domain_type_counts(
            domain_type_counts,
            domain_counts=domains,
            normalized_type_counts=question_types,
        )
    elif (
        domains == DEFAULT_BALANCED_DOMAIN_COUNTS
        and question_types == DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS
    ):
        resolved_domain_type_counts = _normalized_domain_type_counts(
            DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS,
            domain_counts=domains,
            normalized_type_counts=question_types,
        )
    else:
        resolved_domain_type_counts = _derive_domain_type_counts(
            candidates,
            seed=normalized_seed,
            domain_counts=domains,
            normalized_type_counts=question_types,
            max_questions_per_document=document_cap,
        )
    return _select_balanced_questions_for_matrix(
        candidates,
        seed=normalized_seed,
        domain_type_counts=resolved_domain_type_counts,
        max_questions_per_document=document_cap,
    )


def _select_question_index(
    *, seed: str, domain: str, doc_id: int, question_count: int
) -> int:
    if question_count <= 0:
        raise SelectionValidationError(f"document {doc_id} has no questions")
    return min(
        range(question_count),
        key=lambda index: (
            _rank_digest(seed, domain, doc_id, index),
            index,
        ),
    )


def generate_selection_manifest(
    data_root: Path,
    *,
    seed: str = DEFAULT_SEED,
    domain_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """生成绑定本地上游文件、且不含正文的 manifest。"""

    data_root = Path(data_root)
    normalized_seed = _validated_seed(seed)
    counts = _normalized_domain_counts(domain_counts)
    selected = select_document_ids(seed=normalized_seed, domain_counts=counts)
    cases: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        for doc_id in selected[domain]:
            pdf_path, qa_path = _document_paths(data_root, doc_id)
            records = _read_qa_records(qa_path)
            question_index = _select_question_index(
                seed=normalized_seed,
                domain=domain,
                doc_id=doc_id,
                question_count=len(records),
            )
            record = records[question_index]
            cases.append(
                {
                    "case_id": f"docbench:{doc_id}:{question_index}",
                    "doc_id": doc_id,
                    "question_index": question_index,
                    "domain": domain,
                    "question_type": record["type"],
                    "pdf_filename": pdf_path.name,
                    "pdf_sha256": _sha256_file(pdf_path),
                    "qa_sha256": _sha256_file(qa_path),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "seed": normalized_seed,
        "case_count": len(cases),
        "domain_counts": counts,
        "cases": cases,
    }


def write_selection_manifest(
    output_path: Path,
    *,
    data_root: Path,
    seed: str = DEFAULT_SEED,
    domain_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    manifest = generate_selection_manifest(
        data_root,
        seed=seed,
        domain_counts=domain_counts,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _empty_domain_type_counts() -> dict[str, dict[str, int]]:
    return {
        domain: {question_type: 0 for question_type in NORMALIZED_TYPE_ORDER}
        for domain in DOMAIN_ORDER
    }


def generate_balanced_selection_manifest(
    data_root: Path,
    *,
    seed: str = DEFAULT_BALANCED_SEED,
    domain_counts: Mapping[str, int] | None = None,
    normalized_type_counts: Mapping[str, int] | None = None,
    domain_type_counts: Mapping[str, Mapping[str, int]] | None = None,
    max_questions_per_document: int = DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
) -> dict[str, Any]:
    """将一份均衡问题选集冻结到所选 PDF 与 QA 字节。"""

    root = Path(data_root)
    candidate_inventory_sha256 = _candidate_inventory_sha256(root)
    normalized_seed = _validated_seed(seed)
    domains = _normalized_balanced_domain_counts(domain_counts)
    question_types = _normalized_balanced_type_counts(normalized_type_counts)
    document_cap = _validated_document_question_cap(
        max_questions_per_document
    )
    selected = select_balanced_questions(
        root,
        seed=normalized_seed,
        domain_counts=domains,
        normalized_type_counts=question_types,
        domain_type_counts=domain_type_counts,
        max_questions_per_document=document_cap,
    )
    document_authority: dict[
        int,
        tuple[Path, Path, tuple[dict[str, str], ...], str, str],
    ] = {}
    cases: list[dict[str, Any]] = []
    actual_domain_type_counts = _empty_domain_type_counts()
    for item in selected:
        authority = document_authority.get(item.doc_id)
        if authority is None:
            pdf_path, qa_path = _document_paths(root, item.doc_id)
            authority = (
                pdf_path,
                qa_path,
                _read_qa_records(qa_path),
                _sha256_file(pdf_path),
                _sha256_file(qa_path),
            )
            document_authority[item.doc_id] = authority
        pdf_path, _qa_path, records, pdf_sha256, qa_sha256 = authority
        record = records[item.question_index]
        if (
            record["type"] != item.question_type
            or normalize_question_type(record["type"])
            != item.normalized_question_type
        ):
            raise SelectionValidationError(
                f"question candidate drift for {item.case_id}"
            )
        actual_domain_type_counts[item.domain][item.normalized_question_type] += 1
        cases.append(
            {
                "case_id": item.case_id,
                "doc_id": item.doc_id,
                "question_index": item.question_index,
                "domain": item.domain,
                "question_type": item.question_type,
                "normalized_question_type": item.normalized_question_type,
                "pdf_filename": pdf_path.name,
                "pdf_sha256": pdf_sha256,
                "qa_sha256": qa_sha256,
            }
        )
    if Counter(item.domain for item in selected) != Counter(domains):
        raise AssertionError("balanced selection violated frozen domain quotas")
    if Counter(
        item.normalized_question_type for item in selected
    ) != Counter(question_types):
        raise AssertionError("balanced selection violated frozen type quotas")
    return {
        "schema_version": BALANCED_SCHEMA_VERSION,
        "selection_algorithm": BALANCED_SELECTION_ALGORITHM,
        "type_normalization_version": TYPE_NORMALIZATION_VERSION,
        "candidate_inventory_sha256": candidate_inventory_sha256,
        "seed": normalized_seed,
        "case_count": len(cases),
        "domain_counts": domains,
        "normalized_type_counts": question_types,
        "domain_type_counts": actual_domain_type_counts,
        "max_questions_per_document": document_cap,
        "cases": cases,
    }


def write_balanced_selection_manifest(
    output_path: Path,
    *,
    data_root: Path,
    seed: str = DEFAULT_BALANCED_SEED,
    domain_counts: Mapping[str, int] | None = None,
    normalized_type_counts: Mapping[str, int] | None = None,
    domain_type_counts: Mapping[str, Mapping[str, int]] | None = None,
    max_questions_per_document: int = DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
) -> dict[str, Any]:
    manifest = generate_balanced_selection_manifest(
        data_root,
        seed=seed,
        domain_counts=domain_counts,
        normalized_type_counts=normalized_type_counts,
        domain_type_counts=domain_type_counts,
        max_questions_per_document=max_questions_per_document,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _expect_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SelectionValidationError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _expect_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SelectionValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_manifest_shape(raw: Any) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise SelectionValidationError("selection manifest must be a JSON object")
    _expect_exact_keys(raw, _TOP_LEVEL_KEYS, "manifest")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SelectionValidationError(
            f"unsupported selection schema: {raw['schema_version']!r}"
        )
    if raw["selection_algorithm"] != SELECTION_ALGORITHM:
        raise SelectionValidationError(
            f"unsupported selection algorithm: {raw['selection_algorithm']!r}"
        )
    seed = _validated_seed(raw["seed"])
    counts = _normalized_domain_counts(raw["domain_counts"])
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases:
        raise SelectionValidationError("cases must be a non-empty list")
    if type(raw["case_count"]) is not int or raw["case_count"] != len(cases):
        raise SelectionValidationError("case_count does not match cases")
    if sum(counts.values()) != len(cases):
        raise SelectionValidationError("domain_counts do not match cases")
    if any(not isinstance(case, dict) for case in cases):
        raise SelectionValidationError("every case must be an object")
    return seed, counts, cases


def _normalized_domain_type_counts(
    value: object,
    *,
    domain_counts: Mapping[str, int],
    normalized_type_counts: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != set(DOMAIN_ORDER):
        raise SelectionValidationError(
            "domain_type_counts must contain exactly: "
            + ", ".join(DOMAIN_ORDER)
        )
    normalized = _empty_domain_type_counts()
    for domain in DOMAIN_ORDER:
        row = value[domain]
        if not isinstance(row, Mapping) or set(row) != set(NORMALIZED_TYPE_ORDER):
            raise SelectionValidationError(
                f"domain_type_counts[{domain!r}] must contain exactly: "
                + ", ".join(NORMALIZED_TYPE_ORDER)
            )
        for question_type in NORMALIZED_TYPE_ORDER:
            count = row[question_type]
            if type(count) is not int or count < 0:
                raise SelectionValidationError(
                    "domain_type_counts"
                    f"[{domain!r}][{question_type!r}] must be a "
                    "non-negative integer"
                )
            normalized[domain][question_type] = count
        if sum(normalized[domain].values()) != domain_counts[domain]:
            raise SelectionValidationError(
                f"domain_type_counts[{domain!r}] does not match domain_counts"
            )
    for question_type in NORMALIZED_TYPE_ORDER:
        total = sum(
            normalized[domain][question_type]
            for domain in DOMAIN_ORDER
        )
        if total != normalized_type_counts[question_type]:
            raise SelectionValidationError(
                "domain_type_counts column does not match "
                f"normalized_type_counts[{question_type!r}]"
            )
    return normalized


def _validated_balanced_manifest_shape(
    raw: Any,
) -> tuple[
    str,
    str,
    dict[str, int],
    dict[str, int],
    dict[str, dict[str, int]],
    int,
    list[dict[str, Any]],
]:
    if not isinstance(raw, dict):
        raise SelectionValidationError("selection manifest must be a JSON object")
    _expect_exact_keys(raw, _BALANCED_TOP_LEVEL_KEYS, "manifest")
    if raw["schema_version"] != BALANCED_SCHEMA_VERSION:
        raise SelectionValidationError(
            f"unsupported selection schema: {raw['schema_version']!r}"
        )
    if raw["selection_algorithm"] != BALANCED_SELECTION_ALGORITHM:
        raise SelectionValidationError(
            f"unsupported selection algorithm: {raw['selection_algorithm']!r}"
        )
    if raw["type_normalization_version"] != TYPE_NORMALIZATION_VERSION:
        raise SelectionValidationError(
            "unsupported type normalization version: "
            f"{raw['type_normalization_version']!r}"
        )
    candidate_inventory_sha256 = _expect_sha256(
        raw["candidate_inventory_sha256"],
        "candidate_inventory_sha256",
    )
    seed = _validated_seed(raw["seed"])
    domain_counts = _normalized_balanced_domain_counts(raw["domain_counts"])
    normalized_type_counts = _normalized_balanced_type_counts(
        raw["normalized_type_counts"]
    )
    if sum(domain_counts.values()) != sum(normalized_type_counts.values()):
        raise SelectionValidationError(
            "domain_counts and normalized_type_counts must have equal totals"
        )
    domain_type_counts = _normalized_domain_type_counts(
        raw["domain_type_counts"],
        domain_counts=domain_counts,
        normalized_type_counts=normalized_type_counts,
    )
    document_cap = _validated_document_question_cap(
        raw["max_questions_per_document"]
    )
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases:
        raise SelectionValidationError("cases must be a non-empty list")
    if type(raw["case_count"]) is not int or raw["case_count"] != len(cases):
        raise SelectionValidationError("case_count does not match cases")
    if sum(domain_counts.values()) != len(cases):
        raise SelectionValidationError("domain_counts do not match cases")
    if any(not isinstance(case, dict) for case in cases):
        raise SelectionValidationError("every case must be an object")
    return (
        seed,
        candidate_inventory_sha256,
        domain_counts,
        normalized_type_counts,
        domain_type_counts,
        document_cap,
        cases,
    )


def _validated_expected_count(
    expected_count: int | None,
    *,
    actual_count: int,
) -> None:
    if expected_count is None:
        return
    if type(expected_count) is not int or expected_count < 1:
        raise SelectionValidationError("expected_count must be a positive integer")
    if actual_count != expected_count:
        raise SelectionValidationError(
            f"expected {expected_count} cases, manifest has {actual_count}"
        )


def _load_balanced_selection_manifest(
    raw: dict[str, Any],
    *,
    manifest_bytes: bytes,
    data_root: Path,
    expected_count: int | None,
) -> LoadedSelection:
    (
        seed,
        candidate_inventory_sha256,
        domain_counts,
        normalized_type_counts,
        declared_domain_type_counts,
        document_cap,
        raw_cases,
    ) = _validated_balanced_manifest_shape(raw)
    _validated_expected_count(expected_count, actual_count=len(raw_cases))

    actual_candidate_inventory_sha256 = _candidate_inventory_sha256(data_root)
    if actual_candidate_inventory_sha256 != candidate_inventory_sha256:
        raise SelectionValidationError("candidate inventory drift")

    expected = select_balanced_questions(
        data_root,
        seed=seed,
        domain_counts=domain_counts,
        normalized_type_counts=normalized_type_counts,
        domain_type_counts=declared_domain_type_counts,
        max_questions_per_document=document_cap,
    )
    if len(expected) != len(raw_cases):
        raise SelectionValidationError(
            "manifest cases do not match the deterministic balanced selection"
        )

    document_authority: dict[
        int,
        tuple[Path, Path, tuple[dict[str, str], ...], str, str],
    ] = {}
    loaded_cases: list[DocBenchRunCase] = []
    seen_case_ids: set[str] = set()
    seen_pairs: set[tuple[int, int]] = set()
    document_counts: Counter[int] = Counter()
    actual_domain_counts: Counter[str] = Counter()
    actual_type_counts: Counter[str] = Counter()
    actual_domain_type_counts = _empty_domain_type_counts()

    for position, (raw_case, expected_case) in enumerate(
        zip(raw_cases, expected, strict=True)
    ):
        _expect_exact_keys(
            raw_case,
            _BALANCED_CASE_KEYS,
            f"case[{position}]",
        )
        domain = raw_case["domain"]
        doc_id = raw_case["doc_id"]
        question_index = raw_case["question_index"]
        if not isinstance(domain, str) or domain not in DOMAIN_RANGES:
            raise SelectionValidationError(f"case[{position}] has an invalid domain")
        if type(doc_id) is not int or doc_id not in DOMAIN_RANGES[domain]:
            raise SelectionValidationError(f"case[{position}] has an invalid doc_id")
        if type(question_index) is not int or question_index < 0:
            raise SelectionValidationError(
                f"case[{position}] has an invalid question_index"
            )
        case_id = raw_case["case_id"]
        expected_case_id = f"docbench:{doc_id}:{question_index}"
        if case_id != expected_case_id:
            raise SelectionValidationError(
                f"case[{position}] case_id must be {expected_case_id!r}"
            )
        pair = (doc_id, question_index)
        if case_id in seen_case_ids or pair in seen_pairs:
            raise SelectionValidationError(
                "case_id and (doc_id, question_index) must be unique"
            )
        seen_case_ids.add(case_id)
        seen_pairs.add(pair)
        document_counts[doc_id] += 1
        if document_counts[doc_id] > document_cap:
            raise SelectionValidationError(
                f"document {doc_id} exceeds max_questions_per_document"
            )

        question_type = raw_case["question_type"]
        normalized_question_type = raw_case["normalized_question_type"]
        if not isinstance(question_type, str) or question_type not in QUESTION_TYPES:
            raise SelectionValidationError(
                f"case[{position}] has an invalid question_type"
            )
        if (
            not isinstance(normalized_question_type, str)
            or normalized_question_type not in NORMALIZED_TYPE_ORDER
            or normalize_question_type(question_type) != normalized_question_type
        ):
            raise SelectionValidationError(
                f"case[{position}] has an invalid normalized_question_type"
            )
        if (
            case_id != expected_case.case_id
            or doc_id != expected_case.doc_id
            or question_index != expected_case.question_index
            or domain != expected_case.domain
            or question_type != expected_case.question_type
            or normalized_question_type
            != expected_case.normalized_question_type
        ):
            raise SelectionValidationError(
                "manifest cases do not match the deterministic balanced selection"
            )

        pdf_filename = raw_case["pdf_filename"]
        if (
            not isinstance(pdf_filename, str)
            or Path(pdf_filename).name != pdf_filename
            or Path(pdf_filename).suffix.lower() != ".pdf"
        ):
            raise SelectionValidationError(
                f"case[{position}] has an invalid pdf_filename"
            )
        pdf_sha256 = _expect_sha256(
            raw_case["pdf_sha256"],
            f"case[{position}].pdf_sha256",
        )
        qa_sha256 = _expect_sha256(
            raw_case["qa_sha256"],
            f"case[{position}].qa_sha256",
        )

        authority = document_authority.get(doc_id)
        if authority is None:
            pdf_path, qa_path = _document_paths(data_root, doc_id)
            authority = (
                pdf_path,
                qa_path,
                _read_qa_records(qa_path),
                _sha256_file(pdf_path),
                _sha256_file(qa_path),
            )
            document_authority[doc_id] = authority
        pdf_path, qa_path, records, actual_pdf_sha256, actual_qa_sha256 = authority
        if pdf_path.name != pdf_filename:
            raise SelectionValidationError(
                f"PDF filename drift for document {doc_id}"
            )
        if actual_pdf_sha256 != pdf_sha256:
            raise SelectionValidationError(
                f"PDF sha256 drift for document {doc_id}"
            )
        if actual_qa_sha256 != qa_sha256:
            raise SelectionValidationError(
                f"QA sha256 drift for document {doc_id}"
            )
        if question_index >= len(records):
            raise SelectionValidationError(
                f"question index drift for document {doc_id}"
            )
        record = records[question_index]
        if record["type"] != question_type:
            raise SelectionValidationError(
                f"question type drift for document {doc_id}"
            )

        actual_domain_counts[domain] += 1
        actual_type_counts[normalized_question_type] += 1
        actual_domain_type_counts[domain][normalized_question_type] += 1
        loaded_cases.append(
            DocBenchRunCase(
                case_id=case_id,
                doc_id=doc_id,
                question_index=question_index,
                domain=domain,
                question_type=question_type,
                pdf_path=pdf_path,
                qa_path=qa_path,
                question=record["question"],
                answer=record["answer"],
                evidence=record["evidence"],
                pdf_sha256=pdf_sha256,
                qa_sha256=qa_sha256,
            )
        )

    if Counter(domain_counts) != actual_domain_counts:
        raise SelectionValidationError("domain_counts do not match selected cases")
    if Counter(normalized_type_counts) != actual_type_counts:
        raise SelectionValidationError(
            "normalized_type_counts do not match selected cases"
        )
    if actual_domain_type_counts != declared_domain_type_counts:
        raise SelectionValidationError(
            "domain_type_counts do not match selected cases"
        )
    return LoadedSelection(
        raw=raw,
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        cases=tuple(loaded_cases),
    )


def load_selection_manifest(
    path: Path,
    *,
    data_root: Path,
    expected_count: int | None = None,
) -> LoadedSelection:
    """加载 manifest，验证其选集与文件，并关联私有 QA 正文。"""

    path = Path(path)
    manifest_bytes, raw = _read_selection_manifest(path)
    if isinstance(raw, dict) and raw.get("schema_version") == DERIVED_SCHEMA_VERSION:
        try:
            spec = parse_derived_selection(raw, path=path)
            _validated_expected_count(expected_count, actual_count=spec.case_count)
            base_bytes, base_raw = _read_selection_manifest(spec.base_path)
            if hashlib.sha256(base_bytes).hexdigest() != spec.base_sha256:
                raise SelectionValidationError("derived base manifest sha256 drift")
            if not isinstance(base_raw, dict) or base_raw.get("schema_version") not in {
                SCHEMA_VERSION, BALANCED_SCHEMA_VERSION,
            }:
                raise SelectionValidationError("derived base must be a non-derived basic manifest")
            # 使用已经读出并校验哈希的同一字节快照；不递归读派生路径，不缩减基础校验。
            base = _load_basic_selection_manifest(base_raw, base_bytes, data_root=data_root)
            validate_derived_case_binding(spec, verified_base=base.raw)
        except DerivedSelectionError as exc:
            raise SelectionValidationError(str(exc)) from exc
        excluded = set(spec.excluded_case_ids)
        return LoadedSelection(
            raw=raw,
            sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            cases=tuple(case for case in base.cases if case.case_id not in excluded),
        )
    return _load_basic_selection_manifest(
        raw, manifest_bytes, data_root=data_root, expected_count=expected_count,
    )


def _read_selection_manifest(path: Path) -> tuple[bytes, Any]:
    try:
        manifest_bytes = path.read_bytes()
        manifest_text = manifest_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise SelectionValidationError(f"cannot read selection manifest {path}: {exc}") from exc
    return manifest_bytes, _json_loads_strict(manifest_text, source=path)


def _load_basic_selection_manifest(
    raw: Any,
    manifest_bytes: bytes,
    *,
    data_root: Path,
    expected_count: int | None = None,
) -> LoadedSelection:
    """原有基础选集校验的唯一执行路径；派生选集也须完整经过这里。"""

    if isinstance(raw, dict) and raw.get("schema_version") == BALANCED_SCHEMA_VERSION:
        return _load_balanced_selection_manifest(
            raw,
            manifest_bytes=manifest_bytes,
            data_root=Path(data_root),
            expected_count=expected_count,
        )
    seed, counts, raw_cases = _validated_manifest_shape(raw)
    _validated_expected_count(expected_count, actual_count=len(raw_cases))

    selected_ids = select_document_ids(seed=seed, domain_counts=counts)
    expected_pairs = [
        (domain, doc_id)
        for domain in DOMAIN_ORDER
        for doc_id in selected_ids[domain]
    ]
    actual_pairs: list[tuple[str, int]] = []
    loaded_cases: list[DocBenchRunCase] = []
    seen_case_ids: set[str] = set()
    seen_doc_ids: set[int] = set()
    data_root = Path(data_root)

    for position, raw_case in enumerate(raw_cases):
        _expect_exact_keys(raw_case, _CASE_KEYS, f"case[{position}]")
        domain = raw_case["domain"]
        doc_id = raw_case["doc_id"]
        question_index = raw_case["question_index"]
        if not isinstance(domain, str) or domain not in DOMAIN_RANGES:
            raise SelectionValidationError(f"case[{position}] has an invalid domain")
        if type(doc_id) is not int or doc_id not in DOMAIN_RANGES[domain] or doc_id == 0:
            raise SelectionValidationError(f"case[{position}] has an invalid doc_id")
        if type(question_index) is not int or question_index < 0:
            raise SelectionValidationError(f"case[{position}] has an invalid question_index")
        case_id = raw_case["case_id"]
        expected_case_id = f"docbench:{doc_id}:{question_index}"
        if case_id != expected_case_id:
            raise SelectionValidationError(
                f"case[{position}] case_id must be {expected_case_id!r}"
            )
        if case_id in seen_case_ids or doc_id in seen_doc_ids:
            raise SelectionValidationError("case_id and doc_id must be unique")
        seen_case_ids.add(case_id)
        seen_doc_ids.add(doc_id)
        actual_pairs.append((domain, doc_id))

        question_type = raw_case["question_type"]
        if not isinstance(question_type, str) or question_type not in QUESTION_TYPES:
            raise SelectionValidationError(f"case[{position}] has an invalid question_type")
        pdf_filename = raw_case["pdf_filename"]
        if (
            not isinstance(pdf_filename, str)
            or Path(pdf_filename).name != pdf_filename
            or Path(pdf_filename).suffix.lower() != ".pdf"
        ):
            raise SelectionValidationError(f"case[{position}] has an invalid pdf_filename")
        pdf_sha256 = _expect_sha256(raw_case["pdf_sha256"], f"case[{position}].pdf_sha256")
        qa_sha256 = _expect_sha256(raw_case["qa_sha256"], f"case[{position}].qa_sha256")

        pdf_path, qa_path = _document_paths(data_root, doc_id)
        if pdf_path.name != pdf_filename:
            raise SelectionValidationError(f"PDF filename drift for document {doc_id}")
        if _sha256_file(pdf_path) != pdf_sha256:
            raise SelectionValidationError(f"PDF sha256 drift for document {doc_id}")
        if _sha256_file(qa_path) != qa_sha256:
            raise SelectionValidationError(f"QA sha256 drift for document {doc_id}")
        records = _read_qa_records(qa_path)
        if question_index >= len(records):
            raise SelectionValidationError(f"question index drift for document {doc_id}")
        expected_question_index = _select_question_index(
            seed=seed,
            domain=domain,
            doc_id=doc_id,
            question_count=len(records),
        )
        if question_index != expected_question_index:
            raise SelectionValidationError(
                f"question selection drift for document {doc_id}"
            )
        record = records[question_index]
        if record["type"] != question_type:
            raise SelectionValidationError(f"question type drift for document {doc_id}")
        loaded_cases.append(
            DocBenchRunCase(
                case_id=case_id,
                doc_id=doc_id,
                question_index=question_index,
                domain=domain,
                question_type=question_type,
                pdf_path=pdf_path,
                qa_path=qa_path,
                question=record["question"],
                answer=record["answer"],
                evidence=record["evidence"],
                pdf_sha256=pdf_sha256,
                qa_sha256=qa_sha256,
            )
        )

    if actual_pairs != expected_pairs:
        raise SelectionValidationError(
            "manifest cases do not match the deterministic document selection"
        )
    return LoadedSelection(
        raw=raw,
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        cases=tuple(loaded_cases),
    )


def _parse_domain_counts(per_domain: int) -> dict[str, int]:
    return {domain: per_domain for domain in DOMAIN_ORDER}


def _add_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_legacy_options: bool = True,
) -> None:
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="generate the frozen balanced 125-question selection",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed")
    if include_legacy_options:
        parser.add_argument("--per-domain", type=int, default=20)


def _run_cli(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise SelectionValidationError(
            f"refusing to overwrite an existing selection: {args.output}"
        )

    if args.balanced:
        seed = DEFAULT_BALANCED_SEED if args.seed is None else args.seed
        write_balanced_selection_manifest(
            args.output,
            data_root=args.data_root,
            seed=seed,
            domain_counts=DEFAULT_BALANCED_DOMAIN_COUNTS,
            normalized_type_counts=DEFAULT_BALANCED_NORMALIZED_TYPE_COUNTS,
            domain_type_counts=DEFAULT_BALANCED_DOMAIN_TYPE_COUNTS,
            max_questions_per_document=DEFAULT_MAX_QUESTIONS_PER_DOCUMENT,
        )
        load_selection_manifest(
            args.output,
            data_root=args.data_root,
            expected_count=sum(DEFAULT_BALANCED_DOMAIN_COUNTS.values()),
        )
        return 0

    seed = DEFAULT_SEED if args.seed is None else args.seed
    write_selection_manifest(
        args.output,
        data_root=args.data_root,
        seed=seed,
        domain_counts=_parse_domain_counts(args.per_domain),
    )
    load_selection_manifest(
        args.output,
        data_root=args.data_root,
        expected_count=args.per_domain * len(DOMAIN_ORDER),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a formal DocBench selection")
    _add_cli_arguments(parser)
    return _run_cli(parser.parse_args(argv))
