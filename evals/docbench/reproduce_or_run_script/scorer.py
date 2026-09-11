"""兼容 DocBench prompt 并为每个用例持久化证据的 LLM 评审。

本模块刻意不声称复现官方分数。DocBench 发布的 evaluator 硬编码了 OpenAI 模型；
本实现使用固定的上游 prompt，并由调用方选择兼容 OpenAI 的 judge。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

from personagraph.model_io.api_quota_controller import (
    DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS,
    ApiQuotaAdmissionError,
    PreparedApiQuotaRequest,
    prepare_api_quota_request,
)
from personagraph.model_io.dialects import build_request_controls
from personagraph.model_io.endpoint_profiles import ModelProfileQuota


SCORING_PROTOCOL = "docbench_prompt_compatible"
OFFICIAL_COMPARABLE = False
OFFICIAL_PROMPT_SHA256 = (
    "784a6d6cf4f8151765b169c633f46e62d12dcd77f8941172e72edbf8b30f9bde"
)
JUDGE_SYSTEM_MESSAGE = "You are a helpful evaluator."
JUDGE_MAX_OUTPUT_TOKENS = 256
_PLACEHOLDERS = ("question", "sys_ans", "ref_ans", "ref_text")
_PLACEHOLDER_RE = re.compile(r"{{(question|sys_ans|ref_ans|ref_text)}}")
_ANY_PLACEHOLDER_RE = re.compile(r"{{[^{}]+}}")
_OPTIONAL_CORRECTNESS_LABEL_RE = re.compile(
    r"^(?:-\s*)?Correctness\s*:\s*", re.IGNORECASE
)


class DocBenchScoringError(RuntimeError):
    """输入格式错误或 scorer 来源无效时使用的基础异常。"""


class ScoreParseError(DocBenchScoringError):
    """judge 响应没有以无歧义的二元 token 开头。"""


@dataclass(frozen=True, slots=True)
class JudgeRetryPolicy:
    """单个 OpenAI 兼容 judge 的有界重试与节奏控制策略。"""

    max_attempts: int = 6
    fallback_backoff_s: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 30.0)
    inter_case_delay_s: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("judge max_attempts must be positive")
        if len(self.fallback_backoff_s) < self.max_attempts - 1:
            raise ValueError(
                "judge fallback_backoff_s must cover every retry attempt"
            )
        if any(
            not math.isfinite(float(delay)) or float(delay) < 0
            for delay in self.fallback_backoff_s
        ):
            raise ValueError(
                "judge fallback_backoff_s values must be finite and non-negative"
            )
        if (
            not math.isfinite(float(self.inter_case_delay_s))
            or self.inter_case_delay_s < 0
        ):
            raise ValueError(
                "judge inter_case_delay_s must be finite and non-negative"
            )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "fallback_backoff_s": [
                float(delay) for delay in self.fallback_backoff_s
            ],
            "inter_case_delay_s": float(self.inter_case_delay_s),
            "retryable_http_statuses": [408, 429, "5xx"],
            "retryable_transport_errors": ["httpx.TransportError"],
        }


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """由调用方提供的 OpenAI 兼容端点配置。

    ``api_key`` 只保留在内存中，绝不写入评分产物。预期用途之一是 DeepSeek 的标准
    OpenAI 兼容 base URL，但传输层刻意保持 provider 中立，调用方也可使用兼容的
    测试端点。
    """

    base_url: str
    model: str
    api_key: str
    provider: str = "deepseek"
    request_dialect: str = "deepseek"
    timeout_s: float = 120.0
    temperature: float | None = 0.0
    retry_policy: JudgeRetryPolicy = JudgeRetryPolicy()
    quota: ModelProfileQuota = field(default_factory=ModelProfileQuota)
    quota_database_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("judge base_url must not be empty")
        if not self.model.strip():
            raise ValueError("judge model must not be empty")
        if not self.api_key.strip():
            raise ValueError("judge api_key must not be empty")
        if not self.provider.strip():
            raise ValueError("judge provider must not be empty")
        if not self.request_dialect.strip():
            raise ValueError("judge request_dialect must not be empty")
        if self.request_dialect.strip().casefold() != "deepseek":
            raise ValueError("judge request_dialect must be deepseek")
        if self.timeout_s <= 0:
            raise ValueError("judge timeout_s must be positive")
        if not isinstance(self.quota, ModelProfileQuota):
            raise TypeError("judge quota must be ModelProfileQuota")
        if self.quota_database_path is not None:
            if not isinstance(self.quota_database_path, Path):
                raise TypeError("judge quota_database_path must be a Path")
            if not self.quota_database_path.expanduser().is_absolute():
                raise ValueError("judge quota_database_path must be absolute")

    @property
    def chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def public_metadata(self) -> dict[str, Any]:
        quota_enabled = any(
            value is not None
            for value in (
                self.quota.requests_per_minute,
                self.quota.tokens_per_minute,
                self.quota.tokens_per_week,
                self.quota.max_in_flight,
            )
        )
        metadata = {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "request_dialect": self.request_dialect,
            "temperature": self.temperature,
            "thinking_enabled": False,
            "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
            "retry_policy": self.retry_policy.public_metadata(),
            "quota_enabled": quota_enabled,
            "quota": self.quota.to_dict(),
        }
        return metadata


@dataclass(frozen=True, slots=True)
class JudgeInput:
    case_id: str
    question: str
    system_answer: str
    reference_answer: str
    reference_text: str
    domain: str
    question_type: str
    doc_id: str | None = None
    question_index: int | None = None

    def prompt_values(self) -> dict[str, str]:
        return {
            "question": self.question,
            "sys_ans": self.system_answer,
            "ref_ans": self.reference_answer,
            "ref_text": self.reference_text,
        }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_official_prompt(
    path: str | Path,
    *,
    expected_sha256: str = OFFICIAL_PROMPT_SHA256,
) -> tuple[str, str]:
    """读取并验证固定的上游 prompt，且不对其进行规范化。"""

    prompt_path = Path(path)
    raw = prompt_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise DocBenchScoringError(
            "DocBench evaluation prompt sha256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        template = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocBenchScoringError(
            "DocBench evaluation prompt is not valid UTF-8"
        ) from exc
    _validate_prompt_placeholders(template)
    return template, actual_sha256


def render_official_prompt(template: str, judge_input: JudgeInput) -> str:
    """通过一次非递归替换精确填充四个官方占位符。"""

    _validate_prompt_placeholders(template)
    values = judge_input.prompt_values()
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)


def parse_binary_score(response_text: str) -> int:
    """解析首个评分 token，只允许上游标签作为前缀例外。

    接受纯 ``0``/``1``，以及以 ``Correctness: 0|1`` 开头的响应（前面可带 prompt 的
    ``-`` 列表标记）。``10``、``1.`` 或正文前缀等 token 会被拒绝，不会像上游
    notebook 那样仅因包含数字就予以分类。
    """

    if not isinstance(response_text, str):
        raise ScoreParseError("judge response content is not text")
    remaining = response_text.strip()
    remaining = _OPTIONAL_CORRECTNESS_LABEL_RE.sub("", remaining, count=1)
    match = re.match(r"(\S+)", remaining)
    if match is None or match.group(1) not in {"0", "1"}:
        token = match.group(1) if match else "<empty>"
        raise ScoreParseError(
            f"first non-empty score token must be exactly 0 or 1, got {token!r}"
        )
    return int(match.group(1))


def build_judge_inputs(
    *,
    run_results: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    frozen_cases: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    strict: bool = True,
) -> tuple[JudgeInput, ...]:
    """按用例 ID 对齐隔离运行输出与不可变 benchmark 用例。"""

    result_records = _record_sequence(run_results, label="run results")
    case_records = _record_sequence(frozen_cases, label="frozen cases")
    results_by_id = _index_records(result_records, label="run result")
    cases_by_id = _index_records(case_records, label="frozen case")

    missing = sorted(set(cases_by_id) - set(results_by_id))
    extra = sorted(set(results_by_id) - set(cases_by_id))
    if missing:
        raise DocBenchScoringError(
            "run results are missing frozen cases: " + ", ".join(missing)
        )
    if strict and extra:
        raise DocBenchScoringError(
            "run results contain cases outside the frozen set: "
            + ", ".join(extra)
        )

    aligned: list[JudgeInput] = []
    for case_record in case_records:
        case_id = _case_id(case_record)
        result = results_by_id[case_id]
        question = _required_text(case_record, ("question", "benchmark_question"))
        reference_answer = _required_text(
            case_record, ("answer", "reference_answer", "ref_ans")
        )
        reference_text = _required_text(
            case_record, ("evidence", "reference_text", "ref_text")
        )
        doc_id = _optional_text(case_record, ("doc_id", "file", "document_id"))
        question_index = _optional_int(
            case_record, ("question_index", "question_idx", "index")
        )
        domain = _optional_text(
            case_record, ("domain", "category", "document_category")
        ) or _domain_from_doc_id(doc_id)
        raw_type = _optional_text(
            case_record, ("type", "question_type", "qa_type")
        ) or "unknown"
        aligned.append(
            JudgeInput(
                case_id=case_id,
                question=question,
                system_answer=_system_answer(result),
                reference_answer=reference_answer,
                reference_text=reference_text,
                domain=domain,
                question_type=_normalize_question_type(raw_type),
                doc_id=doc_id,
                question_index=question_index,
            )
        )
    return tuple(aligned)


def score_run(
    *,
    run_results: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    frozen_cases: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    prompt_path: str | Path,
    judge: JudgeConfig,
    expected_prompt_sha256: str = OFFICIAL_PROMPT_SHA256,
    resume: bool = True,
    allow_live: bool = False,
    client: httpx.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """评审一次运行，并原子持久化审计证据与聚合结果。

    ``resume`` 为 true 时复用已完成且来源匹配的用例 checkpoint，失败的 checkpoint
    则会重试。若 checkpoint 使用了不同的 prompt、输入、模型或端点，将直接拒绝，
    而不是在同一运行目录中悄悄混用协议。
    """

    if not allow_live:
        raise DocBenchScoringError(
            "DocBench scoring requires allow_live=True before judge calls"
        )
    template, prompt_sha256 = load_official_prompt(
        prompt_path, expected_sha256=expected_prompt_sha256
    )
    judge_inputs = build_judge_inputs(
        run_results=run_results, frozen_cases=frozen_cases
    )
    root = Path(output_dir)
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    owned_client = client is None
    active_client = client or httpx.Client()
    records: list[dict[str, Any]] = []
    new_case_count = 0
    active_now_fn = now_fn or _utc_now
    try:
        for judge_input in judge_inputs:
            filled_prompt = render_official_prompt(template, judge_input)
            filled_prompt_sha256 = sha256_text(filled_prompt)
            checkpoint = cases_dir / _checkpoint_name(judge_input.case_id)
            expected_provenance = {
                "case_id": judge_input.case_id,
                "prompt_sha256": prompt_sha256,
                "filled_prompt_sha256": filled_prompt_sha256,
                "judge_provider": judge.provider,
                "judge_base_url": judge.base_url,
                "judge_model": judge.model,
                "judge_request_dialect": judge.request_dialect,
            }
            existing = _load_checkpoint(
                checkpoint,
                expected_provenance=expected_provenance,
                resume=resume,
            )
            if existing is not None and _is_completed_checkpoint(
                existing, path=checkpoint
            ):
                records.append(existing)
                continue
            initial_wait_s = (
                0.0
                if new_case_count == 0
                else float(judge.retry_policy.inter_case_delay_s)
            )
            new_case_count += 1
            record = _score_one(
                judge_input=judge_input,
                filled_prompt=filled_prompt,
                prompt_sha256=prompt_sha256,
                filled_prompt_sha256=filled_prompt_sha256,
                judge=judge,
                client=active_client,
                previous_attempts=_checkpoint_attempts(existing),
                initial_wait_s=initial_wait_s,
                sleep_fn=sleep_fn,
                now_fn=active_now_fn,
                checkpoint=checkpoint,
            )
            _write_json_atomic(checkpoint, record)
            records.append(record)
    finally:
        if owned_client:
            active_client.close()

    aggregates = aggregate_scores(records)
    overall = aggregates["overall"]
    summary = {
        "schema_version": "docbench-compatible-score-v1",
        "scoring_protocol": SCORING_PROTOCOL,
        "official_comparable": OFFICIAL_COMPARABLE,
        "status": "complete" if overall["errors"] == 0 else "partial",
        "prompt_sha256": prompt_sha256,
        "judge": judge.public_metadata(),
        "case_count": len(records),
        "correct_count": overall["correct"],
        "score": overall["accuracy"] if overall["errors"] == 0 else None,
        "aggregates": aggregates,
        "cases": records,
    }
    _write_json_atomic(root / "summary.json", summary)
    return summary


def aggregate_scores(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    return {
        "overall": _aggregate_bucket(materialized),
        "by_domain": _aggregate_groups(materialized, "domain"),
        "by_type": _aggregate_groups(materialized, "question_type"),
    }


def _score_one(
    *,
    judge_input: JudgeInput,
    filled_prompt: str,
    prompt_sha256: str,
    filled_prompt_sha256: str,
    judge: JudgeConfig,
    client: httpx.Client,
    previous_attempts: Sequence[Mapping[str, Any]] = (),
    initial_wait_s: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    active_now_fn = now_fn or _utc_now
    attempt_history = [dict(attempt) for attempt in previous_attempts]
    record: dict[str, Any] = {
        "schema_version": "docbench-compatible-case-score-v1",
        "scoring_protocol": SCORING_PROTOCOL,
        "official_comparable": OFFICIAL_COMPARABLE,
        "case_id": judge_input.case_id,
        "doc_id": judge_input.doc_id,
        "question_index": judge_input.question_index,
        "domain": judge_input.domain,
        "question_type": judge_input.question_type,
        "question": judge_input.question,
        "system_answer": judge_input.system_answer,
        "reference_answer": judge_input.reference_answer,
        "reference_text": judge_input.reference_text,
        "prompt_sha256": prompt_sha256,
        "filled_prompt_sha256": filled_prompt_sha256,
        "judge_provider": judge.provider,
        "judge_base_url": judge.base_url,
        "judge_model": judge.model,
        "judge_request_dialect": judge.request_dialect,
        "judge_retry_policy": judge.retry_policy.public_metadata(),
        "judge_http_status": None,
        "judge_response_raw": None,
        "judge_response_text": None,
        "judge_attempts": attempt_history,
        "judge_attempt_count": len(attempt_history),
        "score_source": "judge",
        "score": None,
        "error": None,
        "scored_at": _as_utc(active_now_fn()).isoformat(),
    }
    # 固定的 DocBench prompt 明确规定空回答必须得 0 分。这里直接执行该确定性
    # 规则，既避免裁判违背 rubric，也不为已知答案浪费一次外部模型调用。
    if not judge_input.system_answer.strip():
        record["score_source"] = "official_prompt_empty_answer_rule"
        record["score"] = 0
        record["status"] = "completed"
        if checkpoint is not None:
            _write_json_atomic(checkpoint, record)
        return record

    body: dict[str, Any] = {
        "model": judge.model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": filled_prompt},
        ],
    }
    controls = build_request_controls(
        provider="openai-compatible",
        dialect=judge.request_dialect,
        thinking_enabled=False,
        reasoning_effort=None,
        max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
        temperature=judge.temperature,
        json_mode=False,
    )
    body.update(controls.payload_fields)
    prepared_quota = prepare_api_quota_request(
        base_url=judge.base_url,
        credential=judge.api_key,
        quota=judge.quota,
        # 没有供应商 count-tokens 接口时，以完整请求 JSON 的 UTF-8 字节数作为保守
        # token 预留；不能用 char/4，否则中文与高熵内容会系统性低估 TPM 占用。
        estimated_input_tokens=len(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        output_token_limit=JUDGE_MAX_OUTPUT_TOKENS,
        provider_timeout_seconds=judge.timeout_s,
        queue_database_path=judge.quota_database_path,
    )

    wait_before_attempt_s = float(initial_wait_s)
    for request_attempt in range(1, judge.retry_policy.max_attempts + 1):
        if wait_before_attempt_s > 0:
            sleep_fn(wait_before_attempt_s)
        response: httpx.Response | None = None
        attempted_at = _as_utc(active_now_fn()).isoformat()
        record["judge_http_status"] = None
        record["judge_response_raw"] = None
        record["judge_response_text"] = None
        try:
            response = _post_judge_request(
                client=client,
                judge=judge,
                body=body,
                prepared_quota=prepared_quota,
                logical_call_id=(
                    "docbench-judge:"
                    f"{filled_prompt_sha256[:16]}:{len(attempt_history) + 1}"
                ),
            )
            record["judge_http_status"] = response.status_code
            record["judge_response_raw"] = _response_body(response)
            response.raise_for_status()
            payload = response.json()
            response_text = _response_content(payload)
            record["judge_response_text"] = response_text
            record["score"] = parse_binary_score(response_text)
        except Exception as exc:
            if response is not None and record["judge_response_raw"] is None:
                record["judge_response_raw"] = _response_body(response)
            retry_reason = _retry_reason(exc, response=response)
            retryable = retry_reason is not None
            will_retry = (
                retryable and request_attempt < judge.retry_policy.max_attempts
            )
            retry_after_header: str | None = None
            wait_before_next_attempt_s: float | None = None
            wait_source: str | None = None
            if response is not None:
                retry_after_header = response.headers.get("Retry-After")
            if will_retry:
                parsed_retry_after = _retry_after_seconds(
                    retry_after_header,
                    now=_as_utc(active_now_fn()),
                )
                if parsed_retry_after is not None:
                    wait_before_next_attempt_s = parsed_retry_after
                    wait_source = "retry_after"
                else:
                    wait_before_next_attempt_s = float(
                        judge.retry_policy.fallback_backoff_s[
                            request_attempt - 1
                        ]
                    )
                    wait_source = "exponential_backoff"
            error = _safe_error(exc, response=response)
            attempt_history.append(
                {
                    "attempt_number": len(attempt_history) + 1,
                    "attempted_at": attempted_at,
                    "wait_before_attempt_s": wait_before_attempt_s,
                    "http_status": (
                        response.status_code if response is not None else None
                    ),
                    "response_raw": record["judge_response_raw"],
                    "status": "error",
                    "error": error,
                    "retryable": retryable,
                    "retry_reason": retry_reason,
                    "retry_after_header": retry_after_header,
                    "wait_before_next_attempt_s": wait_before_next_attempt_s,
                    "wait_source": wait_source,
                }
            )
            record["judge_attempt_count"] = len(attempt_history)
            record["status"] = "retrying" if will_retry else "error"
            record["error"] = error
            if checkpoint is not None:
                _write_json_atomic(checkpoint, record)
            if not will_retry:
                return record
            wait_before_attempt_s = wait_before_next_attempt_s or 0.0
            continue

        attempt_history.append(
            {
                "attempt_number": len(attempt_history) + 1,
                "attempted_at": attempted_at,
                "wait_before_attempt_s": wait_before_attempt_s,
                "http_status": (
                    response.status_code if response is not None else None
                ),
                "response_raw": record["judge_response_raw"],
                "status": "completed",
                "error": None,
                "retryable": False,
                "retry_reason": None,
                "retry_after_header": None,
                "wait_before_next_attempt_s": None,
                "wait_source": None,
            }
        )
        record["judge_attempt_count"] = len(attempt_history)
        record["status"] = "completed"
        record["error"] = None
        if checkpoint is not None:
            _write_json_atomic(checkpoint, record)
        return record

    raise AssertionError("judge retry loop exhausted without returning")


def _validate_prompt_placeholders(template: str) -> None:
    seen = _ANY_PLACEHOLDER_RE.findall(template)
    expected = [f"{{{{{name}}}}}" for name in _PLACEHOLDERS]
    if sorted(seen) != sorted(expected):
        raise DocBenchScoringError(
            "DocBench prompt must contain each official placeholder exactly once"
        )


def _record_sequence(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, label: str
) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        nested = value.get("cases")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            records = list(nested)
        elif value and all(isinstance(item, Mapping) for item in value.values()):
            records = []
            for key, item in value.items():
                record = dict(item)
                record.setdefault("case_id", str(key))
                records.append(record)
        else:
            raise DocBenchScoringError(f"{label} must contain a cases sequence")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = list(value)
    else:
        raise DocBenchScoringError(f"{label} must be a mapping or sequence")
    if not all(isinstance(record, Mapping) for record in records):
        raise DocBenchScoringError(f"{label} contains a non-object record")
    return records


def _index_records(
    records: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        case_id = _case_id(record)
        if case_id in indexed:
            raise DocBenchScoringError(f"duplicate {label} id: {case_id}")
        indexed[case_id] = record
    return indexed


def _case_id(record: Mapping[str, Any]) -> str:
    for key in ("case_id", "dataset_item_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    doc_id = _optional_text(record, ("doc_id", "file", "document_id"))
    question_index = _optional_int(
        record, ("question_index", "question_idx", "index")
    )
    if doc_id is not None and question_index is not None:
        return f"docbench:{doc_id}:{question_index}"
    raise DocBenchScoringError(
        "case record needs case_id/dataset_item_id or doc_id + question_index"
    )


def _required_text(record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    raise DocBenchScoringError(
        "case record is missing required text field: " + "/".join(keys)
    )


def _optional_text(
    record: Mapping[str, Any], keys: tuple[str, ...]
) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value).strip()
            if text:
                return text
    return None


def _optional_int(
    record: Mapping[str, Any], keys: tuple[str, ...]
) -> int | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
    return None


def _system_answer(result: Mapping[str, Any]) -> str:
    for key in ("sys_ans", "system_answer", "reply", "prediction", "output"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    return ""


def _domain_from_doc_id(doc_id: str | None) -> str:
    if doc_id is None or not doc_id.isdigit():
        return "unknown"
    value = int(doc_id)
    if 0 <= value < 49:
        return "aca"
    if value < 89:
        return "fin"
    if value < 133:
        return "gov"
    if value < 179:
        return "law"
    if value < 229:
        return "new"
    return "unknown"


def _normalize_question_type(value: str) -> str:
    normalized = value.strip().casefold()
    return {
        "text-only": "text",
        "multimodal-f": "multimodal",
        "multimodal-t": "multimodal",
        "multimodal": "multimodal",
        "meta-data": "metadata",
        "una": "unanswerable",
        "una-web": "unanswerable",
    }.get(normalized, normalized or "unknown")


def _response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text


def _response_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DocBenchScoringError(
            "judge response lacks choices[0].message.content"
        ) from exc
    if not isinstance(content, str):
        raise DocBenchScoringError("judge response content is not text")
    return content


def _post_judge_request(
    *,
    client: httpx.Client,
    judge: JudgeConfig,
    body: Mapping[str, Any],
    prepared_quota: PreparedApiQuotaRequest | None,
    logical_call_id: str,
) -> httpx.Response:
    """通过与生成链路相同的共享准入发送一次 judge HTTP 请求。"""

    permit = None
    if prepared_quota is not None:
        permit = prepared_quota.acquire(
            logical_call_id=logical_call_id,
            wait_timeout_seconds=DEFAULT_API_QUOTA_WAIT_TIMEOUT_SECONDS,
        )
        try:
            permit.mark_dispatched()
        except BaseException:
            try:
                permit.abandon_before_dispatch(disposition="cancel")
            except ApiQuotaAdmissionError as cleanup_exc:
                raise DocBenchScoringError(
                    "judge quota reservation could not be reconciled"
                ) from cleanup_exc
            raise

    response: httpx.Response | None = None
    provider_error: BaseException | None = None
    try:
        response = client.post(
            judge.chat_completions_url,
            headers={
                "Authorization": f"Bearer {judge.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=judge.timeout_s,
        )
    except BaseException as exc:
        provider_error = exc

    if permit is not None:
        cooldown_error: ApiQuotaAdmissionError | None = None
        if response is not None and response.status_code == 429:
            try:
                permit.apply_rate_limit_cooldown(
                    _retry_after_seconds(
                        response.headers.get("Retry-After"),
                        now=_utc_now(),
                    )
                    or 1.0
                )
            except ApiQuotaAdmissionError as exc:
                cooldown_error = exc
        try:
            permit.settle(
                outcome=(
                    "succeeded"
                    if provider_error is None
                    and response is not None
                    and response.is_success
                    else "failed"
                ),
                actual_tokens=(
                    _response_usage_tokens(response)
                    if response is not None
                    else None
                ),
            )
        except ApiQuotaAdmissionError as exc:
            raise DocBenchScoringError(
                "judge quota dispatch could not be reconciled"
            ) from exc
        if cooldown_error is not None:
            raise DocBenchScoringError(
                "judge rate-limit cooldown could not be recorded"
            ) from cooldown_error

    if provider_error is not None:
        raise provider_error
    assert response is not None
    return response


def _response_usage_tokens(response: httpx.Response) -> int | None:
    """只在供应商同时给出有效输入、输出用量时替换保守预留。"""

    try:
        usage = response.json()["usage"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(usage, Mapping):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (prompt_tokens, completion_tokens)
    ):
        return None
    return prompt_tokens + completion_tokens


def _safe_error(
    exc: BaseException, *, response: httpx.Response | None
) -> dict[str, Any]:
    message = str(exc).replace("\n", " ").strip() or type(exc).__name__
    return {
        "type": type(exc).__name__,
        "message": message[:2_000],
        "http_status": response.status_code if response is not None else None,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _retry_reason(
    exc: BaseException,
    *,
    response: httpx.Response | None,
) -> str | None:
    if response is not None:
        status = response.status_code
        if status in {408, 429} or 500 <= status <= 599:
            return f"http_{status}"
    if isinstance(exc, httpx.TransportError):
        return "transport_error"
    return None


def _retry_after_seconds(
    raw_value: str | None,
    *,
    now: datetime,
) -> float | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        delay = -1.0
    if math.isfinite(delay) and delay >= 0:
        return delay
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, (_as_utc(retry_at) - _as_utc(now)).total_seconds())


def _checkpoint_name(case_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip("-_") or "case"
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:80]}-{digest}.json"


def _load_checkpoint(
    path: Path,
    *,
    expected_provenance: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any] | None:
    if not resume or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocBenchScoringError(f"invalid case checkpoint: {path}") from exc
    if not isinstance(record, dict):
        raise DocBenchScoringError(f"case checkpoint is not an object: {path}")
    for key, expected in expected_provenance.items():
        if record.get(key) != expected:
            raise DocBenchScoringError(
                f"case checkpoint provenance mismatch for {key}: {path}"
            )
    return record


def _is_completed_checkpoint(record: Mapping[str, Any], *, path: Path) -> bool:
    if record.get("status") != "completed":
        return False
    if record.get("score") not in (0, 1):
        raise DocBenchScoringError(f"completed checkpoint has invalid score: {path}")
    return True


def _checkpoint_attempts(
    record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    if record is None:
        return ()
    attempts = record.get("judge_attempts")
    if isinstance(attempts, list) and all(
        isinstance(attempt, Mapping) for attempt in attempts
    ):
        return tuple(dict(attempt) for attempt in attempts)

    # 旧 scorer checkpoint 只在顶层保存最终请求；升级并重写 checkpoint 前，
    # 先显式保留该请求。
    if record.get("status") not in {"completed", "error", "retrying"}:
        return ()
    status = "completed" if record.get("status") == "completed" else "error"
    http_status = record.get("judge_http_status")
    retry_reason: str | None = None
    if isinstance(http_status, int) and (
        http_status in {408, 429} or 500 <= http_status <= 599
    ):
        retry_reason = f"http_{http_status}"
    return (
        {
            "attempt_number": 1,
            "attempted_at": record.get("scored_at"),
            "wait_before_attempt_s": None,
            "http_status": http_status,
            "response_raw": record.get("judge_response_raw"),
            "status": status,
            "error": record.get("error"),
            "retryable": retry_reason is not None,
            "retry_reason": retry_reason,
            "retry_after_header": None,
            "wait_before_next_attempt_s": None,
            "wait_source": None,
            "legacy_checkpoint": True,
        },
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _aggregate_groups(
    records: list[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = str(record.get(field) or "unknown")
        groups.setdefault(key, []).append(record)
    return {key: _aggregate_bucket(groups[key]) for key in sorted(groups)}


def _aggregate_bucket(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [record.get("score") for record in records]
    scored = [score for score in scores if score in (0, 1)]
    correct = sum(score == 1 for score in scored)
    total = len(records)
    scored_count = len(scored)
    return {
        "total": total,
        "scored": scored_count,
        "correct": correct,
        "errors": total - scored_count,
        "accuracy": correct / scored_count if scored_count else None,
        "strict_accuracy": correct / total if total else None,
        "coverage": scored_count / total if total else None,
    }


__all__ = [
    "DocBenchScoringError",
    "JudgeConfig",
    "JudgeInput",
    "JudgeRetryPolicy",
    "OFFICIAL_COMPARABLE",
    "OFFICIAL_PROMPT_SHA256",
    "SCORING_PROTOCOL",
    "ScoreParseError",
    "aggregate_scores",
    "build_judge_inputs",
    "load_official_prompt",
    "parse_binary_score",
    "render_official_prompt",
    "score_run",
]
