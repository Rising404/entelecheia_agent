from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from evals.docbench.reproduce_or_run_script.scorer import (
    DocBenchScoringError,
    JudgeConfig,
    JudgeRetryPolicy,
    OFFICIAL_PROMPT_SHA256,
    ScoreParseError,
    build_judge_inputs,
    load_official_prompt,
    parse_binary_score,
    render_official_prompt,
    score_run,
)
from personagraph.model_io.api_quota_queue import ModelApiQuotaQueue
from personagraph.model_io.api_quota_queue import derive_quota_scope_hash
from personagraph.model_io.endpoint_profiles import ModelProfileQuota


# Synthetic contract input, not a copy of the upstream evaluation prompt.
SYNTHETIC_PROMPT = (
    "Synthetic evaluator fixture. Score an empty system answer as 0.\n"
    "Question: {{question}}\n"
    "System Answer: {{sys_ans}}\n"
    "Reference Answer: {{ref_ans}}\n"
    "Reference Text: {{ref_text}}\n"
)
SYNTHETIC_PROMPT_SHA256 = hashlib.sha256(SYNTHETIC_PROMPT.encode()).hexdigest()


@pytest.fixture
def synthetic_prompt_path(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-evaluation-prompt.txt"
    path.write_text(SYNTHETIC_PROMPT, encoding="utf-8")
    return path


def _frozen_cases() -> list[dict[str, object]]:
    return [
        {
            "case_id": "docbench:0:1",
            "doc_id": "0",
            "question_index": 1,
            "question": "What is the reported accuracy?",
            "answer": "65%",
            "evidence": "The reported accuracy is 65%.",
            "type": "text-only",
        },
        {
            "case_id": "docbench:49:0",
            "doc_id": "49",
            "question_index": 0,
            "question": "Is revenue reported?",
            "answer": "Yes",
            "evidence": "Revenue is reported in the table.",
            "type": "multimodal-t",
        },
    ]


def _run_results() -> dict[str, object]:
    return {
        "cases": [
            {
                "dataset_item_id": "docbench:49:0",
                "reply": "No",
            },
            {
                "dataset_item_id": "docbench:0:1",
                "reply": "65%",
            },
        ]
    }


def _judge(
    *,
    model: str = "deepseek-chat",
    retry_policy: JudgeRetryPolicy | None = None,
) -> JudgeConfig:
    return JudgeConfig(
        provider="deepseek",
        base_url="https://judge.example/v1",
        model=model,
        api_key="test-secret-never-persist",
        request_dialect="deepseek",
        timeout_s=5,
        retry_policy=retry_policy
        or JudgeRetryPolicy(inter_case_delay_s=0),
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_judge_config_requires_deepseek_dialect() -> None:
    with pytest.raises(ValueError, match="must be deepseek"):
        JudgeConfig(
            provider="anthropic-compatible",
            request_dialect="anthropic",
            base_url="https://judge.example/v1/messages",
            model="claude",
            api_key="secret",
        )


def test_loads_exact_pinned_prompt_and_fills_four_placeholders_once(
    synthetic_prompt_path: Path,
) -> None:
    template, digest = load_official_prompt(
        synthetic_prompt_path, expected_sha256=SYNTHETIC_PROMPT_SHA256
    )
    judge_input = build_judge_inputs(
        run_results=[
            {
                "case_id": "docbench:0:1",
                "reply": "literal {{ref_ans}} must not be recursively filled",
            }
        ],
        frozen_cases=[_frozen_cases()[0]],
    )[0]

    rendered = render_official_prompt(template, judge_input)

    assert digest == SYNTHETIC_PROMPT_SHA256
    assert template == SYNTHETIC_PROMPT
    assert "Question: What is the reported accuracy?" in rendered
    assert "Reference Answer: 65%" in rendered
    assert "Reference Text: The reported accuracy is 65%." in rendered
    assert "literal {{ref_ans}} must not be recursively filled" in rendered
    assert "{{question}}" not in rendered
    assert "{{sys_ans}}" not in rendered


def test_default_official_pin_rejects_a_valid_synthetic_prompt_before_judge_io(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    assert OFFICIAL_PROMPT_SHA256 == (
        "784a6d6cf4f8151765b169c633f46e62d12dcd77f8941172e72edbf8b30f9bde"
    )
    with pytest.raises(DocBenchScoringError, match=OFFICIAL_PROMPT_SHA256):
        load_official_prompt(synthetic_prompt_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("prompt drift must be rejected before judge I/O")

    with _client(handler) as client:
        with pytest.raises(DocBenchScoringError, match=OFFICIAL_PROMPT_SHA256):
            score_run(
                run_results=_run_results(),
                frozen_cases=_frozen_cases(),
                output_dir=tmp_path / "score",
                prompt_path=synthetic_prompt_path,
                judge=_judge(),
                allow_live=True,
                client=client,
            )


def test_explicit_prompt_pin_rejects_changed_bytes(
    synthetic_prompt_path: Path,
) -> None:
    synthetic_prompt_path.write_text(SYNTHETIC_PROMPT + "\n", encoding="utf-8")

    with pytest.raises(DocBenchScoringError, match="sha256 mismatch"):
        load_official_prompt(
            synthetic_prompt_path, expected_sha256=SYNTHETIC_PROMPT_SHA256
        )


def test_prompt_hash_or_placeholder_drift_is_rejected(tmp_path: Path) -> None:
    changed = tmp_path / "evaluation_prompt.txt"
    changed.write_text("{{question}} {{sys_ans}} {{ref_ans}}\n", encoding="utf-8")

    with pytest.raises(DocBenchScoringError, match="sha256 mismatch"):
        load_official_prompt(changed)

    with pytest.raises(DocBenchScoringError, match="exactly once"):
        load_official_prompt(
            changed,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
        )


def test_build_judge_inputs_aligns_by_id_and_normalizes_groups() -> None:
    inputs = build_judge_inputs(
        run_results=_run_results(), frozen_cases={"cases": _frozen_cases()}
    )

    assert [item.case_id for item in inputs] == [
        "docbench:0:1",
        "docbench:49:0",
    ]
    assert inputs[0].system_answer == "65%"
    assert inputs[0].domain == "aca"
    assert inputs[0].question_type == "text"
    assert inputs[1].domain == "fin"
    assert inputs[1].question_type == "multimodal"


def test_build_judge_inputs_maps_real_una_web_type_to_unanswerable() -> None:
    case = dict(_frozen_cases()[0])
    case["type"] = "una-web"
    inputs = build_judge_inputs(
        run_results=[{"case_id": case["case_id"], "reply": "answer"}],
        frozen_cases=[case],
    )
    assert inputs[0].question_type == "unanswerable"


def test_build_judge_inputs_rejects_missing_extra_and_duplicate_cases() -> None:
    with pytest.raises(DocBenchScoringError, match="missing frozen cases"):
        build_judge_inputs(
            run_results=[_run_results()["cases"][0]],
            frozen_cases=_frozen_cases(),
        )

    with pytest.raises(DocBenchScoringError, match="outside the frozen set"):
        build_judge_inputs(
            run_results=_run_results()["cases"]
            + [{"case_id": "docbench:1:0", "reply": "extra"}],
            frozen_cases=_frozen_cases(),
        )

    with pytest.raises(DocBenchScoringError, match="duplicate run result"):
        build_judge_inputs(
            run_results=[_run_results()["cases"][0]] * 2,
            frozen_cases=[_frozen_cases()[1]],
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", 1),
        ("\n 0 additional words", 0),
        ("Correctness: 1", 1),
        ("- Correctness: 0\n", 0),
        ("correctness: 1 explanation", 1),
    ],
)
def test_binary_parser_accepts_only_an_exact_first_score_token(
    raw: str, expected: int
) -> None:
    assert parse_binary_score(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ("", "10", "1.", "The score is 1", "Correctness: 10", "[1]"),
)
def test_binary_parser_rejects_upstream_contains_digit_false_positives(
    raw: str,
) -> None:
    with pytest.raises(ScoreParseError):
        parse_binary_score(raw)


def test_score_run_requires_live_authorization_before_reading_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(DocBenchScoringError, match="allow_live=True"):
        score_run(
            run_results=(),
            frozen_cases=(),
            output_dir=tmp_path / "score",
            prompt_path=tmp_path / "missing-prompt.txt",
            judge=_judge(),
        )


def test_score_run_calls_openai_compatible_endpoint_and_saves_audit_evidence(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    responses = iter(("Correctness: 1", "0"))

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": f"judge-{len(requests)}",
                "choices": [
                    {"message": {"role": "assistant", "content": next(responses)}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    with _client(handler) as client:
        summary = score_run(
            run_results=_run_results(),
            frozen_cases=_frozen_cases(),
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(),
            allow_live=True,
            client=client,
        )

    assert len(requests) == 2
    for request in requests:
        assert str(request.url) == "https://judge.example/v1/chat/completions"
        assert request.headers["authorization"] == (
            "Bearer test-secret-never-persist"
        )
        body = json.loads(request.content)
        assert body["model"] == "deepseek-chat"
        assert body["temperature"] == 0.0
        assert body["thinking"] == {"type": "disabled"}
        assert body["messages"][0] == {
            "role": "system",
            "content": "You are a helpful evaluator.",
        }

    assert summary["scoring_protocol"] == "docbench_prompt_compatible"
    assert summary["official_comparable"] is False
    assert summary["prompt_sha256"] == SYNTHETIC_PROMPT_SHA256
    assert summary["score"] == 0.5
    assert summary["correct_count"] == 1
    assert summary["judge"]["request_dialect"] == "deepseek"
    assert summary["aggregates"]["overall"] == {
        "total": 2,
        "scored": 2,
        "correct": 1,
        "errors": 0,
        "accuracy": 0.5,
        "strict_accuracy": 0.5,
        "coverage": 1.0,
    }
    assert summary["aggregates"]["by_domain"]["aca"]["accuracy"] == 1.0
    assert summary["aggregates"]["by_domain"]["fin"]["accuracy"] == 0.0
    assert summary["aggregates"]["by_type"]["text"]["accuracy"] == 1.0

    case_files = sorted((tmp_path / "score/cases").glob("*.json"))
    assert len(case_files) == 2
    persisted = [json.loads(path.read_text(encoding="utf-8")) for path in case_files]
    assert {item["score"] for item in persisted} == {0, 1}
    assert all(item["judge_response_raw"]["id"].startswith("judge-") for item in persisted)
    assert all(len(item["filled_prompt_sha256"]) == 64 for item in persisted)
    assert not list((tmp_path / "score").rglob("*.tmp"))
    all_artifacts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "score").rglob("*.json")
    )
    assert "test-secret-never-persist" not in all_artifacts


def test_score_run_uses_the_shared_quota_database(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    database = tmp_path / "run/model_api_quota.sqlite3"
    judge = JudgeConfig(
        provider="deepseek",
        base_url="https://judge.example/v1",
        model="deepseek-chat",
        api_key="quota-secret-never-persist",
        request_dialect="deepseek",
        timeout_s=5,
        retry_policy=JudgeRetryPolicy(inter_case_delay_s=0),
        quota=ModelProfileQuota(
            requests_per_minute=10,
            tokens_per_minute=1_000_000,
            tokens_per_week=1_000_000_000,
            max_in_flight=2,
            quota_group="docbench-judge-test",
        ),
        quota_database_path=database,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "1"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    with _client(handler) as client:
        summary = score_run(
            run_results=[_run_results()["cases"][1]],
            frozen_cases=[_frozen_cases()[0]],
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=judge,
            allow_live=True,
            client=client,
        )

    assert summary["status"] == "complete"
    assert "quota_scope_hash" not in summary["judge"]
    scope_hash = derive_quota_scope_hash(
        base_url=judge.base_url,
        credential=judge.api_key,
        quota_group=judge.quota.quota_group,
    )
    usage = ModelApiQuotaQueue(database).usage_snapshot(scope_hash)
    assert usage.requests_last_minute == 1
    assert usage.tokens_last_minute == 11


def test_score_run_scores_blank_answer_zero_without_calling_judge(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("the pinned DocBench prompt defines blank answers as score 0")

    with _client(handler) as client:
        summary = score_run(
            run_results=[
                {
                    "case_id": "docbench:0:1",
                    "reply": "  \n\t",
                }
            ],
            frozen_cases=[_frozen_cases()[0]],
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(),
            allow_live=True,
            client=client,
        )

    assert summary["score"] == 0.0
    assert summary["correct_count"] == 0
    assert summary["case_count"] == 1
    case = summary["cases"][0]
    assert case["status"] == "completed"
    assert case["score"] == 0
    assert case["score_source"] == "official_prompt_empty_answer_rule"
    assert case["judge_attempt_count"] == 0
    assert case["judge_attempts"] == []
    assert case["judge_http_status"] is None
    assert case["judge_response_raw"] is None


def test_resume_reuses_success_and_retries_error(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    first_call_count = 0

    def first_handler(request: httpx.Request) -> httpx.Response:
        nonlocal first_call_count
        first_call_count += 1
        content = "1" if first_call_count == 1 else "ambiguous"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    output_dir = tmp_path / "score"
    with _client(first_handler) as client:
        first = score_run(
            run_results=_run_results(),
            frozen_cases=_frozen_cases(),
            output_dir=output_dir,
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(),
            allow_live=True,
            client=client,
        )
    assert first_call_count == 2
    assert first["status"] == "partial"
    assert first["score"] is None
    assert first["aggregates"]["overall"]["scored"] == 1
    assert first["aggregates"]["overall"]["errors"] == 1
    assert first["cases"][1]["error"]["type"] == "ScoreParseError"
    assert first["cases"][1]["judge_response_text"] == "ambiguous"

    resumed_call_count = 0

    def resumed_handler(request: httpx.Request) -> httpx.Response:
        nonlocal resumed_call_count
        resumed_call_count += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "0"}}]},
        )

    with _client(resumed_handler) as client:
        resumed = score_run(
            run_results=_run_results(),
            frozen_cases=_frozen_cases(),
            output_dir=output_dir,
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(),
            allow_live=True,
            client=client,
        )

    assert resumed_call_count == 1
    assert resumed["status"] == "complete"
    assert resumed["aggregates"]["overall"]["scored"] == 2
    assert [case["score"] for case in resumed["cases"]] == [1, 0]
    retried_case = resumed["cases"][1]
    assert [attempt["status"] for attempt in retried_case["judge_attempts"]] == [
        "error",
        "completed",
    ]
    assert retried_case["judge_attempts"][0]["error"]["type"] == (
        "ScoreParseError"
    )


def test_resume_preserves_a_legacy_error_checkpoint_as_attempt_history(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    output_dir = tmp_path / "score"
    one_result = [_run_results()["cases"][1]]
    one_case = [_frozen_cases()[0]]

    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    no_retry = JudgeRetryPolicy(max_attempts=1, inter_case_delay_s=0)
    with _client(limited) as client:
        score_run(
            run_results=one_result,
            frozen_cases=one_case,
            output_dir=output_dir,
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(retry_policy=no_retry),
            allow_live=True,
            client=client,
        )

    checkpoint = next((output_dir / "cases").glob("*.json"))
    legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
    legacy.pop("judge_attempts")
    legacy.pop("judge_attempt_count")
    legacy.pop("judge_retry_policy")
    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")

    def recovered(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "1"}}]},
        )

    with _client(recovered) as client:
        summary = score_run(
            run_results=one_result,
            frozen_cases=one_case,
            output_dir=output_dir,
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(retry_policy=no_retry),
            allow_live=True,
            client=client,
        )

    attempts = summary["cases"][0]["judge_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["legacy_checkpoint"] is True
    assert attempts[0]["http_status"] == 429
    assert attempts[0]["response_raw"] == {"error": "rate limited"}
    assert attempts[1]["status"] == "completed"


def test_resume_rejects_model_provenance_mix_without_network(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "1"}}]},
        )

    output_dir = tmp_path / "score"
    one_case = [_frozen_cases()[0]]
    one_result = [_run_results()["cases"][1]]
    with _client(handler) as client:
        score_run(
            run_results=one_result,
            frozen_cases=one_case,
            output_dir=output_dir,
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(),
            allow_live=True,
            client=client,
        )
    assert calls == 1

    with _client(handler) as client:
        with pytest.raises(DocBenchScoringError, match="provenance mismatch"):
            score_run(
                run_results=one_result,
                frozen_cases=one_case,
                output_dir=output_dir,
                prompt_path=synthetic_prompt_path,
                expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
                judge=_judge(model="deepseek-reasoner"),
                allow_live=True,
                client=client,
            )
    assert calls == 1


def test_http_error_is_atomically_recorded_with_raw_response(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    with _client(handler) as client:
        summary = score_run(
            run_results=[_run_results()["cases"][1]],
            frozen_cases=[_frozen_cases()[0]],
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(
                retry_policy=JudgeRetryPolicy(
                    max_attempts=1,
                    inter_case_delay_s=0,
                )
            ),
            allow_live=True,
            client=client,
        )

    case = summary["cases"][0]
    assert case["status"] == "error"
    assert case["score"] is None
    assert case["judge_http_status"] == 429
    assert case["judge_response_raw"] == {"error": {"message": "rate limited"}}
    assert case["error"]["type"] == "HTTPStatusError"
    assert summary["status"] == "partial"
    assert summary["score"] is None
    assert summary["aggregates"]["overall"]["coverage"] == 0.0


def test_transient_429_uses_retry_after_seconds_and_audits_every_attempt(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "1"}}]},
        )

    with _client(handler) as client:
        summary = score_run(
            run_results=[_run_results()["cases"][1]],
            frozen_cases=[_frozen_cases()[0]],
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(
                retry_policy=JudgeRetryPolicy(
                    max_attempts=3,
                    fallback_backoff_s=(2, 4),
                    inter_case_delay_s=0,
                )
            ),
            allow_live=True,
            client=client,
            sleep_fn=sleeps.append,
        )

    assert calls == 2
    assert sleeps == [3.0]
    case = summary["cases"][0]
    assert case["status"] == "completed"
    assert case["judge_attempt_count"] == 2
    assert case["judge_attempts"][0] == {
        "attempt_number": 1,
        "attempted_at": case["judge_attempts"][0]["attempted_at"],
        "wait_before_attempt_s": 0.0,
        "http_status": 429,
        "response_raw": {"error": {"message": "rate limited"}},
        "status": "error",
        "error": case["judge_attempts"][0]["error"],
        "retryable": True,
        "retry_reason": "http_429",
        "retry_after_header": "3",
        "wait_before_next_attempt_s": 3.0,
        "wait_source": "retry_after",
    }
    assert case["judge_attempts"][1]["status"] == "completed"
    assert case["judge_attempts"][1]["wait_before_attempt_s"] == 3.0
    assert summary["judge"]["retry_policy"] == {
        "max_attempts": 3,
        "fallback_backoff_s": [2.0, 4.0],
        "inter_case_delay_s": 0.0,
        "retryable_http_statuses": [408, 429, "5xx"],
        "retryable_transport_errors": ["httpx.TransportError"],
    }


def test_retry_after_http_date_is_respected(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    retry_at = now + timedelta(seconds=7)
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                503,
                headers={"Retry-After": format_datetime(retry_at)},
                json={"error": {"message": "busy"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "1"}}]},
        )

    with _client(handler) as client:
        summary = score_run(
            run_results=[_run_results()["cases"][1]],
            frozen_cases=[_frozen_cases()[0]],
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(
                retry_policy=JudgeRetryPolicy(
                    max_attempts=2,
                    fallback_backoff_s=(2,),
                    inter_case_delay_s=0,
                )
            ),
            allow_live=True,
            client=client,
            sleep_fn=sleeps.append,
            now_fn=lambda: now,
        )

    assert calls == 2
    assert sleeps == [7.0]
    assert summary["cases"][0]["judge_attempts"][0]["wait_source"] == (
        "retry_after"
    )


def test_network_and_5xx_failures_use_finite_exponential_fallback(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary connect failure", request=request)
        if calls == 2:
            return httpx.Response(500, json={"error": "temporary server error"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "1"}}]},
        )

    with _client(handler) as client:
        summary = score_run(
            run_results=[_run_results()["cases"][1]],
            frozen_cases=[_frozen_cases()[0]],
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(
                retry_policy=JudgeRetryPolicy(
                    max_attempts=3,
                    fallback_backoff_s=(2, 4),
                    inter_case_delay_s=0,
                )
            ),
            allow_live=True,
            client=client,
            sleep_fn=sleeps.append,
        )

    assert calls == 3
    assert sleeps == [2.0, 4.0]
    attempts = summary["cases"][0]["judge_attempts"]
    assert [attempt["retry_reason"] for attempt in attempts[:2]] == [
        "transport_error",
        "http_500",
    ]
    assert [attempt["wait_source"] for attempt in attempts[:2]] == [
        "exponential_backoff",
        "exponential_backoff",
    ]


def test_transient_retries_stop_at_the_configured_limit(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(408, json={"error": "request timeout"})

    with _client(handler) as client:
        summary = score_run(
            run_results=[_run_results()["cases"][1]],
            frozen_cases=[_frozen_cases()[0]],
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(
                retry_policy=JudgeRetryPolicy(
                    max_attempts=3,
                    fallback_backoff_s=(2, 4),
                    inter_case_delay_s=0,
                )
            ),
            allow_live=True,
            client=client,
            sleep_fn=sleeps.append,
        )

    assert calls == 3
    assert sleeps == [2.0, 4.0]
    case = summary["cases"][0]
    assert case["status"] == "error"
    assert case["judge_attempt_count"] == 3
    assert case["judge_attempts"][-1]["retryable"] is True
    assert case["judge_attempts"][-1]["wait_before_next_attempt_s"] is None


def test_inter_case_pacing_applies_only_between_new_judge_cases(
    tmp_path: Path,
    synthetic_prompt_path: Path,
) -> None:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "1"}}]},
        )

    with _client(handler) as client:
        summary = score_run(
            run_results=_run_results(),
            frozen_cases=_frozen_cases(),
            output_dir=tmp_path / "score",
            prompt_path=synthetic_prompt_path,
            expected_prompt_sha256=SYNTHETIC_PROMPT_SHA256,
            judge=_judge(
                retry_policy=JudgeRetryPolicy(
                    inter_case_delay_s=1.25,
                )
            ),
            allow_live=True,
            client=client,
            sleep_fn=sleeps.append,
        )

    assert sleeps == [1.25]
    assert summary["cases"][0]["judge_attempts"][0][
        "wait_before_attempt_s"
    ] == 0.0
    assert summary["cases"][1]["judge_attempts"][0][
        "wait_before_attempt_s"
    ] == 1.25
