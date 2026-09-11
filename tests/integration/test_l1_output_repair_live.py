"""显式启用：历史真实拒绝原文 → 现行四消息 → 真实模型格式修复。

只读原实验数据库，首份坏输出来自历史记录而非新模型调用；后续修复才真实外发。
不恢复旧任务、不执行其工具、不评价答案语义，不把格式通过称作 DocBench 答对。
"""

from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import time

import pytest


@pytest.mark.skipif(
    os.environ.get("PERSONAGRAPH_RUN_L1_REPAIR_LIVE") != "1",
    reason="explicit L1 repair API opt-in required",
)
@pytest.mark.parametrize("case_id", ["5:1", "15:1", "54:6", "110:3"])
def test_live_l1_repair_from_recorded_failure(monkeypatch, tmp_path, case_id):
    from personagraph.configuration import app_settings
    from personagraph.model_io import endpoint_profiles
    from personagraph.model_io.gateway import DEFAULT_MODEL_TIMEOUT_S, ModelResult, PreparedModelCall, complete_structured
    from personagraph.model_io.output_validation import ModelOutputValidationError
    from personagraph.model_io.prepared_structured_provider import prepare_structured_repair_request
    from personagraph.model_io.tier_bindings import ModelTier, resolve_tier
    from personagraph.runtime.l1.execution_notes import validate_execution_notes_observation_coverage
    from personagraph.runtime.l1.model import _validate_l1_decision_proposal
    from personagraph.runtime.l1.model_authority import L1_ATTEMPT_RESULT_CONTRACT
    from personagraph.runtime.l1.model_output_budgets import l1_attempt_max_output_tokens
    from personagraph.runtime.model_calls.requests import request_model_with_retry
    from personagraph.runtime.turn_events import RuntimeStage

    run_root = Path(os.environ["PERSONAGRAPH_L1_REPAIR_LIVE_RUN"])
    config_path = Path(os.environ["PERSONAGRAPH_L1_REPAIR_LIVE_CONFIG"])
    assert run_root.is_absolute() and config_path.is_absolute()
    databases = list((run_root / "cases" / f"docbench:{case_id}" / "state" / "sessions").glob("*/session.sqlite"))
    assert len(databases) == 1
    with sqlite3.connect(databases[0].as_uri() + "?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT r.response_text, c.request_json FROM insession_runtime_model_rejected_outputs r "
            "JOIN insession_runtime_model_logical_calls c USING(logical_call_id) "
            "ORDER BY c.created_at DESC, r.physical_ordinal LIMIT 1"
        ).fetchone()
    assert row is not None
    rejected_text = row[0]
    original = json.loads(row[1])["structured_prompt"]
    user_content = original["user_content"]
    payload = json.loads(user_content)

    # 只在明确指定的配置中读取 L1 绑定；从不打印或写入凭据。
    with monkeypatch.context() as settings:
        settings.setattr(app_settings, "CONFIG_PATH", config_path)
        settings.setattr(endpoint_profiles, "CONFIG_PATH", config_path.with_name("model_profiles.json"))
        settings.delenv("PERSONAGRAPH_MODEL_PROVIDER", raising=False)
        binding = resolve_tier(ModelTier.L1)
    assert binding.provider != "mock" and binding.api_key
    monkeypatch.setenv("PERSONAGRAPH_TRAJECTORY", "off")
    # 显式 binding 已冻结真实提供方；测试默认 mock 环境不得覆盖这次 opt-in。
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", binding.provider)

    metrics = []
    sent_feedback = []

    class ObservedProvider:
        def prepare(self, system_prompt, content, **kwargs):
            messages = kwargs.get("repair_messages")
            assert messages is not None
            assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
            assert messages[2]["content"] == rejected_text
            sent_feedback.append(messages[3]["content"])
            prepared = complete_structured.prepare(system_prompt, content, **kwargs)

            def dispatch(model_call_id):
                response = prepared.dispatch(model_call_id=model_call_id)
                metrics.append(response)
                (tmp_path / "live_response.json").write_text(response.reply, encoding="utf-8")
                print(json.dumps({
                    "case_id": case_id, "provider_response_received": True,
                    "reply_sha256": sha256(response.reply.encode()).hexdigest(),
                    "same_as_recorded_rejection": response.reply == rejected_text,
                    "finish_reason": response.finish_reason,
                }, ensure_ascii=False))
                return response

            return PreparedModelCall(_dispatch=dispatch)

    def validate(result):
        try:
            decision = _validate_l1_decision_proposal(json.loads(result.reply), payload=payload)
        except ModelOutputValidationError as exc:
            print(json.dumps({
                "case_id": case_id, "validation_source": "live" if metrics else "recorded",
                "issues": [issue.model_dump(mode="json") for issue in exc.repair_issues or ()],
            }, ensure_ascii=False))
            raise
        # 这些调用与生产复用同一规则；不为现场结果伪造新的工具或引用。
        validate_execution_notes_observation_coverage(decision, request=payload)
        known = {item["tool_call_id"]: item for item in payload.get("prior_tool_results", [])}
        for observation in decision.execution_notes.observations:
            for ref in observation.source_refs:
                if ref.tool_call_id in known:
                    assert ref.result_sha256 == known[ref.tool_call_id]["result_sha256"]
        return decision

    start = time.monotonic()
    repair = prepare_structured_repair_request(
        # 历史 long-ID 协议只能配其冻结提示词，不能混用当前短引用提示词。
        ObservedProvider(), system_prompt=original["system_prompt"], user_content=user_content,
        purpose="runtime_l1_attempt",
        prepare_kwargs=lambda: {
            "mock_payload": {}, "max_tokens": l1_attempt_max_output_tokens(thinking_enabled=binding.thinking_enabled),
            "timeout_s": DEFAULT_MODEL_TIMEOUT_S,
            "json_mode": True, "binding": binding,
        },
    )
    result = request_model_with_retry(
        turn_id=f"isolated-repair-{case_id}", session_id=None, purpose="runtime_l1_attempt",
        stage=RuntimeStage.L1_BOOTSTRAP,
        prepare_request=lambda: PreparedModelCall(_dispatch=lambda model_call_id: ModelResult(
            reply=rejected_text, provider="historical-replay", model="recorded-invalid-output",
            model_call_id=model_call_id, latency_ms=0,
        )),
        prepare_repair_request=repair, repair_target_contract=L1_ATTEMPT_RESULT_CONTRACT,
        validate=validate, emit=lambda _event: None, max_attempts=2,
    )
    assert result.attempts == 2 and len(metrics) == 1
    expected = (
        "/execution_notes/observations/0/source_refs/0/result_sha256"
        if case_id == "54:6" else "/execution_notes/observations"
    )
    assert expected in sent_feedback[0]
    assert "schema.tuple_type" not in sent_feedback[0]
    print(json.dumps({
        "case_id": case_id, "source": "recorded_failure_then_live_repair",
        "model": binding.model, "thinking_enabled": binding.thinking_enabled,
        "real_api_calls": len(metrics), "elapsed_seconds": round(time.monotonic() - start, 2),
        "input_tokens": metrics[0].input_tokens, "output_tokens": metrics[0].output_tokens,
        "schema_and_observation_coverage_valid": True, "action": result.value.action.kind,
    }, ensure_ascii=False))
