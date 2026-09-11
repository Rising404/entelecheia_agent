"""显式启用：真实 L1 自主读文件、短引用观察、验证及交付的一轮冒烟。

只外发本测试生成的非敏感文本，配置只读，状态与工作区隔离；不是 DocBench 质量评测。
工具选择、执行决定和验证结果均来自真实模型，不预制动作或修补模型回复。
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import pytest


@pytest.mark.skipif(
    os.environ.get("PERSONAGRAPH_RUN_L1_SHORT_REFERENCES_LIVE") != "1",
    reason="explicit short-reference API smoke opt-in required",
)
def test_live_l1_short_reference_turn(monkeypatch, tmp_path):
    from personagraph.configuration import app_settings
    from personagraph.model_io import endpoint_profiles
    from personagraph.model_io.gateway import PreparedModelCall, complete_structured
    from personagraph.model_io.tier_bindings import ModelTier, resolve_tier
    from personagraph.runtime import entry
    from personagraph.runtime.entry.ingress import model as ingress_model
    from personagraph.runtime.entry.routing.policy import TurnRoutingPolicy, freeze_turn_routing_policy
    from personagraph.runtime.l1 import execution_config, model, semantic_verification
    from personagraph.session import store

    config_path = Path(os.environ["PERSONAGRAPH_L1_SHORT_REFERENCES_LIVE_CONFIG"])
    assert config_path.is_absolute()
    with monkeypatch.context() as settings:
        settings.setattr(app_settings, "CONFIG_PATH", config_path)
        settings.setattr(endpoint_profiles, "CONFIG_PATH", config_path.with_name("model_profiles.json"))
        settings.delenv("PERSONAGRAPH_MODEL_PROVIDER", raising=False)
        l1_binding = resolve_tier(ModelTier.L1)
        # 此冒烟统一使用用户的 L1 端点；不改写本机 Router 配置或继承其独立实验设置。
        bindings = {ModelTier.L1: l1_binding, ModelTier.ROUTER: replace(l1_binding, tier=ModelTier.ROUTER)}
    assert all(value.provider != "mock" and value.api_key for value in bindings.values())
    monkeypatch.setattr(execution_config, "resolve_tier", bindings.__getitem__)
    monkeypatch.setattr(ingress_model, "resolve_tier", bindings.__getitem__)
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", bindings[ModelTier.L1].provider)
    monkeypatch.setenv("PERSONAGRAPH_TRAJECTORY", "off")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text(
        "项目记录：早期草案中的样本数是 21，该数字已废弃。\n"
        "最终核准的样本数为 37。本文件不含任何真实项目数据。\n", encoding="utf-8",
    )
    session_id = store.create_session("Entelecheia", working_dir=str(workspace))
    metrics = []
    model_views = []

    class ObservedProvider:
        def prepare(self, system_prompt, user_content, **kwargs):
            purpose = kwargs["purpose"]
            if purpose in {"runtime_l1_attempt", "runtime_l1_semantic_verifier"}:
                view = json.loads(user_content)
                model_views.append((purpose, view))
                serialized = json.dumps(view)
                for internal in ("_host_reference_bindings", "result_sha256", "candidate_binding"):
                    assert internal not in serialized
                if kwargs.get("repair_messages"):
                    feedback_text = kwargs["repair_messages"][3]["content"]
                    assert "rejected_response_sha256" not in feedback_text
            prepared = complete_structured.prepare(system_prompt, user_content, **kwargs)

            def dispatch(model_call_id):
                response = prepared.dispatch(model_call_id=model_call_id)
                metrics.append({"purpose": purpose, "latency_ms": response.latency_ms,
                                "is_repair": kwargs.get("repair_messages") is not None,
                                "input_tokens": response.input_tokens, "output_tokens": response.output_tokens})
                print(json.dumps({"api_response": len(metrics), **metrics[-1]}, ensure_ascii=False), flush=True)
                return response

            return PreparedModelCall(_dispatch=dispatch)

    provider = ObservedProvider()
    for module in (model, semantic_verification, ingress_model):
        monkeypatch.setattr(module, "complete_structured", provider)
    start = time.monotonic()
    result = entry.run_entry_turn(
        user_input="请阅读工作目录中的 notes.txt，告诉我最终核准的样本数，并简短说明依据。",
        features={"context_guard_limit": 262_144, "l1_max_attempts": 6,
                  "l1_max_tool_calls_per_attempt": 4, "turn_wall_clock_budget_s": 300},
        session_id=session_id, client_request_id="short-reference-live-smoke",
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=True, l2_enabled=False), source="request_override",
        ), store=store,
    )
    execution = store.get_l1_turn_execution(session_id=session_id, turn_id=result.turn_id)
    report = {"status": result.status, "processing_level": result.processing_level,
              "end_reason": result.end_reason, "error_code": result.error_code,
              "reply": result.reply, "elapsed_seconds": round(time.monotonic() - start, 2),
              "api_calls": metrics, "model": bindings[ModelTier.L1].model,
              "router_binding": "isolated_test_uses_l1_endpoint",
              "thinking_enabled": bindings[ModelTier.L1].thinking_enabled,
              "tool_calls": [{"tool_id": item["tool_id"], "status": item["status"]}
                             for item in (execution or {}).get("tool_calls", [])]}
    (tmp_path / "live_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    assert result.status == "completed" and result.processing_level == "L1"
    assert result.reply and "37" in result.reply
    assert execution and any(item["status"] == "succeeded" for item in execution["tool_calls"])
    assert any(view.get("prior_tool_results") for purpose, view in model_views if purpose == "runtime_l1_attempt")
    assert any(purpose == "runtime_l1_semantic_verifier" for purpose, _ in model_views)
    sent = len(metrics)
    replayed = entry.run_entry_turn(
        user_input="请阅读工作目录中的 notes.txt，告诉我最终核准的样本数，并简短说明依据。",
        features={"context_guard_limit": 24_000}, session_id=session_id,
        client_request_id="short-reference-live-smoke", store=store,
    )
    assert replayed == result and len(metrics) == sent
