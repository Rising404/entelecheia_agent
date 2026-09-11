from __future__ import annotations

import json

import pytest

from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
    ReasoningEffort,
)
from personagraph.model_io import gateway as models
from personagraph.l2.task_execution.work_run.model_providers import (
    WorkRunStructuredModelProfile,
    build_attempt_structured_provider,
    build_verification_structured_provider,
)
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_endpoint_identity,
)


def _profile() -> WorkRunStructuredModelProfile:
    return WorkRunStructuredModelProfile(
        attempt_max_output_tokens=4096,
        verification_max_output_tokens=4096,
        timeout_s=60.0,
    )


def test_structured_endpoint_identity_binds_normalized_physical_profile_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "model-a")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-private-a")
    monkeypatch.setenv(
        "PERSONAGRAPH_BASE_URL",
        "HTTPS://user:password@PRIVATE.EXAMPLE:443/gateway/",
    )
    first = configured_structured_model_endpoint_identity()

    monkeypatch.setenv(
        "PERSONAGRAPH_BASE_URL",
        "https://user:password@private.example/gateway",
    )
    equivalent = configured_structured_model_endpoint_identity()

    assert first == equivalent
    assert first.provider == "anthropic-compatible"
    assert first.protocol == "anthropic-messages-2023-06-01"
    assert len(first.endpoint_fingerprint) == 64
    assert len(first.control_profile_id) == 64
    projection = repr(first)
    assert "sk-private-a" not in projection
    assert "password" not in projection
    assert "private.example" not in projection.lower()


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("PERSONAGRAPH_BASE_URL", "https://other.example/v1"),
        ("PERSONAGRAPH_MODEL", "model-b"),
        ("PERSONAGRAPH_API_KEY", "sk-private-b"),
    ],
)
def test_structured_endpoint_identity_changes_with_effective_physical_profile(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed: str,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "model-a")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-private-a")
    baseline = configured_structured_model_endpoint_identity()

    monkeypatch.setenv(field, changed)
    updated = configured_structured_model_endpoint_identity()

    assert updated.endpoint_fingerprint != baseline.endpoint_fingerprint


def test_openai_endpoint_suffix_and_default_port_are_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "model-a")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-private-a")
    monkeypatch.setenv(
        "PERSONAGRAPH_BASE_URL",
        "HTTPS://API.EXAMPLE:443/v1/",
    )
    assembled = configured_structured_model_endpoint_identity()

    monkeypatch.setenv(
        "PERSONAGRAPH_BASE_URL",
        "https://api.example/v1/chat/completions",
    )
    explicit = configured_structured_model_endpoint_identity()

    assert assembled.endpoint_fingerprint == explicit.endpoint_fingerprint
    assert explicit.protocol == "openai-chat-completions-v1"


def test_mock_attempt_provider_derives_complete_submit_from_exact_payload() -> None:
    provider = build_attempt_structured_provider(_profile())
    result = provider(
        "system",
        json.dumps(
            {
                "node": {
                    "title": "整理计划",
                    "objective": "给出计划",
                    "acceptances": [
                        {"acceptance_id": "answer", "criterion": "给出回答"},
                        {"acceptance_id": "format", "criterion": "Markdown"},
                    ],
                },
                "user_input": {
                    "content": "请整理今天的计划",
                    "prior_waiting_user_question": None,
                },
            },
            ensure_ascii=False,
        ),
        model_call_id="model-attempt",
        purpose="runtime_work_run_attempt_decision",
    )

    payload = json.loads(result.reply)
    assert payload["action"]["kind"] == "submit_output_window"
    assert "请整理今天的计划" in payload["action"]["content"]
    assert [item["acceptance_id"] for item in payload["acceptance_updates"]] == [
        "answer",
        "format",
    ]
    assert all(item["model_claimed_satisfied"] for item in payload["acceptance_updates"])
    assert all(
        item["empty_support_justification"]["reason_code"]
        == "candidate_is_primary_artifact"
        for item in payload["acceptance_updates"]
    )


def test_mock_attempt_provider_keeps_legacy_current_input_compatibility() -> None:
    provider = build_attempt_structured_provider(_profile())
    result = provider(
        "system",
        json.dumps(
            {
                "node": {
                    "title": "整理计划",
                    "objective": "给出计划",
                    "acceptances": [
                        {"acceptance_id": "answer", "criterion": "给出回答"}
                    ],
                },
                "current_user_input": "兼容旧版 Prompt",
            },
            ensure_ascii=False,
        ),
        model_call_id="model-attempt-legacy",
        purpose="runtime_work_run_attempt_decision",
    )

    assert "兼容旧版 Prompt" in json.loads(result.reply)["action"]["content"]


def test_mock_auxiliary_terminal_uses_host_task_graph_source_contract() -> None:
    provider = build_attempt_structured_provider(_profile())
    result = provider(
        "system",
        json.dumps(
            {
                "node": {
                    "title": "形成正式任务图",
                    "objective": "根据已验证依赖形成可执行任务图",
                    "acceptances": [
                        {
                            "acceptance_id": "terminal_ready",
                            "criterion": "任务图可执行",
                            "source_anchor_ids": [
                                "mounted_document_01",
                                "task_creation_source",
                            ],
                        }
                    ],
                },
                "task_graph_proposal_contract": {
                    "schema_version": "insession-task-graph-revision-v2",
                    "allowed_source_anchor_ids": ["task_creation_source"],
                    "required_source_anchor_ids": ["task_creation_source"],
                    "limits": {"max_nodes_per_task": 64, "max_depth": 12},
                },
            },
            ensure_ascii=False,
        ),
        model_call_id="auxiliary-terminal-model",
        purpose="runtime_auxiliary_v2_attempt_decision",
    )

    payload = json.loads(result.reply)
    assert payload["action"]["kind"] == "submit_task_graph"
    root = payload["action"]["proposal"]["root"]["nodes"][0]
    assert root["source_anchor_ids"] == ["task_creation_source"]
    assert root["acceptance_criteria"][0]["source_anchor_ids"] == [
        "task_creation_source"
    ]


def test_mock_positive_terminal_emits_changed_root_revision_lineage() -> None:
    provider = build_attempt_structured_provider(_profile())
    result = provider(
        "system",
        json.dumps(
            {
                "node": {
                    "title": "修订正式任务图",
                    "objective": "根据整体验证失败修订任务图",
                    "acceptances": [
                        {
                            "acceptance_id": "terminal_ready",
                            "criterion": "订正版任务图可执行",
                            "source_anchor_ids": ["task_creation_source"],
                        }
                    ],
                },
                "task_graph_proposal_contract": {
                    "schema_version": "insession-task-graph-revision-v2",
                    "allowed_source_anchor_ids": ["task_creation_source"],
                    "required_source_anchor_ids": ["task_creation_source"],
                    "limits": {"max_nodes_per_task": 64, "max_depth": 12},
                },
                "task_graph_revision_base": {
                    "base_task_graph_revision": 1,
                    "root_node_alias": "base_node_000",
                    "nodes": [
                        {
                            "node_alias": "base_node_000",
                            "parent_node_alias": None,
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ),
        model_call_id="auxiliary-positive-terminal-model",
        purpose="runtime_auxiliary_v2_attempt_decision",
    )

    payload = json.loads(result.reply)
    assert payload["action"]["lineage"] == [
        {
            "proposal_node_key": "root",
            "disposition": "revise",
            "base_node_alias": "base_node_000",
        }
    ]
    assert "修复整体验证失败" in payload["action"]["proposal"]["root"][
        "nodes"
    ][0]["acceptance_criteria"][0]["criterion"]



def test_mock_synthesis_root_uses_only_verified_child_delivery_bodies() -> None:
    provider = build_attempt_structured_provider(_profile())
    result = provider(
        "system",
        json.dumps(
            {
                "node": {
                    "title": "论文综合",
                    "objective": "综合子档案",
                    "acceptances": [
                        {"acceptance_id": "root", "criterion": "综合完成"}
                    ],
                },
                "allowed_tools": [],
                "dependency_deliveries": [
                    {
                        "delivery_id": "delivery-p1",
                        "output_window": {
                            "content": "P1 dossier body [P1:C1]"
                        },
                    },
                    {
                        "delivery_id": "delivery-p2",
                        "output_window": {
                            "content": "P2 dossier body [P2:C1]"
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        model_call_id="paper-root-submit",
        purpose="runtime_work_run_attempt_decision",
    )

    reply = json.loads(result.reply)
    content = reply["action"]["content"]
    assert "P1 dossier body [P1:C1]" in content
    assert "P2 dossier body [P2:C1]" in content
    assert (
        reply["acceptance_updates"][0]["empty_support_justification"][
            "reason_code"
        ]
        == "dependency_delivery_sufficient"
    )


def test_mock_verifier_covers_every_acceptance_in_prompt_order() -> None:
    provider = build_verification_structured_provider(_profile())
    result = provider(
        "system",
        json.dumps(
            {
                "node": {
                    "acceptances": [
                        {"acceptance_id": "one"},
                        {"acceptance_id": "two"},
                    ]
                }
            }
        ),
        model_call_id="model-verifier",
        purpose="runtime_task_node_semantic_verification",
    )

    payload = json.loads(result.reply)
    assert [item["acceptance_id"] for item in payload["acceptance_results"]] == [
        "one",
        "two",
    ]
    assert all(item["verdict"] == "passed" for item in payload["acceptance_results"])


def test_profile_rejects_non_positive_physical_limits() -> None:
    with pytest.raises(ValueError, match="attempt_max_output_tokens"):
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=0,
            verification_max_output_tokens=1,
            timeout_s=1,
        )


def test_openai_structured_call_sends_the_explicit_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def request(endpoint, admitted_request, api_key, timeout_s):
        captured.update(
            endpoint=endpoint,
            payload=json.loads(
                admitted_request.admitted_request.body_for_dispatch()
            ),
            api_key=api_key,
            timeout_s=timeout_s,
        )
        return {
            "choices": [
                {
                    "message": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(models, "_openai_request", request)
    monkeypatch.setattr(models, "record_model_call", lambda **_kwargs: None)
    binding = ModelTierBinding(
        tier=ModelTier.ATTEMPT,
        provider="openai-compatible",
        base_url="https://api.openai.com/v1",
        model="gpt-reasoning",
        api_key="sk-test",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
        reasoning_effort=ReasoningEffort.HIGH,
    )

    models.complete_structured(
        "system",
        "{}",
        mock_payload={},
        model_call_id="openai-effort-call",
        binding=binding,
    )

    assert captured["payload"]["reasoning_effort"] == "high"
    assert captured["payload"]["max_completion_tokens"] == 1200
    assert "max_tokens" not in captured["payload"]
    assert "temperature" not in captured["payload"]


def test_mock_provider_rejects_malformed_prompt_payload() -> None:
    provider = build_attempt_structured_provider(_profile())
    with pytest.raises(ValueError, match="not JSON"):
        provider(
            "system",
            "not-json",
            model_call_id="model-attempt",
            purpose="runtime_work_run_attempt_decision",
        )
