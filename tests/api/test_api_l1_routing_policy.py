"""嵌套式逐轮运行时策略的 API 准入测试。"""

from __future__ import annotations

import json

import pytest

from personagraph.api import service
from personagraph.api.service.errors import ApiError
from personagraph.model_io.gateway import ModelResult
from personagraph.session import store as session_store
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _payload(session_id: str, request_id: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "client_request_id": request_id,
        "message": "hello",
    }


def test_api_rejects_mutually_enabled_l1_and_l2_before_acceptance() -> None:
    session_id = session_store.create_session("Entelecheia")
    payload = _payload(session_id, "double-enabled")
    payload["runtime_policy"] = {"l1_enabled": True, "l2_enabled": True}

    with pytest.raises(ApiError) as captured:
        service.chat_turn(payload)

    assert captured.value.code == "INVALID_RUNTIME_POLICY"
    assert captured.value.status == 400
    assert session_store.get_turns(session_id) == []


def test_session_detail_projects_effective_policy_and_host_capabilities() -> None:
    session_id = session_store.create_session("Entelecheia")

    enabled = service.get_session(session_id)["runtime_routing"]
    assert enabled == {
        "schema_version": 1,
        "source": "default",
        "policy": {
            "schema_version": 1,
            "l1_enabled": True,
            "l2_enabled": False,
        },
        "allowed_processing_levels": ["L0", "L1"],
        "available_processing_levels": ["L0", "L1", "L2"],
    }

def test_system_status_publishes_available_levels_and_product_default() -> None:
    routing = service.system_status()["runtime_routing"]

    assert routing == {
        "schema_version": 1,
        "available_processing_levels": ["L0", "L1", "L2"],
        "default_policy": {
            "schema_version": 1,
            "l1_enabled": True,
            "l2_enabled": False,
        },
        "default_source": "default",
    }


@pytest.mark.parametrize(
    "policy,expected_levels",
    (
        ({"l1_enabled": False, "l2_enabled": True}, ["L0", "L2"]),
        ({"l1_enabled": False, "l2_enabled": False}, ["L0"]),
    ),
)
def test_api_persists_an_explicit_policy_as_the_session_default(
    policy: dict[str, bool],
    expected_levels: list[str],
    tmp_path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    payload = _payload(session_id, "explicit-policy")
    payload["runtime_policy"] = policy

    response = service.chat_turn(payload)

    assert response["result"]["status"] == "completed"
    stored = session_store.get_session_turn_routing_policy(session_id)
    assert stored is not None
    assert json.loads(str(stored["policy_json"])) == {
        "schema_version": 1,
        **policy,
    }
    receipt = session_store.get_turn_execution_for_client_request(
        session_id=session_id,
        client_request_id="explicit-policy",
    )
    assert receipt is not None
    snapshot = json.loads(str(receipt["routing_policy"]["snapshot_json"]))  # type: ignore[index]
    assert snapshot["source"] == "request_override"
    assert snapshot["allowed_processing_levels"] == expected_levels
    detail = service.get_session(session_id)["runtime_routing"]
    assert detail["source"] == "session_default"
    assert detail["allowed_processing_levels"] == expected_levels


def test_accepted_l1_request_replays_from_its_immutable_policy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    bound_partitioned_session,
) -> None:
    monkeypatch.setattr(
        "personagraph.api.service.sessions._schedule_pending_turn_post_commit_jobs",
        lambda _session_id: None,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    payload = _payload(session_id, "l1-policy-replay")
    payload["runtime_policy"] = {"l1_enabled": True, "l2_enabled": False}
    def classify(*_args: object, **kwargs: object) -> ModelResult:
        return ModelResult(
            reply=json.dumps({"processing_level": "L1", "task_matches": []}),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=str(kwargs["model_call_id"]),
        )

    monkeypatch.setattr(
        "personagraph.runtime.entry.ingress.model.complete_structured",
        as_prepared_test_provider(classify),
    )

    first = service.chat_turn(payload)
    replay_payload = _payload(session_id, "l1-policy-replay")
    replayed = service.chat_turn(replay_payload)

    assert first["result"]["processing_level"] == "L1"
    assert first["result"]["status"] == "completed"
    assert first["result"]["error_code"] is None
    settled = session_store.get_turn_execution_for_client_request(
        session_id=session_id,
        client_request_id="l1-policy-replay",
    )
    assert settled is not None
    settled_turn = settled["turn"]
    assert settled_turn["turn_id"] == first["result"]["turn_id"]
    assert settled_turn["status"] == "completed"
    assert settled_turn["processing_level"] == "L1"
    assert settled_turn["error_code"] is None
    assert settled_turn["end_reason"] is None
    assert settled_turn["completed_at"]
    formal_pair = session_store.get_committed_turn_pair(
        session_id,
        f"commit_{first['result']['turn_id']}",
    )
    assert formal_pair is not None
    assert formal_pair["turn_id"] == first["result"]["turn_id"]
    assert formal_pair["assistant_content"] == first["result"]["reply"]
    assert replayed["result"] == first["result"]


def test_accepted_l1_request_replays_with_its_explicit_policy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    bound_partitioned_session,
) -> None:
    monkeypatch.setattr(
        "personagraph.api.service.sessions._schedule_pending_turn_post_commit_jobs",
        lambda _session_id: None,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    payload = _payload(session_id, "l1-explicit-replay")
    payload["runtime_policy"] = {"l1_enabled": True, "l2_enabled": False}

    def classify(*_args: object, **kwargs: object) -> ModelResult:
        return ModelResult(
            reply=json.dumps({"processing_level": "L1", "task_matches": []}),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=str(kwargs["model_call_id"]),
        )

    monkeypatch.setattr(
        "personagraph.runtime.entry.ingress.model.complete_structured",
        as_prepared_test_provider(classify),
    )

    first = service.chat_turn(payload)
    replay = _payload(session_id, "l1-explicit-replay")
    replay["runtime_policy"] = {"l1_enabled": True, "l2_enabled": False}
    replayed = service.chat_turn(replay)

    assert replayed["result"] == first["result"]


def test_replay_rejects_a_changed_runtime_policy(
    tmp_path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    first = _payload(session_id, "policy-replay-mismatch")
    first["runtime_policy"] = {"l1_enabled": False, "l2_enabled": False}
    service.chat_turn(first)

    changed = _payload(session_id, "policy-replay-mismatch")
    changed["runtime_policy"] = {"l1_enabled": False, "l2_enabled": True}
    with pytest.raises(ApiError) as captured:
        service.chat_turn(changed)

    assert captured.value.code == "CLIENT_REQUEST_ID_REUSED"
    assert captured.value.status == 409
