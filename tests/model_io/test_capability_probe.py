from personagraph.model_io import capabilities, probe
from personagraph.model_io.contracts import ToolCall, ToolCallBatch
from personagraph.model_io.gateway import ModelResult


def test_probe_records_safe_matching_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities, "CAPABILITY_RECORD_PATH", tmp_path / "capabilities.json")
    monkeypatch.setattr(probe, "record_native_probe", capabilities.record_native_probe)
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic-compatible")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "model-a")

    def fake_chat(*_args, **kwargs):
        assert kwargs["control_transport"] == "native"
        assert kwargs["tool_choice"]["name"] == probe.PROBE_TOOL
        return ModelResult(
            "", "fake", "model-a", 1,
            output=ToolCallBatch(calls=[ToolCall(
                call_id="toolu_probe",
                tool_name=probe.PROBE_TOOL,
                arguments={"nonce": "r1"},
                transport="native",
            )]),
        )

    monkeypatch.setattr(probe, "anthropic_compatible_chat", fake_chat)
    result = probe.run_native_tool_probe()
    assert result["passed"] is True
    assert result["request_dialect"] == "generic-anthropic"
    stored = capabilities.native_probe_record(
        provider="anthropic-compatible",
        base_url="https://example.test/anthropic",
        model="model-a",
    )
    assert stored is not None
    assert "api_key" not in str(stored)
    assert "raw" not in str(stored)


def test_openai_probe_dispatches_to_the_openai_gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities, "CAPABILITY_RECORD_PATH", tmp_path / "capabilities.json")
    monkeypatch.setattr(probe, "record_native_probe", capabilities.record_native_probe)
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "gpt-test")
    called = []

    def fake_openai(*_args, **kwargs):
        called.append(kwargs)
        return ModelResult(
            "", "openai-compatible", "gpt-test", 1,
            output=ToolCallBatch(calls=[ToolCall(
                call_id="call_probe",
                tool_name=probe.PROBE_TOOL,
                arguments={"nonce": "r1"},
                transport="native",
            )]),
        )

    monkeypatch.setattr(probe, "openai_compatible_chat", fake_openai)
    result = probe.run_native_tool_probe()

    assert result["passed"] is True
    assert result["request_dialect"] == "openai-native"
    assert called[0]["control_transport"] == "native"
    assert called[0]["tool_choice"]["name"] == probe.PROBE_TOOL


def test_probe_record_whitelists_evidence_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities, "CAPABILITY_RECORD_PATH", tmp_path / "capabilities.json")
    record = capabilities.record_native_probe(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="model-a",
        passed=False,
        evidence={
            "error_code": "MODEL_CALL_FAILED",
            "status_code": 400,
            "exception_type": "HTTPStatusError",
            "raw_response": "secret",
            "api_key": "secret",
        },
    )
    assert record["evidence"] == {
        "error_code": "MODEL_CALL_FAILED",
        "status_code": 400,
        "exception_type": "HTTPStatusError",
    }
