import httpx
import pytest

from personagraph.model_io import gateway as models
from personagraph.model_io.gateway import ModelGatewayError


class _FakeClient:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return self.response


class _FakeResponse:
    def __init__(self, *, data=None, json_exc=None, status_exc=None):
        self.data = data
        self.json_exc = json_exc
        self.status_exc = status_exc

    def raise_for_status(self):
        if self.status_exc:
            raise self.status_exc

    def json(self):
        if self.json_exc:
            raise self.json_exc
        return self.data


def _env(monkeypatch):
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("PERSONAGRAPH_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "test-model")


def test_missing_api_key_is_model_call_failed(monkeypatch):
    monkeypatch.delenv("PERSONAGRAPH_API_KEY", raising=False)

    with pytest.raises(ModelGatewayError) as exc:
        models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert exc.value.code == "MODEL_CALL_FAILED"
    assert exc.value.retryable is False
    assert exc.value.details["reason"] == "missing_api_key"


def test_provider_http_error_is_model_call_failed(monkeypatch):
    _env(monkeypatch)
    request = httpx.Request("POST", "https://example.test/anthropic/v1/messages")
    response = httpx.Response(500, request=request)
    status_exc = httpx.HTTPStatusError("server error", request=request, response=response)
    monkeypatch.setattr(models.httpx, "Client", lambda *a, **k: _FakeClient(_FakeResponse(status_exc=status_exc)))

    with pytest.raises(ModelGatewayError) as exc:
        models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert exc.value.code == "MODEL_CALL_FAILED"
    assert exc.value.retryable is True
    assert exc.value.details["status_code"] == 500
    assert exc.value.details["endpoint"] == "https://example.test/anthropic/v1/messages"


def test_provider_timeout_is_distinct_and_uses_the_sixty_second_default(monkeypatch):
    _env(monkeypatch)
    captured: dict = {}

    class _TimeoutClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *_args, **_kwargs):
            raise httpx.ReadTimeout("timed out")

    def build_client(*_args, **kwargs):
        captured.update(kwargs)
        return _TimeoutClient()

    monkeypatch.setattr(models.httpx, "Client", build_client)

    with pytest.raises(ModelGatewayError) as exc:
        models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert exc.value.code == "MODEL_CALL_TIMEOUT"
    assert exc.value.retryable is True
    assert exc.value.details["timeout_s"] == 60.0
    assert captured["timeout"] == 60.0


def test_provider_invalid_json_is_model_bad_response(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(models.httpx, "Client", lambda *a, **k: _FakeClient(_FakeResponse(json_exc=ValueError("bad json"))))

    with pytest.raises(ModelGatewayError) as exc:
        models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert exc.value.code == "MODEL_BAD_RESPONSE"
    assert exc.value.retryable is True
    assert exc.value.details["exception_type"] == "ValueError"


def test_provider_unextractable_text_is_model_bad_response(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(models.httpx, "Client", lambda *a, **k: _FakeClient(_FakeResponse(data={"unexpected": []})))

    with pytest.raises(ModelGatewayError) as exc:
        models.anthropic_compatible_chat([{"role": "user", "content": "hi"}])

    assert exc.value.code == "MODEL_BAD_RESPONSE"
    assert exc.value.retryable is True
    assert exc.value.details["response_keys"] == ["unexpected"]
