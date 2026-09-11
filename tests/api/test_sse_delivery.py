from __future__ import annotations

import errno
from io import BytesIO
from types import MethodType

import pytest

from personagraph.api import router, server
from personagraph.api.sse import SseDelivery, is_client_disconnect


@pytest.mark.parametrize(
    "error",
    [
        BrokenPipeError(),
        ConnectionResetError(),
        ConnectionAbortedError(),
        OSError(errno.ENOTCONN, "not connected"),
    ],
)
def test_disconnect_errors_disable_later_delivery(error):
    attempts: list[str] = []

    def send(event, payload):
        attempts.append(event)
        raise error

    delivery = SseDelivery(send)

    assert delivery.send("running", {}) is False
    assert delivery.send("final", {}) is False
    assert delivery.connected is False
    assert attempts == ["running"]
    assert is_client_disconnect(error) is True


def test_non_connection_errors_are_not_hidden():
    error = ValueError("invalid payload")
    delivery = SseDelivery(lambda event, payload: (_ for _ in ()).throw(error))

    with pytest.raises(ValueError, match="invalid payload"):
        delivery.send("running", {})

    assert delivery.connected is True
    assert is_client_disconnect(error) is False


def _json_handler(write):
    handler = object.__new__(server.ApiHandler)
    handler.path = "/api/status"
    handler.headers = {}
    handler.rfile = BytesIO()
    handler.wfile = type("ResponseWriter", (), {"write": write})()
    handler.send_response = MethodType(lambda self, status: None, handler)
    handler.send_header = MethodType(lambda self, key, value: None, handler)
    handler.end_headers = MethodType(lambda self: None, handler)
    handler._require_allowed_origin = MethodType(lambda self: None, handler)
    handler._require_authorization = MethodType(lambda self: None, handler)
    return handler


def test_json_client_disconnect_does_not_trigger_a_second_error_response(monkeypatch):
    attempts = 0

    def disconnect(_self, _body):
        nonlocal attempts
        attempts += 1
        raise BrokenPipeError("page reloaded")

    handler = _json_handler(disconnect)
    monkeypatch.setattr(
        server,
        "dispatch_response",
        lambda *_args: router.RouteResponse({"ok": True}),
    )

    handler._handle("GET")

    assert attempts == 1


def test_json_non_disconnect_write_error_is_not_hidden():
    def fail(_self, _body):
        raise OSError(errno.EIO, "device error")

    handler = _json_handler(fail)

    with pytest.raises(OSError, match="device error"):
        handler._try_send_json({"ok": True})


def test_route_broken_pipe_is_an_application_error_not_a_client_disconnect(monkeypatch):
    delivered = []
    handler = _json_handler(lambda _self, body: delivered.append(body))
    monkeypatch.setattr(
        server,
        "dispatch_response",
        lambda *_args: (_ for _ in ()).throw(BrokenPipeError("domain subprocess")),
    )

    handler._handle("GET")

    assert len(delivered) == 1
    assert b'"code": "INTERNAL_ERROR"' in delivered[0]
