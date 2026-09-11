"""传输限制与声明式请求契约覆盖。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
from email.message import Message

import pytest

from personagraph.session import project_catalog as session_projects
from personagraph.api import router, server, service
from personagraph.api.contracts import MAX_REQUEST_BODY_BYTES, parse_json_object
from personagraph.workspace.storage import context as document_context
from personagraph.configuration import paths
from personagraph.session import catalog as catalog_module
from personagraph.session.catalog import SessionCatalog
from personagraph.session import store as session_store


def _route_payload(method: str, target: str, body: dict) -> dict:
    return router.dispatch_response(method, target, body).payload


def _handler_with_body(body: bytes, *, content_type: str = "application/json") -> server.ApiHandler:
    handler = object.__new__(server.ApiHandler)
    headers = Message()
    headers["Content-Length"] = str(len(body))
    headers["Content-Type"] = content_type
    handler.headers = headers
    handler.rfile = io.BytesIO(body)
    return handler


def _security_handler(headers: dict[str, str] | None = None, path: str = "/api/status"):
    handler = object.__new__(server.ApiHandler)
    message = Message()
    for key, value in (headers or {}).items():
        message[key] = value
    handler.headers = message
    handler.path = path
    return handler


def _json_response_handler(path: str) -> server.ApiHandler:
    handler = object.__new__(server.ApiHandler)
    handler.path = path
    handler.headers = Message()
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.response_status = None
    handler.response_headers = {}
    handler.send_response = lambda status: setattr(handler, "response_status", status)
    handler.send_header = lambda key, value: handler.response_headers.__setitem__(key, value)
    handler.end_headers = lambda: None
    handler._require_allowed_origin = lambda: None
    handler._require_authorization = lambda: None
    return handler


def test_json_parser_rejects_oversized_or_over_nested_objects():
    with pytest.raises(service.ApiError) as too_large:
        parse_json_object(b"x" * (MAX_REQUEST_BODY_BYTES + 1))
    assert too_large.value.code == "REQUEST_BODY_TOO_LARGE"

    nested: object = "end"
    for _ in range(30):
        nested = {"next": nested}
    with pytest.raises(service.ApiError) as too_deep:
        parse_json_object(json.dumps(nested).encode("utf-8"))
    assert too_deep.value.code == "REQUEST_TOO_COMPLEX"


def test_router_enforces_declared_field_types_and_target_limit():
    with pytest.raises(service.ApiError) as invalid_field:
        _route_payload("POST", "/api/sessions", {"title": 42})
    assert invalid_field.value.code == "INVALID_REQUEST_FIELD"
    assert invalid_field.value.details["field"] == "title"

    with pytest.raises(service.ApiError) as oversized_target:
        _route_payload("GET", "/api/health?" + ("x" * 9000), {})
    assert oversized_target.value.code == "REQUEST_TARGET_TOO_LARGE"


def test_http_boundary_rejects_oversized_and_wrong_media_type_before_dispatch():
    oversized = _handler_with_body(b"", content_type="application/json")
    oversized.headers.replace_header("Content-Length", str(MAX_REQUEST_BODY_BYTES + 1))
    with pytest.raises(service.ApiError) as too_large:
        oversized._read_json()
    assert too_large.value.code == "REQUEST_BODY_TOO_LARGE"

    wrong_media = _handler_with_body(b"{}", content_type="text/plain")
    with pytest.raises(service.ApiError) as unsupported:
        wrong_media._read_json()
    assert unsupported.value.code == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.parametrize("method", ("GET", "POST", "DELETE"))
def test_visual_consent_endpoints_are_not_exposed(method: str) -> None:
    target = "/api/sessions/session-a/vision-disclosures"
    if method == "DELETE":
        target += "/" + "a" * 64

    with pytest.raises(service.ApiError) as retired:
        _route_payload(method, target, {})

    assert retired.value.code == "NOT_FOUND"
    assert retired.value.status == 404


def test_loopback_transport_requires_bearer_and_rejects_unknown_origin(monkeypatch):
    from personagraph.api import security

    monkeypatch.setattr(security, "_API_TOKEN", "x" * 40)
    missing = _security_handler()
    with pytest.raises(service.ApiError) as unauthorized:
        missing._require_authorization()
    assert unauthorized.value.code == "API_AUTH_REQUIRED"

    authenticated = _security_handler({"Authorization": f"Bearer {'x' * 40}"})
    authenticated._require_authorization()

    hostile = _security_handler({"Origin": "https://hostile.example"})
    with pytest.raises(service.ApiError) as forbidden:
        hostile._require_allowed_origin()
    assert forbidden.value.code == "ORIGIN_FORBIDDEN"


def test_cors_preflight_allows_the_binary_attachment_filename_header(monkeypatch):
    handler = _security_handler({"Origin": "http://127.0.0.1:5173"})
    response_headers: dict[str, str] = {}
    handler.send_header = response_headers.__setitem__
    monkeypatch.setattr(server.api_security, "origin_allowed", lambda _origin: True)

    handler._send_cors_headers()

    assert response_headers["Access-Control-Allow-Origin"] == (
        "http://127.0.0.1:5173"
    )
    assert "X-Attachment-Filename" in {
        item.strip()
        for item in response_headers["Access-Control-Allow-Headers"].split(",")
    }


def test_health_is_public_but_origin_policy_still_applies():
    health = _security_handler(path="/api/health")
    health._require_authorization()


def test_dev_auth_bypass_is_limited_to_exact_vite_http_origins(monkeypatch):
    from personagraph.api import security

    monkeypatch.setenv("PERSONAGRAPH_API_ALLOW_UNAUTHENTICATED_DEV_ORIGINS", "1")
    monkeypatch.setenv(
        "PERSONAGRAPH_API_ALLOWED_ORIGINS",
        "http://custom-local.example",
    )
    monkeypatch.setattr(security, "_API_TOKEN", "x" * 40)

    for origin in ("http://127.0.0.1:5174", "http://localhost:5174"):
        assert security.unauthenticated_dev_origin_allowed(origin) is True
        _security_handler({"Origin": origin})._require_authorization()

    for origin in (
        None,
        "null",
        "file://",
        "http://127.0.0.1:5174/",
        "http://custom-local.example",
    ):
        assert security.unauthenticated_dev_origin_allowed(origin) is False
        headers = {} if origin is None else {"Origin": origin}
        with pytest.raises(service.ApiError) as unauthorized:
            _security_handler(headers)._require_authorization()
        assert unauthorized.value.code == "API_AUTH_REQUIRED"


def test_health_challenge_proves_token_ownership_without_echoing_token(monkeypatch):
    from personagraph.api import security

    api_token = "server-owned-token-".ljust(48, "x")
    challenge = base64.urlsafe_b64encode(b"c" * 32).rstrip(b"=").decode("ascii")
    monkeypatch.setattr(security, "_API_TOKEN", api_token)

    payload = _route_payload(
        "GET",
        f"/api/health?challenge={challenge}",
        {},
    )
    expected = base64.urlsafe_b64encode(
        hmac.new(
            api_token.encode("utf-8"),
            security.API_IDENTITY_CONTEXT + challenge.encode("ascii"),
            hashlib.sha256,
        ).digest()
    ).rstrip(b"=").decode("ascii")

    assert payload == {
        "ok": True,
        "service": "personagraph-api",
        "identity": {
            "scheme": security.API_IDENTITY_SCHEME,
            "proof": expected,
        },
    }
    assert api_token not in json.dumps(payload)

    with pytest.raises(service.ApiError) as invalid:
        _route_payload("GET", "/api/health?challenge=too-short", {})
    assert invalid.value.code == "INVALID_IDENTITY_CHALLENGE"


def test_http_boundary_plain_route_still_returns_200():
    handler = _json_response_handler("/api/health")

    handler._handle("GET")

    assert handler.response_status == 200
    assert json.loads(handler.wfile.getvalue()) == {
        "ok": True,
        "service": "personagraph-api",
    }


def test_route_response_preserves_202_for_the_http_transport(monkeypatch):
    payload = {"accepted": True, "job_id": "job-1"}
    accepted_route = router.Route(
        "POST",
        ("api", "accepted"),
        lambda _params, _body, _query: router.RouteResponse(payload, status=202),
    )
    monkeypatch.setattr(router, "ROUTES", (accepted_route,))

    transport_response = router.dispatch_response("POST", "/api/accepted", {})
    assert transport_response == router.RouteResponse(payload, status=202)

    handler = _json_response_handler("/api/accepted")
    handler._handle("POST")

    assert handler.response_status == 202
    assert json.loads(handler.wfile.getvalue()) == payload


def test_http_boundary_api_error_mapping_is_unchanged():
    handler = _json_response_handler("/api/does-not-exist")

    handler._handle("GET")

    assert handler.response_status == 404
    assert json.loads(handler.wfile.getvalue()) == {
        "error": {
            "code": "NOT_FOUND",
            "message": "API path not found",
            "details": {},
        }
    }


def test_raw_attachment_upload_binds_session_and_project_database_scopes(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "partitioned-state"
    project_root = tmp_path / "project"
    state_dir.mkdir()
    project_root.mkdir()
    shared_catalog = state_dir / "project_catalog.sqlite"
    monkeypatch.setattr(session_store, "DB_PATH", session_store._DEFAULT_DB_PATH)
    monkeypatch.setattr(
        catalog_module.paths,
        "PROJECT_CATALOG_DB_PATH",
        shared_catalog,
    )
    monkeypatch.setattr(paths, "PROJECTS_DIR", state_dir / "projects")
    monkeypatch.setattr(session_projects, "DB_PATH", shared_catalog)
    project = session_projects.remember(str(project_root))
    SessionCatalog().create_session(session_id="s1", project_id=project.project_id)

    observed = []
    handler = _json_response_handler("/api/sessions/s1/attachments")
    handler.headers["Content-Length"] = "0"

    def fake_upload(session_id):
        observed.append(
            (
                session_store._SESSION_DATABASE_BINDING.get(),
                document_context.current(),
            )
        )
        return {"attachment": {"session_id": session_id}}

    monkeypatch.setattr(handler, "_handle_attachment_upload", fake_upload)

    handler._handle("POST")

    assert handler.response_status == 200
    assert observed and observed[0][0][0] == "s1"
    assert observed[0][1] is not None
    assert observed[0][1].project_id == project.project_id
    assert session_store._SESSION_DATABASE_BINDING.get() is None
    assert document_context.current() is None
