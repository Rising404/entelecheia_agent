"""L1 默认外发，但保留精确来源、路径与持久调用边界。"""

from __future__ import annotations

import pytest
from PIL import Image

from personagraph.input_processing.vision.contracts import VisionCapabilitySnapshot, VisionPurpose
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.session.local_file_authority import SqliteSessionFileAuthority
from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
from personagraph.workspace.storage.context import require_current


class _ExternalVision:
    transmits_externally = True

    def capabilities(self):
        return VisionCapabilitySnapshot(
            available=True, provider="configured", model="configured",
            endpoint_identity="configured", processor_fingerprint="configured-external@1",
            supported_purposes=tuple(VisionPurpose),
        )


@pytest.fixture
def visual_runtime(monkeypatch, tmp_path, bound_partitioned_session):
    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        _ExternalVision,
    )
    root = tmp_path / "workspace"
    root.mkdir()
    session_id = bound_partitioned_session(working_dir=root)
    return root, session_id


def _prepare(runtime, path="chart.png", *, purpose="chart", question=None):
    arguments = {"path": path, "purpose": purpose, "detail": "standard", "region": "page"}
    if question is not None:
        arguments["question"] = question
    return runtime.prepare(
        tool_id="analyze_image",
        arguments=arguments,
        remaining_tool_calls=1,
    )


@pytest.mark.parametrize("origin", [FileSource.USER_UPLOAD, FileSource.AGENT_OUTPUT])
@pytest.mark.parametrize("purpose", ["chart", "question"])
def test_managed_file_gets_exact_egress_binding_without_consent_state(visual_runtime, origin, purpose):
    root, session_id = visual_runtime
    Image.new("RGB", (20, 20)).save(root / "chart.png")
    WorkspaceFileAuthority(require_current()).register_path(
        "chart.png", source=origin, media_type="image/png",
    )
    runtime = build_l1_tool_runtime(session_id)
    assert {"analyze_image", "analyze_pdf_page"} <= set(runtime.registrations_by_tool_id)
    prepared = _prepare(
        runtime, purpose=purpose,
        question="请寻找图中的标题。" if purpose == "question" else None,
    )
    assert prepared.rejected_outcome is None
    assert prepared.requires_protected_dispatch is True
    assert prepared.policy["disposition"] == "allow"
    assert prepared.protected_authority is not None
    assert prepared.protected_authority.approval_receipt_ids[0].startswith("auto_visual_egress_")
    assert prepared.protected_authority.revalidate()


@pytest.mark.parametrize("registered", [False, True])
def test_ordinary_workspace_file_uses_default_egress(visual_runtime, registered):
    root, session_id = visual_runtime
    path = root / "chart.png"
    Image.new("RGB", (20, 20)).save(path)
    if registered:
        WorkspaceFileAuthority(require_current()).register_path(
            "chart.png", source=FileSource.WORKSPACE_EXISTING, media_type="image/png",
        )
    runtime = build_l1_tool_runtime(session_id)
    prepared = _prepare(runtime)
    assert prepared.rejected_outcome is None
    assert prepared.protected_authority.approval_receipt_ids[0].startswith("auto_visual_egress_")
    assert prepared.protected_authority.revalidate()


def test_prepared_visual_authority_rejects_source_change_and_local_revocation(visual_runtime):
    root, session_id = visual_runtime
    path = root / "chart.png"
    Image.new("RGB", (20, 20)).save(path)
    WorkspaceFileAuthority(require_current()).register_path(
        "chart.png", source=FileSource.USER_UPLOAD, media_type="image/png",
    )
    runtime = build_l1_tool_runtime(session_id)
    prepared = _prepare(runtime)
    Image.new("RGB", (30, 30)).save(path)
    assert prepared.protected_authority.revalidate() is False
    refreshed = _prepare(runtime)
    SqliteSessionFileAuthority().revoke(
        session_id=session_id,
        grant_id=runtime.workspace_corpus_authority.local_file_authorization_receipt_id,
    )
    assert refreshed.protected_authority.revalidate() is False


def test_default_visual_authority_does_not_escape_workspace(visual_runtime, tmp_path):
    root, session_id = visual_runtime
    Image.new("RGB", (20, 20)).save(tmp_path / "outside.png")
    (root / "escape.png").symlink_to(tmp_path / "outside.png")
    runtime = build_l1_tool_runtime(session_id)
    for path in ("../outside.png", "escape.png"):
        denied = _prepare(runtime, path)
        assert denied.rejected_outcome.error.code == "workspace_path_blocked"


def test_in_workspace_alias_does_not_require_separate_egress_consent(visual_runtime):
    root, session_id = visual_runtime
    Image.new("RGB", (20, 20)).save(root / "chart.png")
    WorkspaceFileAuthority(require_current()).register_path(
        "chart.png", source=FileSource.USER_UPLOAD, media_type="image/png",
    )
    (root / "alias.png").symlink_to(root / "chart.png")
    runtime = build_l1_tool_runtime(session_id)
    assert _prepare(runtime).rejected_outcome is None
    assert _prepare(runtime, "alias.png").rejected_outcome is None


def test_invalid_visual_arguments_are_rejected_before_authority_resolution(visual_runtime):
    _, session_id = visual_runtime
    runtime = build_l1_tool_runtime(session_id)
    denied = runtime.prepare(tool_id="analyze_image", arguments={"path": "missing.png"}, remaining_tool_calls=1)
    assert denied.rejected_outcome.error.code == "invalid_tool_input"


def test_visual_question_allows_derived_state_without_user_approval(visual_runtime):
    from personagraph.tools.documents.external_visual_authority import build_external_visual_authority_resolver
    from personagraph.tools.effects import EffectAction, EffectResource, EffectScopeKind
    from personagraph.tools.policy import ScopeGrant
    from personagraph.tools.workspace.workspace_tools import FrozenWorkspaceToolBoundary

    root, session_id = visual_runtime
    Image.new("RGB", (20, 20)).save(root / "question.png")
    WorkspaceFileAuthority(require_current()).register_path(
        "question.png", source=FileSource.USER_UPLOAD, media_type="image/png",
    )
    resolve = build_external_visual_authority_resolver(
        boundary=FrozenWorkspaceToolBoundary(session_id=session_id, root=root),
        tool_id="analyze_image", capabilities=_ExternalVision().capabilities(),
         backend_identity_sha256="a" * 64,
    )
    authority = resolve({
        "path": "question.png", "purpose": "question", "question": "图中有什么？",
        "detail": "standard", "region": "page",
    })
    assert ScopeGrant(
        EffectResource.RUNTIME_STATE, EffectAction.UPDATE,
        EffectScopeKind.SESSION, session_id,
    ) in authority.authority.grants
    assert authority.authority.approval_grants == ()
    assert all(grant.scope == session_id for grant in authority.authority.grants)
    assert authority.protected_authority.revalidate()


@pytest.mark.parametrize("origin", [FileSource.USER_UPLOAD, FileSource.AGENT_OUTPUT])
def test_file_visual_binding_uses_default_egress_receipts(visual_runtime, monkeypatch, origin):
    root, session_id = visual_runtime
    monkeypatch.setattr(
        "personagraph.tools.visual.file_visual_adapter.default_vision_adapter", _ExternalVision,
    )
    Image.new("RGB", (20, 20)).save(root / "chart.png")
    file = WorkspaceFileAuthority(require_current()).register_path(
        "chart.png", source=origin, media_type="image/png",
    )
    runtime = build_l1_tool_runtime(session_id, execution_features={
        "l1_retrieval_tools_enabled": True, "file_retrieval_read_enabled": True,
    })
    prepared = runtime.prepare(
        tool_id="read_file_visuals", remaining_tool_calls=1,
        arguments={"requests": [{
            "file_id": file.file.file_id, "file_version_id": file.version.file_version_id,
            "visual_unit_id": "whole_file", "purpose": "chart",
            "detail": "standard", "region": "detected",
        }]},
    )
    assert prepared.rejected_outcome is None
    assert prepared.protected_authority.approval_receipt_ids[0].startswith("auto_visual_egress_")
    assert prepared.protected_authority.revalidate()
