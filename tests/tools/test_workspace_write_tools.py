"""V2 冻结工作区文本写入器的边界测试。"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
)
from personagraph.tools.execution import (
    ResolvedInvocation,
    ToolBusinessFailure,
    ToolExecutor,
)
from personagraph.tools.policy import (
    AuthorityFacts,
    PolicyDisposition,
    PolicyRequest,
    ScopeGrant,
    ToolPolicyCore,
)
from personagraph.tools.workspace.workspace_tools import FrozenWorkspaceToolBoundary
from personagraph.tools.workspace.workspace_write_tools import (
    MAX_CONTENT_BYTES,
    build_workspace_write_tool_registrations,
)


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@pytest.fixture
def root(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def writer(root: Path):
    boundary = FrozenWorkspaceToolBoundary(session_id="session-1", root=root)
    (registration,) = build_workspace_write_tool_registrations(boundary)
    return boundary, registration


def _error_code(callable_) -> str:
    with pytest.raises(ToolBusinessFailure) as raised:
        callable_()
    return raised.value.error.code


def test_registration_declares_valid_v2_schemas_and_real_output(writer):
    _, registration = writer
    Draft202012Validator.check_schema(_plain(registration.spec.input_schema))
    validator = Draft202012Validator(_plain(registration.spec.output_schema))

    result = registration.handler(
        {
            "path": "drafts/plan.md",
            "content": "# Plan\n",
            "create_parents": True,
        }
    )

    validator.validate(result)
    assert result == {
        "path": "drafts/plan.md",
        "mode": "overwrite",
        "created": True,
        "overwrote": False,
        "previous_size": None,
        "new_size": 7,
        "written_bytes": 7,
        "content_sha256": hashlib.sha256(b"# Plan\n").hexdigest(),
        "created_parent_count": 1,
    }


def test_parent_creation_is_explicit_and_overwrite_is_atomic(root: Path, writer, monkeypatch):
    _, registration = writer

    assert _error_code(
        lambda: registration.handler({"path": "new/note.txt", "content": "first"})
    ) == "parent_directory_missing"
    assert not (root / "new").exists()

    target = root / "note.txt"
    target.write_text("old content", encoding="utf-8")

    def _fail_after_temp_write(_fd: int, _content: bytes) -> None:
        raise OSError("simulated write interruption")

    from personagraph.tools.workspace import workspace_write_tools

    monkeypatch.setattr(workspace_write_tools, "_write_all", _fail_after_temp_write)
    assert _error_code(
        lambda: registration.handler({"path": "note.txt", "content": "new content"})
    ) == "workspace_write_failed"
    assert target.read_text(encoding="utf-8") == "old content"


def test_overwrite_and_append_report_the_actual_file_state(root: Path, writer):
    _, registration = writer
    target = root / "report.txt"
    target.write_text("old", encoding="utf-8")

    overwritten = registration.handler(
        {"path": "report.txt", "content": "new", "mode": "overwrite"}
    )
    appended = registration.handler(
        {"path": "report.txt", "content": "+", "mode": "append"}
    )
    created_by_append = registration.handler(
        {"path": "created.txt", "content": "one", "mode": "append"}
    )

    assert target.read_text(encoding="utf-8") == "new+"
    assert overwritten["created"] is False
    assert overwritten["overwrote"] is True
    assert overwritten["previous_size"] == 3
    assert overwritten["new_size"] == 3
    assert appended["created"] is False
    assert appended["overwrote"] is False
    assert appended["previous_size"] == 3
    assert appended["new_size"] == 4
    assert created_by_append["created"] is True
    assert created_by_append["previous_size"] is None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlink support")
def test_paths_cannot_escape_or_follow_links(root: Path, writer, tmp_path: Path):
    _, registration = writer
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / "linked")

    for path in (
        "../outside.txt",
        str((outside / "absolute.txt").resolve()),
        "linked/escape.txt",
        ".personagraph/output/session-1/draft.txt",
    ):
        assert _error_code(
            lambda path=path: registration.handler(
                {"path": path, "content": "blocked", "create_parents": True}
            )
        ) == "workspace_path_blocked"
    assert not (outside / "escape.txt").exists()


def test_local_config_tree_is_not_writable_but_public_configs_remain_available(
    root: Path,
    writer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personagraph.configuration import paths

    _, registration = writer
    local_config = root / "configs" / "local"
    monkeypatch.setattr(paths, "LOCAL_CONFIG_DIR", local_config.resolve())

    assert _error_code(
        lambda: registration.handler(
            {
                "path": "configs/local/new.json",
                "content": "blocked",
                "create_parents": True,
            }
        )
    ) == "workspace_path_blocked"
    assert not (local_config / "new.json").exists()

    result = registration.handler(
        {
            "path": "configs/preset.json",
            "content": "public",
            "create_parents": True,
        }
    )
    assert result["created"] is True
    assert (root / "configs" / "preset.json").read_text(encoding="utf-8") == "public"


def test_append_rejects_hard_linked_files(root: Path, writer, tmp_path: Path):
    _, registration = writer
    outside = tmp_path / "outside.txt"
    outside.write_text("trusted", encoding="utf-8")
    os.link(outside, root / "linked.txt")

    assert _error_code(
        lambda: registration.handler(
            {"path": "linked.txt", "content": "untrusted", "mode": "append"}
        )
    ) == "workspace_path_blocked"
    assert outside.read_text(encoding="utf-8") == "trusted"


def test_content_limit_and_invalid_direct_handler_arguments_are_business_failures(
    writer,
    monkeypatch: pytest.MonkeyPatch,
):
    _, registration = writer
    from personagraph.tools.workspace import workspace_write_tools

    monkeypatch.setattr(workspace_write_tools, "MAX_CONTENT_BYTES", 3)
    assert _error_code(
        lambda: registration.handler({"path": "note.txt", "content": "four"})
    ) == "content_too_large"
    assert _error_code(
        lambda: registration.handler(
            {"path": "note.txt", "content": "ok", "mode": "replace"}
        )
    ) == "invalid_request"
    assert _error_code(
        lambda: registration.handler(
            {"path": "note.txt", "content": "ok", "create_parents": "yes"}
        )
    ) == "invalid_request"


def test_workspace_authority_is_checked_before_and_after_the_write(
    root: Path,
    writer,
    monkeypatch: pytest.MonkeyPatch,
):
    boundary, registration = writer
    calls: list[str] = []
    original = FrozenWorkspaceToolBoundary.require_current_root

    def _record(self: FrozenWorkspaceToolBoundary) -> Path:
        calls.append("root")
        return original(self)

    monkeypatch.setattr(FrozenWorkspaceToolBoundary, "require_current_root", _record)
    registration.handler({"path": "note.txt", "content": "ok"})
    assert len(calls) >= 3
    assert (root / "note.txt").read_text(encoding="utf-8") == "ok"

    moved = root.with_name("workspace-moved")
    root.rename(moved)
    root.mkdir()
    assert _error_code(
        lambda: registration.handler({"path": "must-not-write.txt", "content": "no"})
    ) == "workspace_authority_changed"
    assert not (root / "must-not-write.txt").exists()
    assert boundary.root == root


def test_write_rejects_a_root_replaced_after_path_validation(
    root: Path,
    writer,
    monkeypatch: pytest.MonkeyPatch,
):
    _, registration = writer
    from personagraph.tools.workspace import workspace_write_tools

    original_open = workspace_write_tools._open_directory_path
    moved = root.with_name("workspace-original")

    def _replace_before_open(path: Path) -> int:
        root.rename(moved)
        root.mkdir()
        return original_open(path)

    monkeypatch.setattr(
        workspace_write_tools,
        "_open_directory_path",
        _replace_before_open,
    )
    assert _error_code(
        lambda: registration.handler({"path": "must-not-write.txt", "content": "no"})
    ) == "workspace_authority_changed"
    assert not (root / "must-not-write.txt").exists()
    assert not (moved / "must-not-write.txt").exists()


def test_effect_profile_requires_workspace_authorization_and_approval(writer):
    boundary, registration = writer
    effects = registration.effect_profile.effects
    assert [(effect.resource, effect.action, effect.scope_kind) for effect in effects] == [
        (EffectResource.FILESYSTEM, EffectAction.READ, EffectScopeKind.WORKSPACE),
        (EffectResource.FILESYSTEM, EffectAction.UPDATE, EffectScopeKind.WORKSPACE),
    ]
    assert effects[0].data_egress is DataEgress.METADATA
    assert effects[0].idempotency is Idempotency.IDEMPOTENT
    assert effects[1].data_egress is DataEgress.NONE
    assert effects[1].idempotency is Idempotency.NOT_IDEMPOTENT
    assert effects[1].reversibility is Reversibility.IRREVERSIBLE
    assert effects[1].resource_argument == "path"

    arguments = {"path": "draft.txt", "content": "body"}
    policy = ToolPolicyCore()
    unauthorized = policy.evaluate(PolicyRequest.from_registration(registration, arguments))
    assert unauthorized.disposition is PolicyDisposition.AUTHORIZATION_REQUIRED

    grants = tuple(
        ScopeGrant(
            EffectResource.FILESYSTEM,
            effect.action,
            EffectScopeKind.WORKSPACE,
            str(boundary.root),
        )
        for effect in effects
    )
    needs_approval = policy.evaluate(
        PolicyRequest.from_registration(
            registration,
            arguments,
            authority=AuthorityFacts(grants=grants),
        )
    )
    assert needs_approval.disposition is PolicyDisposition.APPROVAL_REQUIRED
    assert policy.evaluate(
        PolicyRequest.from_registration(
            registration,
            arguments,
            authority=AuthorityFacts(grants=grants, approval_grants=grants),
        )
    ).disposition is PolicyDisposition.ALLOW


def test_effectful_registration_disables_transparent_retries_and_executor_validates_input(writer):
    _, registration = writer
    assert registration.execution_profile.max_transparent_retries == 0
    outcome = ToolExecutor().execute(
        # 执行器必须在处理器写入任何内容之前拒绝该请求。
        # JSON Schema 失败属于提案错误，而不是文件系统错误。
        ResolvedInvocation(registration, {"path": "note.txt", "content": 42})
    )
    assert outcome.error is not None
    assert outcome.error.code == "invalid_tool_input"


def test_module_never_imports_v1_file_tools_or_workspace_policy():
    from personagraph.tools.workspace import workspace_write_tools

    module = ast.parse(Path(workspace_write_tools.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(module)
        for alias in (node.names if isinstance(node, ast.Import) else [])
    }
    imports |= {
        node.module or ""
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        forbidden in imported
        for imported in imports
        for forbidden in ("file_tools", "workspace_policy", "runtime", "graph")
    )


def test_documented_per_call_limit_is_stable():
    assert MAX_CONTENT_BYTES == 128 * 1024
