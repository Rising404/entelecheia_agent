"""Current native File contracts share persistent definitions and revocable bindings."""

import json

from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.session.local_file_authority import SqliteSessionFileAuthority
from personagraph.tools.catalog.binding import BoundToolRegistration
from personagraph.tools.files.file_catalog import file_tool_definition_manifest
from personagraph.tools.retrieval.file_retrieval_catalog import build_file_retrieval_tool_definition_manifest
from personagraph.tools.documents.file_chunk_catalog import build_file_chunk_tool_definition_manifest


def test_native_file_bindings_match_default_definitions_and_revoke(tmp_path, bound_partitioned_session):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "brief.md").write_text("# Brief\n\nStable File binding evidence.")
    session_id = bound_partitioned_session(working_dir=workspace)
    runtime = build_l1_tool_runtime(session_id, execution_features={
        "file_retrieval_write_enabled": True, "file_retrieval_read_enabled": True,
        "l1_retrieval_tools_enabled": True,
    })
    manifest = (*file_tool_definition_manifest(), *build_file_retrieval_tool_definition_manifest(),
                *build_file_chunk_tool_definition_manifest())
    bindings = []
    for item in manifest:
        registration = runtime.registrations_by_tool_id[item.definition.spec.tool_id]
        binding = registration.binding
        assert BoundToolRegistration(item.definition, binding).descriptor() == registration.descriptor()
        bindings.append(binding.descriptor())
    encoded = json.dumps(bindings)
    assert session_id not in encoded and str(workspace) not in encoded
    grants = SqliteSessionFileAuthority().list_grants(session_id=session_id)
    active_grant = next(grant for grant in grants if grant.revoked_at is None)
    assert active_grant.grant_id not in encoded
    prepare = runtime.registrations_by_tool_id["prepare_files"]
    protected = runtime.protected_authority_by_key[(prepare.tool_id, prepare.contract_version)]
    assert protected.revalidate()
    SqliteSessionFileAuthority().revoke(session_id=session_id, grant_id=active_grant.grant_id)
    assert not protected.revalidate()
    result = prepare.handler({"files": [{"path": "brief.md"}]})
    assert result["results"][0]["status"] == "unavailable"
    assert result["results"][0]["reason_code"] == "file_authority_denied"
