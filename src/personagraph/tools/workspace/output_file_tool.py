"""指定 output/ 产物目录的 CREATE 工具：稳定协议、当前项目绑定和结果投影。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from ...workspace.files import (
    ProjectFileError, ProjectFilePathError, ProjectOutputConflict, ProjectOutputService,
)
from ...workspace.files.outputs import (
    MAX_OUTPUT_CONTENT_BYTES, MAX_OUTPUT_PATH_CHARS, OUTPUT_DIRECTORY,
)
from ...workspace.storage.context import current as current_project_database
from ..catalog.binding import BoundToolRegistration, ToolBinding, ToolDefinition
from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress, EffectAction, EffectDescriptor, EffectResource, EffectScopeKind,
    Idempotency, Reversibility, ToolEffectProfile,
)
from ..execution import ToolBusinessFailure
from ..files.file_authority import SessionFileToolAuthority
from ..policy import AuthorityFacts, ProtectedToolExecutionAuthority, ScopeGrant
from ..registration import ToolExecutionProfile, ToolRegistration
from .workspace_tools import FrozenWorkspaceToolBoundary


CREATE_OUTPUT_FILE_TOOL_ID = "create_output_file"


@dataclass(frozen=True, slots=True)
class OutputFileToolSource:
    binding: ToolBinding
    authority: AuthorityFacts
    protected_authority: ProtectedToolExecutionAuthority


def build_output_file_definition() -> ToolDefinition:
    registration = _registration(_definition_only, scope="*")
    return ToolDefinition(
        spec=registration.spec,
        implementation_version=registration.implementation_version,
        implementation_ref=f"builtin/{CREATE_OUTPUT_FILE_TOOL_ID}",
        implementation_digest=_digest({
            "behavior": "create-output-file-handler-1",
            "registration": registration.descriptor(),
        }),
        effect_template=registration.effect_profile,
        execution_profile=registration.execution_profile,
    )


def build_session_output_file_tool_source(
    *, session_id: str, boundary: FrozenWorkspaceToolBoundary, workspace_grant_id: str,
    protected_dispatch_sha256: str, operation_ledger_sha256: str,
) -> OutputFileToolSource | None:
    """绑定根目录的既有回执允许 output CREATE，不引入工作区 UPDATE 授予。"""

    database = current_project_database()
    if database is None:
        return None
    if boundary.session_id != session_id or boundary.root != database.project_root:
        raise ValueError("output source crossed the bound Session/Project root")
    access = SessionFileToolAuthority(session_id, boundary, workspace_grant_id)
    if not access.validate_current():
        raise ValueError("current project authority is unavailable")
    service = ProjectOutputService(
        database, root_device=boundary.root_device, root_inode=boundary.root_inode,
    )

    def create(payload):
        if not access.validate_current():
            raise ToolBusinessFailure("workspace_authority_revoked", "Project authority changed.")
        try:
            result = service.create_text(path=payload["path"], content=payload["content"])
        except ProjectOutputConflict as exc:
            raise ToolBusinessFailure("output_already_exists", "Choose a new output path; existing files are never overwritten.") from exc
        except ProjectFilePathError as exc:
            raise ToolBusinessFailure("workspace_path_blocked", "Output path is outside the permitted area or is unsafe.") from exc
        except ProjectFileError as exc:
            raise ToolBusinessFailure("output_creation_failed", "The output could not be safely created and registered.") from exc
        except ValueError as exc:
            raise ToolBusinessFailure("invalid_request", str(exc)) from exc
        return {
            "file_id": result.file.file_id,
            "file_version_id": result.version.file_version_id,
            "relative_path": result.file.relative_path,
            "size_bytes": result.version.size_bytes,
            "content_sha256": result.version.content_sha256,
        }

    scope = str(boundary.root / OUTPUT_DIRECTORY)
    definition = build_output_file_definition()
    registration = _registration(create, scope=scope)
    assertion = {
        "schema_version": "project-output-create-binding-v1",
        "policy": "bound-project-output-create-only",
        "authority_sha256": _digest({
            "session_id": session_id, "project_id": database.project_id,
            "database": str(database.db_path), "root": str(boundary.root),
            "root_device": boundary.root_device, "root_inode": boundary.root_inode,
            "workspace_read_grant": workspace_grant_id,
        }),
        "scope_sha256": _digest(scope),
        "protected_dispatch_sha256": protected_dispatch_sha256,
        "operation_ledger_sha256": operation_ledger_sha256,
    }
    binding = ToolBinding(
        identity=definition.identity, definition_digest=definition.digest,
        source=replace(registration.source, fingerprint=_digest(assertion)),
        handler=registration.handler, effect_profile=registration.effect_profile,
        binding_assertion=assertion,
    )
    BoundToolRegistration(definition, binding)
    create_grant = ScopeGrant(
        EffectResource.FILESYSTEM, EffectAction.CREATE, EffectScopeKind.WORKSPACE, scope,
    )
    return OutputFileToolSource(
        binding=binding,
        authority=AuthorityFacts(grants=(create_grant,), approval_grants=(create_grant,)),
        protected_authority=ProtectedToolExecutionAuthority(
            approval_receipt_ids=(workspace_grant_id,),
            execution_backend_identity_sha256=_digest(assertion),
            revalidate=access.validate_current,
        ),
    )


def _registration(handler, *, scope: str) -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=CREATE_OUTPUT_FILE_TOOL_ID, contract_version="output-file-v1",
            name="Create a new output text file",
            description=(
                "Create a new UTF-8 text artifact under the project output/ directory. "
                "path is relative to output/, e.g. reports/summary.md. Parent directories "
                "are created as needed. Existing files are NEVER overwritten. Returns "
                "the project-relative path and File IDs for reading or prepare_files. "
                "Creating a new output needs no additional user approval."
            ),
            input_schema={
                "type": "object", "additionalProperties": False,
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": MAX_OUTPUT_PATH_CHARS},
                    "content": {"type": "string", "maxLength": MAX_OUTPUT_CONTENT_BYTES},
                },
            },
            output_schema={
                "type": "object", "additionalProperties": False,
                "required": ["file_id", "file_version_id", "relative_path", "size_bytes", "content_sha256"],
                "properties": {
                    "file_id": {"type": "string", "minLength": 1},
                    "file_version_id": {"type": "string", "minLength": 1},
                    "relative_path": {"type": "string", "minLength": 1},
                    "size_bytes": {"type": "integer", "minimum": 0, "maximum": MAX_OUTPUT_CONTENT_BYTES},
                    "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
            catalog_tags=("file", "artifact", "write"),
        ),
        implementation_version="1",
        source=ToolSourceDescriptor(
            kind=ToolSourceKind.LOCAL, source_id="personagraph.workspace.output",
            display_name="Project output creator",
        ),
        handler=handler,
        effect_profile=ToolEffectProfile((EffectDescriptor(
            resource=EffectResource.FILESYSTEM, action=EffectAction.CREATE,
            scope_kind=EffectScopeKind.WORKSPACE, default_scope=scope,
            data_egress=DataEgress.NONE, idempotency=Idempotency.NOT_IDEMPOTENT,
            reversibility=Reversibility.COMPENSATABLE, resource_argument="path",
        ),)),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=10.0, hard_timeout_s=20.0, max_output_bytes=4_000,
            max_transparent_retries=0, concurrency_class="workspace_write",
        ),
    )


def _definition_only(_payload):
    raise RuntimeError("output tool definition is not executable")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
