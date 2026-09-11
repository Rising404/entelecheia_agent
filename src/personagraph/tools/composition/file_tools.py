"""文件工具的唯一跨域装配：File 访问、按需准备、检索、精读和视觉。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from collections.abc import Callable

from personagraph.retrieval.lifecycle.generation import require_exact_active_generation
from personagraph.retrieval.operations.document_maintenance import build_document_retrieval_composition
from personagraph.retrieval.operations.picture_index import PictureObservationIndex
from personagraph.retrieval.sources.picture import PictureObservationSourceAdapter
from personagraph.retrieval.tooling.contracts import FrozenFileVersionBinding
from personagraph.workspace.files.access import FileAccess
from personagraph.workspace.files.turn_inputs import ResolvedTurnInputFile
from personagraph.workspace.ingestion.composition import resolve_document_ingest_owner
from personagraph.workspace.ingestion.application import FileIngestionService
from personagraph.workspace.ingestion.indexing_ports import IngestionGenerationIdentity
from personagraph.workspace.storage.context import require_current
from ..catalog.binding import ToolBinding
from ..documents.file_chunk_catalog import FileChunkBindingFacts, build_file_chunk_tool_bindings
from ..documents.file_chunk_reader import build_file_chunk_reader_runtime
from ..documents.file_inspection_adapter import FileInspectionRuntime
from ..documents.file_inspection_catalog import bind_file_inspection_tools
from ..effects import EffectAction, EffectResource, EffectScopeKind
from ..files.file_adapter import FileStateRuntime
from ..files.file_authority import SessionFileToolAuthority
from ..files.file_catalog import bind_file_state_tools
from ..files.file_tools import PREPARE_FILES_TOOL_ID
from ..policy import (
    AuthorityFacts, ProtectedToolExecutionAuthority, ScopeGrant, ToolInvocationAuthorityResolver,
)
from ..registration import ToolRegistration
from ..retrieval.file_retrieval_adapter import build_file_retrieval_runtime
from ..retrieval.file_retrieval_catalog import FileRetrievalBindingFacts, build_file_retrieval_tool_bindings
from ..retrieval.picture_file_authority import PictureRetrievalFileAuthority
from ..visual.file_visual_adapter import build_file_visual_runtime
from ..workspace.workspace_tools import FrozenWorkspaceToolBoundary


@dataclass(frozen=True, slots=True)
class FileToolSource:
    registrations: tuple[ToolRegistration, ...]
    bindings: tuple[ToolBinding, ...]
    authority: AuthorityFacts
    protected_authority_by_key: dict[tuple[str, str], ProtectedToolExecutionAuthority]
    invocation_authority_resolver_by_key: dict[tuple[str, str], ToolInvocationAuthorityResolver]
    retrieval_data_version: str


def build_file_tool_source(
    *, session_id: str, turn_id: str,
    workspace_boundary: FrozenWorkspaceToolBoundary, workspace_grant_id: str,
    file_retrieval_data_version: str | None = None,
    attachment_sources: tuple[ResolvedTurnInputFile, ...] = (),
    document_composition_factory: Callable = build_document_retrieval_composition,
    resolve_ingest_owner: Callable = resolve_document_ingest_owner,
    visual_runtime_factory: Callable = build_file_visual_runtime,
) -> FileToolSource:
    database = require_current()
    if workspace_boundary.session_id != session_id or workspace_boundary.root != database.project_root:
        raise ValueError("file tools crossed the bound Session/Project root")
    if any(source.project_id != database.project_id for source in attachment_sources):
        raise ValueError("Turn attachments crossed Project identity")
    access_authority = SessionFileToolAuthority(session_id, workspace_boundary, workspace_grant_id)
    if not access_authority.validate_current():
        raise ValueError("current file read authority is unavailable")
    access = FileAccess(database=database, validate_path=access_authority.validate_path)
    authority_sha256 = _digest({
        "contract": "session-file-access-v1", "session_id": session_id,
        "turn_id": turn_id, "project_id": database.project_id,
        "root": str(database.project_root), "database": str(database.db_path),
        "root_device": workspace_boundary.root_device,
        "root_inode": workspace_boundary.root_inode, "read_grant": workspace_grant_id,
    })
    scope_id = "session-files:" + authority_sha256
    effect_scope = "file-corpus:" + _text_digest(scope_id)

    def picture_bindings():
        if not access_authority.validate_current():
            raise ValueError("file authority changed")
        # 已登记语料枚举不是目录扫描；原文件在被选中/输出前仍由 FileAccess 重验。
        return tuple(FrozenFileVersionBinding(
            project_id=database.project_id, file_id=file_id, file_version_id=version_id,
        ) for file_id, version_id in access.authorized_versions())

    picture_authority = PictureRetrievalFileAuthority(
        session_id=session_id, authorized_file_bindings=picture_bindings,
    )
    composition = document_composition_factory(
        picture_source_adapter=PictureObservationSourceAdapter(access_authority=picture_authority),
    )
    generation = composition.generation_spec
    if file_retrieval_data_version is not None and file_retrieval_data_version != generation.version_id:
        raise RuntimeError("accepted File retrieval generation differs from current recipe")
    if composition.foundation.catalog.active_data_version() is not None:
        require_exact_active_generation(composition.foundation.catalog, generation)
    generation_identity = IngestionGenerationIdentity(generation.version_id, generation.fingerprint)

    ingestion = FileIngestionService(
        session_id=session_id, access=access, generation_identity=generation_identity,
        chunking_profile=composition.chunking_profile,
        validate_source=access_authority.validate_source, resolve_owner=resolve_ingest_owner,
    )
    state_runtime = FileStateRuntime(access, effect_scope, ingestion.check, ingestion.prepare)
    retrieve_runtime = build_file_retrieval_runtime(
        session_id=session_id, scope_id=scope_id, generation_spec=generation,
        resolve_file=access.resolve_file, revalidate=access.revalidate,
        file_foundation=composition.foundation, trajectory_turn_id=turn_id,
        picture_file_authority=picture_authority, filesystem_scope_kind=EffectScopeKind.SESSION,
    )
    reader_runtime = build_file_chunk_reader_runtime(
        session_id=session_id, scope_id=scope_id,
        resolve_file=access.resolve_file, revalidate=access.revalidate,
        filesystem_scope_kind=EffectScopeKind.SESSION,
    )
    inspection_runtime = FileInspectionRuntime(
        session_id=session_id, effect_scope=effect_scope,
        resolve_file=access.resolve_file, revalidate=access.revalidate,
    )
    visual = visual_runtime_factory(
        session_id=session_id, resolve_file=access.resolve_file, revalidate_source=access.revalidate,
        # 视觉问答沿用本轮冻结的 FILE 索引配方；不在工具内重建另一份配置。
        picture_index=PictureObservationIndex(
            composition=composition, connect_documents=database.open_connection,
        ),
    )
    generation_sha256 = _text_digest(generation.canonical_json)
    state_registrations = state_runtime.registrations
    state_bindings = bind_file_state_tools(
        state_registrations, authority_sha256=authority_sha256, generation_sha256=generation_sha256,
        scope_sha256=_text_digest(effect_scope),
    )
    state_registrations = tuple(replace(registration, source=binding.source)
                                for registration, binding in zip(state_registrations, state_bindings, strict=True))
    retrieval_registration = retrieve_runtime.registration()
    reader_registration = reader_runtime.registration()
    inspection_registrations = inspection_runtime.registrations
    bindings = (
        *state_bindings,
        *build_file_retrieval_tool_bindings((retrieval_registration,), facts=FileRetrievalBindingFacts(
            scope_sha256=_text_digest(effect_scope), retrieval_generation_sha256=generation_sha256,
            retrieval_service_sha256=_digest({
                "recipe": "native-file-retrieval-v1", "authority": authority_sha256,
                "generation": generation_sha256,
                "requested_profile": composition.requested_profile.fingerprint(),
                "effective_profile": composition.effective_profile.fingerprint(),
                "encoder": composition.encoder.fingerprint(),
                "reranker": composition.reranker.fingerprint() if composition.reranker else None,
                "methods": [method.value for method in composition.foundation.retrieval_methods],
            }),
        )),
        *build_file_chunk_tool_bindings((reader_registration,), facts=FileChunkBindingFacts(
            scope_sha256=_text_digest(effect_scope), chunk_reader_authority_sha256=authority_sha256,
        )),
        *bind_file_inspection_tools(inspection_registrations,
                                    scope_sha256=_text_digest(effect_scope), authority_sha256=authority_sha256),
        *visual.file_visual_bindings,
    )
    registrations = (*state_registrations, retrieval_registration, reader_registration,
                     *inspection_registrations, *visual.registrations)
    grants = tuple(dict.fromkeys(
        ScopeGrant(effect.resource, effect.action, effect.scope_kind, effect.default_scope)
        for registration in (*state_registrations, retrieval_registration, reader_registration, *inspection_registrations)
        for effect in registration.effect_profile.effects
    ))
    protected = {}
    invocation_resolvers = {}
    for registration in registrations:
        if registration.tool_id != PREPARE_FILES_TOOL_ID and not any(
            effect.resource is EffectResource.NETWORK and effect.action is EffectAction.TRANSMIT
            for effect in registration.effect_profile.effects
        ):
            continue
        if registration.tool_id != PREPARE_FILES_TOOL_ID:
            invocation_resolvers[(registration.tool_id, registration.contract_version)] = (
                visual.prepare_read_authority
            )
            continue
        protected[(registration.tool_id, registration.contract_version)] = ProtectedToolExecutionAuthority(
            approval_receipt_ids=(workspace_grant_id,),
            execution_backend_identity_sha256=_digest({
                "authority": authority_sha256, "generation": generation_sha256,
                "registration": registration.descriptor(),
            }), revalidate=access_authority.validate_current,
        )
    return FileToolSource(
        registrations=registrations, bindings=bindings,
        authority=AuthorityFacts(
            grants=tuple(dict.fromkeys((*grants, *visual.authority.grants))),
            approval_grants=visual.authority.approval_grants,
        ), protected_authority_by_key=protected,
        invocation_authority_resolver_by_key=invocation_resolvers,
        retrieval_data_version=generation.version_id,
    )


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _digest(value) -> str:
    return _text_digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
