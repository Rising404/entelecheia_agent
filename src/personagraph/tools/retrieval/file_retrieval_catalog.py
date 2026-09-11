"""Stable definition and contextual binding for retrieve_files only."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from ..catalog.binding import BoundToolRegistration, ToolBinding, ToolDefinition
from ..effects import EffectScopeKind
from .file_retrieval_tools import build_retrieve_files_registration


FILE_RETRIEVAL_DEFINITION_MANIFEST_SCHEMA = "file-retrieval-definition-manifest-v2"
FILE_RETRIEVAL_BINDING_ASSERTION_SCHEMA = "file-retrieval-binding-assertion-v2"


class FileRetrievalCatalogError(ValueError):
    """A File retrieval definition or binding drifted."""


@dataclass(frozen=True, slots=True)
class FileRetrievalToolDefinitionManifestItem:
    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition


@dataclass(frozen=True, slots=True)
class FileRetrievalBindingFacts:
    scope_sha256: str
    retrieval_generation_sha256: str
    retrieval_service_sha256: str

    def __post_init__(self):
        if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in asdict(self).values()):
            raise ValueError("File retrieval binding facts must be SHA-256 digests")


def _surface():
    return build_retrieve_files_registration(
        handler=_definition_handler, effect_scope="*", filesystem_scope_kind=EffectScopeKind.SESSION,
    )


def _definition_handler(_payload):
    raise RuntimeError("a definition handler cannot execute")


def build_file_retrieval_tool_definition_manifest():
    surface = _surface()
    implementation_ref, revision = "builtin/retrieve_files", "file-retrieval-handler-5"
    digest = hashlib.sha256(json.dumps(
        [FILE_RETRIEVAL_DEFINITION_MANIFEST_SCHEMA, implementation_ref, revision, surface.descriptor()],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    definition = ToolDefinition(
        spec=surface.spec, implementation_version=surface.implementation_version,
        implementation_ref=implementation_ref, implementation_digest=digest,
        effect_template=surface.effect_profile, execution_profile=surface.execution_profile,
    )
    return (FileRetrievalToolDefinitionManifestItem(implementation_ref, revision, definition),)


def build_file_retrieval_tool_bindings(registrations, *, facts):
    values = tuple(registrations)
    if len(values) != 1 or not isinstance(facts, FileRetrievalBindingFacts):
        raise FileRetrievalCatalogError("File retrieval requires one registration and binding facts")
    registration = values[0]
    definition = build_file_retrieval_tool_definition_manifest()[0].definition
    if registration.source != _surface().source:
        raise FileRetrievalCatalogError("File retrieval implementation source drifted")
    scopes = {effect.default_scope for effect in registration.effect_profile.effects}
    if len(scopes) != 1 or hashlib.sha256(next(iter(scopes)).encode()).hexdigest() != facts.scope_sha256:
        raise FileRetrievalCatalogError("File retrieval authority scope drifted")
    binding = ToolBinding(
        identity=definition.identity, definition_digest=definition.digest,
        source=registration.source, handler=registration.handler,
        effect_profile=registration.effect_profile,
        binding_assertion={"schema_version": FILE_RETRIEVAL_BINDING_ASSERTION_SCHEMA,
                           "binding_kind": "file_retrieval", **asdict(facts)},
    )
    try:
        if BoundToolRegistration(definition, binding).descriptor() != registration.descriptor():
            raise ValueError("registration descriptor drifted")
    except (TypeError, ValueError) as exc:
        raise FileRetrievalCatalogError("File retrieval registration drifted") from exc
    return (binding,)
