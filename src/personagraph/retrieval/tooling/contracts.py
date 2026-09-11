"""Version-neutral contracts between retrieval tool adapters and data planes.

These values describe frozen Host authority and verified retrieval results.  They
do not define a model-visible tool or a particular runtime lane.  Public wire
contracts translate to and from these types at the tool-adapter boundary.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Protocol

from ..contracts import (
    MAX_FILE_RETRIEVAL_ITEMS,
    MAX_FILE_RETRIEVAL_QUERIES,
    RetrievalQueryMatch,
    SourceAccess,
    SourceAvailability,
    SourceFilter,
    SourceIndexBinding,
    SourceType,
    source_index_binding_manifest,
)
from ..query_guard import (
    DEFAULT_MAX_RETRIEVAL_QUERY_CHARACTERS,
    DEFAULT_MAX_RETRIEVAL_QUERY_TOKENS,
)


DEFAULT_RETRIEVAL_TOKEN_LIMIT = 8_000
DEFAULT_FILE_RETRIEVAL_RESULT_LIMIT = MAX_FILE_RETRIEVAL_ITEMS
DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT = 96_000
MAX_FROZEN_HISTORY_BINDINGS = 4_096
MAX_FILE_RETRIEVAL_TOKEN_LIMIT = 96_000
MAX_PRIVATE_CITATION_FIELDS = 32
MAX_PRIVATE_CITATION_VALUE_CHARS = 4_096
MAX_RETRIEVAL_ITEMS = 20
MAX_RETRIEVAL_QUERY_CHARS = DEFAULT_MAX_RETRIEVAL_QUERY_CHARACTERS
MAX_RETRIEVAL_QUERY_TOKENS = DEFAULT_MAX_RETRIEVAL_QUERY_TOKENS

_CURRENT_SESSION_CUTOFF = re.compile(
    r"assistant_turn_idx:(0|[1-9][0-9]{0,17})\Z"
)
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[/\\]")


class RetrievalCorpus(StrEnum):
    FILES = "files"
    HISTORY = "history"


class FileRetrievalScope(StrEnum):
    SESSION_CORPUS = "session_corpus"
    SELECTED_FILES = "selected_files"


class FileRetrievalReadinessStatus(StrEnum):
    READY = "ready"
    PENDING = "pending"
    BLOCKED = "blocked"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class FileRetrievalReadinessResult:
    """Read-only exact File-to-Document resolution for the retrieval service."""

    status: FileRetrievalReadinessStatus
    file_id: str
    reason_code: str | None = None
    retrieval_data_version: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None


class HistoryScope(StrEnum):
    CURRENT_SESSION = "current_session"
    LONG_TERM_USER = "long_term_user"
    CURRENT_TASK = "current_task"


class RetrievalStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class FileRetrievalOrigin(StrEnum):
    WORKSPACE = "workspace"
    USER_UPLOAD = "user_upload"
    AGENT_OUTPUT = "agent_output"
    MOUNTED_DOCUMENT = "mounted_document"


@dataclass(frozen=True, slots=True)
class FrozenFileRetrievalBinding:
    """A public source selector bound to one private, Host-authorized file."""

    source_id: str
    authority_id: str
    origin: FileRetrievalOrigin
    project_id: str | None = field(default=None, repr=False)
    file_id: str | None = field(default=None, repr=False)
    file_version_id: str | None = field(default=None, repr=False)
    expected_source_revision: str | None = None
    expected_content_sha256: str | None = None
    exact_read_tool_id: str | None = None
    exact_read_arguments: tuple[tuple[str, str | int | bool], ...] = ()

    def __post_init__(self) -> None:
        _public_source_id(self.source_id)
        _private_identifier(self.authority_id, "authority_id")
        if not isinstance(self.origin, FileRetrievalOrigin):
            raise ValueError("origin must be a FileRetrievalOrigin")
        project_binding = (self.project_id, self.file_id, self.file_version_id)
        if any(value is not None for value in project_binding):
            if not all(value is not None for value in project_binding):
                raise ValueError(
                    "project_id, file_id, and file_version_id must be supplied together"
                )
            _private_identifier(self.project_id, "project_id")
            _private_identifier(self.file_id, "file_id")
            _private_identifier(self.file_version_id, "file_version_id")
        if self.expected_source_revision is not None:
            _private_identifier(
                self.expected_source_revision,
                "expected_source_revision",
            )
        if self.expected_content_sha256 is not None:
            _sha256(self.expected_content_sha256, "expected_content_sha256")
        if self.exact_read_tool_id is not None:
            _safe_code(self.exact_read_tool_id, "exact_read_tool_id")
        arguments = tuple(self.exact_read_arguments)
        if (self.exact_read_tool_id is None) != (not arguments):
            raise ValueError(
                "exact-read tool and fixed arguments must be supplied together"
            )
        if len(arguments) > 8:
            raise ValueError("exact-read route has too many fixed arguments")
        names: list[str] = []
        for item in arguments:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("exact-read arguments must be key/value pairs")
            name, value = item
            _safe_code(name, "exact-read argument name")
            names.append(name)
            if isinstance(value, str):
                _public_source_id(value)
            elif isinstance(value, bool):
                pass
            elif not isinstance(value, int):
                raise ValueError(
                    "exact-read argument values must be string, integer, or bool"
                )
        if len(names) != len(set(names)):
            raise ValueError("exact-read argument names must be unique")
        object.__setattr__(
            self,
            "exact_read_arguments",
            tuple(sorted(arguments, key=lambda item: item[0])),
        )


@dataclass(frozen=True, slots=True)
class FrozenFileVersionBinding:
    """One Host-private, exact Project file version in a Session inventory."""

    project_id: str = field(repr=False)
    file_id: str = field(repr=False)
    file_version_id: str = field(repr=False)

    def __post_init__(self) -> None:
        _private_identifier(self.project_id, "project_id")
        _private_identifier(self.file_id, "file_id")
        _private_identifier(self.file_version_id, "file_version_id")


@dataclass(frozen=True, slots=True)
class FrozenHistorySourceSnapshot:
    """A pointer-only manifest for one long-term History source."""

    source_type: SourceType
    bindings: tuple[SourceIndexBinding, ...]

    def __post_init__(self) -> None:
        if self.source_type not in {
            SourceType.LONG_TERM_USER,
            SourceType.LONG_TERM_TASK,
        }:
            raise ValueError("History source snapshots are long-term memory only")
        bindings = tuple(self.bindings)
        if any(not isinstance(item, SourceIndexBinding) for item in bindings):
            raise ValueError("History snapshot bindings must be SourceIndexBinding values")
        if len(bindings) > MAX_FROZEN_HISTORY_BINDINGS:
            raise ValueError("frozen History source exceeds its manifest ceiling")
        ordered = tuple(
            sorted(
                bindings,
                key=lambda item: (
                    item.source_unit_id,
                    item.source_revision,
                    item.indexed_content_hash,
                ),
            )
        )
        identities = {
            (item.source_unit_id, item.source_revision) for item in ordered
        }
        if len(identities) != len(ordered):
            raise ValueError("History snapshot bindings must have unique identities")
        object.__setattr__(self, "bindings", ordered)

    @property
    def source_snapshot_id(self) -> str:
        digest, _ = source_index_binding_manifest(self.bindings)
        return f"{self.source_type.value}:{digest}"

    @property
    def source_revision_map(self) -> Mapping[str, str]:
        _, revision_map = source_index_binding_manifest(self.bindings)
        return revision_map

    def expected_access(self, source_filter: SourceFilter) -> SourceAccess:
        if source_filter.source_type is not self.source_type:
            raise ValueError("History snapshot filter has another SourceType")
        return SourceAccess(
            source_type=self.source_type,
            source_filter=source_filter,
            availability=(
                SourceAvailability.READY
                if self.bindings
                else SourceAvailability.EMPTY
            ),
            source_snapshot_id=self.source_snapshot_id,
            source_revision_map=self.source_revision_map,
        )

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "source_type": self.source_type.value,
            "source_snapshot_id": self.source_snapshot_id,
            "bindings": [
                {
                    "source_unit_id": item.source_unit_id,
                    "source_revision": item.source_revision,
                    "indexed_content_hash": item.indexed_content_hash,
                }
                for item in self.bindings
            ],
        }


def _canonical_history_authority(
    *,
    scopes: tuple[HistoryScope, ...],
    current_session_cutoff: str | None,
    long_term_task_id: str | None,
    source_snapshots: tuple[FrozenHistorySourceSnapshot, ...],
    scopes_field: str,
    snapshots_field: str,
    empty_scopes_error: str,
) -> tuple[
    tuple[HistoryScope, ...],
    tuple[FrozenHistorySourceSnapshot, ...],
]:
    frozen_scopes = tuple(scopes)
    if not frozen_scopes:
        raise ValueError(empty_scopes_error)
    if any(not isinstance(item, HistoryScope) for item in frozen_scopes):
        raise ValueError(f"{scopes_field} contains an invalid scope")
    if len(frozen_scopes) != len(set(frozen_scopes)):
        raise ValueError(f"{scopes_field} must be unique")
    canonical_scopes = tuple(
        scope for scope in HistoryScope if scope in frozen_scopes
    )

    if HistoryScope.CURRENT_SESSION in canonical_scopes:
        if current_session_cutoff is None:
            raise ValueError("current_session scope requires a committed-turn cutoff")
        parse_current_session_cutoff(current_session_cutoff)
    elif current_session_cutoff is not None:
        raise ValueError("current_session_cutoff requires current_session scope")

    if HistoryScope.CURRENT_TASK in canonical_scopes:
        if long_term_task_id is None:
            raise ValueError("current_task scope requires a trusted Task ID")
        _private_identifier(long_term_task_id, "long_term_task_id")
    elif long_term_task_id is not None:
        raise ValueError("long_term_task_id requires current_task scope")

    frozen_snapshots = tuple(source_snapshots)
    if any(
        not isinstance(item, FrozenHistorySourceSnapshot)
        for item in frozen_snapshots
    ):
        raise ValueError(f"{snapshots_field} contains an invalid snapshot")
    snapshot_sources = [item.source_type for item in frozen_snapshots]
    if len(snapshot_sources) != len(set(snapshot_sources)):
        raise ValueError("History source snapshots must be unique per Source")
    required_sources: set[SourceType] = set()
    if HistoryScope.LONG_TERM_USER in canonical_scopes:
        required_sources.add(SourceType.LONG_TERM_USER)
    if HistoryScope.CURRENT_TASK in canonical_scopes:
        required_sources.add(SourceType.LONG_TERM_TASK)
    if set(snapshot_sources) != required_sources:
        raise ValueError(
            "History scope must freeze every selected long-term Source exactly"
        )
    return canonical_scopes, tuple(
        sorted(frozen_snapshots, key=lambda item: item.source_type.value)
    )


@dataclass(frozen=True, slots=True)
class FrozenHistoryRetrievalScope:
    """An immutable History scope and committed-turn cutoff."""

    session_id: str
    scope_snapshot_id: str
    retrieval_data_version: str
    allowed_scopes: tuple[HistoryScope, ...]
    current_session_cutoff: str | None = None
    long_term_task_id: str | None = None
    long_term_source_snapshots: tuple[FrozenHistorySourceSnapshot, ...] = ()
    max_items: int = 20
    context_token_limit: int = DEFAULT_RETRIEVAL_TOKEN_LIMIT

    def __post_init__(self) -> None:
        _private_identifier(self.session_id, "session_id")
        _private_identifier(self.scope_snapshot_id, "scope_snapshot_id")
        _private_identifier(self.retrieval_data_version, "retrieval_data_version")
        scopes, snapshots = _canonical_history_authority(
            scopes=self.allowed_scopes,
            current_session_cutoff=self.current_session_cutoff,
            long_term_task_id=self.long_term_task_id,
            source_snapshots=self.long_term_source_snapshots,
            scopes_field="allowed_scopes",
            snapshots_field="long_term_source_snapshots",
            empty_scopes_error="history scope must allow at least one source",
        )
        object.__setattr__(self, "allowed_scopes", scopes)
        object.__setattr__(self, "long_term_source_snapshots", snapshots)
        _retrieval_limits(
            self.max_items,
            self.context_token_limit,
            max_items=MAX_RETRIEVAL_ITEMS,
        )

    @property
    def effect_scope(self) -> str:
        digest = hashlib.sha256(self.scope_snapshot_id.encode("utf-8")).hexdigest()
        return f"history-corpus:{digest[:24]}"

    def source_snapshot_for(
        self,
        source_type: SourceType,
    ) -> FrozenHistorySourceSnapshot | None:
        return next(
            (
                item
                for item in self.long_term_source_snapshots
                if item.source_type is source_type
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class RetrievalToolRequest:
    """Private, frozen request sent to a retrieval data-plane port.

    File requests must declare whether they address the current Session's
    authorized Document corpus or an exact File subset.  The scope is not
    inferred from binding emptiness, so an omitted binding list cannot silently
    broaden retrieval authority.
    """

    request_id: str
    corpus: RetrievalCorpus
    query: str
    limit: int
    context_token_limit: int
    session_id: str
    scope_snapshot_id: str
    retrieval_data_version: str
    query_variants: tuple[str, ...] = ()
    file_scope: FileRetrievalScope | None = None
    file_bindings: tuple[FrozenFileRetrievalBinding, ...] = ()
    session_file_bindings: tuple[FrozenFileVersionBinding, ...] = ()
    file_inventory_complete: bool = True
    history_scopes: tuple[HistoryScope, ...] = ()
    current_session_cutoff: str | None = None
    long_term_task_id: str | None = None
    history_source_snapshots: tuple[FrozenHistorySourceSnapshot, ...] = ()

    def __post_init__(self) -> None:
        _private_identifier(self.request_id, "request_id")
        if not isinstance(self.corpus, RetrievalCorpus):
            raise ValueError("corpus must be RetrievalCorpus")
        queries = tuple(
            _query(value)
            for value in (self.query, *tuple(self.query_variants))
        )
        if (
            len(queries) > MAX_FILE_RETRIEVAL_QUERIES
            and self.corpus is RetrievalCorpus.FILES
        ):
            raise ValueError(
                f"File retrieval accepts at most {MAX_FILE_RETRIEVAL_QUERIES} queries"
            )
        if len(queries) != len(set(queries)):
            raise ValueError("retrieval queries must be unique after trimming")
        object.__setattr__(self, "query", queries[0])
        object.__setattr__(self, "query_variants", queries[1:])
        _retrieval_limits(
            self.limit,
            self.context_token_limit,
            max_items=(
                MAX_FILE_RETRIEVAL_ITEMS
                if self.corpus is RetrievalCorpus.FILES
                else MAX_RETRIEVAL_ITEMS
            ),
            max_context_tokens=(
                MAX_FILE_RETRIEVAL_TOKEN_LIMIT
                if self.corpus is RetrievalCorpus.FILES
                else None
            ),
        )
        _private_identifier(self.session_id, "session_id")
        _private_identifier(self.scope_snapshot_id, "scope_snapshot_id")
        _private_identifier(self.retrieval_data_version, "retrieval_data_version")
        if self.corpus is RetrievalCorpus.FILES:
            self._validate_file_authority()
        else:
            self._validate_history_authority()

    @property
    def queries(self) -> tuple[str, ...]:
        return (self.query, *self.query_variants)

    def _validate_file_authority(self) -> None:
        if not isinstance(self.file_scope, FileRetrievalScope):
            raise ValueError("file request requires an explicit File retrieval scope")
        if not isinstance(self.file_inventory_complete, bool):
            raise ValueError("file_inventory_complete must be a bool")
        if (
            self.history_scopes
            or self.current_session_cutoff is not None
            or self.long_term_task_id is not None
            or self.history_source_snapshots
        ):
            raise ValueError("file request cannot carry history authority")
        bindings = tuple(self.file_bindings)
        session_bindings = tuple(self.session_file_bindings)
        if any(
            not isinstance(item, FrozenFileRetrievalBinding)
            for item in bindings
        ):
            raise ValueError("file_bindings contains an invalid binding")
        if any(
            not isinstance(item, FrozenFileVersionBinding)
            for item in session_bindings
        ):
            raise ValueError("session_file_bindings contains an invalid binding")
        session_binding_keys = [
            (item.project_id, item.file_id, item.file_version_id)
            for item in session_bindings
        ]
        if len(session_binding_keys) != len(set(session_binding_keys)):
            raise ValueError("session file-version bindings must be unique")
        source_ids = [item.source_id for item in bindings]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("file binding source IDs must be unique")
        authority_ids = [item.authority_id for item in bindings]
        if len(authority_ids) != len(set(authority_ids)):
            raise ValueError("file binding authority IDs must be unique")
        if self.file_scope is FileRetrievalScope.SESSION_CORPUS and bindings:
            raise ValueError("session_corpus File scope cannot carry file bindings")
        if (
            self.file_scope is FileRetrievalScope.SELECTED_FILES
            and session_bindings
        ):
            raise ValueError(
                "selected_files File scope cannot carry Session inventory"
            )
        if (
            self.file_scope is FileRetrievalScope.SELECTED_FILES
            and not bindings
        ):
            raise ValueError("selected_files File scope requires file bindings")
        object.__setattr__(self, "file_bindings", bindings)
        object.__setattr__(self, "session_file_bindings", session_bindings)

    def _validate_history_authority(self) -> None:
        if self.query_variants:
            raise ValueError("History retrieval accepts exactly one query")
        if self.file_scope is not None:
            raise ValueError("history request cannot carry a File retrieval scope")
        if self.file_inventory_complete is not True:
            raise ValueError("history request cannot carry File inventory coverage")
        if tuple(self.file_bindings):
            raise ValueError("history request cannot carry file authority")
        if tuple(self.session_file_bindings):
            raise ValueError("history request cannot carry Session file authority")
        scopes, snapshots = _canonical_history_authority(
            scopes=self.history_scopes,
            current_session_cutoff=self.current_session_cutoff,
            long_term_task_id=self.long_term_task_id,
            source_snapshots=self.history_source_snapshots,
            scopes_field="history_scopes",
            snapshots_field="history_source_snapshots",
            empty_scopes_error="history request requires selected scopes",
        )
        object.__setattr__(self, "file_bindings", ())
        object.__setattr__(self, "session_file_bindings", ())
        object.__setattr__(self, "history_scopes", scopes)
        object.__setattr__(self, "history_source_snapshots", snapshots)


@dataclass(frozen=True, slots=True)
class RetrievalToolEvidence:
    """Verified evidence returned before a public tool-result projection."""

    source_type: SourceType
    source_unit_id: str
    source_revision: str
    indexed_content_hash: str
    content: str
    estimated_tokens: int
    rank: int
    query_index: int = 0
    authority_id: str | None = None
    citation: Mapping[str, str] = field(default_factory=dict)
    query_matches: tuple[RetrievalQueryMatch, ...] = ()

    def __post_init__(self) -> None:
        matches = tuple(self.query_matches)
        if len(matches) > MAX_FILE_RETRIEVAL_QUERIES:
            raise ValueError("query_matches accepts at most four queries")
        if any(not isinstance(match, RetrievalQueryMatch) for match in matches):
            raise ValueError("query_matches must contain RetrievalQueryMatch values")
        if len({match.query_index for match in matches}) != len(matches):
            raise ValueError("query_matches must be unique by query_index")
        object.__setattr__(self, "query_matches", matches)
        if self.source_type not in {
            SourceType.DOCUMENT,
            SourceType.PICTURE,
            SourceType.CURRENT_SESSION,
            SourceType.LONG_TERM_USER,
            SourceType.LONG_TERM_TASK,
        }:
            raise ValueError("retrieval tool evidence has an unsupported SourceType")
        _private_identifier(self.source_unit_id, "source_unit_id")
        _private_identifier(self.source_revision, "source_revision")
        _sha256(self.indexed_content_hash, "indexed_content_hash")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.indexed_content_hash:
            raise ValueError("indexed_content_hash must match authoritative content")
        if (
            isinstance(self.estimated_tokens, bool)
            or not isinstance(self.estimated_tokens, int)
            or self.estimated_tokens <= 0
        ):
            raise ValueError("estimated_tokens must be positive")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be positive")
        if (
            isinstance(self.query_index, bool)
            or not isinstance(self.query_index, int)
            or self.query_index < 0
        ):
            raise ValueError("query_index must be non-negative")
        if self.authority_id is not None:
            _private_identifier(self.authority_id, "authority_id")
        if not isinstance(self.citation, Mapping):
            raise ValueError("citation must be a mapping")
        citation = dict(self.citation)
        if len(citation) > MAX_PRIVATE_CITATION_FIELDS:
            raise ValueError("citation has too many private fields")
        for key, value in citation.items():
            if not isinstance(key, str) or not key:
                raise ValueError("citation keys must be non-empty strings")
            if not isinstance(value, str) or len(value) > MAX_PRIVATE_CITATION_VALUE_CHARS:
                raise ValueError("citation values must be bounded strings")
        object.__setattr__(self, "citation", MappingProxyType(citation))


@dataclass(frozen=True, slots=True)
class RetrievalToolGap:
    code: str
    blocking: bool
    source_type: SourceType | None = None
    authority_id: str | None = None
    known_count: int | None = None

    def __post_init__(self) -> None:
        _safe_code(self.code, "gap code")
        if not isinstance(self.blocking, bool):
            raise ValueError("blocking must be a bool")
        if self.source_type is not None and self.source_type not in {
            SourceType.DOCUMENT,
            SourceType.PICTURE,
            SourceType.CURRENT_SESSION,
            SourceType.LONG_TERM_USER,
            SourceType.LONG_TERM_TASK,
        }:
            raise ValueError("gap has an unsupported SourceType")
        if self.authority_id is not None:
            _private_identifier(self.authority_id, "authority_id")
        if self.known_count is not None and (
            isinstance(self.known_count, bool)
            or not isinstance(self.known_count, int)
            or self.known_count < 0
        ):
            raise ValueError("known_count must be non-negative")


@dataclass(frozen=True, slots=True)
class RetrievalToolResult:
    status: RetrievalStatus
    scope_snapshot_id: str
    retrieval_data_version: str
    evidence: tuple[RetrievalToolEvidence, ...] = ()
    gaps: tuple[RetrievalToolGap, ...] = ()
    configured_token_limit: int = DEFAULT_RETRIEVAL_TOKEN_LIMIT
    packed_tokens: int = 0
    truncated: bool = False
    scope_is_current: bool = True
    encoder_fingerprint: str | None = None
    reranker_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RetrievalStatus):
            raise ValueError("status must be RetrievalStatus")
        _private_identifier(self.scope_snapshot_id, "scope_snapshot_id")
        _private_identifier(self.retrieval_data_version, "retrieval_data_version")
        if any(not isinstance(item, RetrievalToolEvidence) for item in self.evidence):
            raise ValueError("evidence contains an invalid item")
        ranks = tuple(item.rank for item in self.evidence)
        if len(ranks) != len(set(ranks)):
            raise ValueError("evidence ranks must be unique within one port result")
        if any(not isinstance(item, RetrievalToolGap) for item in self.gaps):
            raise ValueError("gaps contains an invalid item")
        if len(self.gaps) > 64:
            raise ValueError("port result has too many gaps")
        if (
            isinstance(self.configured_token_limit, bool)
            or not isinstance(self.configured_token_limit, int)
            or self.configured_token_limit <= 0
        ):
            raise ValueError("configured_token_limit must be positive")
        if (
            isinstance(self.packed_tokens, bool)
            or not isinstance(self.packed_tokens, int)
            or self.packed_tokens < 0
            or self.packed_tokens > self.configured_token_limit
        ):
            raise ValueError("packed_tokens must fit configured_token_limit")
        if not isinstance(self.truncated, bool) or not isinstance(
            self.scope_is_current,
            bool,
        ):
            raise ValueError("port result flags must be bools")
        for name, value in (
            ("encoder_fingerprint", self.encoder_fingerprint),
            ("reranker_fingerprint", self.reranker_fingerprint),
        ):
            if value is not None:
                _private_identifier(value, name)


class RetrievalToolPort(Protocol):
    def retrieve(self, request: RetrievalToolRequest) -> RetrievalToolResult: ...


def format_current_session_cutoff(assistant_turn_idx: int) -> str:
    if (
        isinstance(assistant_turn_idx, bool)
        or not isinstance(assistant_turn_idx, int)
        or assistant_turn_idx < 0
        or assistant_turn_idx >= 10**18
    ):
        raise ValueError("assistant_turn_idx must be a supported non-negative integer")
    return f"assistant_turn_idx:{assistant_turn_idx}"


def parse_current_session_cutoff(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("current_session_cutoff must be a string")
    matched = _CURRENT_SESSION_CUTOFF.fullmatch(value)
    if matched is None:
        raise ValueError(
            "current_session_cutoff must be assistant_turn_idx:<non-negative decimal>"
        )
    return int(matched.group(1))


def _private_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2_000:
        raise ValueError(f"{name} must be a bounded non-empty string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _public_source_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("source_id must be a bounded non-empty string")
    normalized = value.strip().replace("\\", "/")
    if (
        normalized.startswith(("/", "../", "~/", "file://"))
        or _WINDOWS_ABSOLUTE_PATH.match(normalized)
        or "/../" in normalized
        or normalized.endswith("/..")
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("source_id must be a safe public relative identifier")
    return value


def _query(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("query must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > MAX_RETRIEVAL_QUERY_CHARS:
        raise ValueError("query must be a bounded non-empty string")
    if any(ord(character) < 32 and character not in "\n\t\r" for character in normalized):
        raise ValueError("query must not contain control characters")
    return normalized


def _retrieval_limits(
    item_count: int,
    context_token_limit: int,
    *,
    max_items: int,
    max_context_tokens: int | None = None,
) -> None:
    if (
        isinstance(item_count, bool)
        or not isinstance(item_count, int)
        or not 1 <= item_count <= max_items
    ):
        raise ValueError(f"max_items must be from 1 to {max_items}")
    if (
        isinstance(context_token_limit, bool)
        or not isinstance(context_token_limit, int)
        or context_token_limit <= 0
        or (
            max_context_tokens is not None
            and context_token_limit > max_context_tokens
        )
    ):
        suffix = (
            f" to {max_context_tokens}"
            if max_context_tokens is not None
            else ""
        )
        raise ValueError(f"context_token_limit must be from 1{suffix}")


def _safe_code(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_CODE.fullmatch(value):
        raise ValueError(f"{name} must be a safe code")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


__all__ = (
    "DEFAULT_FILE_RETRIEVAL_RESULT_LIMIT",
    "DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT",
    "DEFAULT_RETRIEVAL_TOKEN_LIMIT",
    "FileRetrievalOrigin",
    "FileRetrievalReadinessResult",
    "FileRetrievalReadinessStatus",
    "FileRetrievalScope",
    "FrozenFileRetrievalBinding",
    "FrozenFileVersionBinding",
    "FrozenHistoryRetrievalScope",
    "FrozenHistorySourceSnapshot",
    "HistoryScope",
    "MAX_FILE_RETRIEVAL_ITEMS",
    "MAX_FILE_RETRIEVAL_QUERIES",
    "MAX_FILE_RETRIEVAL_TOKEN_LIMIT",
    "MAX_FROZEN_HISTORY_BINDINGS",
    "MAX_RETRIEVAL_ITEMS",
    "MAX_RETRIEVAL_QUERY_CHARS",
    "MAX_RETRIEVAL_QUERY_TOKENS",
    "RetrievalCorpus",
    "RetrievalStatus",
    "RetrievalToolEvidence",
    "RetrievalToolGap",
    "RetrievalToolPort",
    "RetrievalToolRequest",
    "RetrievalToolResult",
    "format_current_session_cutoff",
    "parse_current_session_cutoff",
)
