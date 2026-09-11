"""Tool Platform 的版本化快照目录。

Catalog 只持有注册发现和生命周期。之后 Runtime 可以将快照绑定到 Attempt，
但本模块绝不创建 Attempt，也不存储其标识符。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from ..registration import ExecutableToolRegistration
from .binding import BoundToolRegistration, ToolIdentity


class CatalogStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"
    RETIRED = "retired"


class CatalogError(ValueError):
    code = "catalog_error"


class CatalogConflictError(CatalogError):
    code = "catalog_revision_conflict"


class ToolResolutionError(CatalogError):
    code = "tool_not_resolvable"


@dataclass(frozen=True, order=True)
class ToolKey:
    tool_id: str
    contract_version: str


@dataclass(frozen=True)
class CatalogEntry:
    key: ToolKey
    registration: ExecutableToolRegistration
    status: CatalogStatus
    created_revision: int
    updated_revision: int

    def descriptor(self) -> dict[str, object]:
        return {
            "tool_id": self.key.tool_id,
            "contract_version": self.key.contract_version,
            "status": self.status.value,
            "created_revision": self.created_revision,
            "updated_revision": self.updated_revision,
            "registration": self.registration.descriptor(),
        }

    def bound_descriptor(self) -> dict[str, object]:
        """投影带 Definition/Binding digest 的新 execution snapshot entry。"""

        if not isinstance(self.registration, BoundToolRegistration):
            raise CatalogError(
                "bound catalog descriptor requires only BoundToolRegistration entries"
            )
        return {
            "tool_id": self.key.tool_id,
            "contract_version": self.key.contract_version,
            "status": self.status.value,
            "created_revision": self.created_revision,
            "updated_revision": self.updated_revision,
            "registration": self.registration.bound_descriptor(),
        }


@dataclass(frozen=True)
class CatalogChange:
    revision: int
    action: str
    key: ToolKey | None
    detail: str | None = None


@dataclass(frozen=True)
class CatalogSnapshot:
    revision: int
    entries: tuple[CatalogEntry, ...]

    def resolve(
        self,
        tool_id: str,
        contract_version: str | None = None,
    ) -> ExecutableToolRegistration:
        candidates = [
            entry
            for entry in self.entries
            if entry.key.tool_id == tool_id and (contract_version is None or entry.key.contract_version == contract_version)
        ]
        if not candidates:
            raise ToolResolutionError(f"unknown tool or contract version: {tool_id!r}")
        resolvable = [entry for entry in candidates if entry.status in {CatalogStatus.ACTIVE, CatalogStatus.DEPRECATED}]
        if not resolvable:
            states = sorted({entry.status.value for entry in candidates})
            raise ToolResolutionError(f"tool {tool_id!r} is not resolvable in statuses {states!r}")
        active = [entry for entry in resolvable if entry.status is CatalogStatus.ACTIVE]
        selected = active or resolvable
        if len(selected) != 1:
            versions = sorted(entry.key.contract_version for entry in selected)
            raise ToolResolutionError(f"ambiguous contract version for {tool_id!r}: {versions!r}")
        return selected[0].registration

    def exposed(self, *, include_deprecated: bool = True) -> tuple[CatalogEntry, ...]:
        statuses = {CatalogStatus.ACTIVE}
        if include_deprecated:
            statuses.add(CatalogStatus.DEPRECATED)
        return tuple(entry for entry in self.entries if entry.status in statuses)

    def to_descriptor(self) -> dict[str, object]:
        return {"revision": self.revision, "entries": [entry.descriptor() for entry in self.entries]}

    def to_bound_descriptor(self) -> dict[str, object]:
        """显式生成不可丢失 Definition/Binding digest 的新快照投影。"""

        return {
            "revision": self.revision,
            "entries": [entry.bound_descriptor() for entry in self.entries],
        }


ReferenceGuard = Callable[[CatalogEntry], bool]


class ToolCatalog:
    """具有 CAS、审计历史、回滚和快照能力的应用级目录。"""

    def __init__(self) -> None:
        self._entries: dict[ToolKey, CatalogEntry] = {}
        self._definition_digests: dict[ToolIdentity, str] = {}
        self._revision = 0
        self._history: dict[int, dict[ToolKey, CatalogEntry]] = {0: {}}
        self._audit: list[CatalogChange] = []

    @property
    def revision(self) -> int:
        return self._revision

    def audit_log(self) -> tuple[CatalogChange, ...]:
        return tuple(self._audit)

    def register(
        self,
        registration: ExecutableToolRegistration,
        *,
        status: CatalogStatus | None = None,
        expected_revision: int | None = None,
        replace: bool = False,
    ) -> CatalogEntry:
        self._check_revision(expected_revision)
        self._check_definition_identity(registration)
        key = ToolKey(registration.tool_id, registration.contract_version)
        previous = self._entries.get(key)
        if previous and not replace:
            raise CatalogConflictError(f"tool already registered: {key!r}")
        next_revision = self._revision + 1
        desired_status = status if status is not None else previous.status if previous else CatalogStatus.ACTIVE
        entry = CatalogEntry(
            key=key,
            registration=registration,
            status=desired_status,
            created_revision=previous.created_revision if previous else next_revision,
            updated_revision=next_revision,
        )
        self._entries[key] = entry
        if isinstance(registration, BoundToolRegistration):
            self._definition_digests[
                registration.identity
            ] = registration.definition_digest
        self._commit("replace" if previous else "register", key)
        return entry

    def set_status(
        self,
        key: ToolKey,
        status: CatalogStatus,
        *,
        expected_revision: int | None = None,
    ) -> CatalogEntry:
        self._check_revision(expected_revision)
        previous = self._entry(key)
        if status is previous.status:
            return previous
        validate_catalog_status_transition(previous.status, status)
        next_revision = self._revision + 1
        entry = CatalogEntry(key, previous.registration, status, previous.created_revision, next_revision)
        self._entries[key] = entry
        self._commit("set_status", key, f"{previous.status.value}->{status.value}")
        return entry

    def rollback(self, target_revision: int, *, expected_revision: int | None = None) -> int:
        self._check_revision(expected_revision)
        try:
            target = self._history[target_revision]
        except KeyError as exc:
            raise CatalogError(f"unknown catalog revision {target_revision}") from exc
        self._entries = dict(target)
        self._commit("rollback", None, f"to_revision={target_revision}")
        return self._revision

    def purge_retired(
        self,
        key: ToolKey,
        *,
        reference_guard: ReferenceGuard,
        expected_revision: int | None = None,
    ) -> None:
        self._check_revision(expected_revision)
        entry = self._entry(key)
        if entry.status is not CatalogStatus.RETIRED:
            raise CatalogError("only retired registrations may be physically purged")
        if reference_guard(entry):
            raise CatalogError("retired registration remains protected by a historical reference")
        del self._entries[key]
        self._commit("purge", key)

    def get(self, key: ToolKey) -> CatalogEntry:
        return self._entry(key)

    def entries_for_source(self, source_id: str) -> tuple[CatalogEntry, ...]:
        return tuple(entry for entry in self._entries.values() if entry.registration.source.source_id == source_id)

    def snapshot(self) -> CatalogSnapshot:
        # 元组固定此目录修订中的成员及注册标识。
        return CatalogSnapshot(self._revision, tuple(sorted(self._entries.values(), key=lambda entry: entry.key)))

    def _entry(self, key: ToolKey) -> CatalogEntry:
        try:
            return self._entries[key]
        except KeyError as exc:
            raise CatalogError(f"unknown tool registration: {key!r}") from exc

    def _check_revision(self, expected_revision: int | None) -> None:
        if expected_revision is not None and expected_revision != self._revision:
            raise CatalogConflictError(
                f"catalog revision mismatch: expected {expected_revision}, current {self._revision}"
            )

    def _check_definition_identity(
        self,
        registration: ExecutableToolRegistration,
    ) -> None:
        if isinstance(registration, BoundToolRegistration):
            previous_digest = self._definition_digests.get(registration.identity)
            if (
                previous_digest is not None
                and previous_digest != registration.definition_digest
            ):
                raise CatalogConflictError(
                    "tool definition changed under an existing ToolIdentity"
                )
            return
        identity = ToolIdentity(
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
        )
        key = ToolKey(registration.tool_id, registration.contract_version)
        previous = self._entries.get(key)
        if identity in self._definition_digests or (
            previous is not None
            and isinstance(previous.registration, BoundToolRegistration)
        ):
            raise CatalogConflictError(
                "unbound ToolRegistration cannot replace an established bound identity"
            )

    def _commit(self, action: str, key: ToolKey | None, detail: str | None = None) -> None:
        self._revision += 1
        self._history[self._revision] = dict(self._entries)
        self._audit.append(CatalogChange(self._revision, action, key, detail))


_ALLOWED_TRANSITIONS: dict[CatalogStatus, set[CatalogStatus]] = {
    CatalogStatus.DRAFT: {CatalogStatus.ACTIVE, CatalogStatus.RETIRED},
    CatalogStatus.ACTIVE: {CatalogStatus.DEPRECATED, CatalogStatus.DISABLED, CatalogStatus.RETIRED},
    CatalogStatus.DEPRECATED: {CatalogStatus.ACTIVE, CatalogStatus.DISABLED, CatalogStatus.RETIRED},
    CatalogStatus.DISABLED: {CatalogStatus.ACTIVE, CatalogStatus.RETIRED},
    CatalogStatus.RETIRED: set(),
}


def validate_catalog_status_transition(
    previous: CatalogStatus,
    next_status: CatalogStatus,
) -> None:
    if next_status not in _ALLOWED_TRANSITIONS[previous]:
        raise CatalogError(f"invalid catalog lifecycle transition: {previous.value} -> {next_status.value}")
