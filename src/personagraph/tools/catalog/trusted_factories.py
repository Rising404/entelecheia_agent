"""Resolve default tools through one explicit trusted factory allowlist.

Factory references are opaque keys, not Python import paths. A persisted value can
only select an entry registered explicitly by the running process; catalog data is
never interpreted as a module or attribute name.

Contextual factories keep provider instances, resource scope, credentials, and
effective effects exclusively in the live :class:`ToolBinding`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any

from .binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
    ToolIdentity,
)
from ..contracts import ToolSourceDescriptor
from ..effects import ToolEffectProfile
from ..registration import (
    ToolHandler,
    ToolRegistration,
    _registration_descriptor,
)


DECLARED_IMPLEMENTATION_SCHEMA = "trusted-tool-implementation-v1"
_FACTORY_KEY = re.compile(
    r"[a-z0-9][a-z0-9_-]*(?:/[a-z0-9][a-z0-9_-]*)+\Z"
)


class TrustedCatalogResolutionError(RuntimeError):
    """A persisted definition cannot be bound without weakening its identity."""


class UnknownToolFactoryError(TrustedCatalogResolutionError):
    """The persisted implementation key is absent from the explicit allowlist."""


class ToolFactoryDriftError(TrustedCatalogResolutionError):
    """Trusted code no longer matches the exact persisted definition."""


class ToolFactoryUnavailableError(RuntimeError):
    """A known factory cannot currently provide its declared implementation."""

    def __init__(self, reason: str) -> None:
        normalized = str(reason).strip()
        if not normalized:
            raise ValueError("factory unavailable reason must not be empty")
        super().__init__(normalized)
        self.reason = normalized


RegistrationFactory = Callable[[], ToolRegistration]


@dataclass(frozen=True)
class TrustedContextualBinding:
    """Live binding inputs whose serializable fields contain no credentials."""

    source: ToolSourceDescriptor
    handler: ToolHandler
    effect_profile: ToolEffectProfile
    binding_assertion: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.source, ToolSourceDescriptor):
            raise TypeError("contextual binding source must be a ToolSourceDescriptor")
        if self.source.fingerprint is None:
            raise ValueError("contextual binding source fingerprint must not be empty")
        if not callable(self.handler):
            raise TypeError("contextual binding handler must be callable")
        if not isinstance(self.effect_profile, ToolEffectProfile):
            raise TypeError("contextual binding effects must be a ToolEffectProfile")
        if not isinstance(self.binding_assertion, Mapping):
            raise TypeError("contextual binding assertion must be a mapping")


ContextualBindingFactory = Callable[[], TrustedContextualBinding]


@dataclass(frozen=True)
class TrustedStaticDefaultToolFactory:
    """One process-static implementation and its declared behavior revision.

    ``declared_behavior_revision`` is a reviewed declaration, not a measurement of
    handler source or bytecode. The implementation digest also commits the full
    registration descriptor. Changing either requires a new immutable definition.
    A future trusted build-manifest digest must use a new encoding/version rather
    than silently changing this declaration's meaning.
    """

    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition
    source: ToolSourceDescriptor
    registration_factory: RegistrationFactory

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_ref, str) or _FACTORY_KEY.fullmatch(
            self.implementation_ref
        ) is None:
            raise ValueError(
                "implementation_ref must be an opaque namespaced factory key"
            )
        declared_revision = str(self.declared_behavior_revision).strip()
        if not declared_revision:
            raise ValueError("declared_behavior_revision must not be empty")
        object.__setattr__(
            self,
            "declared_behavior_revision",
            declared_revision,
        )
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        if self.definition.implementation_ref != self.implementation_ref:
            raise ValueError("definition implementation_ref does not match factory key")
        if not isinstance(self.source, ToolSourceDescriptor):
            raise TypeError("source must be a ToolSourceDescriptor")
        if self.source.fingerprint is None:
            raise ValueError("trusted factory source fingerprint must not be empty")
        if not callable(self.registration_factory):
            raise TypeError("registration_factory must be callable")
        expected_digest = _declared_implementation_digest(
            implementation_ref=self.implementation_ref,
            declared_behavior_revision=self.declared_behavior_revision,
            registration_descriptor=self._expected_registration_descriptor(),
        )
        if self.definition.implementation_digest != expected_digest:
            raise ValueError(
                "definition implementation digest does not match factory artifact"
            )

    @classmethod
    def from_registration(
        cls,
        *,
        implementation_ref: str,
        declared_behavior_revision: str,
        registration: ToolRegistration,
        registration_factory: RegistrationFactory | None = None,
    ) -> "TrustedStaticDefaultToolFactory":
        """Freeze a serializable definition from one trusted registration manifest."""

        if not isinstance(registration, ToolRegistration):
            raise TypeError("registration manifest must be a ToolRegistration")
        if registration.source.fingerprint is None:
            raise ValueError("registration source fingerprint must not be empty")
        descriptor = registration.descriptor()
        implementation_digest = _declared_implementation_digest(
            implementation_ref=implementation_ref,
            declared_behavior_revision=declared_behavior_revision,
            registration_descriptor=descriptor,
        )
        definition = ToolDefinition(
            spec=registration.spec,
            implementation_version=registration.implementation_version,
            implementation_ref=implementation_ref,
            implementation_digest=implementation_digest,
            effect_template=registration.effect_profile,
            execution_profile=registration.execution_profile,
        )
        factory = (
            registration_factory
            if registration_factory is not None
            else lambda: registration
        )
        return cls(
            implementation_ref=implementation_ref,
            declared_behavior_revision=declared_behavior_revision,
            definition=definition,
            source=registration.source,
            registration_factory=factory,
        )

    @property
    def identity(self) -> ToolIdentity:
        return self.definition.identity

    def build_registration(self) -> ToolRegistration:
        """Build and revalidate the live registration against the frozen manifest."""

        try:
            registration = self.registration_factory()
        except ToolFactoryUnavailableError:
            raise
        if not isinstance(registration, ToolRegistration):
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} returned a non-ToolRegistration"
            )
        self._validate_registration(registration)
        return registration

    def bind(self, definition: ToolDefinition) -> BoundToolRegistration:
        """Bind one exact persisted definition to the verified live handler."""

        _require_definition_match(self.definition, definition)
        registration = self.build_registration()
        binding = ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=registration.source,
            handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion={
                "factory_key": self.implementation_ref,
                "binding_kind": "process_static_default",
                "implementation_digest": definition.implementation_digest,
                "source_fingerprint": registration.source.fingerprint,
            },
        )
        return BoundToolRegistration(definition, binding)

    def _expected_registration_descriptor(self) -> dict[str, object]:
        return _registration_descriptor(
            spec=self.definition.spec,
            implementation_version=self.definition.implementation_version,
            source=self.source,
            effect_profile=self.definition.effect_template,
            execution_profile=self.definition.execution_profile,
        )

    def _validate_registration(self, registration: ToolRegistration) -> None:
        actual_identity = ToolIdentity(
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
        )
        if actual_identity != self.identity:
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} changed tool identity"
            )
        if registration.spec != self.definition.spec:
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} changed tool spec"
            )
        if registration.effect_profile != self.definition.effect_template:
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} changed effect template"
            )
        if registration.execution_profile != self.definition.execution_profile:
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} changed execution profile"
            )
        if registration.source != self.source:
            detail = (
                "source fingerprint"
                if registration.source.fingerprint != self.source.fingerprint
                else "source descriptor"
            )
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} changed {detail}"
            )
        implementation_digest = _declared_implementation_digest(
            implementation_ref=self.implementation_ref,
            declared_behavior_revision=self.declared_behavior_revision,
            registration_descriptor=registration.descriptor(),
        )
        if implementation_digest != self.definition.implementation_digest:
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} changed implementation digest"
            )


@dataclass(frozen=True)
class TrustedContextualDefaultToolFactory:
    """Environment-independent Definition with a materialization-time Binding."""

    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition
    source: ToolSourceDescriptor
    binding_factory: ContextualBindingFactory

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_ref, str) or _FACTORY_KEY.fullmatch(
            self.implementation_ref
        ) is None:
            raise ValueError(
                "implementation_ref must be an opaque namespaced factory key"
            )
        declared_revision = str(self.declared_behavior_revision).strip()
        if not declared_revision:
            raise ValueError("declared_behavior_revision must not be empty")
        object.__setattr__(
            self,
            "declared_behavior_revision",
            declared_revision,
        )
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        if self.definition.implementation_ref != self.implementation_ref:
            raise ValueError("definition implementation_ref does not match factory key")
        if not isinstance(self.source, ToolSourceDescriptor):
            raise TypeError("source must be a ToolSourceDescriptor")
        if self.source.fingerprint is None:
            raise ValueError("trusted factory source fingerprint must not be empty")
        if not callable(self.binding_factory):
            raise TypeError("binding_factory must be callable")
        expected_digest = _declared_implementation_digest(
            implementation_ref=self.implementation_ref,
            declared_behavior_revision=self.declared_behavior_revision,
            registration_descriptor=self._expected_registration_descriptor(),
        )
        if self.definition.implementation_digest != expected_digest:
            raise ValueError(
                "definition implementation digest does not match contextual artifact"
            )

    @classmethod
    def from_registration_surface(
        cls,
        *,
        implementation_ref: str,
        declared_behavior_revision: str,
        registration: ToolRegistration,
        binding_factory: ContextualBindingFactory,
    ) -> "TrustedContextualDefaultToolFactory":
        """Create the legacy-compatible Definition from a stable registration surface.

        The manifest source is part of the immutable implementation identity. The
        materialized provider pipeline is not: it belongs only to the live binding.
        """

        if not isinstance(registration, ToolRegistration):
            raise TypeError("registration manifest must be a ToolRegistration")
        if registration.source.fingerprint is None:
            raise ValueError("registration source fingerprint must not be empty")
        definition = ToolDefinition(
            spec=registration.spec,
            implementation_version=registration.implementation_version,
            implementation_ref=implementation_ref,
            implementation_digest=_declared_implementation_digest(
                implementation_ref=implementation_ref,
                declared_behavior_revision=declared_behavior_revision,
                registration_descriptor=registration.descriptor(),
            ),
            effect_template=registration.effect_profile,
            execution_profile=registration.execution_profile,
        )
        return cls(
            implementation_ref=implementation_ref,
            declared_behavior_revision=declared_behavior_revision,
            definition=definition,
            source=registration.source,
            binding_factory=binding_factory,
        )

    @property
    def identity(self) -> ToolIdentity:
        return self.definition.identity

    def bind(self, definition: ToolDefinition) -> BoundToolRegistration:
        _require_definition_match(self.definition, definition)
        try:
            contextual = self.binding_factory()
        except ToolFactoryUnavailableError:
            raise
        if not isinstance(contextual, TrustedContextualBinding):
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} returned an invalid binding"
            )
        assertion = dict(contextual.binding_assertion)
        reserved = {
            "binding_kind",
            "factory_key",
            "implementation_digest",
        }
        if reserved.intersection(assertion):
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} replaced reserved assertions"
            )
        binding = ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=contextual.source,
            handler=contextual.handler,
            effect_profile=contextual.effect_profile,
            binding_assertion={
                "factory_key": self.implementation_ref,
                "binding_kind": "trusted_contextual_default",
                "implementation_digest": definition.implementation_digest,
                **assertion,
            },
        )
        try:
            return BoundToolRegistration(definition, binding)
        except ValueError as exc:
            raise ToolFactoryDriftError(
                f"factory {self.implementation_ref!r} produced an invalid binding"
            ) from exc

    def _expected_registration_descriptor(self) -> dict[str, object]:
        return _registration_descriptor(
            spec=self.definition.spec,
            implementation_version=self.definition.implementation_version,
            source=self.source,
            effect_profile=self.definition.effect_template,
            execution_profile=self.definition.execution_profile,
        )


TrustedDefaultToolFactory = (
    TrustedStaticDefaultToolFactory | TrustedContextualDefaultToolFactory
)


class TrustedDefaultFactoryRegistry:
    """Immutable allowlist for static and trusted contextual default factories."""

    def __init__(self, factories: Sequence[TrustedDefaultToolFactory]) -> None:
        if isinstance(factories, (str, bytes)):
            raise TypeError("factories must be a sequence of trusted factories")
        normalized = tuple(factories)
        if any(
            not isinstance(
                item,
                (
                    TrustedStaticDefaultToolFactory,
                    TrustedContextualDefaultToolFactory,
                ),
            )
            for item in normalized
        ):
            raise TypeError("factories must contain only trusted factory values")
        by_ref: dict[str, TrustedDefaultToolFactory] = {}
        by_identity: dict[ToolIdentity, TrustedDefaultToolFactory] = {}
        for factory in normalized:
            if factory.implementation_ref in by_ref:
                raise ValueError(
                    f"duplicate trusted factory key: {factory.implementation_ref!r}"
                )
            if factory.identity in by_identity:
                raise ValueError(
                    f"duplicate trusted factory identity: {factory.identity!r}"
                )
            by_ref[factory.implementation_ref] = factory
            by_identity[factory.identity] = factory
        self._factories: Mapping[
            str,
            TrustedDefaultToolFactory,
        ] = MappingProxyType(by_ref)
        self._factories_by_identity: Mapping[
            ToolIdentity,
            TrustedDefaultToolFactory,
        ] = MappingProxyType(by_identity)
        self._ordered = normalized

    def factories(self) -> tuple[TrustedDefaultToolFactory, ...]:
        return self._ordered

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(factory.definition for factory in self._ordered)

    def owns(self, definition: ToolDefinition) -> bool:
        """Report exact trusted ownership without constructing a live binding."""

        if not isinstance(definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        factory_by_ref = self._factories.get(definition.implementation_ref)
        factory_by_identity = self._factories_by_identity.get(definition.identity)
        if factory_by_ref is None and factory_by_identity is None:
            return False
        if factory_by_ref is None:
            raise ToolFactoryDriftError(
                "trusted tool identity changed its implementation reference"
            )
        if factory_by_identity is None:
            raise ToolFactoryDriftError(
                "trusted factory key is owned by a different tool identity"
            )
        if factory_by_ref is not factory_by_identity:
            raise ToolFactoryDriftError(
                "trusted factory key and tool identity resolve to different owners"
            )
        # Ownership is also an integrity assertion.  A disabled optional entry may
        # skip live binding, but it must never hide a Definition drift behind an
        # availability outcome.
        _require_definition_match(factory_by_ref.definition, definition)
        return True

    def resolve(
        self,
        definition: ToolDefinition,
        *,
        expected_definition_digest: str | None = None,
    ) -> BoundToolRegistration:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        if (
            expected_definition_digest is not None
            and definition.digest != expected_definition_digest
        ):
            raise ToolFactoryDriftError(
                "persisted definition digest does not match the selected profile"
            )
        factory = self._factories.get(definition.implementation_ref)
        if factory is None:
            raise UnknownToolFactoryError(
                f"unknown trusted factory key: {definition.implementation_ref!r}"
            )
        return factory.bind(definition)


def _require_definition_match(
    expected: ToolDefinition,
    actual: ToolDefinition,
) -> None:
    if actual.identity != expected.identity:
        raise ToolFactoryDriftError("persisted ToolDefinition identity drifted")
    if actual.spec != expected.spec:
        raise ToolFactoryDriftError("persisted ToolDefinition spec drifted")
    if actual.implementation_ref != expected.implementation_ref:
        raise ToolFactoryDriftError(
            "persisted ToolDefinition implementation reference drifted"
        )
    if actual.implementation_digest != expected.implementation_digest:
        raise ToolFactoryDriftError(
            "persisted ToolDefinition implementation digest drifted"
        )
    if actual.effect_template != expected.effect_template:
        raise ToolFactoryDriftError("persisted ToolDefinition effect template drifted")
    if actual.execution_profile != expected.execution_profile:
        raise ToolFactoryDriftError(
            "persisted ToolDefinition execution profile drifted"
        )
    if actual.digest != expected.digest:
        raise ToolFactoryDriftError("persisted ToolDefinition digest drifted")


def _declared_implementation_digest(
    *,
    implementation_ref: str,
    declared_behavior_revision: str,
    registration_descriptor: Mapping[str, object],
) -> str:
    """Hash a reviewed declaration, never bytecode, paths, or callable reprs."""

    artifact = {
        "schema_version": DECLARED_IMPLEMENTATION_SCHEMA,
        "implementation_ref": implementation_ref,
        # Keep the v1 wire key stable so this honest Python rename does not alter
        # already-published Definition digests.
        "semantic_artifact": declared_behavior_revision,
        "registration": dict(registration_descriptor),
    }
    return _artifact_digest(artifact)


def _artifact_digest(artifact: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            dict(artifact),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("trusted implementation artifact must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ContextualBindingFactory",
    "DECLARED_IMPLEMENTATION_SCHEMA",
    "ToolFactoryDriftError",
    "ToolFactoryUnavailableError",
    "TrustedCatalogResolutionError",
    "TrustedContextualBinding",
    "TrustedContextualDefaultToolFactory",
    "TrustedDefaultFactoryRegistry",
    "TrustedDefaultToolFactory",
    "TrustedStaticDefaultToolFactory",
    "UnknownToolFactoryError",
]
