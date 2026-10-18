"""Choose the supported L0/L1 processing route for an accepted Turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...turn.contracts import AcceptedEntryTurn, ProcessingRouteAuthorityError
from ..ingress.model_contracts import EntryClassification
from .policy import ProcessingLevel


@dataclass(frozen=True, slots=True)
class EntryResponseProcessingRoute:
    processing_level: Literal["L0"] = "L0"
    kind: Literal["generate_response"] = "generate_response"


@dataclass(frozen=True, slots=True)
class EntryL1ProcessingRoute:
    kind: Literal["l1"] = "l1"
    processing_level: Literal["L1"] = "L1"


EntryProcessingRoute = EntryResponseProcessingRoute | EntryL1ProcessingRoute


def select_entry_processing_route(
    *,
    accepted: AcceptedEntryTurn,
    classification: EntryClassification,
    processing_level: ProcessingLevel,
) -> EntryProcessingRoute:
    """Reject disabled or mismatched routes before entering an executor."""
    if (
        classification.processing_level != processing_level
        or not accepted.routing_policy.allows(processing_level)
    ):
        raise ProcessingRouteAuthorityError("processing route differs from accepted authority")
    if processing_level == "L1":
        return EntryL1ProcessingRoute()
    if processing_level == "L0":
        return EntryResponseProcessingRoute()
    raise ProcessingRouteAuthorityError(f"unsupported processing level: {processing_level}")
