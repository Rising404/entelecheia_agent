"""Current Session 派生检索的窄合同与冻结绑定。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

from ....input_processing.documents.chunking import ChunkingProfile
from ...foundation import RetrievalFoundation
from ...lifecycle.generation import RetrievalGenerationSpec
from ...ports import BgeM3EncoderPort, RerankerPort
from ...profile import DocumentRetrievalCapability, DocumentRetrievalProfile

if TYPE_CHECKING:
    from .projection import CurrentSessionSourceAdapter


class SessionStoreReadPort(Protocol):
    def get_session(self, session_id: str) -> Mapping[str, object] | None: ...

    def get_committed_turn_pair(
        self,
        session_id: str,
        run_id: str,
    ) -> Mapping[str, object] | None: ...

    def list_committed_turn_pairs(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> Sequence[Mapping[str, object]]: ...


class SessionRetrievalNotReady(RuntimeError):
    """当前 Session 无法被完整发布到其精确派生 generation。"""


@dataclass(frozen=True, slots=True)
class SessionRetrievalComposition:
    foundation: RetrievalFoundation
    generation_spec: RetrievalGenerationSpec
    chunking_profile: ChunkingProfile
    source_adapter: CurrentSessionSourceAdapter
    encoder: BgeM3EncoderPort
    reranker: RerankerPort | None
    requested_profile: DocumentRetrievalProfile
    effective_profile: DocumentRetrievalProfile
    capability: DocumentRetrievalCapability
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SessionTurnRetrievalBinding:
    """AcceptedTurn 冻结的 Session generation 与已提交 Turn 截止点。"""

    composition: SessionRetrievalComposition
    data_version_id: str
    assistant_turn_cutoff: int | None

    def __post_init__(self) -> None:
        if self.data_version_id != self.composition.generation_spec.version_id:
            raise ValueError("Session Turn binding changed retrieval generation")
        cutoff = self.assistant_turn_cutoff
        if cutoff is not None and (
            isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 0
        ):
            raise ValueError("Session retrieval cutoff must be non-negative")


__all__ = [
    "SessionRetrievalComposition",
    "SessionRetrievalNotReady",
    "SessionStoreReadPort",
    "SessionTurnRetrievalBinding",
]
