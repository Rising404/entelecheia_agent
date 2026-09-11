"""L1 Turn attachment tool-source contracts.

These values describe the frozen public attachment surface owned by an L1 Turn.
They intentionally do not import the mounted-document planning implementation, so
the L1 controller and catalog contracts stay loadable without initializing L2.
"""

from __future__ import annotations

class L1TurnAttachmentToolRuntimeError(RuntimeError):
    """The frozen attachment authority cannot be replayed safely."""

    code = "l1_turn_attachment_authority_unavailable"

    def __init__(self) -> None:
        super().__init__(self.code)

__all__ = [
    "L1TurnAttachmentToolRuntimeError",
]
