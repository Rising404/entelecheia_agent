"""Live Tool Catalog model exposed to Runtime callers.

Durable control-plane, snapshot-codec, and materialization APIs live in their
explicit subpackages.  Importing this package deliberately loads only the
in-memory catalog model.
"""

from .model import (
    CatalogChange,
    CatalogConflictError,
    CatalogEntry,
    CatalogError,
    CatalogSnapshot,
    CatalogStatus,
    ToolCatalog,
    ToolKey,
    ToolResolutionError,
    validate_catalog_status_transition,
)

__all__ = [
    "CatalogChange",
    "CatalogConflictError",
    "CatalogEntry",
    "CatalogError",
    "CatalogSnapshot",
    "CatalogStatus",
    "ToolCatalog",
    "ToolKey",
    "ToolResolutionError",
    "validate_catalog_status_transition",
]
