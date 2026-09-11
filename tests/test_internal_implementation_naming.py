"""Prevent implementation generations from becoming parallel internal APIs."""

from __future__ import annotations

import io
from pathlib import Path
import re
import tokenize


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "personagraph"
_VERSIONED_NAME = re.compile(
    r"(?:^|_)[vV][0-9]+(?:_|$)|V[0-9]+(?=[A-Z_]|$)"
)
_MILESTONE_NAME = re.compile(
    r"^M[0-9]+(?=[A-Z_]|$)|_[mM][0-9]+[a-zA-Z]?(?:_|$)"
)

# These names describe external model identities or ordered storage migrations. They
# are not implementation generations and must retain the version carried by the
# identity they represent.
_IDENTITY_NAMES = frozenset(
    {
        "BGE_V2_M3",
        "BGE_V2_M3_RERANKER_MANIFEST",
        "BGE_V2_M3_RERANKER_MODEL_ID",
        "BGE_V2_M3_RERANKER_REVISION",
        "_SCHEMA_V1_STATEMENTS",
        "_migration_v2_method_indexes",
        "_migration_v3_method_index_manifest",
        "_migration_v4_source_coverage_lookup",
    }
)


def _is_external_identity_name(name: str) -> bool:
    return (
        name in _IDENTITY_NAMES
        or name == "BGE_M3"
        or name.startswith("BGE_M3_")
        or name.startswith("BGE_V2_M3_")
    )


def test_product_python_names_do_not_encode_implementation_generations() -> None:
    unexpected: list[str] = []

    for source_path in sorted(_SOURCE_ROOT.rglob("*.py")):
        relative_path = source_path.relative_to(_SOURCE_ROOT)
        for part in relative_path.parts:
            module_part = part.removesuffix(".py")
            if _VERSIONED_NAME.search(module_part) or _MILESTONE_NAME.search(
                module_part
            ):
                unexpected.append(
                    f"implementation-generation module path: {relative_path}"
                )
                break

        source = source_path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if (
                token.type == tokenize.NAME
                and (
                    _VERSIONED_NAME.search(token.string)
                    or _MILESTONE_NAME.search(token.string)
                )
                and not _is_external_identity_name(token.string)
            ):
                unexpected.append(
                    f"{relative_path}:{token.start[0]}: implementation-generation Python name "
                    f"{token.string!r}"
                )

    assert not unexpected, "\n".join(unexpected)
