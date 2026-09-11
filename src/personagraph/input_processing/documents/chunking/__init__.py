"""结构优先文档切块的稳定入口。"""

from .contracts import (
    CHUNKER_NAME,
    CHUNKER_VERSION,
    CHUNK_LOCATION_HASH_LABEL,
    CHUNK_LOCATION_MAX_CHARS,
    ChunkSpan,
    ChunkingProfile,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_SPLIT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    DocumentChunk,
    HEURISTIC_TOKENIZER_ID,
    bounded_chunk_location,
)
from .structure_first import chunk_document, chunker_fingerprint

__all__ = [
    "CHUNKER_NAME",
    "CHUNKER_VERSION",
    "CHUNK_LOCATION_HASH_LABEL",
    "CHUNK_LOCATION_MAX_CHARS",
    "ChunkSpan",
    "ChunkingProfile",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MIN_TOKENS",
    "DEFAULT_SPLIT_OVERLAP_TOKENS",
    "DEFAULT_TARGET_TOKENS",
    "DocumentChunk",
    "HEURISTIC_TOKENIZER_ID",
    "bounded_chunk_location",
    "chunk_document",
    "chunker_fingerprint",
]
