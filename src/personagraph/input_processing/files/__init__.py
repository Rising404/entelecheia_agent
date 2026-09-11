"""不可信输入文件的有界识别与容器预检。"""

from .contracts import DetectedType, FileKind
from .detection import MAX_STORED_NAME_LENGTH, detect_type, sanitize_original_name
from .ooxml import (
    DEFAULT_OOXML_LIMITS,
    OoxmlFailureKind,
    OoxmlKind,
    OoxmlLimits,
    OoxmlValidationError,
    ValidatedOoxmlSource,
    probe_ooxml_kind,
    read_validated_ooxml,
    validate_ooxml,
)
from .snapshot import (
    MAX_DOCUMENT_FILE_BYTES,
    SourceChangedDuringReadError,
    SourceFingerprint,
    SourceSizeLimitError,
    fingerprint_file,
)

__all__ = [
    "DEFAULT_OOXML_LIMITS",
    "DetectedType",
    "FileKind",
    "MAX_STORED_NAME_LENGTH",
    "MAX_DOCUMENT_FILE_BYTES",
    "OoxmlFailureKind",
    "OoxmlKind",
    "OoxmlLimits",
    "OoxmlValidationError",
    "SourceChangedDuringReadError",
    "SourceFingerprint",
    "SourceSizeLimitError",
    "ValidatedOoxmlSource",
    "detect_type",
    "fingerprint_file",
    "probe_ooxml_kind",
    "read_validated_ooxml",
    "sanitize_original_name",
    "validate_ooxml",
]
