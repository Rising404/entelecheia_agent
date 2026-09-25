"""输入文件的类型检测与不可信名称处理。

客户端发送的文件名和 ``Content-Type`` 均可被攻击者控制。本模块不会信任二者来做安全
决策：存储的媒体类型来自文件字节，存储路径也绝不包含所提供名称中任何可逃逸目录的部分。
"""

from __future__ import annotations

import re
import unicodedata

from .contracts import DetectedType, FileKind
from .ooxml import OoxmlKind, probe_ooxml_kind


MAX_STORED_NAME_LENGTH = 120
_UNSAFE_NAME_CHARS = re.compile(r"[^\w.\-() 一-鿿]", re.UNICODE)
_COLLAPSE_DOTS = re.compile(r"\.{2,}")


# 按签名长度从长到短检查，避免容器格式因恰好共享较短前缀而被误判。
_MAGIC_SIGNATURES: tuple[tuple[bytes, str, FileKind, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png", FileKind.IMAGE, ".png"),
    (b"\xff\xd8\xff", "image/jpeg", FileKind.IMAGE, ".jpg"),
    (b"GIF87a", "image/gif", FileKind.IMAGE, ".gif"),
    (b"GIF89a", "image/gif", FileKind.IMAGE, ".gif"),
    (b"BM", "image/bmp", FileKind.IMAGE, ".bmp"),
    (b"%PDF-", "application/pdf", FileKind.DOCUMENT, ".pdf"),
    (b"ID3", "audio/mpeg", FileKind.AUDIO, ".mp3"),
    (b"OggS", "audio/ogg", FileKind.AUDIO, ".ogg"),
    (b"fLaC", "audio/flac", FileKind.AUDIO, ".flac"),
    (b"\x1a\x45\xdf\xa3", "video/x-matroska", FileKind.VIDEO, ".mkv"),
)

_OLE_COMPOUND_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_OOXML_TYPES: dict[OoxmlKind, tuple[str, FileKind, str]] = {
    OoxmlKind.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        FileKind.DOCUMENT,
        ".docx",
    ),
    OoxmlKind.PPTX: (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        FileKind.DOCUMENT,
        ".pptx",
    ),
    OoxmlKind.XLSX: (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        FileKind.DOCUMENT,
        ".xlsx",
    ),
}

_TYPE_HEAD_BYTES = 8192
_ZIP_EOCD_WINDOW_BYTES = 65_535 + 22

_TEXT_EXTENSIONS: dict[str, str] = {
    ".txt": "text/plain", ".md": "text/markdown", ".markdown": "text/markdown",
    ".csv": "text/csv", ".tsv": "text/tab-separated-values", ".json": "application/json",
    ".log": "text/plain", ".yaml": "application/yaml", ".yml": "application/yaml",
    ".py": "text/x-python", ".js": "text/javascript", ".ts": "text/typescript",
    ".html": "text/html", ".css": "text/css", ".sql": "application/sql",
    ".sh": "application/x-sh", ".toml": "application/toml", ".xml": "application/xml",
    ".ini": "text/plain", ".rs": "text/x-rust", ".go": "text/x-go", ".java": "text/x-java",
    ".c": "text/x-c", ".h": "text/x-c", ".cpp": "text/x-c++", ".rb": "text/x-ruby",
}


def sanitize_original_name(raw: str) -> str:
    """将不可信文件名化简为可安全存储和显示的形式。

    所有目录组成部分都会被丢弃，而不是转义：附件位置由 Host 分配，名称无须携带路径
    结构；允许它携带路径只会制造目录遍历攻击面。
    """

    name = unicodedata.normalize("NFC", raw or "").strip()
    # 无论 Host OS 是什么，两种分隔符都会被移除：Linux 上收到的 Windows 风格名称仍是在
    # 尝试寻址目录。
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(char for char in name if unicodedata.category(char)[0] != "C")
    name = _UNSAFE_NAME_CHARS.sub("_", name).strip(" .")
    name = _COLLAPSE_DOTS.sub(".", name)
    if not name:
        return "file"
    if len(name) > MAX_STORED_NAME_LENGTH:
        stem, dot, suffix = name.rpartition(".")
        if dot and 0 < len(suffix) <= 12:
            keep = MAX_STORED_NAME_LENGTH - len(suffix) - 1
            name = f"{stem[:keep]}.{suffix}"
        else:
            name = name[:MAX_STORED_NAME_LENGTH]
    return name or "file"


def detect_type(payload: bytes, sanitized_name: str) -> DetectedType:
    """对 payload 分类，优先采用字节证据而不是名称。

    只有对没有区分性文件头的类型族——纯文本和源代码——才参考名称；此时扩展名是唯一可用
    信号，且不能借此宣称比文本更丰富的能力。
    """

    head = payload[:_TYPE_HEAD_BYTES]
    for signature, media_type, kind, extension in _MAGIC_SIGNATURES:
        if head.startswith(signature):
            return DetectedType(media_type, kind, extension)

    if head.startswith(_OLE_COMPOUND_SIGNATURE):
        suffix = _suffix_of(sanitized_name)
        if suffix == ".doc":
            return DetectedType(
                "application/msword", FileKind.DOCUMENT, suffix
            )
        if suffix == ".ppt":
            return DetectedType(
                "application/vnd.ms-powerpoint", FileKind.DOCUMENT, suffix
            )
        # 共享容器无法证明它属于哪个旧 Office 应用。将其保留为未知类型，不信任其他扩展名，
        # 也不尝试转换活动内容。
        return DetectedType(
            "application/x-ole-storage", FileKind.UNKNOWN, suffix or ".ole"
        )

    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return DetectedType("image/webp", FileKind.IMAGE, ".webp")
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return DetectedType("audio/wav", FileKind.AUDIO, ".wav")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in {b"avif", b"heic", b"heix", b"mif1"}:
            return DetectedType("image/avif", FileKind.IMAGE, ".avif")
        return DetectedType("video/mp4", FileKind.VIDEO, ".mp4")
    has_zip_directory = b"PK\x05\x06" in payload[-_ZIP_EOCD_WINDOW_BYTES:]
    if has_zip_directory:
        ooxml_kind = probe_ooxml_kind(payload)
        if ooxml_kind is not None:
            media_type, kind, extension = _OOXML_TYPES[ooxml_kind]
            return DetectedType(media_type, kind, extension)
    if has_zip_directory or head.startswith(b"PK\x03\x04"):
        # 通用归档文件被有意保持为未知：当前实现从不解包，因此宣称可读类型会承诺并不存在的能力。
        return DetectedType("application/zip", FileKind.UNKNOWN, ".zip")

    suffix = _suffix_of(sanitized_name)
    if suffix in _TEXT_EXTENSIONS and _looks_like_text(head):
        return DetectedType(_TEXT_EXTENSIONS[suffix], FileKind.TEXT, suffix)
    if _looks_like_text(head):
        # 已被内容证伪的后缀不能进入 reader 路由。存储路径以此扩展名为键，因此在此保留
        # `.pdf`、`.doc` 或 `.png` 会把已确认为文本的内容送入错误解析器。
        return DetectedType("text/plain", FileKind.TEXT, ".txt")
    return DetectedType("application/octet-stream", FileKind.UNKNOWN, suffix)


def _suffix_of(name: str) -> str:
    _, dot, suffix = name.rpartition(".")
    return f".{suffix.lower()}" if dot and suffix else ""


    # 制表符、换行符和回车符是真实文本会使用的仅有控制字节。
_ALLOWED_CONTROL_BYTES = frozenset(b"\t\n\r\f")
_MAX_CONTROL_BYTE_RATIO = 0.05


def _looks_like_text(head: bytes) -> bool:
    """判断 payload 能否视为可解码文本。

    仅能解码并不足够：ELF 文件头等短二进制内容可能恰好是有效 UTF-8，因此即使能解码，
    控制字节密集的 payload 也会被拒绝。把二进制当作文本准入，会向模型交付无法与真实内容
    区分的乱码。
    """

    if not head:
        return True
    if b"\x00" in head:
        return False
    control = sum(
        1 for byte in head if byte < 0x20 and byte not in _ALLOWED_CONTROL_BYTES
    )
    if control > max(1, int(len(head) * _MAX_CONTROL_BYTE_RATIO)):
        return False
    for encoding in ("utf-8", "utf-16", "gb18030"):
        try:
            head.decode(encoding)
        except UnicodeDecodeError:
            continue
        else:
            return True
        # 读取边界处被截断的多字节字符不属于数据损坏。
    try:
        head[:-4].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


__all__ = ["MAX_STORED_NAME_LENGTH", "detect_type", "sanitize_original_name"]
