"""纯文本、Markdown、CSV 和源代码文件 reader。

文本没有页面或几何信息，因此 locator 携带字符范围；对于 Markdown，还携带内容块所在的
heading path。此处无须知道其他格式使用页面，这正是共享契约的意义。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..contracts import (
    DiagnosticCode,
    DocumentElement,
    DocumentLocator,
    ElementKind,
    ProcessingDiagnostic,
    ProcessingResult,
    ProcessorFingerprint,
    make_element_id,
)


READER_NAME = "plain_text"
READER_VERSION = "4"

MAX_ELEMENT_CHARS = 20000
MAX_ELEMENTS = 20000
MAX_TEXT_BYTES = 16 * 1024 * 1024

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def read_plain_text(path: Path) -> ProcessingResult:
    """将文本拆分为块，并在存在时跟踪 Markdown heading path。"""

    processor = ProcessorFingerprint(READER_NAME, READER_VERSION)
    try:
        with path.open("rb") as source:
            payload = source.read(MAX_TEXT_BYTES + 1)
    except PermissionError as exc:
        return _fatal_result(processor, DiagnosticCode.PERMISSION_DENIED, exc)
    except OSError as exc:
        return _fatal_result(processor, DiagnosticCode.CORRUPT_SOURCE, exc)
    if len(payload) > MAX_TEXT_BYTES:
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                DiagnosticCode.LIMIT_REACHED,
                detail="plain-text source byte limit reached",
            ),),
        )
    try:
    # UTF-8 是当前实现能够证明的唯一无损文本交换契约。未知旧编码需要未来显式转换步骤；
    # 猜测 GB18030 会把任意字节变成所谓“完整”文本。
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return _fatal_result(
            processor,
            DiagnosticCode.CORRUPT_SOURCE,
            exc,
            detail="plain-text source is not valid UTF-8",
        )
    if not text.strip():
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(DiagnosticCode.EMPTY_SOURCE),),
        )

    source_key = hashlib.sha256(payload).hexdigest()
    elements: list[DocumentElement] = []
    diagnostics: list[ProcessingDiagnostic] = []
    section: list[str] = []
    ordinal = 0
    element_budget_exhausted = False

    for block, start, _end in _blocks(text):
        stripped = block.strip()
        content_start = start
        heading = _ATX_HEADING.match(stripped) if "\n" not in stripped else None
        if heading is not None:
            depth = len(heading.group(1))
            title = heading.group(2)
            section = [*section[: depth - 1], title]
            kind = ElementKind.HEADING
            content = title
            content_start += len(block) - len(block.lstrip()) + heading.start(2)
        else:
            kind = ElementKind.PARAGRAPH
        # 保留精确源切片。前导缩进可能改变 Python、Markdown、YAML 和列表语义；裁剪它会把
        # 机械完整读取变成被修改的内容。
            content = block
        if not content.strip():
            continue
        for segment_offset in range(0, len(content), MAX_ELEMENT_CHARS):
            segment = content[segment_offset:segment_offset + MAX_ELEMENT_CHARS]
            segment_start = content_start + segment_offset
            locator = DocumentLocator(
                ordinal=ordinal,
                section_path=tuple(section),
                char_range=(segment_start, segment_start + len(segment)),
            )
            if len(elements) >= MAX_ELEMENTS:
                diagnostics.append(ProcessingDiagnostic(
                    DiagnosticCode.LIMIT_REACHED,
                    locator,
                    detail="element budget exhausted",
                ))
                element_budget_exhausted = True
                break
            elements.append(DocumentElement(
                element_id=make_element_id(source_key, locator, segment),
                kind=kind,
                text=segment,
                locator=locator,
            ))
            ordinal += 1
        if element_budget_exhausted:
            break

    return ProcessingResult(
        elements=tuple(elements),
        processor=processor,
        diagnostics=tuple(diagnostics),
    )


def _fatal_result(
    processor: ProcessorFingerprint,
    code: DiagnosticCode,
    error: BaseException,
    *,
    detail: str | None = None,
) -> ProcessingResult:
    return ProcessingResult(
        elements=(),
        processor=processor,
        diagnostics=(ProcessingDiagnostic(
            code,
            detail=detail or type(error).__name__,
        ),),
    )


def _blocks(text: str):
    """生成由空行分隔的块及其字符 span。"""

    start = 0
    for raw in re.split(r"(\n\s*\n)", text):
        if not raw:
            continue
        end = start + len(raw)
        if raw.strip():
            yield raw, start, end
        start = end
