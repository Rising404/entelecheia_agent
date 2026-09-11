"""在进程内构建 PDF 夹具，因此测试套件无需二进制测试资源。"""

from __future__ import annotations

import io
import zlib

import pytest


def _pdf(objects: list[bytes]) -> bytes:
    out = io.BytesIO()
    out.write(b"%PDF-1.7\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def _text_stream(lines: list[str], *, draw_image: bool = False) -> bytes:
    parts: list[str] = []
    if draw_image:
        # 必须真正绘制图片：解析器读取内容流，而不只是资源字典。
        parts += ["q", "200 0 0 200 100 400 cm", "/Im0 Do", "Q"]
    parts += ["BT", "/F1 12 Tf"]
    y = 700
    for line in lines:
        parts.append(f"1 0 0 1 72 {y} Tm ({line}) Tj")
        y -= 20
    parts.append("ET")
    content = "\n".join(parts).encode("latin-1")
    return b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"


def _image_stream() -> bytes:
    raw = bytes([200, 100, 50] * 16)
    data = zlib.compress(raw)
    return (
        b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Filter /FlateDecode /Length " + str(len(data)).encode()
        + b" >>\nstream\n" + data + b"\nendstream"
    )


def _build(pages: list[dict]) -> bytes:
    objects: list[bytes] = [b"", b"", b""]
    kids, page_objs, extra = [], [], []
    next_id = 4
    for page in pages:
        content_id = next_id
        extra.append(page["content"])
        next_id += 1
        resources = "/Font << /F1 3 0 R >>"
        if page.get("image"):
            extra.append(_image_stream())
            resources += f" /XObject << /Im0 {next_id} 0 R >>"
            next_id += 1
        page_id = next_id
        next_id += 1
        kids.append(f"{page_id} 0 R")
        page_objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << {resources} >> /Contents {content_id} 0 R >>".encode()
        )
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    # 按 ID 顺序交错排列内容/图片对象与对应页面对象。
    body: list[bytes] = []
    cursor = 0
    for page in pages:
        body.append(extra[cursor])
        cursor += 1
        if page.get("image"):
            body.append(extra[cursor])
            cursor += 1
        body.append(page_objs[pages.index(page)])
    return _pdf(objects + body)


@pytest.fixture
def text_pdf(tmp_path):
    path = tmp_path / "text.pdf"
    path.write_bytes(_build([
        {"content": _text_stream(["First line of the report", "Second line with detail"])},
    ]))
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    """内容只有一张图片的页面：典型的静默失败案例。"""
    path = tmp_path / "scan.pdf"
    path.write_bytes(_build([{"content": _text_stream([], draw_image=True), "image": True}]))
    return path


@pytest.fixture
def mixed_pdf(tmp_path):
    path = tmp_path / "mixed.pdf"
    path.write_bytes(_build([
        {"content": _text_stream(["Readable heading text here", "And a second line"])},
        {"content": _text_stream([], draw_image=True), "image": True},
    ]))
    return path


@pytest.fixture
def empty_pdf(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(_build([{"content": _text_stream([])}]))
    return path


@pytest.fixture
def corrupt_pdf(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not a real pdf body\n%%EOF\n")
    return path


@pytest.fixture
def styled_pdf(tmp_path):
    """包含大字号标题与自动换行正文的页面。

    标题没有标记，只有更大字号；这正是几何读取器无法恢复、而布局分析能够
    恢复的结构。
    """
    def styled(items):
        parts = ["BT"]
        y = 700
        for text, size, gap in items:
            parts += [f"/F1 {size} Tf", f"1 0 0 1 72 {y} Tm ({text}) Tj"]
            y -= gap
        parts.append("ET")
        content = "\n".join(parts).encode("latin-1")
        return b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"

    path = tmp_path / "styled.pdf"
    path.write_bytes(_build([{"content": styled([
        ("Retrieval Design Notes", 20, 34),
        ("The system must bind every answer to a source", 11, 14),
        ("version so a stale chunk is never presented as", 11, 14),
        ("current fact under any circumstances.", 11, 30),
    ])}]))
    return path
