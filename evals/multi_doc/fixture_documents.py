"""Deterministic PDF and raster-table fixtures for the synthetic evaluation."""

from dataclasses import dataclass
from html import escape
from itertools import groupby
import os
from pathlib import Path
import textwrap


@dataclass(frozen=True)
class TableImage:
    name: str
    title: str
    subtitle: str
    rows: tuple[tuple[str, str, str, str, str], ...]


def write_pdf(path: Path, content: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen.canvas import Canvas

    pages = content.split("\f")
    chinese = any("\u3400" <= character <= "\u9fff" for character in content)
    if chinese:
        pdfmetrics.registerFont(TTFont("FixtureCJK", str(chinese_font_path()), subfontIndex=0))
    canvas = Canvas(str(path), pagesize=A4, invariant=1, pageCompression=1)
    canvas.setTitle(path.stem)
    canvas.setAuthor("Entelecheia synthetic evaluation")
    width, height = A4
    for number, page in enumerate(pages, 1):
        heading, _, body = page.partition("\n")
        canvas.setFillColorRGB(0.12, 0.24, 0.36)
        canvas.setFont("FixtureCJK" if chinese else "Helvetica-Bold", 14)
        canvas.drawString(48, height - 54, heading)
        canvas.setStrokeColorRGB(0.7, 0.77, 0.82)
        canvas.line(48, height - 66, width - 48, height - 66)
        canvas.setFillColorRGB(0.1, 0.1, 0.1)
        if chinese:
            bottom = _draw_chinese_body(canvas, body, width - 96, height - 90)
        else:
            # Preserve the original English fixtures' layout and byte identity.
            text = canvas.beginText(48, height - 90)
            text.setFont("Helvetica", 10)
            text.setLeading(14)
            for paragraph in body.split("\n"):
                for line in textwrap.wrap(paragraph, width=91, break_long_words=False):
                    text.textLine(line)
                text.textLine("")
            bottom = text.getY()
            canvas.drawText(text)
        if bottom < 54:
            raise ValueError(f"fixture page overflow: {path.name}, page {number}")
        canvas.setFont("Helvetica", 9)
        canvas.drawString(48, 30, "Synthetic records | physical page numbers")
        canvas.drawRightString(width - 48, 30, f"{number} / {len(pages)}")
        canvas.showPage()
    canvas.save()


def chinese_font_path() -> Path:
    path = Path(os.environ.get("ENTELECHEIA_EVAL_CJK_FONT", "/System/Library/Fonts/STHeiti Light.ttc"))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("set ENTELECHEIA_EVAL_CJK_FONT to an absolute CJK TrueType font path")
    return path


def _draw_chinese_body(canvas, body: str, width: float, top: float) -> float:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    style = ParagraphStyle("fixture_body", fontName="FixtureCJK", fontSize=10.5,
                           leading=17, wordWrap="CJK", textColor=colors.HexColor("#192633"))
    heading = ParagraphStyle("fixture_heading", parent=style, fontSize=11.5,
                             textColor=colors.HexColor("#1f4967"))
    for is_table, lines in groupby(body.splitlines(), key=lambda line: line.startswith("|")):
        if is_table:
            rows = [[Paragraph(escape(cell.strip()), style) for cell in line.strip("|").split("|")]
                    for line in lines]
            table = Table(rows, colWidths=[width / len(rows[0])] * len(rows[0]))
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8c9d6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            blocks = [table]
        else:
            blocks = [Paragraph(escape(line), heading if len(line) > 1 and line[1] == "、" else style)
                      for line in lines if line]
        for block in blocks:
            _, height = block.wrap(width, 1000)
            top -= height
            block.drawOn(canvas, 48, top)
            top -= 10
    return top


def write_table_image(path: Path, spec: TableImage) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1500, 1000), "#f6f8fb")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=40)
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=26)
    draw.rectangle((0, 0, 1500, 165), fill="#18354e")
    draw.text((55, 32), spec.title, fill="white", font=title)
    draw.text((55, 100), spec.subtitle, fill="white", font=small)
    columns = (65, 330, 640, 900, 1210)
    for x, label in zip(columns, ("SKU", "LOT", "ON HAND", "RESERVED", "STATE")):
        draw.text((x, 205), label, fill="#18354e", font=small)
    palette = {"released": "#238636", "quarantine": "#e69b20", "blocked": "#c63131"}
    for index, row in enumerate(spec.rows):
        y = 275 + index * 112
        draw.rectangle((45, y - 12, 1455, y + 85), fill="white")
        for x, value in zip(columns[:4], row[:4]):
            draw.text((x, y + 15), value, fill="#152634", font=font)
        draw.ellipse((1230, y + 10, 1285, y + 65), fill=palette[row[4]])
    for index, (label, color) in enumerate(palette.items()):
        x = 65 + index * 475
        draw.ellipse((x, 890, x + 40, 930), fill=color)
        draw.text((x + 55, 892), label.upper(), fill="#152634", font=small)
    image.save(path)
