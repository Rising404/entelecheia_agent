"""Deterministic PDF and raster-table fixtures for the synthetic evaluation."""

from dataclasses import dataclass
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
    from reportlab.pdfgen.canvas import Canvas

    pages = content.split("\f")
    canvas = Canvas(str(path), pagesize=A4, invariant=1, pageCompression=1)
    canvas.setTitle(path.stem)
    canvas.setAuthor("Entelecheia synthetic evaluation")
    width, height = A4
    for number, page in enumerate(pages, 1):
        heading, _, body = page.partition("\n")
        canvas.setFillColorRGB(0.12, 0.24, 0.36)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(48, height - 54, heading)
        canvas.setStrokeColorRGB(0.7, 0.77, 0.82)
        canvas.line(48, height - 66, width - 48, height - 66)
        canvas.setFillColorRGB(0.1, 0.1, 0.1)
        text = canvas.beginText(48, height - 90)
        text.setFont("Helvetica", 10)
        text.setLeading(14)
        for paragraph in body.split("\n"):
            for line in textwrap.wrap(paragraph, width=91, break_long_words=False):
                text.textLine(line)
            text.textLine("")
        if text.getY() < 54:
            raise ValueError(f"fixture page overflow: {path.name}, page {number}")
        canvas.drawText(text)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(48, 30, "Synthetic records | physical page numbers")
        canvas.drawRightString(width - 48, 30, f"{number} / {len(pages)}")
        canvas.showPage()
    canvas.save()


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
