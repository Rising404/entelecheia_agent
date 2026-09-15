"""Build an original, fictional three-page document without starting the agent."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def validate_output(raw_path: str) -> Path:
    """Require an exact, new PDF file outside this source checkout."""
    requested = Path(raw_path)
    if not requested.is_absolute():
        raise ValueError("--output must be an absolute file path.")
    if requested.suffix.lower() != ".pdf":
        raise ValueError("--output must name a .pdf file, not a directory.")
    if requested.is_symlink():
        raise ValueError("--output must not be a symbolic link.")

    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("The output parent must be an existing directory.")
    output = parent / requested.name
    checkout = Path(__file__).resolve().parents[2]
    if output.is_relative_to(checkout):
        raise ValueError("--output must be outside the source checkout.")
    if output.exists():
        raise FileExistsError("The output already exists; overwriting is not allowed.")
    return output


def build_pdf() -> bytes:
    """Return deterministic PDF bytes for the same source and ReportLab version."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="Harbor Learning Room Handbook",
        author="Entelecheia synthetic examples",
        subject="Original fictional document for a local document-QA walkthrough",
        creator="Entelecheia document QA example generator",
        invariant=1,
        pageCompression=1,
    )
    navy = colors.HexColor("#17324D")
    body = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=11, leading=17, spaceAfter=12
    )
    title = ParagraphStyle(
        "Title", parent=body, fontName="Helvetica-Bold", fontSize=25,
        leading=30, textColor=navy, spaceAfter=20,
    )
    heading = ParagraphStyle(
        "Heading", parent=body, fontName="Helvetica-Bold", fontSize=15,
        leading=20, textColor=navy, spaceBefore=12,
    )

    def paragraph(text: str) -> Paragraph:
        return Paragraph(text, body)

    def footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(54, 42, A4[0] - 54, 42)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(54, 29, "Fictional example - no real organization or users")
        canvas.drawRightString(A4[0] - 54, 29, f"{doc.page} / 3")
        canvas.restoreState()

    returns = Table(
        [
            ["Day", "Return deadline", "Return location"],
            ["Monday to Thursday", "16:00", "Maple Desk"],
            ["Friday", "16:30", "Maple Desk"],
        ],
        colWidths=[185, 120, document.width - 305],
        hAlign="LEFT",
    )
    returns.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F9")),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))

    story = [
        Paragraph("Harbor Learning Room<br/>Handbook", title),
        paragraph(
            "This is an original fictional handbook for a made-up community learning "
            "room. All names, schedules and quantities were invented for this example. "
            "It contains no real user records or external dataset content."
        ),
        Paragraph("1. Visiting and returning materials", heading),
        paragraph(
            "The learning room has a reading corner, a materials cabinet and a small "
            "workshop table. During weekday opening hours, visitors may borrow "
            "drawing boards for use inside the room. The cabinet holds the boards; "
            "the Maple Desk handles their return."
        ),
        paragraph(
            "Borrowed materials must be returned by the deadline for the relevant "
            "day. The return schedule below applies to drawing boards and other "
            "reusable room materials."
        ),
        returns,
        Spacer(1, 18),
        paragraph(
            "Weekday lending and the Saturday workshop are separate activities. "
            "The weekday return deadlines do not describe workshop starting times. "
            "Pages 2 and 3 describe workshop supplies and the participant roster."
        ),
        PageBreak(),
        Paragraph("Workshop materials", title),
        Paragraph("2. Packing rules", heading),
        paragraph(
            "For each workshop session, prepare two colored cards for every registered "
            "participant. The cards are used for a short design activity. Each "
            "participant receives their own pair; cards are not shared between pairs "
            "of participants."
        ),
        paragraph(
            "In addition to the participant cards, prepare six spare colored cards "
            "for the session as a whole. This is one shared reserve, not six extra "
            "cards per participant. Do not include any additional card allowance."
        ),
        Paragraph("Reusable equipment", heading),
        paragraph(
            "Place one reusable pencil at each participant seat. Pencils are collected "
            "after the activity and stored separately from the colored cards. A tray "
            "of demonstration shapes stays at the workshop table for everyone to see."
        ),
        Paragraph("Who receives a pack", heading),
        paragraph(
            "Only registered participants receive a card pack. Facilitators use the "
            "shared demonstration shapes and receive no colored cards. Use the "
            "participant count in the session roster on page 3 when preparing supplies."
        ),
        paragraph(
            "Unused spare cards go back into the materials cabinet after the session. "
            "The next session is packed independently from its own roster."
        ),
        PageBreak(),
        Paragraph("Saturday workshop", title),
        Paragraph("3. Session roster and setup", heading),
        paragraph(
            "The Saturday design workshop consists of one session, from 10:00 to "
            "11:00. Its confirmed roster contains 18 registered participants. There "
            "are also two facilitators; they are not included in the participant count."
        ),
        paragraph(
            "No additional participants are expected for this session. Prepare the "
            "colored cards using the packing rules on page 2 and the confirmed "
            "participant count above. The two facilitators arrange the room and "
            "demonstrate the activity before participants begin."
        ),
        Paragraph("Room arrangement", heading),
        paragraph(
            "Set the participant seats around three tables. Leave the reading corner "
            "available for quiet use. Keep the shared demonstration tray on the "
            "workshop table and place the reserve cards in a clearly marked envelope."
        ),
        Paragraph("After the session", heading),
        paragraph(
            "Collect the reusable pencils and demonstration shapes. Participants may "
            "take their completed card designs with them. Return unused reserve cards "
            "to the cabinet and clear the tables for the next room activity."
        ),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fictional three-page PDF offline, outside this checkout."
    )
    parser.add_argument(
        "--output", required=True, metavar="ABSOLUTE_PATH.pdf",
        help="New PDF file in an existing directory outside the source checkout.",
    )
    args = parser.parse_args()
    try:
        output = validate_output(args.output)
        pdf_bytes = build_pdf()
        with output.open("xb") as stream:
            stream.write(pdf_bytes)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot create sample: {exc}\n")
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
