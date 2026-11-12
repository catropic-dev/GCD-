"""Place Chinese reading notes beside the corresponding original PDF pages."""

import argparse
from html import escape
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


FONT = "STSong-Light"
PANEL_WIDTH = 350
MARGIN = 26


def read_notes(path, page_count):
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict) or not source:
        raise ValueError("Provide a JSON object keyed by one-based PDF page number")
    notes = {}
    for key, entries in source.items():
        if not key.isascii() or not key.isdecimal() or not 1 <= int(key) <= page_count:
            raise ValueError(f"Unknown PDF page: {key}")
        number = int(key)
        if number in notes or not isinstance(entries, list) or not entries:
            raise ValueError(f"Page {number} must have one nonempty list of notes")
        for entry in entries:
            if not isinstance(entry, dict) or any(
                not isinstance(entry.get(field), str) or not entry[field].strip()
                for field in ("anchor", "explanation")
            ):
                raise ValueError(f"Every note on page {number} needs an anchor and explanation")
        notes[number] = entries
    return notes


def draw_paragraph(canvas, text, y, size, leading, color):
    style = ParagraphStyle(
        "note", fontName=FONT, fontSize=size, leading=leading,
        textColor=HexColor(color), wordWrap="CJK",
    )
    paragraph = Paragraph(escape(text).replace("\n", "<br/>"), style)
    _, height = paragraph.wrap(PANEL_WIDTH - 2 * MARGIN, y)
    if y - height < MARGIN:
        raise ValueError("Notes exceed the sidebar. Condense the explanations on this page.")
    paragraph.drawOn(canvas, MARGIN, y - height)
    return y - height


def panel_pdf(number, entries, height):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(PANEL_WIDTH, height))
    canvas.setFillColor(HexColor("#F3F7FC"))
    canvas.rect(0, 0, PANEL_WIDTH, height, stroke=0, fill=1)
    canvas.setFillColor(HexColor("#167C74"))
    canvas.rect(0, 0, 5, height, stroke=0, fill=1)
    y = draw_paragraph(canvas, f"中文详解 · 原文第 {number} 页", height - MARGIN, 19, 26, "#10233B") - 18
    for entry in entries:
        y = draw_paragraph(canvas, entry["anchor"], y, 14, 20, "#12665E") - 6
        y = draw_paragraph(canvas, entry["explanation"], y, 12, 19, "#1C3048") - 16
    canvas.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def explain(source, notes_path, output):
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output file for the explained PDF")
    reader = PdfReader(source)
    notes = read_notes(notes_path, len(reader.pages))
    writer = PdfWriter(clone_from=reader)
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    for number, entries in notes.items():
        page = writer.pages[number - 1]
        if page.rotation:
            page.transfer_rotation_to_content()
        crop = page.cropbox
        left, bottom = map(float, crop.lower_left)
        right, top = map(float, crop.upper_right)
        height = max(top - bottom, 640)
        try:
            panel = panel_pdf(number, entries, height)
        except ValueError as error:
            raise ValueError(f"Page {number}: {error}") from error
        page.merge_transformed_page(panel, Transformation().translate(right, top - height))
        bounds = (left, top - height, right + PANEL_WIDTH, top)
        for box in (page.mediabox, page.cropbox):
            box.lower_left = bounds[:2]
            box.upper_right = bounds[2:]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("notes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        explain(args.paper, args.notes, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"PDF generation failed: {error}\n")


if __name__ == "__main__":
    main()
