#!/usr/bin/env python3
"""Render 大作业报告.md to PDF with a CJK font.

Usage:
    python tools/render_report_pdf.py
    python tools/render_report_pdf.py --md 大作业报告.md --pdf 大作业报告.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown
from fpdf import FPDF

FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese course report to PDF.")
    parser.add_argument("--md", default="大作业报告.md")
    parser.add_argument("--pdf", default="大作业报告.pdf")
    return parser.parse_args()


def find_font() -> Path:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    raise SystemExit("No CJK font found. Install fonts-wqy-microhei or pass a TTF/TTC.")


def size_image(match: re.Match) -> str:
    """Give fpdf2 an explicit height so the next paragraph is not drawn on the figure."""
    from PIL import Image

    src = match.group(2)
    width_mm = 150.0
    try:
        with Image.open(src) as image:
            width_px, height_px = image.size
        height_mm = width_mm * height_px / width_px
    except OSError:
        height_mm = 90.0
    return f'<img src="{src}" width="{width_mm:.1f}" height="{height_mm:.1f}"><br>'


def prepare_markdown(text: str, root: Path) -> str:
    def replace_image(match: re.Match) -> str:
        alt = match.group(1)
        raw = match.group(2).strip()
        path = Path(raw)
        if not path.is_absolute():
            path = (root / path).resolve()
        if not path.exists():
            return f"[{alt}: missing {raw}]"
        return f'![{alt}]({path.as_posix()})'

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, text)


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    source = Path(args.md)
    if not source.is_absolute():
        source = root / source
    text = prepare_markdown(source.read_text(encoding="utf-8"), source.parent)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # fpdf2 draws code in Courier, which cannot encode CJK or arrows.
    body = re.sub(r"</?code[^>]*>", "", body)
    body = re.sub(r"</?pre[^>]*>", "", body)
    body = re.sub(r"<img\b([^>]*?)src=\"([^\"]+)\"([^>]*)/?>", size_image, body)
    html = f"""
    <style>
    h1 {{ font-size: 18pt; text-align: center; }}
    h2 {{ font-size: 14pt; }}
    h3 {{ font-size: 12pt; }}
    p, li {{ font-size: 11pt; line-height: 1.45; }}
    table {{ width: 100%; border: 1px solid #444; }}
    th, td {{ border: 1px solid #444; padding: 2px; font-size: 9pt; }}
    code, pre {{ font-size: 8pt; }}
    </style>
    {body}
    """
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    font_path = str(find_font())
    for style in ("", "B", "I", "BI"):
        pdf.add_font("hei", style=style, fname=font_path)
    pdf.set_font("hei", size=11)
    pdf.add_page()
    pdf.write_html(html)
    output = Path(args.pdf)
    if not output.is_absolute():
        output = root / output
    pdf.output(str(output))
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
