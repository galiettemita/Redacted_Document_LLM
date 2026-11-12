from __future__ import annotations

from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent


def generate_visible_layout() -> Path:
    output = HERE / "visible-layout.pdf"
    document = pymupdf.open()
    for page_number in range(2):
        page = document.new_page(width=612, height=792)
        page.insert_text((205, 35), "VISIBLE BULLETIN HEADER", fontsize=11)
        page.insert_text((72, 120), f"LEFT COLUMN PAGE {page_number + 1}", fontsize=12)
        page.insert_text(
            (340, 120), f"RIGHT COLUMN PAGE {page_number + 1}", fontsize=12
        )
        page.insert_text((72, 180), "before SECRET after", fontsize=12)
        page.draw_rect(
            pymupdf.Rect(108, 166, 160, 186), color=(0, 0, 0), fill=(0, 0, 0)
        )
        page.insert_text((72, 250), "visible words", fontsize=12)
        page.insert_text((72, 320), "white before", fontsize=12)
        page.insert_text((240, 320), "after", fontsize=12)
    document.save(output)
    document.close()
    return output


def generate_scanned_layout(source: Path) -> Path:
    output = HERE / "scanned-visible-layout.pdf"
    source_document = pymupdf.open(source)
    pixmap = source_document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    image_bytes = pixmap.tobytes("jpeg", jpg_quality=85)
    source_document.close()

    scanned = pymupdf.open()
    page = scanned.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image_bytes)
    scanned.save(output)
    scanned.close()
    return output


if __name__ == "__main__":
    visible = generate_visible_layout()
    generate_scanned_layout(visible)
