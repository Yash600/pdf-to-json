"""
PDF Parser — PyMuPDF based
Extracts:
  - Embedded raster images with exact bounding boxes (digital PDFs)
  - Page renders (PNG bytes) for Groq Vision — ALL pages, always
  - is_scanned flag for downstream use

Both digital and scanned PDFs render every page for Groq Vision.
Digital PDFs additionally extract embedded images at native resolution
(better quality than cropping from a rendered PNG).
"""

import fitz  # PyMuPDF
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int
    block_no: int


@dataclass
class ImageBlock:
    xref: int               # PyMuPDF internal ref
    page_num: int
    x0: float               # position on page (points)
    y0: float
    x1: float
    y1: float
    width: int              # native image size
    height: int
    image_bytes: bytes      # raw PNG/JPEG bytes
    ext: str                # png / jpeg


@dataclass
class ParsedPage:
    page_num: int           # 0-indexed
    width: float
    height: float
    text_blocks: list[TextBlock] = field(default_factory=list)
    image_blocks: list[ImageBlock] = field(default_factory=list)
    has_images: bool = False
    render_bytes: Optional[bytes] = None   # always filled — every page rendered


@dataclass
class ParsedPDF:
    pdf_hash: str
    page_count: int
    pages: list[ParsedPage] = field(default_factory=list)
    is_scanned: bool = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_scanned(pages: list[ParsedPage], threshold: float = 0.1) -> bool:
    """True if fewer than 10% of pages have meaningful embedded text."""
    text_pages = sum(
        1 for p in pages
        if any(len(b.text.strip()) > 10 for b in p.text_blocks)
    )
    return (text_pages / max(len(pages), 1)) < threshold


# ── main parser ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str, render_dpi: int = 150) -> ParsedPDF:
    """
    Parse a PDF and return a ParsedPDF object.

    Every page is rendered as a PNG for Groq Vision regardless of whether
    the PDF is digital or scanned — Groq handles question/option/answer
    extraction from the page image.

    For digital PDFs, embedded raster images are also extracted at native
    resolution (better quality than cropping from a rendered PNG).
    """
    pdf_hash = _sha256(pdf_path)
    doc = fitz.open(pdf_path)
    pages: list[ParsedPage] = []
    mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pw, ph = page.rect.width, page.rect.height

        # ── 1. Extract embedded text (used only for is_scanned detection) ─────
        raw_blocks = page.get_text("blocks")
        text_blocks: list[TextBlock] = []
        for b in raw_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0:
                continue
            clean = text.strip()
            if not clean:
                continue
            text_blocks.append(TextBlock(
                text=clean,
                x0=x0, y0=y0, x1=x1, y1=y1,
                page_num=page_idx,
                block_no=block_no,
            ))

        # ── 2. Extract embedded raster images (digital PDFs) ──────────────────
        # Kept for image quality — native resolution > cropping from render.
        image_blocks: list[ImageBlock] = []
        seen_xrefs: set[int] = set()
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            img_rects = page.get_image_rects(xref)
            if not img_rects:
                continue
            rect = img_rects[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            area_pts = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
            if area_pts < 1600:   # skip tiny decorative images
                continue
            image_blocks.append(ImageBlock(
                xref=xref,
                page_num=page_idx,
                x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1,
                width=base_image["width"],
                height=base_image["height"],
                image_bytes=base_image["image"],
                ext=base_image["ext"],
            ))

        # ── 3. Detect vector drawings ─────────────────────────────────────────
        has_vector_diagrams = False
        try:
            drawings = page.get_drawings()
            complex_paths = sum(
                1 for d in drawings
                if d.get("type") in ("f", "fs", "s")
                and len(d.get("items", [])) > 2
            )
            has_vector_diagrams = complex_paths > 15
        except Exception:
            pass

        has_images = len(image_blocks) > 0 or has_vector_diagrams

        # ── 4. Render every page as PNG for Groq Vision ───────────────────────
        pix = page.get_pixmap(matrix=mat, alpha=False)
        render_bytes = pix.tobytes("png")
        pix = None  # release pixmap memory immediately

        pages.append(ParsedPage(
            page_num=page_idx,
            width=pw,
            height=ph,
            text_blocks=text_blocks,
            image_blocks=image_blocks,
            has_images=has_images,
            render_bytes=render_bytes,
        ))

    doc.close()

    is_scanned = _is_scanned(pages)
    logger.info(
        f"Parsed {len(pages)} pages | scanned={is_scanned} | "
        f"pages_with_embedded_images={sum(1 for p in pages if p.image_blocks)}"
    )

    return ParsedPDF(
        pdf_hash=pdf_hash,
        page_count=len(pages),
        pages=pages,
        is_scanned=is_scanned,
    )


def get_page_full_text(parsed_pdf: ParsedPDF, page_num: int) -> str:
    page = parsed_pdf.pages[page_num]
    return "\n".join(b.text for b in page.text_blocks)


def get_all_text(parsed_pdf: ParsedPDF) -> str:
    parts = []
    for page in parsed_pdf.pages:
        for block in page.text_blocks:
            parts.append(block.text)
    return "\n".join(parts)
