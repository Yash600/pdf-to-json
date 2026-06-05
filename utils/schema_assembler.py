"""
Schema Assembler
----------------
Combines:
  - RawQuestion objects (from question_segmenter)
  - PageVisionResult objects (from groq_vision)
  - StoredImage objects (from image_pipeline)

Into the final schema-compliant JSON list.

Image placement logic:
  - loc.option_key is set   → Option.imageDetails
  - loc.is_table_cell       → tables[row][col].imageDetails
  - neither                 → question.imageDetails + hasImage=True
"""

import re
import logging
from typing import Optional

from .question_segmenter import RawQuestion, RawOption, RawTable
from .groq_vision import PageVisionResult, ImageLocation
from .image_pipeline import (
    StoredImage, process_image_location,
    crop_image_from_render, upload_image, points_to_0_1000,
)
from .pdf_parser import ParsedPDF, ImageBlock

logger = logging.getLogger(__name__)

_OPTION_KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}
_RE_OPT_LABEL = re.compile(r'^[\(\[]?([A-D1-4])[\)\.]]\s*$', re.IGNORECASE)


# ─────────────────────── raster-image spatial assignment ─────────────────────

def _question_column(rq, page, page_mid_x: float) -> str:
    """Return 'L' or 'R' for a question based on its first text block x-centre."""
    best_tb, best_dist = None, float('inf')
    for tb in page.text_blocks:
        d = abs(tb.y0 - rq.y0)
        if d < best_dist:
            best_dist = d
            best_tb = tb
    if best_tb is not None and best_dist < 50:
        return "L" if (best_tb.x0 + best_tb.x1) / 2 < page_mid_x else "R"
    return "?"


def _classify_images(imgs, rq, page) -> list[tuple]:
    """
    Given a list of ImageBlock objects belonging to a question, decide
    which are stem images and which are option images.

    Returns list of (ImageBlock, opt_key_or_None) where opt_key=None means stem.

    Rules (applied in priority order):
      1. Count match:
           n == n_opts (>1)     → all options, assign A/B/C/D by y-position
           n == n_opts + 1      → first (topmost) is stem, rest are options
           n == 1               → stem
      2. X-position:
           image x-centre in left margin (< 150pts) → stem (full-width diagram)
           otherwise → option
      3. Label proximity:
           look for (A)/(1) text label near image → use that as option key
      4. Default → stem
    """
    _KEYS = ["A", "B", "C", "D"]
    _KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}
    _RE_LABEL = re.compile(r'^[\(\[]?([A-D1-4])[\)\.\]]\s*$', re.IGNORECASE)

    n_opts = len(rq.options) if rq else 0
    n_imgs = len(imgs)
    imgs_sorted = sorted(imgs, key=lambda i: i.y0)  # top → bottom

    # ── Rule 1: count match ───────────────────────────────────────────────────
    if n_imgs == n_opts and n_opts > 1:
        # All images are option graphs
        return [(img, key) for img, key in zip(imgs_sorted, _KEYS[:n_opts])]

    if n_imgs == n_opts + 1 and n_opts > 0:
        # First image is stem, rest are options
        result = [(imgs_sorted[0], None)]
        result += [(img, key) for img, key in zip(imgs_sorted[1:], _KEYS[:n_opts])]
        return result

    if n_imgs == 1:
        return [(imgs_sorted[0], None)]  # single → stem

    # ── Rule 2: x-position heuristic ─────────────────────────────────────────
    # Find where option labels (A/B/C/D) sit horizontally
    opt_label_xs = []
    for tb in page.text_blocks:
        if _RE_LABEL.match(tb.text.strip()):
            opt_label_xs.append((tb.x0 + tb.x1) / 2)

    result = []
    used_keys: set[str] = set()

    for img in imgs_sorted:
        img_cx = (img.x0 + img.x1) / 2
        img_cy = (img.y0 + img.y1) / 2

        # ── Rule 3: label proximity ───────────────────────────────────────────
        opt_key = None
        for tb in page.text_blocks:
            m = _RE_LABEL.match(tb.text.strip())
            if not m:
                continue
            vert_ok = (tb.y1 > img.y0 - 50) and (tb.y0 < img.y1 + 50)
            horiz_ok = tb.x0 < img.x1 + 20
            if vert_ok and horiz_ok:
                raw_key = m.group(1).upper()
                k = _KEY_MAP.get(raw_key, raw_key)
                if k not in used_keys:
                    opt_key = k
                    used_keys.add(k)
                    break

        if opt_key:
            result.append((img, opt_key))
            continue

        # ── Rule 2: x-position ────────────────────────────────────────────────
        # If image x-centre is in the left margin (narrow), likely a stem diagram
        if img_cx < 150:
            result.append((img, None))
            continue

        # If option labels exist and image x aligns with them → option
        if opt_label_xs:
            closest_label_x_dist = min(abs(img_cx - lx) for lx in opt_label_xs)
            if closest_label_x_dist < 80:
                # Assign next unused key
                for k in _KEYS:
                    if k not in used_keys:
                        used_keys.add(k)
                        result.append((img, k))
                        break
                else:
                    result.append((img, None))
                continue

        # ── Default: stem ─────────────────────────────────────────────────────
        result.append((img, None))

    return result


def _assign_raster_images(
    parsed_pdf: ParsedPDF,
    raw_questions: list[RawQuestion],
    job_id: str,
    start_idx: int,
) -> tuple[dict, dict]:
    """
    Per-question image extraction and classification.

    For each question:
      1. Collect all embedded raster images whose bbox falls within the
         question's column + y-region.
      2. Classify each image as stem or option using 3-rule logic
         (count match → x-position → label proximity).
      3. Crop + upload each image.

    Returns (q_images, opt_images).
    """
    q_images: dict[int, list[StoredImage]] = {}
    opt_images: dict[tuple, StoredImage] = {}
    idx = start_idx
    _KEYS = ["A", "B", "C", "D"]

    # Build qnum → RawQuestion and page → ParsedPage lookups
    qnum_to_rq: dict[int, RawQuestion] = {rq.number: rq for rq in raw_questions}
    pagenum_to_page = {p.page_num: p for p in parsed_pdf.pages}

    # Count how many pages each xref appears on → watermark filter
    xref_page_count: dict[int, int] = {}
    for page in parsed_pdf.pages:
        for img in (page.image_blocks or []):
            xref_page_count[img.xref] = xref_page_count.get(img.xref, 0) + 1

    # Build per-page question list sorted by y0
    page_to_qs: dict[int, list[RawQuestion]] = {}
    for rq in raw_questions:
        page_to_qs.setdefault(rq.page_num, []).append(rq)
    for pnum in page_to_qs:
        page_to_qs[pnum].sort(key=lambda q: q.y0)

    # ── Per-question: collect images in question's region ────────────────────
    for rq in raw_questions:
        page = pagenum_to_page.get(rq.page_num)
        if page is None or not page.image_blocks or page.render_bytes is None:
            continue

        page_mid_x = page.width / 2
        page_area = page.width * page.height

        # Question's column
        rq_col = _question_column(rq, page, page_mid_x)

        # Next question on same page (to bound the bottom of this question)
        page_qs = page_to_qs.get(rq.page_num, [])
        rq_idx = next((i for i, q in enumerate(page_qs) if q.number == rq.number), None)
        if rq_idx is not None and rq_idx + 1 < len(page_qs):
            next_same_col = next(
                (q for q in page_qs[rq_idx+1:]
                 if _question_column(q, page, page_mid_x) == rq_col),
                None
            )
            y_bottom = next_same_col.y0 if next_same_col else page.height
        else:
            y_bottom = page.height

        # Collect images in this question's region
        q_imgs = []
        for img in page.image_blocks:
            # Watermark / repeat filter
            if xref_page_count.get(img.xref, 1) >= 3:
                continue
            # Size filters
            img_area = (img.x1 - img.x0) * (img.y1 - img.y0)
            if page_area > 0 and img_area / page_area > 0.55:
                continue
            if img_area < 500:
                continue

            img_cy = (img.y0 + img.y1) / 2
            img_cx = (img.x0 + img.x1) / 2
            img_col = "L" if img_cx < page_mid_x else "R"

            # Must be in question's y-region
            if not (rq.y0 <= img_cy < y_bottom):
                continue

            # Must be in same column (unless question column is unknown)
            if rq_col != "?" and img_col != rq_col:
                continue

            q_imgs.append(img)

        if not q_imgs:
            continue

        # ── Classify: stem vs option ──────────────────────────────────────────
        classified = _classify_images(q_imgs, rq, page)

        # ── Crop + upload ─────────────────────────────────────────────────────
        for img, opt_key in classified:
            bbox = points_to_0_1000(img.x0, img.y0, img.x1, img.y1, page.width, page.height)
            try:
                img_bytes = crop_image_from_render(page.render_bytes, bbox)
            except Exception as e:
                logger.warning(f"Raster crop failed q{rq.number}: {e}")
                continue

            q_part = f"q{rq.number}"
            opt_part = f"_opt{opt_key}" if opt_key else ""
            public_id = f"{job_id}_{q_part}{opt_part}_{idx}"
            mapping_name = f"{{{{IMAGE:img_{q_part}{opt_part}_{idx}}}}}"
            image_type = "option" if opt_key else "question"
            alt = f"Image for option {opt_key}" if opt_key else f"Diagram for question {rq.number}"

            try:
                url = upload_image(img_bytes, public_id)
            except Exception as e:
                logger.error(f"Upload failed {mapping_name}: {e}")
                url = ""

            stored = StoredImage(
                url=url,
                mapping_image_name=mapping_name,
                alt_text=alt,
                image_type=image_type,
                figure_bbox=bbox,
            )
            idx += 1

            if opt_key:
                opt_images[(rq.number, opt_key)] = stored
            else:
                q_images.setdefault(rq.number, []).append(stored)

    # Free render_bytes from all pages to release memory
    for page in parsed_pdf.pages:
        page.render_bytes = None

    return q_images, opt_images

# ─────────────────────── schema builders ─────────────────────────────────────

def _build_image_details(stored: StoredImage) -> dict:
    return {
        "url": stored.url,
        "altText": stored.alt_text,
        "imageType": stored.image_type,
        "mappingImageName": stored.mapping_image_name,
        # figure_bbox kept for debug — assembler will strip before final output
        "_figure_bbox": stored.figure_bbox,
    }


def _build_option(raw_opt: RawOption) -> dict:
    return {
        "key": raw_opt.key,
        "optionType": "text",   # updated later if image found
        "text": raw_opt.text or None,
        "imageDetails": None,
    }


def _build_table(raw_table: RawTable) -> dict:
    rows = []
    for row in raw_table.rows:
        built_row = []
        for cell in row:
            built_row.append({
                "text": cell.text or None,
                "imageDetails": None,  # filled in image placement pass
            })
        rows.append(built_row)
    return {
        "tableId": "",          # filled below
        "headers": raw_table.headers,
        "rows": rows,
    }


def _strip_figure_bbox(question: dict) -> dict:
    """Remove internal _figure_bbox fields before final output."""
    for img in question.get("imageDetails", []):
        img.pop("_figure_bbox", None)
    for opt in question.get("options", []):
        if opt.get("imageDetails"):
            opt["imageDetails"].pop("_figure_bbox", None)
    for tbl in question.get("tables", []):
        for row in tbl.get("rows", []):
            for cell in row:
                if cell.get("imageDetails"):
                    cell["imageDetails"].pop("_figure_bbox", None)
    return question


# ─────────────────────── image ↔ question mapping ────────────────────────────

def _match_images_to_questions(
    vision_results: dict[int, PageVisionResult],
    raw_questions: list[RawQuestion],
) -> dict[int, list[tuple[ImageLocation, int]]]:
    """
    Returns dict: question_number → list of (ImageLocation, page_num).
    Uses question_number from vision result directly when available,
    falls back to spatial proximity matching.
    """
    mapping: dict[int, list[tuple[ImageLocation, int]]] = {}

    # Build a quick lookup: page_num → sorted question numbers on that page
    page_to_qnums: dict[int, list[int]] = {}
    for rq in raw_questions:
        page_to_qnums.setdefault(rq.page_num, []).append(rq.number)
    for pnum in page_to_qnums:
        page_to_qnums[pnum].sort()

    for page_num, vr in vision_results.items():
        for loc in vr.image_locations:
            qnum = loc.question_number

            # If Groq didn't give us a question number, infer from page
            if qnum is None:
                qnums_on_page = page_to_qnums.get(page_num, [])
                if qnums_on_page:
                    # Pick the question whose y-range covers the image's y centre
                    img_y_centre = (loc.bbox[1] + loc.bbox[3]) / 2  # on 0-1000 grid
                    best = qnums_on_page[0]
                    for rq in raw_questions:
                        if rq.page_num != page_num:
                            continue
                        # Convert question y0 to 0-1000 grid
                        page_obj = None
                        # Use page height — approximate via 792pt standard
                        rq_y0_norm = (rq.y0 / 792) * 1000
                        rq_y1_norm = (rq.y1 / 792) * 1000
                        if rq_y0_norm <= img_y_centre <= rq_y1_norm:
                            best = rq.number
                            break
                    qnum = best
                else:
                    qnum = 0  # unknown

            mapping.setdefault(qnum, []).append((loc, page_num))

    return mapping


# ─────────────────────── main assembler ──────────────────────────────────────

def assemble(
    parsed_pdf: ParsedPDF,
    raw_questions: list[RawQuestion],
    vision_results: dict[int, PageVisionResult],
    job_id: str,
    is_scanned: bool = False,
    groq_num_map: dict | None = None,
) -> list[dict]:
    """
    Full assembly: text + images -> schema-compliant list[Question].

    Image source priority (highest wins):
      1. Raster extraction (PyMuPDF exact coords) -- most accurate
      2. Groq vision bbox -- fallback for pages where raster fails

    Bug fixes applied:
      FIX-1: When raster assigns option images for ALL options of a question,
             any Groq detections that leaked into stem are discarded.
      FIX-2: Raster option images always overwrite Groq option images for the
             same slot -- no duplicates.
    """
    # Step 1: map Groq vision images to question numbers
    img_mapping = _match_images_to_questions(vision_results, raw_questions)

    # Step 2: crop + upload Groq-detected images
    q_images: dict[int, list[StoredImage]] = {}
    opt_images: dict[tuple, StoredImage] = {}
    cell_images: dict[tuple, StoredImage] = {}

    # For scanned PDFs, Groq returns question_number 1-N within each section.
    # Our raw_questions are globally numbered. Build a page→questions map to
    # remap Groq's within-section index to the correct global question number.
    _page_to_rqs: dict[int, list[RawQuestion]] = {}
    for _rq in raw_questions:
        _page_to_rqs.setdefault(_rq.page_num, []).append(_rq)

    def _resolve_qnum(groq_qnum, page_num):
        if not is_scanned:
            return groq_qnum
        # Use groq_num_map first: maps (page_num, exam_question_number) -> global_qnum
        # This is more reliable than index-based mapping because Groq returns the
        # actual exam question number (e.g., "5"), not a 1-based page index.
        if groq_num_map and groq_qnum is not None:
            mapped = groq_num_map.get((page_num, groq_qnum))
            if mapped is not None:
                return mapped
        # Fallback: spatial — find the question on this page whose y-range
        # best covers the image. Index-based (groq_qnum as page-index) is a
        # last resort since it's unreliable for multi-section papers.
        page_rqs = sorted(_page_to_rqs.get(page_num, []), key=lambda q: q.y0)
        if not page_rqs:
            return groq_qnum
        if groq_qnum is not None and 1 <= groq_qnum <= len(page_rqs):
            return page_rqs[groq_qnum - 1].number
        return page_rqs[0].number

    global_idx = 0
    for groq_qnum, loc_list in img_mapping.items():
        for loc, page_num in loc_list:
            if loc.latex_content:
                continue

            page = parsed_pdf.pages[page_num]
            if page.render_bytes is None:
                continue

            qnum = _resolve_qnum(groq_qnum, page_num)
            if qnum is None:
                continue

            stored = process_image_location(
                loc=loc,
                render_bytes=page.render_bytes,
                job_id=job_id,
                image_index=global_idx,
            )
            global_idx += 1

            if loc.is_table_cell and loc.table_row is not None and loc.table_col is not None:
                cell_images[(qnum, loc.table_row, loc.table_col)] = stored
            elif loc.option_key:
                _KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}
                norm_key = _KEY_MAP.get(loc.option_key.upper(), loc.option_key.upper())
                opt_images[(qnum, norm_key)] = stored
            else:
                q_images.setdefault(qnum, []).append(stored)

    # Step 3: raster extraction (more accurate than Groq bboxes)
    raster_q, raster_opt = _assign_raster_images(
        parsed_pdf=parsed_pdf,
        raw_questions=raw_questions,
        job_id=job_id,
        start_idx=global_idx,
    )

    # Raster option images always win over Groq option images
    for key, si in raster_opt.items():
        opt_images[key] = si

    # Raster stem images replace Groq stem images (exact crop > bbox guess)
    for qnum, imgs in raster_q.items():
        if imgs:
            q_images[qnum] = imgs

    # FIX-1: Drop false Groq stem images
    # When raster assigned option images for EVERY option of a question, the
    # raster images ARE the 4 option graphs -- there is no stem image. Any Groq
    # detections that ended up in q_images for that question are spurious
    # (Groq re-detected the same option graphs but with no option_key -> stem).
    for rq in raw_questions:
        all_opts = [k for k in ['A', 'B', 'C', 'D'][:len(rq.options)]]
        raster_covers_all_opts = (
            len(all_opts) > 1 and
            all((rq.number, k) in raster_opt for k in all_opts)
        )
        if raster_covers_all_opts:
            q_images.pop(rq.number, None)

    # Step 4: build output
    output: list[dict] = []

    for rq in raw_questions:
        # Options
        options = []
        for raw_opt in rq.options:
            opt_dict = _build_option(raw_opt)
            si = opt_images.get((rq.number, raw_opt.key))
            if si:
                opt_dict["imageDetails"] = _build_image_details(si)
                opt_dict["optionType"] = "text_and_image" if raw_opt.text else "image"
                if not raw_opt.text:
                    opt_dict["text"] = None
            options.append(opt_dict)

        # Tables
        tables = []
        for t_idx, raw_tbl in enumerate(rq.tables):
            tbl_dict = _build_table(raw_tbl)
            tbl_dict["tableId"] = f"tbl_q{rq.number}_{t_idx+1}"
            for r_idx, row in enumerate(tbl_dict["rows"]):
                for c_idx, cell in enumerate(row):
                    si = cell_images.get((rq.number, r_idx, c_idx))
                    if si:
                        cell["imageDetails"] = _build_image_details(si)
            tables.append(tbl_dict)

        # Stem images
        q_img_list = q_images.get(rq.number, [])
        image_details = [_build_image_details(si) for si in q_img_list]

        # Inject LaTeX found as images back into questionText
        for loc, _ in img_mapping.get(rq.number, []):
            if loc.latex_content and loc.latex_content not in rq.question_text:
                rq.question_text += f" {loc.latex_content}"

        question = {
            "questionText": rq.question_text,
            "questionType": rq.question_type or "MCQ_SINGLE",
            "options": options,
            "answer": {
                "key": rq.answer_key,
                "explanation": rq.answer_explanation or "",
            },
            "hasImage": len(image_details) > 0,
            "imageDetails": image_details,
            "tables": tables,
            "subject": rq.subject,
            "topic": rq.topic,
            "paragraphContext": rq.paragraph_context,
        }

        question = _strip_figure_bbox(question)

        # Remove empty optional fields for clean output
        for field in ("tables", "subject", "topic", "paragraphContext"):
            val = question.get(field)
            if val is None or val == [] or val == {}:
                question.pop(field, None)

        output.append(question)

    return output


# ─────────────────────── marker-pdf assembler ────────────────────────────────

def assemble_from_marker(
    raw_questions: list[RawQuestion],
    q_images: dict[int, list[tuple[str, bytes]]],
    opt_images: dict[tuple[int, str], tuple[str, bytes]],
    job_id: str,
) -> list[dict]:
    """
    Build final question JSON from marker-pdf extracted questions + images.
    No Groq Vision needed — marker-pdf already extracted and positioned images.

    Args:
        raw_questions  — from markdown_segmenter
        q_images       — {qnum: [(name, bytes)]}  stem images
        opt_images     — {(qnum, key): (name, bytes)}  option images
        job_id         — for image naming
    """
    output: list[dict] = []
    img_idx = 0

    for rq in raw_questions:
        # ── Options ────────────────────────────────────────────────────────────
        options = []
        for raw_opt in rq.options:
            opt_dict = _build_option(raw_opt)
            opt_key = (rq.number, raw_opt.key)
            if opt_key in opt_images:
                name, img_bytes = opt_images[opt_key]
                public_id = f"{job_id}_q{rq.number}_opt{raw_opt.key}_{img_idx}"
                mapping = f"{{{{IMAGE:img_q{rq.number}_opt{raw_opt.key}_{img_idx}}}}}"
                try:
                    url = upload_image(img_bytes, public_id)
                except Exception as e:
                    logger.error(f"Upload failed {public_id}: {e}")
                    url = ""
                opt_dict["imageDetails"] = {
                    "url": url,
                    "altText": f"Graph for option {raw_opt.key}",
                    "imageType": "option",
                    "mappingImageName": mapping,
                }
                opt_dict["optionType"] = "text_and_image" if raw_opt.text else "image"
                if not raw_opt.text:
                    opt_dict["text"] = None
                img_idx += 1
            options.append(opt_dict)

        # ── Stem images ────────────────────────────────────────────────────────
        image_details = []
        for name, img_bytes in q_images.get(rq.number, []):
            public_id = f"{job_id}_q{rq.number}_{img_idx}"
            mapping = f"{{{{IMAGE:img_q{rq.number}_{img_idx}}}}}"
            try:
                url = upload_image(img_bytes, public_id)
            except Exception as e:
                logger.error(f"Upload failed {public_id}: {e}")
                url = ""
            image_details.append({
                "url": url,
                "altText": f"Diagram for question {rq.number}",
                "imageType": "question",
                "mappingImageName": mapping,
            })
            img_idx += 1

        question = {
            "questionText": rq.question_text,
            "questionType": rq.question_type or "MCQ_SINGLE",
            "options": options,
            "answer": {
                "key": rq.answer_key,
                "explanation": rq.answer_explanation or "",
            },
            "hasImage": len(image_details) > 0,
            "imageDetails": image_details,
        }

        # Optional fields
        for field_name, attr in (
            ("subject", "subject"),
            ("topic", "topic"),
            ("paragraphContext", "paragraph_context"),
        ):
            val = getattr(rq, attr, None)
            if val:
                question[field_name] = val

        output.append(question)

    return output
