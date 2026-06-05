"""
Pipeline Orchestrator
---------------------
Unified single code path for ALL PDFs (digital and scanned):

  parse_pdf → Groq Vision ALL pages (SCANNED_PAGE_PROMPT) → assemble → JSON

Groq Vision extracts questions, options, answers, and image locations directly
from rendered page images — format-agnostic, no regex dependency.
For digital PDFs, PyMuPDF embedded images are also extracted at native resolution
and used in the assembler for higher-quality image crops.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from utils.pdf_parser import parse_pdf
from utils.question_segmenter import RawQuestion, RawOption
from utils.groq_vision import analyse_all_image_pages, PageVisionResult
from utils.schema_assembler import assemble, assemble_from_marker

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _build_raw_questions_from_vision(
    vision_results: dict[int, "PageVisionResult"],
    ocr_questions: list["RawQuestion"] | None = None,
) -> tuple[list["RawQuestion"], dict]:
    """
    Build RawQuestion objects from Groq's extracted_questions (scanned PDF path).

    Key design: scanned papers often repeat question numbers across sections
    (e.g. Physics Q1-16, Chemistry Q1-16, Maths Q1-16). We assign a
    GLOBAL sequential number so all questions are preserved.
    """
    import re as _re
    _KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}
    questions: list[RawQuestion] = []
    groq_num_map: dict[tuple, int] = {}  # (page_num, groq_qnum) -> global_qnum
    groq_pages: set[int] = set()  # pages where Groq returned questions
    global_num = 1  # sequential across all pages/sections

    for page_num in sorted(vision_results.keys()):
        vr = vision_results[page_num]
        for q_data in vr.extracted_questions:
            qtext = (q_data.get("question_text") or "").strip()
            if not qtext:
                continue  # skip empty extractions

            # Build options
            options = []
            for opt in q_data.get("options", []):
                key = str(opt.get("key", "")).upper()
                key = _KEY_MAP.get(key, key)
                text = opt.get("text") or ""
                if key:
                    options.append(RawOption(key=key, text=text.strip()))

            # Parse answer key
            ans_raw = str(q_data.get("answer_key") or "")
            ans_key = None
            if ans_raw:
                letters = _re.findall(r'[A-D]', ans_raw.upper())
                if letters:
                    ans_key = ",".join(dict.fromkeys(letters))
                elif _re.match(r'^-?\d+(\.\d+)?$', ans_raw.strip()):
                    ans_key = ans_raw.strip()
                else:
                    ans_key = ans_raw.strip() or None

            # Store mapping from (page_num, groq_section_qnum) to global_qnum
            groq_qnum_raw = q_data.get("number")
            if groq_qnum_raw is not None:
                try:
                    groq_num_map[(page_num, int(groq_qnum_raw))] = global_num
                except (ValueError, TypeError):
                    pass
            groq_pages.add(page_num)

            rq = RawQuestion(
                number=global_num,
                question_text=qtext,
                options=options,
                answer_key=ans_key,
                question_type=q_data.get("question_type") or "MCQ_SINGLE",
                page_num=page_num,
                y0=float(global_num * 100),
                y1=float(global_num * 100 + 99),
            )
            questions.append(rq)
            global_num += 1

    # For pages where Groq returned no questions, use OCR fallback
    if ocr_questions:
        groq_page_set = set(q.page_num for q in questions)
        for ocr_q in ocr_questions:
            if ocr_q.page_num not in groq_page_set:
                # Renumber to continue from current global_num
                ocr_q.number = global_num
                ocr_q.y0 = float(global_num * 100)
                ocr_q.y1 = float(global_num * 100 + 99)
                questions.append(ocr_q)
                global_num += 1

    return questions, groq_num_map


async def _run_pipeline_async(pdf_path: str, job_id: str) -> dict:
    start = time.monotonic()

    # ── Step 1: Parse PDF (detect scanned vs digital) ─────────────────────────
    logger.info(f"[{job_id}] Parsing PDF: {pdf_path}")
    parsed = parse_pdf(pdf_path)
    logger.info(
        f"[{job_id}] Parsed {parsed.page_count} pages | "
        f"scanned={parsed.is_scanned} | "
        f"image_pages={sum(1 for p in parsed.pages if p.has_images)}"
    )

    # ── Unified path: Groq Vision for ALL PDFs ────────────────────────────────
    # Both digital and scanned PDFs go through the same pipeline.
    # Groq reads rendered page images → extracts questions, options, answers,
    # and image bounding boxes regardless of PDF format or layout.
    # PyMuPDF embedded images (digital only) give native-res image crops.
    logger.info(
        f"[{job_id}] {'Scanned' if parsed.is_scanned else 'Digital'} PDF → "
        f"Groq Vision ({len(parsed.pages)} pages, {len(set(k for k in [1]))} unified path)"
    )

    vision_results = await analyse_all_image_pages(
        pages=parsed.pages,
        is_scanned=parsed.is_scanned,   # passed for logging only — prompt is always SCANNED_PAGE_PROMPT
    )
    logger.info(f"[{job_id}] Vision complete | {len(vision_results)} pages analysed")

    raw_questions, groq_num_map = _build_raw_questions_from_vision(vision_results)
    logger.info(f"[{job_id}] Extracted {len(raw_questions)} questions from Groq Vision")

    logger.info(f"[{job_id}] Assembling JSON...")
    questions = assemble(
        parsed_pdf=parsed,
        raw_questions=raw_questions,
        vision_results=vision_results,
        job_id=job_id,
        is_scanned=parsed.is_scanned,
        groq_num_map=groq_num_map,
    )

    elapsed = time.monotonic() - start
    logger.info(f"[{job_id}] Done in {elapsed:.1f}s | {len(questions)} questions")

    return {
        "job_id": job_id,
        "pdf_hash": parsed.pdf_hash,
        "page_count": parsed.page_count,
        "question_count": len(questions),
        "is_scanned": parsed.is_scanned,
        "elapsed_seconds": round(elapsed, 2),
        "questions": questions,
    }


def run_pipeline(pdf_path: str, job_id: str) -> dict:
    """Synchronous wrapper — used by Celery worker."""
    return asyncio.run(_run_pipeline_async(pdf_path, job_id))


def run_pipeline_and_save(pdf_path: str, job_id: str) -> str:
    """
    Run pipeline and save JSON to output dir.
    Returns path to output JSON file.
    """
    result = run_pipeline(pdf_path, job_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{job_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"[{job_id}] Output saved → {out_path}")
    return str(out_path)
