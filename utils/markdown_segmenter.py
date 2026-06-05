"""
Markdown Question Segmenter
---------------------------
Parses marker-pdf markdown output into RawQuestion objects.

Handles all JEE/NEET question formats:
  - MCQ Single / Multiple correct
  - Numerical Integer / Decimal
  - Matrix Match (List-I / List-II)
  - Assertion-Reason

Image assignment logic:
  - n_images == 1                      → stem image
  - n_images == n_options, opts empty  → all option images (e.g. 4 graphs)
  - n_images == n_options + 1          → first is stem, rest are option images
  - otherwise                          → all stem images
"""

import re
import logging
from typing import Optional

from .question_segmenter import RawQuestion, RawOption

logger = logging.getLogger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Question start: "Q. 1", "Q1.", "1.", "**Q. 1**", "(1)"
RE_QSTART = re.compile(
    r'(?:^|\n)\s*\**\s*Q\.?\s*(\d{1,3})\s*[\.:\)]\s*\**\s*',
    re.MULTILINE,
)

# Option: "(A) text", "A) text", "(A)" alone, "**A)**"
RE_OPT = re.compile(
    r'^\s*\**\s*[\(\[]\s*([A-D])\s*[\)\]]\s*\**\.?\s*(.*)',
    re.IGNORECASE,
)
RE_OPT2 = re.compile(
    r'^\s*\**\s*([A-D])\s*[\.:\)]\s*\**\s+(.*)',
    re.IGNORECASE,
)

# Answer key: "Ans: D", "Answer: A, C", "**Ans. (C)**"
RE_ANS = re.compile(
    r'^\s*\**\s*Ans(?:wer)?\.?\s*:?\s*\**\s*[\(\[]?\s*([A-D,\s\d\./]+)',
    re.IGNORECASE,
)

# Section header (markdown heading or all-caps line)
RE_SECTION = re.compile(
    r'^#+\s*(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|BIOLOGY|BOTANY|ZOOLOGY)\b'
    r'|^\s*(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|BIOLOGY|BOTANY|ZOOLOGY)\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# Image reference
RE_IMG = re.compile(r'!\[([^\]]*)\]\(([^\)]+)\)')

# Lines to discard (instructions, page numbers, watermarks)
RE_NOISE = re.compile(
    r'Full Marks|Zero Marks|Negative Marks|Maximum Marks|'
    r'marking scheme|This section contains|Each question has|'
    r'choose the option corresponding|Answer to each question|'
    r'SECTION\s+\d+|^\s*\d{1,3}\s*$|^\s*[-–—]+\s*$|'
    r'PupilTree|Brilliant|FIITJEE|Allen|Resonance|Aakash',
    re.IGNORECASE,
)

_KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm_key(k: str) -> str:
    return _KEY_MAP.get(k.upper(), k.upper())


def _parse_answer(raw: str) -> Optional[str]:
    raw = raw.strip().strip('()[].')
    if re.match(r'^-?\d+(\.\d+)?$', raw.strip()):
        return raw.strip()
    letters = list(dict.fromkeys(re.findall(r'[A-D]', raw.upper())))
    return ",".join(letters) if letters else (raw.strip() or None)


def _classify(text: str, options: list[RawOption]) -> str:
    t = text.lower()
    if re.search(r'list[\s\-]*(i{1,2}|1|2)', t):
        return "MATRIX_MATCH"
    if re.search(r'assert\w*.*reason', t):
        return "ASSERTION_REASON"
    if re.search(r'one or more|more than one|correct.*statement.*are|may be more', t):
        return "MCQ_MULTIPLE"
    if options:
        return "MCQ_SINGLE"
    if re.search(r'nearest integer|integer value', t):
        return "NUMERICAL_INTEGER"
    if re.search(r'decimal', t):
        return "NUMERICAL_DECIMAL"
    return "NUMERICAL_INTEGER"


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if RE_NOISE.search(s):
        return True
    return False


def _assign_images(
    img_names: list[str],
    images: dict[str, bytes],
    options: list[RawOption],
    qnum: int,
) -> tuple[list[tuple[str, bytes]], dict[str, tuple[str, bytes]]]:
    """
    Decide which images are stem images and which are option images.

    Returns:
        (stem_images, option_images)
        stem_images:   [(name, bytes), ...]
        option_images: {opt_key: (name, bytes)}
    """
    _KEYS = ["A", "B", "C", "D"]
    n = len(img_names)
    n_opts = len(options)
    opts_have_text = any(o.text and o.text.strip() for o in options)

    imgs = [(name, images[name]) for name in img_names if name in images]
    imgs_sorted = sorted(imgs, key=lambda x: x[0])  # stable order

    if n == 0:
        return [], {}

    if n == 1:
        return [imgs_sorted[0]], {}

    if n == n_opts and n_opts > 1 and not opts_have_text:
        # All images are option graphs (options have no text)
        opt_imgs = {_KEYS[i]: imgs_sorted[i] for i in range(min(n, 4))}
        return [], opt_imgs

    if n == n_opts + 1 and n_opts > 0:
        # First image is stem, rest are option images
        stem = [imgs_sorted[0]]
        opt_imgs = {_KEYS[i]: imgs_sorted[i + 1] for i in range(min(n_opts, 4))}
        return stem, opt_imgs

    # Default: all stem images
    return imgs_sorted, {}


# ── Main segmenter ─────────────────────────────────────────────────────────────

def segment_from_markdown(
    markdown: str,
    images: dict[str, bytes],
    job_id: str = "",
) -> tuple[
    list[RawQuestion],
    dict[int, list[tuple[str, bytes]]],   # qnum -> stem images
    dict[tuple[int, str], tuple[str, bytes]],  # (qnum, key) -> option image
]:
    """
    Parse marker-pdf markdown into RawQuestion objects plus image assignments.

    Returns:
        questions      — list of RawQuestion
        q_images       — {qnum: [(name, bytes), ...]}  stem images
        opt_images     — {(qnum, key): (name, bytes)}  option images
    """
    questions: list[RawQuestion] = []
    q_images: dict[int, list[tuple[str, bytes]]] = {}
    opt_images: dict[tuple[int, str], tuple[str, bytes]] = {}

    # Build section map: char_position → section_name
    section_map: list[tuple[int, str]] = []
    for m in RE_SECTION.finditer(markdown):
        name = (m.group(1) or m.group(2)).upper()
        if name == "MATHS":
            name = "MATHEMATICS"
        section_map.append((m.start(), name))

    def get_section(pos: int) -> Optional[str]:
        result = None
        for sec_pos, sec_name in section_map:
            if sec_pos <= pos:
                result = sec_name
        return result

    # Find all question starts
    q_matches = list(RE_QSTART.finditer(markdown))
    if not q_matches:
        logger.warning("No questions found in markdown — check marker-pdf output")
        return [], {}, {}

    global_num = 1

    for i, qm in enumerate(q_matches):
        local_qnum = int(qm.group(1))
        block_start = qm.start()
        block_end = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(markdown)
        block = markdown[block_start:block_end]

        subject = get_section(block_start)

        # Collect images referenced in this block
        block_img_names: list[str] = []
        for img_m in RE_IMG.finditer(block):
            name = img_m.group(2)
            if name in images:
                block_img_names.append(name)

        # Strip image markdown from block
        clean = RE_IMG.sub("", block).strip()
        lines = clean.splitlines()

        # ── Parse lines ────────────────────────────────────────────────────────
        q_text_lines: list[str] = []
        option_texts: dict[str, list[str]] = {}
        cur_opt: Optional[str] = None
        answer_key: Optional[str] = None

        for j, line in enumerate(lines):
            stripped = line.strip()

            # Skip noise
            if _is_noise(stripped):
                continue

            # Answer key
            ans_m = RE_ANS.match(stripped)
            if ans_m:
                answer_key = _parse_answer(ans_m.group(1))
                cur_opt = None
                continue

            # Option line
            opt_m = RE_OPT.match(stripped) or RE_OPT2.match(stripped)
            if opt_m:
                key = _norm_key(opt_m.group(1))
                if key in ("A", "B", "C", "D"):
                    if cur_opt:
                        pass  # already saved lines
                    cur_opt = key
                    option_texts[key] = [opt_m.group(2).strip()] if opt_m.group(2).strip() else []
                    continue

            # Continue building current option or question text
            if cur_opt:
                # Check if this looks like a new option starting without proper prefix
                # (sometimes marker-pdf indents continuation)
                option_texts[cur_opt].append(stripped)
            else:
                # Skip the "Q.N" prefix on first line
                if j == 0:
                    stripped = re.sub(r'^\**\s*Q\.?\s*\d+\s*[\.:\)]\s*\**\s*', '', stripped).strip()
                if stripped:
                    q_text_lines.append(stripped)

        q_text = " ".join(q_text_lines).strip()
        if not q_text and not block_img_names:
            global_num += 1
            continue

        raw_opts = [
            RawOption(key=k, text=" ".join(v).strip())
            for k, v in sorted(option_texts.items())
        ]

        rq = RawQuestion(
            number=global_num,
            question_text=q_text,
            options=raw_opts,
            answer_key=answer_key,
            subject=subject,
            page_num=0,
            y0=float(global_num * 100),
            y1=float(global_num * 100 + 99),
        )
        rq.question_type = _classify(q_text, raw_opts)

        # Assign images
        if block_img_names:
            stem_imgs, opt_imgs = _assign_images(
                block_img_names, images, raw_opts, global_num
            )
            if stem_imgs:
                q_images[global_num] = stem_imgs
            for key, img_data in opt_imgs.items():
                opt_images[(global_num, key)] = img_data

        global_num += 1
        questions.append(rq)

    # Remove duplicate question numbers (keep first occurrence)
    seen: set[int] = set()
    unique: list[RawQuestion] = []
    for rq in questions:
        if rq.number not in seen:
            seen.add(rq.number)
            unique.append(rq)

    logger.info(f"Extracted {len(unique)} questions from markdown")
    return unique, q_images, opt_images
