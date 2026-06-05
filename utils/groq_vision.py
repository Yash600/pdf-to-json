"""
Groq Vision Module
------------------
Sends rendered page PNGs to Groq Llama Vision and returns structured
question data for scanned PDFs, or image locations for digital PDFs.

Multi-key support: set GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3 in .env.
Pages split evenly across available keys — each key has its own DualRateLimiter
and processes its chunk sequentially. All chunks run concurrently via asyncio.gather.
With 2 keys: ~2x speed. With 3 keys: ~3x speed.

Rate limiting (per key): dual token-bucket (RPM + TPM).
  RPM bucket : 28 req/min  (stays under 30 RPM free-tier ceiling)
  TPM bucket : 28000 tok/min (stays under 30k TPM free-tier ceiling)

On 429 errors: parses the exact "try again in Xs" from Groq's message.
"""

import os
import re
import base64
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()


class _TPDExhaustedError(Exception):
    """Raised when Groq tokens-per-day quota is exhausted."""
    pass


logger = logging.getLogger(__name__)

GROQ_RPM_LIMIT = int(os.getenv("GROQ_RPM_LIMIT", "28"))
GROQ_TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "28000"))
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL     = "llama-3.3-70b-versatile"

# Collect all available API keys (primary + optional extras)
def _load_api_keys() -> list[str]:
    keys = []
    for env_var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        k = os.getenv(env_var, "").strip()
        if k:
            keys.append(k)
    if not keys:
        raise ValueError("No Groq API keys found. Set GROQ_API_KEY in .env")
    return keys

GROQ_API_KEYS = _load_api_keys()


# ─────────────────────── dual rate limiter ───────────────────────────────────

class DualRateLimiter:
    """
    Async dual token-bucket rate limiter — enforces both RPM and TPM limits.

    Before each API call, acquire(tokens_needed) must be awaited.
    It will sleep only as long as necessary for both buckets to have capacity.
    """

    def __init__(self, rpm: int, tpm: int):
        self._rpm_tokens   = float(rpm)
        self._rpm_max      = float(rpm)
        self._rpm_refill   = rpm / 60.0       # tokens per second

        self._tpm_tokens   = float(tpm)
        self._tpm_max      = float(tpm)
        self._tpm_refill   = tpm / 60.0       # tokens per second

        self._last_refill  = time.monotonic()
        self._lock         = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._rpm_tokens = min(self._rpm_max, self._rpm_tokens + elapsed * self._rpm_refill)
        self._tpm_tokens = min(self._tpm_max, self._tpm_tokens + elapsed * self._tpm_refill)
        self._last_refill = now

    async def acquire(self, token_cost: int):
        """Block until both RPM (1 request) and TPM (token_cost tokens) are available."""
        async with self._lock:
            self._refill()

            # How long until RPM bucket has 1 token?
            rpm_wait = max(0.0, (1 - self._rpm_tokens) / self._rpm_refill)
            # How long until TPM bucket has token_cost tokens?
            tpm_wait = max(0.0, (token_cost - self._tpm_tokens) / self._tpm_refill)
            wait = max(rpm_wait, tpm_wait)

            if wait > 0:
                logger.debug(f"Rate limiter sleeping {wait:.2f}s (rpm_wait={rpm_wait:.2f}, tpm_wait={tpm_wait:.2f})")
                await asyncio.sleep(wait)
                self._refill()

            self._rpm_tokens -= 1
            self._tpm_tokens -= token_cost

    async def wait_for_retry(self, error_message: str, attempt: int):
        """
        Parse Groq's exact retry-after from the 429 error message and sleep that duration.
        Falls back to exponential backoff if parsing fails.

        Groq message formats:
          "Please try again in 532ms."
          "Please try again in 6.378s."
          "Please try again in 1m30s."
        """
        wait = None

        # Try parsing "Xs" or "Xms" or "XmYs"
        match_s  = re.search(r'try again in (\d+(?:\.\d+)?)s', error_message)
        match_ms = re.search(r'try again in (\d+(?:\.\d+)?)ms', error_message)
        match_ms_first = re.search(r'try again in (\d+)m(\d+(?:\.\d+)?)s', error_message)

        if match_ms_first:
            wait = float(match_ms_first.group(1)) * 60 + float(match_ms_first.group(2))
        elif match_s:
            wait = float(match_s.group(1))
        elif match_ms:
            wait = float(match_ms.group(1)) / 1000.0

        if wait is None:
            wait = min(2 ** attempt, 30)  # fallback: capped exponential backoff

        wait += 0.5  # small buffer
        logger.info(f"Rate limit hit — sleeping {wait:.2f}s (from Groq retry-after)")
        await asyncio.sleep(wait)


# One DualRateLimiter per API key — each key has independent Groq limits
_rate_limiters: dict[str, DualRateLimiter] = {
    key: DualRateLimiter(rpm=GROQ_RPM_LIMIT, tpm=GROQ_TPM_LIMIT)
    for key in GROQ_API_KEYS
}


# ─────────────────────── response data classes ───────────────────────────────

@dataclass
class ImageLocation:
    bbox: list[int]
    question_number: Optional[int] = None
    option_key: Optional[str] = None
    is_table_cell: bool = False
    table_row: Optional[int] = None
    table_col: Optional[int] = None
    latex_content: Optional[str] = None
    description: str = ""


@dataclass
class PageVisionResult:
    page_num: int
    image_locations: list[ImageLocation] = field(default_factory=list)
    extracted_questions: list[dict] = field(default_factory=list)
    raw_response: str = ""


# ─────────────────────── prompts ─────────────────────────────────────────────

IMAGE_MAPPING_PROMPT = """You are analyzing a page from an exam paper (NEET/JEE style).

Your job: find every figure, diagram, graph, or image on this page and return their locations.

Return ONLY valid JSON in this exact format:
{
  "images": [
    {
      "bbox": [x0, y0, x1, y1],
      "question_number": <int or null>,
      "option_key": <"A"/"B"/"C"/"D" or null>,
      "is_table_cell": <true/false>,
      "table_row": <int or null>,
      "table_col": <int or null>,
      "latex_content": <"$...$" string if this image is a math equation, else null>,
      "description": "<one line description>"
    }
  ]
}

CRITICAL RULES — read carefully:
- bbox: ALL four values must be INTEGERS on a 0-1000 grid. (0,0)=top-left, (1000,1000)=bottom-right. No floats.
- Crop TIGHTLY around each individual image — do NOT group multiple graphs into one bbox.
- ONE entry per image. If a question has 4 option graphs (labeled 1/2/3/4 or A/B/C/D), output 4 SEPARATE entries.
- question_number: integer from the nearest question label (e.g. "1." → 1). Required when identifiable.
- option_key: set to "A","B","C","D" when the image is an answer option.
  - Options labeled 1) 2) 3) 4) → map to A B C D respectively.
  - Options labeled (A) (B) (C) (D) → use A B C D directly.
  - When option_key is set, the bbox must tightly wrap ONLY that single option's graph.
- is_table_cell: true ONLY if the image is inside a printed table grid.
- For a WHOLE TABLE (e.g. Column-I / Column-II matching tables): treat the entire table as ONE image.
  Set is_table_cell=false, option_key=null. The bbox MUST span the FULL WIDTH of the table (x0 at left
  border, x1 at right border) and FULL HEIGHT (y0 at top border, y1 at bottom border including all rows).
  Do NOT clip the right column — x1 must reach the rightmost cell of the table.
- latex_content: ONLY for pure math equation images — write the LaTeX string.
- Skip: watermarks, page numbers, decorative borders, logos, header/footer art.
- If no real content images exist on the page, return {"images": []}.

Return ONLY the JSON object. No markdown, no explanation."""


SCANNED_PAGE_PROMPT = """You are a precise exam paper parser. Extract EVERY question from this page completely and accurately.

Return ONLY valid JSON:
{
  "questions": [
    {
      "number": <int — the printed question number>,
      "question_text": "<COMPLETE question text. Include all sub-parts, conditions, given data. Write ALL math/formulas in LaTeX: inline $...$ or display $$...$$. Do NOT truncate.>",
      "options": [
        {"key": "A", "text": "<full option text, LaTeX for math>", "has_image": <bool>},
        {"key": "B", "text": "...", "has_image": <bool>},
        {"key": "C", "text": "...", "has_image": <bool>},
        {"key": "D", "text": "...", "has_image": <bool>}
      ],
      "answer_key": "<see rules below>",
      "question_type": "<see rules below>"
    }
  ],
  "images": [
    {
      "bbox": [x0, y0, x1, y1],
      "question_number": <int — EXACT question number this image belongs to>,
      "option_key": <"A"/"B"/"C"/"D" or null>,
      "is_table": <true if this is a matching/column table>,
      "latex_content": <"$...$" only if the image IS a math equation, else null>,
      "description": "<one-line description of what the image shows>"
    }
  ]
}

QUESTION EXTRACTION RULES:
1. Extract EVERY question on this page — scan top to bottom, left to right. Do NOT stop early.
2. A typical exam page has 4–8 questions. If you find fewer than 3, look again carefully.
3. question_text: Copy the FULL text. Never truncate. Include all "Given:", "Consider:", conditions.
4. Math: ALL formulas, equations, symbols must be in LaTeX. Greek letters: $\alpha$, $\beta$ etc.
5. options: MCQ questions ALWAYS have 4 options (A B C D).
   - If options are printed as (1)(2)(3)(4) or 1. 2. 3. 4. → map to A B C D respectively.
   - If an option contains a diagram/graph → set has_image=true, text="" or describe briefly.
6. answer_key rules:
   - Single correct MCQ: "A" or "B" or "C" or "D"
   - Multiple correct MCQ: "A,C" or "A,B,D" etc.
   - Numerical integer: the integer value as string e.g. "12"
   - Numerical decimal: e.g. "3.14"
   - If answer is not printed on this page: null
   - Answer key section at page bottom (e.g. "1.(B) 2.(A)..."): extract each answer for its question.
7. question_type:
   - MCQ_SINGLE: one correct option
   - MCQ_MULTIPLE: one or more correct options ("one or more", "may be correct")
   - NUMERICAL_INTEGER: answer is an integer (no options)
   - NUMERICAL_DECIMAL: answer is a decimal (no options)
   - MATRIX_MATCH: matching/Column-I Column-II / List-I List-II
   - PARAGRAPH_BASED: questions based on a common passage
8. If a question is partially cut off at page edge: still extract what is visible, mark question_text ending with "[continued on next page]".
9. Cover pages, instructions, blank pages: return {"questions": [], "images": []}.

IMAGE EXTRACTION RULES:
1. Find EVERY diagram, graph, figure, circuit, chemical structure, geometric shape on this page.
2. bbox: [x0, y0, x1, y1] as INTEGERS on 0–1000 grid. (0,0)=top-left. Crop TIGHTLY — no whitespace padding.
3. question_number: Look at the question label DIRECTLY above or adjacent to the image. Use that EXACT number.
   - If image is between Q5 text and Q6 text → it belongs to Q5.
   - If image is inside an option row → set option_key to A/B/C/D accordingly.
4. option_key: Set ONLY when image IS an answer option graphic (inside option A/B/C/D block).
   - Options labeled 1/2/3/4 → A/B/C/D mapping.
5. is_table: true for Column-I/Column-II, List-I/List-II matching tables. Bbox must cover the FULL table.
6. SKIP: page numbers, exam headers, watermarks, logos, decorative borders, ruling lines.
7. One entry per image. Four option-graphs → four separate entries with option_key A/B/C/D.

Return ONLY the JSON object. No markdown code fences. No explanation text before or after."""


# ─────────────────────── core API call ───────────────────────────────────────

def _parse_retry_after(err_str: str) -> Optional[float]:
    """Extract wait seconds from Groq 429 error message."""
    m = re.search(r'try again in (\d+)m(\d+(?:\.\d+)?)s', err_str)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r'try again in (\d+(?:\.\d+)?)s', err_str)
    if m:
        return float(m.group(1))
    m = re.search(r'try again in (\d+(?:\.\d+)?)ms', err_str)
    if m:
        return float(m.group(1)) / 1000.0
    return None


async def _call_vision(
    api_key: str,
    limiter: DualRateLimiter,
    image_bytes: bytes,
    prompt: str,
    max_tokens: int = 3500,
    max_retries: int = 6,
) -> str:
    """Call Groq vision model using a specific API key and its own rate limiter."""
    client = AsyncGroq(api_key=api_key)
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{b64}"

    for attempt in range(max_retries):
        try:
            await limiter.acquire(token_cost=max_tokens)
            response = await client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            err_str = str(e)

            if "tokens per day" in err_str or "TPD" in err_str:
                raise _TPDExhaustedError()

            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = _parse_retry_after(err_str)
                if wait is None:
                    wait = min(2 ** attempt, 60)
                wait += 0.5
                logger.warning(f"429 on attempt {attempt + 1} — sleeping {wait:.1f}s")
                await asyncio.sleep(wait)
                continue

            logger.warning(f"Vision attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2)

    logger.error("All Groq vision retries exhausted.")
    return '{"questions": [], "images": []}'


# ─────────────────────── response parsing ────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"questions": [], "images": []}


def _normalise_bbox(bbox: list) -> list[int] | None:
    try:
        normalised = []
        for v in bbox:
            v = float(v)
            normalised.append(int(round(v * 1000)) if 0 < v < 1 else int(round(v)))
        x0, y0, x1, y1 = normalised
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(1000, x1), min(1000, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]
    except Exception:
        return None


# ─────────────────────── public API ──────────────────────────────────────────

def _build_page_result(page_num: int, parsed: dict) -> PageVisionResult:
    """Convert Groq's parsed JSON into a PageVisionResult."""
    result = PageVisionResult(page_num=page_num)
    for q in parsed.get("questions", []):
        result.extracted_questions.append(q)
    for img_data in parsed.get("images", []):
        bbox_raw = img_data.get("bbox")
        if not bbox_raw or not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            continue
        bbox = _normalise_bbox(bbox_raw)
        if bbox is None:
            continue
        result.image_locations.append(ImageLocation(
            bbox=bbox,
            question_number=img_data.get("question_number"),
            option_key=img_data.get("option_key"),
            is_table_cell=bool(img_data.get("is_table", False) or img_data.get("is_table_cell", False)),
            table_row=img_data.get("table_row"),
            table_col=img_data.get("table_col"),
            latex_content=img_data.get("latex_content"),
            description=img_data.get("description", ""),
        ))
    return result


def _split_page_overlapping(render_bytes: bytes) -> tuple[bytes, bytes, float]:
    """
    Split page into two OVERLAPPING halves (20% overlap zone).

    Top crop  : rows 0%  → 60% of page height
    Bottom crop: rows 40% → 100% of page height

    Any question crossing the midpoint appears in both halves → never missed.
    Returns (top_bytes, bottom_bytes, bottom_y_start_fraction).
    """
    from PIL import Image as _PIL
    import io as _io

    img = _PIL.open(_io.BytesIO(render_bytes))
    w, h = img.size
    top_end   = int(h * 0.60)
    bot_start = int(h * 0.40)

    buf_top, buf_bot = _io.BytesIO(), _io.BytesIO()
    img.crop((0, 0,         w, top_end)).save(buf_top, format="PNG")
    img.crop((0, bot_start, w, h      )).save(buf_bot, format="PNG")
    return buf_top.getvalue(), buf_bot.getvalue(), bot_start / h


def _remap_bboxes(parsed: dict, y_start_frac: float, y_end_frac: float) -> dict:
    """
    Remap image bboxes from a crop's local 0-1000 grid back to the full-page grid.

    y_start_frac: top of crop as fraction of full page (e.g. 0.0 for top half, 0.4 for bottom)
    y_end_frac:   bottom of crop as fraction (e.g. 0.6 for top half, 1.0 for bottom)
    """
    span = y_end_frac - y_start_frac
    for img in parsed.get("images", []):
        bbox = img.get("bbox")
        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            img["bbox"] = [
                x0,
                int((y_start_frac + y0 / 1000.0 * span) * 1000),
                x1,
                int((y_start_frac + y1 / 1000.0 * span) * 1000),
            ]
    return parsed


def _richer_question(q1: dict, q2: dict) -> dict:
    """Return whichever of two duplicate questions has more content."""
    score = lambda q: (
        len(q.get("question_text", "")) +
        len(q.get("options", [])) * 20 +
        (10 if q.get("answer_key") else 0)
    )
    return q1 if score(q1) >= score(q2) else q2


async def _process_page(
    page,
    api_key: str,
    limiter: DualRateLimiter,
    is_scanned: bool,
    key_label: str,
) -> tuple[int, PageVisionResult]:
    """
    Process a page using overlapping top/bottom splits (60%/60% with 20% overlap).

    Each crop has ~3-4 questions — comfortably within 3500 tokens.
    The 20% overlap zone ensures questions spanning the midpoint appear in both
    halves and are merged (keeping the richer/more complete version).
    """
    MAX_TOKENS = 3500

    try:
        top_bytes, bot_bytes, bot_frac = _split_page_overlapping(page.render_bytes)

        # Sequential under same rate limiter — 2 calls per page
        raw_top = await _call_vision(api_key, limiter, top_bytes, SCANNED_PAGE_PROMPT, max_tokens=MAX_TOKENS)
        raw_bot = await _call_vision(api_key, limiter, bot_bytes, SCANNED_PAGE_PROMPT, max_tokens=MAX_TOKENS)

        # Remap image bboxes back to full-page coordinates
        p_top = _remap_bboxes(_parse_json_response(raw_top), y_start_frac=0.0,      y_end_frac=0.60)
        p_bot = _remap_bboxes(_parse_json_response(raw_bot), y_start_frac=bot_frac, y_end_frac=1.0)

        # Merge questions — on duplicate question number, keep the richer version
        result = PageVisionResult(page_num=page.page_num)
        q_map: dict[int, dict] = {}
        for q in p_top.get("questions", []) + p_bot.get("questions", []):
            num = q.get("number")
            if num is None:
                result.extracted_questions.append(q)
                continue
            if num in q_map:
                q_map[num] = _richer_question(q_map[num], q)
            else:
                q_map[num] = q
        result.extracted_questions.extend(sorted(q_map.values(), key=lambda q: q.get("number", 0)))

        # Merge images — deduplicate by (question_number, option_key, approx bbox)
        seen_imgs: set = set()
        for parsed in (p_top, p_bot):
            for img_data in parsed.get("images", []):
                bbox = _normalise_bbox(img_data.get("bbox", []))
                if bbox is None:
                    continue
                # Deduplicate: same question + option + similar y position
                dedup_key = (
                    img_data.get("question_number"),
                    img_data.get("option_key"),
                    bbox[1] // 50,   # y bucket (50-unit tolerance)
                )
                if dedup_key in seen_imgs:
                    continue
                seen_imgs.add(dedup_key)
                result.image_locations.append(ImageLocation(
                    bbox=bbox,
                    question_number=img_data.get("question_number"),
                    option_key=img_data.get("option_key"),
                    is_table_cell=bool(img_data.get("is_table", False) or img_data.get("is_table_cell", False)),
                    table_row=img_data.get("table_row"),
                    table_col=img_data.get("table_col"),
                    latex_content=img_data.get("latex_content"),
                    description=img_data.get("description", ""),
                ))

        logger.info(
            f"[{key_label}] page {page.page_num} → "
            f"{len(result.extracted_questions)}q {len(result.image_locations)}img"
        )
        return page.page_num, result

    except _TPDExhaustedError:
        logger.warning(f"[{key_label}] TPD quota hit on page {page.page_num}")
        raise
    except Exception as e:
        logger.error(f"[{key_label}] page {page.page_num} failed: {e}")
        return page.page_num, PageVisionResult(page_num=page.page_num)


async def _worker(
    queue: asyncio.Queue,
    results: dict[int, PageVisionResult],
    api_key: str,
    limiter: DualRateLimiter,
    is_scanned: bool,
    key_label: str,
):
    """
    Worker coroutine — pulls pages from shared queue until empty or TPD hit.
    If this key's TPD is exhausted, remaining pages stay in queue for other workers.
    """
    while True:
        try:
            page = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        try:
            page_num, vr = await _process_page(page, api_key, limiter, is_scanned, key_label)
            results[page_num] = vr
            queue.task_done()
        except _TPDExhaustedError:
            logger.warning(f"[{key_label}] TPD exhausted — exiting worker, {queue.qsize()} pages remain in queue")
            queue.task_done()
            break
        except Exception as e:
            logger.error(f"[{key_label}] page {page.page_num} error: {e}")
            results[page.page_num] = PageVisionResult(page_num=page.page_num)
            queue.task_done()


async def analyse_all_image_pages(
    pages: list,
    is_scanned: bool = False,
) -> dict[int, "PageVisionResult"]:
    """
    Analyse ALL pages using SCANNED_PAGE_PROMPT regardless of PDF type.

    Both digital and scanned PDFs go through Groq Vision for question/option/answer
    extraction — format-agnostic, no regex dependency.

    All API keys share a single queue. If one key's TPD exhausts, others continue.
    """
    # Send every page — every page has render_bytes now
    all_pages = [p for p in pages if p.render_bytes is not None]
    logger.info(
        f"Vision: {len(all_pages)} pages, {len(GROQ_API_KEYS)} worker(s) — SCANNED_PAGE_PROMPT"
    )

    if not all_pages:
        return {}

    # Shared queue — all workers pull from it
    queue: asyncio.Queue = asyncio.Queue()
    for page in all_pages:
        await queue.put(page)

    results: dict[int, PageVisionResult] = {}

    # Launch one worker per key — all pull from the same queue
    # is_scanned=True forces SCANNED_PAGE_PROMPT in _process_page
    await asyncio.gather(*[
        _worker(queue, results, key, _rate_limiters[key], is_scanned=True, key_label=f"key{i+1}")
        for i, key in enumerate(GROQ_API_KEYS)
    ])

    if not queue.empty():
        remaining = queue.qsize()
        logger.warning(f"Vision incomplete — {remaining} pages unprocessed (all keys exhausted TPD)")

    return results
