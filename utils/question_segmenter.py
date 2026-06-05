"""
Question Segmenter — Block-level parsing (PyMuPDF text blocks)
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from .pdf_parser import ParsedPDF


@dataclass
class RawOption:
    key: str
    text: str

@dataclass
class RawTableCell:
    text: str
    row: int
    col: int

@dataclass
class RawTable:
    headers: list
    rows: list

@dataclass
class RawQuestion:
    number: int
    question_text: str
    options: list = field(default_factory=list)
    answer_key: Optional[str] = None
    answer_explanation: Optional[str] = None
    question_type: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    paragraph_context: Optional[str] = None
    tables: list = field(default_factory=list)
    page_num: int = 0
    y0: float = 0.0
    y1: float = 0.0


# ── Regexes ──────────────────────────────────────────────────────────────────

RE_QBLOCK          = re.compile(r'^(?:Q[\s.]*)?[\(]?(\d{1,3})[.).]?\]?\s*[\|]?\s*\n?(.*)', re.DOTALL)
RE_QBLOCK_EMBEDDED = re.compile(r'(?:^|\n)\s*(?:Q[\s.]*)?[\(]?(\d{1,3})[.).]?\]?\s*[\|]?\s*([\s\S]+)')

RE_OPT_LINE  = re.compile(r'^[\(\[]?([A-D1-4])[\)\]]\s*(.*)', re.IGNORECASE)
RE_ANS_PREFIX = re.compile(
    r'^(?:Sol(?:ution)?\.?\s*|Ans(?:wer)?[:\.]?\s*|Ans:\s*|MathonGo\s+Answer\s+Key\s*:\s*)',
    re.IGNORECASE)
RE_ANS_BLOCK = re.compile(
    r'^(?:Sol(?:ution)?\.?\s*|Ans(?:wer)?[:\.]?\s*|Ans:\s*|MathonGo\s+Answer\s+Key\s*:\s*)(.+?)\s*$',
    re.IGNORECASE | re.DOTALL)
RE_ANS_INLINE = re.compile(
    r'\n(?:Sol(?:ution)?\.?\s*|Ans(?:wer)?[:\.]?\s*|Ans:\s*|MathonGo\s+Answer\s+Key\s*:\s*)(.+?)\s*$',
    re.IGNORECASE | re.DOTALL)


RE_SUBJECT  = re.compile(r'^(PHYSICS|CHEMISTRY|BIOLOGY|MATHEMATICS|BOTANY|ZOOLOGY)\s*$', re.IGNORECASE)
RE_NOISE    = re.compile(r'^(RCC\*+\s*){3,}|^www\.mathongo\.com\s*$')
RE_PAGE_HDR = re.compile(
    r'^(NEET|JEE Advanced|JEE|FINAL NEET|TEST BOOKLET|CODE\s*:|SET CODE|ROLL NUMBER|'
    r'Date\s*:|Time\s*:|ANSWER KEY|\(Held on)',
    re.IGNORECASE)
RE_MATRIX   = re.compile(r'(?:column|list)[\s\-]*(I{1,3}|[12])', re.IGNORECASE)
RE_PARA     = re.compile(r'^(?:passage|paragraph|read the following|directions?)', re.IGNORECASE)

# Answer key section header — when seen, stop parsing questions entirely
RE_ANS_KEY_SECTION = re.compile(r'answer\s*key|hints?\s*&?\s*solutions?', re.IGNORECASE)

# Single answer-key entries like "1. (A)" or "45. C" — noise in answer key section
RE_ANS_ENTRY = re.compile(r'^\d{1,3}[.)\s]+[\(\[]?[A-D1-4][\)\]]?\s*$')

# Instruction / booklet text that appears on test pages — not question content
RE_INSTR = re.compile(
    r'^(?:write this|question booklet|answer sheet|roll no|sr\.?\s*no|'
    r'serial no|candidate|invigilator|this question|space for rough|rough work)',
    re.IGNORECASE)

_KEY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}

def _nk(k):
    return _KEY_MAP.get(k.upper(), k.upper())

def _parse_answer_key(raw: str) -> str:
    """Parse answer key string.
    - '(C)'         -> 'C'
    - '(A), (B)'    -> 'A,B'
    - '1915'        -> '1915'  (numerical answer, keep as-is)
    - '0.46'        -> '0.46'
    """
    import re as _re
    raw = raw.strip().strip('()')
    # If it looks like a pure number (integer or decimal), return as-is
    if _re.match(r'^-?\d+(\.\d+)?$', raw.strip()):
        return raw.strip()
    # Extract A-D option letters
    keys = _re.findall(r'[A-D]', raw.upper())
    if keys:
        seen, mapped = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                mapped.append(k)
        return ",".join(mapped)
    # Fallback: return raw
    return raw.strip()

def _is_noise(t):
    t = t.strip()
    if not t: return True
    if RE_NOISE.match(t): return True
    if RE_PAGE_HDR.match(t): return True
    if RE_ANS_ENTRY.match(t): return True
    if RE_INSTR.match(t): return True
    if re.match(r'^\d{1,3}$', t): return True
    return False

def _clean(text):
    lines = [l for l in text.splitlines() if not _is_noise(l)]
    return " ".join(l.strip() for l in lines if l.strip())

def _block_has_options(raw):
    """Return True if ANY line in block matches an option pattern."""
    for line in raw.splitlines():
        if RE_OPT_LINE.match(line.strip()):
            return True
    return False

def _parse_opts(block_text):
    """
    Parse options from a block. Strips embedded Sol./Ans. lines.
    Returns (list[RawOption], answer_key_or_None).
    """
    embedded_ans = None

    # Strip trailing inline Sol./Ans.
    ans_m = RE_ANS_INLINE.search(block_text)
    if ans_m:
        embedded_ans = _parse_answer_key(ans_m.group(1).strip())
        block_text = block_text[:ans_m.start()]

    # Also strip standalone Sol./Ans. lines anywhere in the block
    clean_lines = []
    for line in block_text.splitlines():
        sm = RE_ANS_BLOCK.match(line.strip())
        if sm:
            if embedded_ans is None:
                embedded_ans = _parse_answer_key(sm.group(1).strip())
        else:
            clean_lines.append(line)
    block_text = "\n".join(clean_lines)

    opts = []
    cur_key = None
    cur_lines = []

    for line in block_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = RE_OPT_LINE.match(line)
        if m:
            if cur_key is not None:
                opts.append(RawOption(key=_nk(cur_key), text=" ".join(cur_lines).strip()))
            cur_key = m.group(1)
            cur_lines = [m.group(2).strip()] if m.group(2).strip() else []
        elif cur_key is not None:
            cur_lines.append(line)

    if cur_key is not None:
        opts.append(RawOption(key=_nk(cur_key), text=" ".join(cur_lines).strip()))

    return opts, embedded_ans


def _classify(rq):
    qt = rq.question_text.lower()
    if rq.tables: return "MATRIX_MATCH"
    if rq.paragraph_context: return "PARAGRAPH_BASED"
    if re.search(r'\bassert\w*\b.*\breas\w*\b', qt): return "ASSERTION_REASON"
    if re.search(r'\btrue\b.*\bfalse\b|\bfalse\b.*\btrue\b', qt): return "TRUE_FALSE"
    if rq.options:
        if re.search(r'more than one|one or more|multiple.*correct', qt): return "MCQ_MULTIPLE"
        return "MCQ_SINGLE"
    if re.search(r'nearest integer|integer value', qt): return "NUMERICAL_INTEGER"
    if re.search(r'decimal|upto.*decimal', qt): return "NUMERICAL_DECIMAL"
    return "NUMERICAL_INTEGER" if not rq.options else "MCQ_SINGLE"


def _parse_matrix(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cidxs = [i for i, l in enumerate(lines) if RE_MATRIX.search(l)]
    if len(cidxs) < 2: return None
    headers = [lines[i] for i in cidxs]
    data = lines[cidxs[-1]+1:]
    rows = []
    for i, line in enumerate(data):
        parts = re.split(r'\s{3,}|\t', line)
        if len(parts) >= 2:
            rows.append([RawTableCell(text=p.strip(), row=i, col=j) for j, p in enumerate(parts)])
        else:
            rows.append([RawTableCell(text=line, row=i, col=0)])
    return RawTable(headers=headers, rows=rows) if rows else None


# ── Main segmenter ────────────────────────────────────────────────────────────

def segment_questions(parsed_pdf: ParsedPDF):
    questions = []
    current_q = None
    cur_section = None
    cur_topic = None
    pending_para = []
    _answer_key_mode = False   # once True, stop parsing new questions

    def finalise():
        nonlocal current_q
        if current_q is None:
            return
        current_q.question_type = _classify(current_q)
        if pending_para:
            current_q.paragraph_context = " ".join(pending_para)
        questions.append(current_q)
        current_q = None

    def make_question(qnum, rest, pnum, y0, y1):
        """Create a RawQuestion, splitting options and answer out of rest."""
        opt_split = re.split(r'\n(?=[\(\[]?[A-D1-4][\)\]])', rest, maxsplit=1)
        qt = _clean(opt_split[0])
        embedded_opts, emb_ans = _parse_opts(opt_split[1]) if len(opt_split) > 1 else ([], None)
        rq = RawQuestion(
            number=qnum, question_text=qt, options=embedded_opts,
            subject=cur_section, topic=cur_topic,
            page_num=pnum, y0=y0, y1=y1,
        )
        if emb_ans:
            rq.answer_key = emb_ans
        if RE_MATRIX.search(qt):
            tbl = _parse_matrix(rest)
            if tbl:
                rq.tables.append(tbl)
        return rq

    all_blocks = []
    for page in parsed_pdf.pages:
        for block in page.text_blocks:
            all_blocks.append((block.text, page.page_num, block.y0, block.y1))

    for block_text, pnum, y0, y1 in all_blocks:
        raw = block_text.strip()
        if not raw or _is_noise(raw):
            continue

        # ── Answer key section — stop all question parsing ────────────────────
        if RE_ANS_KEY_SECTION.search(raw) and RE_PAGE_HDR.match(raw):
            _answer_key_mode = True
            finalise()
            continue

        if _answer_key_mode:
            continue

        # ── Subject header ────────────────────────────────────────────────────
        if RE_SUBJECT.match(raw):
            finalise()
            cur_section = raw.upper().strip()
            cur_topic = None
            pending_para.clear()
            continue

        # ── Standalone answer block ───────────────────────────────────────────
        ans_m = RE_ANS_BLOCK.match(raw)
        if ans_m and current_q is not None:
            current_q.answer_key = _parse_answer_key(ans_m.group(1).strip())
            finalise()
            continue

        # ── Question block: "N." at position 0 ───────────────────────────────
        q_m = RE_QBLOCK.match(raw)
        if q_m:
            finalise()
            current_q = make_question(int(q_m.group(1)), q_m.group(2).strip(), pnum, y0, y1)
            continue

        # ── Embedded question: "Topic\n N. text" ─────────────────────────────
        # Works whether current_q is open or not — always finalise first
        emb_m = RE_QBLOCK_EMBEDDED.search(raw)
        if emb_m:
            # Only treat as a new question if the embedded number makes sense
            # (avoids false positives on formula lines like "x = 1. something")
            before = raw[:emb_m.start()].strip()
            qnum = int(emb_m.group(1))
            rest = emb_m.group(2).strip()
            # Heuristic: the "before" part should look like a topic (short, no digits at start)
            # or be empty. Reject if before text starts with a digit or special char.
            is_topic_before = (not before) or (len(before) < 120 and before[0].isalpha())
            if is_topic_before:
                finalise()
                if before and len(before) < 80 and '\n' not in before:
                    cur_topic = _clean(before)
                current_q = make_question(qnum, rest, pnum, y0, y1)
                # If we already got an answer from make_question, flush immediately
                if current_q.answer_key:
                    finalise()
                continue

        # ── Option block (any line matches option pattern) ────────────────────
        if current_q is not None and _block_has_options(raw):
            opts, emb_ans = _parse_opts(raw)
            if opts:
                existing = {o.key for o in current_q.options}
                for o in opts:
                    if o.key not in existing:
                        current_q.options.append(o)
                        existing.add(o.key)
            if emb_ans and current_q.answer_key is None:
                current_q.answer_key = emb_ans
                finalise()
            continue

        # ── Topic / paragraph / continuation ─────────────────────────────────
        if current_q is None:
            if RE_PARA.match(raw):
                pending_para.append(_clean(raw))
            elif len(raw) < 80 and raw[0].isupper() and '\n' not in raw:
                cur_topic = raw.strip()
        else:
            cleaned = _clean(raw)
            if cleaned:
                current_q.question_text = (current_q.question_text + " " + cleaned).strip()

    finalise()

    # ── Post-process: remove invalid questions ───────────────────────────────
    # 1. Drop Q0 (spurious matches from table cells, figure labels, etc.)
    # 2. If the same question number appears multiple times (e.g. Q3 again
    #    inside an option block), keep only the first occurrence.
    seen_nums: set[int] = set()
    clean_questions: list[RawQuestion] = []
    for rq in questions:
        if rq.number == 0:
            continue
        if rq.number in seen_nums:
            # Merge text into the previous occurrence instead of discarding
            prev = next(q for q in reversed(clean_questions) if q.number == rq.number)
            if rq.question_text and rq.question_text not in prev.question_text:
                prev.question_text = (prev.question_text + " " + rq.question_text).strip()
            continue
        seen_nums.add(rq.number)
        clean_questions.append(rq)
    questions = clean_questions

    # Set y1 boundaries
    for i, rq in enumerate(questions):
        if i + 1 < len(questions) and questions[i+1].page_num == rq.page_num:
            rq.y1 = questions[i+1].y0
        else:
            rq.y1 = 9999.0

    return questions
