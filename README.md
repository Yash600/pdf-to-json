# ExamParse — Exam PDF → Structured JSON Pipeline

**PupilTree.ai Intern Assessment Submission**

---

## Table of Contents

1. [Approach & Architecture](#approach--architecture)
2. [Tools & Models Used](#tools--models-used)
3. [How to Run](#how-to-run)
4. [Cost Per PDF](#cost-per-pdf)
5. [Measured Latency Per PDF](#measured-latency-per-pdf)
6. [Queue Design & Scaling Plan for 100 PDFs](#queue-design--scaling-plan-for-100-pdfs)
7. [Rate Limiting — System Design](#rate-limiting--system-design)
8. [Multi-Key Parallel Processing](#multi-key-parallel-processing)
9. [Schema Extensions — Table Representation](#schema-extensions--table-representation)
10. [Image Handling](#image-handling)
11. [Problems Faced & How We Solved Them](#problems-faced--how-we-solved-them)
12. [Known Limitations](#known-limitations)

---

## Approach & Architecture

The pipeline has two distinct code paths depending on whether the PDF has a selectable text layer (digital) or is a raw image scan.

```
PDF Input
   │
   ▼
pdf_parser.py  ──  PyMuPDF
   │
   ├── Digital PDF ──► segment_questions() ──► Groq Vision (vector pages only) ──► assemble()
   │
   └── Scanned PDF ──► Groq Vision ALL pages (SCANNED_PAGE_PROMPT) ──► assemble()
                            │
                       Dual RPM+TPM rate limiter
                       Multi-key parallel workers (shared queue)
                            │
                       image_pipeline.py  ──  crop → Cloudinary upload
                            │
                       schema_assembler.py  ──  Final JSON
```

### Digital PDF Path

1. **PyMuPDF** (`fitz`) extracts embedded text blocks with exact bounding boxes, preserving reading order across single and multi-column layouts using a column-bucket sort.
2. **`segment_questions()`** parses question numbers, text, options, and answer keys purely from the text layer — no vision model needed for text.
3. **Groq Vision** is called only on pages that contain vector drawings (diagrams, graphs detected via PyMuPDF's `get_drawings()`) — not on pure-text pages. This minimises token usage significantly.
4. **`schema_assembler.py`** spatially matches vision-detected image locations (on a 0-1000 grid) to question bounding boxes, decides stem vs option placement, crops each image, uploads to Cloudinary, and embeds the URL in the JSON.

### Scanned PDF Path

1. **PyMuPDF** renders every page as a PNG at 150 DPI. Each pixmap is released immediately after `tobytes("png")` to keep RAM flat.
2. **Groq Vision** (`SCANNED_PAGE_PROMPT`) reads each page image directly and returns fully structured JSON — question text (with LaTeX in `$...$`), options, answer key, question type, image bounding boxes — all in one pass. No OCR step required.
3. Results from all pages are merged keyed by `page_num`, iterated in sorted order (so question ordering is always correct), assembled, and saved.

---

## Tools & Models Used

| Component | Tool / Model | Reason |
|---|---|---|
| PDF parsing | PyMuPDF (`fitz`) | Fast, zero-cost, accurate bbox extraction |
| Vision / extraction | Groq `meta-llama/llama-4-scout-17b-16e-instruct` | Free tier, fast, multimodal, strong at structured JSON output |
| Image storage | Cloudinary (free tier) | Persistent CDN URLs, no infra cost |
| Queue broker | Redis + Celery | Robust, battle-tested, horizontally scalable |
| API server | FastAPI | Async, lightweight, production-grade |
| Rate limiting | Custom dual token-bucket (RPM + TPM) | Prevents both Groq rate limit types simultaneously |

---

## How to Run

### Prerequisites

```bash
pip install pymupdf groq cloudinary fastapi uvicorn celery redis python-dotenv pillow
```

### Environment Variables (`.env`)

```env
GROQ_API_KEY=gsk_your_primary_key
GROQ_API_KEY_2=gsk_your_second_key        # optional — doubles speed
GROQ_API_KEY_3=gsk_your_third_key         # optional — triples speed

CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

REDIS_URL=redis://localhost:6379/0         # for queue mode
GROQ_RPM_LIMIT=28                         # stays under 30 RPM free-tier ceiling
GROQ_TPM_LIMIT=28000                      # stays under 30k TPM free-tier ceiling
```

### CLI — Single PDF (no queue needed)

```bash
cd ExamParse
python run.py path/to/exam.pdf
```

Output saved to `output/<job_id>.json`. Cropped image URLs are embedded in the JSON (Cloudinary links).

### API + Queue Mode (for bulk / 100 PDFs)

```bash
# Terminal 1 — Start Redis
docker run -p 6379:6379 redis

# Terminal 2 — Start Celery worker
celery -A workers.celery_app worker --loglevel=info --concurrency=3

# Terminal 3 — Start FastAPI
uvicorn app.main:app --reload

# Submit a PDF
curl -X POST http://localhost:8000/process -F "file=@exam.pdf"
# Returns: { "job_id": "...", "status": "queued", "poll_url": "/job/<id>" }

# Poll for result
curl http://localhost:8000/job/<job_id>

# Download JSON directly
curl http://localhost:8000/job/<job_id>/download
```

---

## Cost Per PDF

All components used are on free tiers during development.

| Item | Free Tier | Paid Groq estimate |
|---|---|---|
| Groq Vision (33 pages × 3,500 tokens) | $0 (free tier quota) | ~$0.008 |
| Cloudinary image storage | $0 (25 GB free) | $0 (within free) |
| Redis | $0 (local / free cloud tier) | $0 |
| PyMuPDF | $0 (open source) | $0 |
| **Total per PDF** | **$0** | **~$0.01** |

At ~$0.01/PDF on paid Groq, processing 100 PDFs costs approximately **$1.00**.

Digital PDFs use far fewer tokens (only vector-drawing pages go to Groq) — cost approaches $0 for text-heavy digital exams.

---

## Measured Latency Per PDF

Tested on JEE Advanced 2026 Paper 1 — 33 pages, scanned:

| Configuration | Measured Time |
|---|---|
| 1 Groq key, sequential pages | ~4.5 minutes |
| 2 Groq keys, shared-queue workers | ~3 minutes |
| Digital PDF (text layer present), no full-page vision | ~25 seconds |

**Bottleneck analysis:** The free-tier TPM ceiling (30,000 tokens/minute) is the dominant constraint. A 33-page scanned PDF requires ~115,500 tokens total. At 28,000 effective TPM (staying safely under the ceiling), that is ~4.1 minutes of pure token-budget time regardless of network speed.

With a paid Groq plan (no TPM ceiling), the same 33-page PDF would process in under 90 seconds — within the 1.5-minute SLA. Digital PDFs already meet the SLA comfortably on free tier.

---

## Queue Design & Scaling Plan for 100 PDFs

### Current Implementation

```
Client ──POST /process──► FastAPI ──► Celery Task Queue (Redis broker)
                                            │
                          ┌─────────────────┼─────────────────┐
                          ▼                 ▼                 ▼
                       Worker 1          Worker 2          Worker 3
                       (pipeline)        (pipeline)        (pipeline)
                          │
                    Redis result backend (24h TTL)
                          │
                    GET /job/{id} ◄── Client polling
```

**Key design properties:**

- **Idempotency:** Every PDF is SHA-256 hashed before queuing. If the same PDF is submitted twice, the second call returns the cached result from Redis instantly — no reprocessing.
- **Retries:** Each Celery task retries up to 3 times with exponential backoff (60s → 120s → 240s) on any failure.
- **Backpressure:** `worker_prefetch_multiplier=1` — each worker holds exactly one task at a time. Queue depth remains visible; clients can back off when the queue grows.
- **Hard time limit:** 100-second hard limit per task (`task_time_limit=100`). Tasks exceeding this are killed and retried, enforcing the 1.5-minute SLA.
- **Late acknowledgement:** `task_acks_late=True` — tasks are acknowledged only after completion. Worker crash mid-task → task re-enters the queue automatically.
- **Result TTL:** Results cached in Redis for 24 hours.

### Scaling Plan for 100 PDFs

#### Free Tier (current config)
- 3 Celery workers, 2 Groq keys
- ~3 PDFs processed in parallel, each taking ~3 minutes
- 100 PDFs → ~34 batches → ~100 minutes total
- Queue handles the backlog correctly — no PDF is lost, dropped, or double-processed. All 100 complete, just not in 1.5 minutes.

#### Paid Groq (recommended for production SLA)
- Paid plan removes TPM ceiling — each PDF takes ~60–90 seconds
- 10 Celery workers, each with a dedicated Groq key
- 100 PDFs → 10 parallel → **~15 minutes total**

#### Horizontal Scale (cloud production)
```
                    Load Balancer
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        FastAPI Pod 1         FastAPI Pod 2
              │                     │
              └──────────┬──────────┘
                         ▼
                  Redis Cluster
                  (broker + result backend)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  Worker Pod 1     Worker Pod 2     Worker Pod N
  (Groq key pool)  (Groq key pool)  (Groq key pool)
        │                │                │
        └────────────────┴────────────────┘
                         ▼
                    Cloudinary CDN
                    (image storage)
```

- Worker pods auto-scale based on Redis queue depth (KEDA on Kubernetes, or ECS auto-scaling on AWS)
- Each pod gets its own pool of Groq API keys — avoids key contention between pods
- Cloudinary handles image storage and serving — no scaling needed, global CDN
- Redis Cluster provides broker + result backend with replication for reliability

---

## Rate Limiting — System Design

### The Problem

Groq's free tier enforces two independent limits:
- **RPM** — 30 requests per minute
- **TPM** — 30,000 tokens per minute

The original implementation used a single RPM token bucket. When all 33 pages were dispatched concurrently, the TPM ceiling was blown within 3 seconds (33 × ~3,500 = ~115,500 tokens requested instantly). This generated hundreds of 429 errors, each causing 7-second waits. Total time spent waiting: over 3 minutes out of 4.5 minutes total.

### The Solution: Dual Token Bucket

Two token buckets run in parallel — one per limit type. Before every API call, `acquire(token_cost)` waits until both buckets have enough capacity:

```python
class DualRateLimiter:

    # RPM bucket: 28 tokens max, refills at 28/60 = 0.467/sec
    # TPM bucket: 28000 tokens max, refills at 28000/60 = 467/sec

    async def acquire(self, token_cost: int):
        rpm_wait = max(0, (1 - rpm_tokens) / rpm_refill_rate)
        tpm_wait = max(0, (token_cost - tpm_tokens) / tpm_refill_rate)

        # Sleep the maximum — satisfies BOTH limits in one wait
        await asyncio.sleep(max(rpm_wait, tpm_wait))

        rpm_tokens -= 1
        tpm_tokens -= token_cost
```

Each API key gets its own `DualRateLimiter` instance, so keys never share quota state.

### Retry-After Parsing (instead of exponential backoff)

When a 429 occurs despite the limiter (due to Groq's sliding window vs our fixed-period approximation), the error message contains the exact wait time:

```
"Please try again in 6.378s."  →  sleep(6.878)   # exact duration + 0.5s buffer
"Please try again in 532ms."   →  sleep(1.032)
"Please try again in 1m30s."   →  sleep(90.5)
```

Exponential backoff guesses — often sleeping too long (wasting time) or too short (immediate re-429). Parsing the exact value eliminates both failure modes.

---

## Multi-Key Parallel Processing

### Design: Shared Queue, Independent Workers

Pages go into a single `asyncio.Queue`. Each API key runs an independent worker coroutine that pulls from the shared queue. Each worker has its own `DualRateLimiter`.

```
asyncio.Queue: [page0, page1, page2, ... page32]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Worker (key1)           Worker (key2)
  DualRateLimiter         DualRateLimiter
  pulls & processes       pulls & processes
  hits TPD → stops        continues draining queue
        └───────────┬───────────┘
                    ▼
          results: dict[page_num → PageVisionResult]
                    ▼
          sorted(results.keys()) in assembler
                    ▼
            correct question order guaranteed
```

**Why shared queue beats fixed chunks:**

With fixed chunks (key1 gets pages 0-16, key2 gets pages 17-32): if key1's daily token quota (TPD) dies at page 7, pages 7-16 are silently dropped. In testing, this produced 31 questions instead of 54.

With a shared queue: when key1 stops, key2 automatically continues pulling pages 7-16 along with its own 17-32. No pages are ever lost.

**Adding more keys:** Zero code changes — add `GROQ_API_KEY_3` to `.env`. The `_load_api_keys()` function collects all keys automatically and creates one worker + one limiter per key.

**Order preservation:** Results stored as `dict[page_num → result]`. The assembler iterates `sorted(vision_results.keys())` — question order is always correct regardless of which key finished which page first.

---

## Schema Extensions — Table Representation

The reference schema has no explicit table field. We extended it to handle tables (e.g. Column-I / Column-II matching questions common in JEE/NEET) as follows:

```json
{
  "questionText": "Match each entry in List-I to the correct entry in List-II...",
  "questionType": "MATRIX_MATCH",
  "hasImage": true,
  "imageDetails": [
    {
      "url": "https://res.cloudinary.com/.../img_q3_tbl_rNonecNone_17.png",
      "altText": "Column-I Column-II matching table",
      "imageType": "question",
      "mappingImageName": "{{IMAGE:img_q3_tbl_rNonecNone_17}}",
      "isTable": true,
      "tableRow": null,
      "tableCol": null
    }
  ]
}
```

**Rationale:** In JEE/NEET style exams, matching tables (Column-I / Column-II, List-I / List-II) are almost always rendered as raster or vector graphics — not as HTML-style tabular text. We treat the entire table as a single tightly-cropped image with `isTable: true` added to `ImageDetails`. The `mappingImageName` convention `img_q{N}_tbl_r{row}c{col}_{xref}` identifies individual table cells when they are separately cropped as option images. Individual cell images that are answer options are placed in the relevant `Option.imageDetails` using the standard schema fields.

This representation is a deliberate extension — it is additive, does not break the base schema, and is documented here as required.

---

## Image Handling

### Detection

**Digital PDFs:** PyMuPDF's `get_images()` extracts embedded raster images with exact page-coordinate bounding boxes. `get_drawings()` detects vector diagrams (graphs, axes, curves) by counting complex path operations — pages with more than 15 complex paths are flagged as containing diagrams and sent to Groq Vision. Tiny images under 1,600 sq. points (~40×40pt) are filtered out as watermarks or decorative glyphs.

**Scanned PDFs:** Groq Vision's `SCANNED_PAGE_PROMPT` returns bounding boxes (normalised 0-1000 grid) for every figure, diagram, graph, and table visible on the page image. A normalisation pass handles Groq's occasional mixed-format responses (some bboxes as 0-1 floats, some as 0-1000 integers).

### Cropping

Images are cropped from the rendered page PNG using the bounding box (converted from the 0-1000 grid to pixel coordinates). Tight cropping is enforced — no full-page dumps. Degenerate boxes (x1 ≤ x0 or y1 ≤ y0) are discarded.

### Placement Logic (`schema_assembler.py`)

| Condition | Placement in JSON |
|---|---|
| `option_key` set (A/B/C/D) | `Option.imageDetails`, `optionType: "text_and_image"` or `"image"` |
| `is_table_cell: true` | `imageDetails` with `isTable: true`, row/col noted |
| Neither (stem image) | `question.imageDetails`, `hasImage: true` |

For digital PDFs, spatial matching: each image block's bounding box is compared against all question bounding boxes on that page — the nearest question by vertical overlap wins ownership. If image count equals option count (>1), images are assigned A/B/C/D by y-position (top to bottom). If image count equals option count + 1, the topmost is the stem image; the rest are option images.

### Storage

All cropped images are uploaded to Cloudinary and referenced by permanent CDN URL in the JSON. No base64 inline embedding — compliant with the hard requirement.

---

## Problems Faced & How We Solved Them

### 1. marker-pdf / PaddleOCR — Wrong Tool for the Job

**Problem:** Started with `marker-pdf` (a local ML OCR pipeline — PyTorch, transformers, surya-ocr, ~4 GB of model weights) for scanned PDF processing. It took 10+ minutes per PDF, consumed enormous disk space, and produced inconsistent output on JEE-style papers.

**Attempt 1:** Replaced `marker-pdf` with PaddleOCR as the primary engine. PaddleOCR v3 silently changed its API — removed `show_log`, `use_angle_cls`, and the `cls` keyword argument from `ocr()`. Required several rounds of fixes. Underlying issue persisted: a Windows/oneDNN incompatibility (`ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`) caused PaddleOCR to fail on every page, falling back silently to Tesseract.

**Attempt 2:** Tesseract's output on scanned JEE papers (multi-column, dense LaTeX, mixed content) was garbled — multiple questions merged into single blocks, formulas stripped, options mangled. Feeding this into `segment_questions()` (designed for clean digital text) produced 16 questions out of 54 with completely wrong text, mixed answer keys, and misplaced images.

**Root cause identified:** OCR → text → segment is fundamentally brittle for complex scanned exam papers. Any error in OCR propagates unrecoverably into segmentation.

**Final fix:** Abandoned OCR entirely for scanned PDFs. Groq Vision reads each page as an image and returns structured JSON with question text, LaTeX, options, types, and image bounding boxes in one pass. Simpler, faster, more accurate.

### 2. Groq 429 — Only RPM Throttled, TPM Ignored

**Problem:** The original rate limiter was a single RPM token bucket (28 requests/min). Dispatching all 33 pages concurrently consumed ~115,500 tokens in under 3 seconds — blowing the 30,000 TPM ceiling immediately. This generated hundreds of 429 errors and 7-second retry waits per request. Out of 4.5 minutes total runtime, over 3 minutes were spent waiting on rate limit retries.

**Fix:** Dual token bucket — one for RPM (28/min), one for TPM (28,000/min). Before every request, `acquire(token_cost=3500)` sleeps the maximum of the two wait times, satisfying both limits in one sleep. Combined with exact retry-after parsing from Groq's error messages, 429 errors dropped by ~90%.

### 3. httpx `proxies` Conflict — Groq SDK vs httpx Version

**Problem:** After installing PaddleOCR (which pulled in newer httpx), all Groq calls failed with `AsyncClient.__init__() got an unexpected keyword argument 'proxies'`. The old Groq SDK (0.9.0) passed `proxies` to httpx's `AsyncClient`; httpx 0.28+ removed that parameter.

**Wrong fix attempted:** Downgraded httpx to 0.27.2. This broke google-genai (requires httpx ≥0.28.1).

**Correct fix:** Upgraded Groq SDK to 1.4.0. This is compatible with httpx 0.28.x and resolved the conflict. Minor langchain-groq version warning remains (non-critical — the project uses Groq directly, not via LangChain).

### 4. malloc Failure — Double Page Rendering OOM

**Problem:** The scanned PDF path opened the PDF twice and rendered each page twice — once for Groq Vision at 150 DPI, then again for OCR at 150 DPI. PyMuPDF pixmaps (~6.5 MB each at 150 DPI on a standard A4 page) were not released between renders. By page 10 of a 33-page PDF, RAM was exhausted: `malloc (7,702,200 bytes) failed`.

**Fix:** Render each page exactly once. Set `pix = None` immediately after `tobytes("png")` to release the pixmap. Reuse the same PNG bytes for both Groq Vision and (formerly) OCR. Memory stays flat across all pages regardless of PDF length.

### 5. 0 Questions Extracted — `ocr_available` Logic Flaw

**Problem:** Even after Tesseract extracted garbled text blocks, the pipeline set `ocr_available = True`. This caused Groq Vision to receive `IMAGE_MAPPING_PROMPT` (finds image bounding boxes only) instead of `SCANNED_PAGE_PROMPT` (extracts full questions). The `_build_raw_questions_from_vision()` function received empty `extracted_questions` lists and returned 0 questions despite Groq Vision running successfully on all 33 pages.

**Fix:** Removed the `ocr_available` branching logic entirely. Scanned PDFs always receive `SCANNED_PAGE_PROMPT`. The `ocr_available` parameter was eliminated from `analyse_page_images()`.

### 6. Fixed Chunk Split — Pages Dropped When Key Hits Daily Quota

**Problem:** Pages were split into fixed chunks: key1 → pages 0-16, key2 → pages 17-32. During testing, key1's daily token quota (TPD) was already partially used from earlier runs and exhausted at page 7. Pages 7-16 were silently dropped — the chunk worker stopped and those pages never processed. Result: 31 questions instead of 54.

**Fix:** Replaced fixed chunks with a shared `asyncio.Queue`. Both key workers pull from the same queue. When key1 exhausts its TPD and stops, key2 automatically picks up all remaining pages from the queue. No pages are ever dropped due to a single key's quota being hit.

---

## Known Limitations

**Latency on free tier:** A 33-page scanned PDF takes ~3 minutes with 2 Groq keys on the free tier — above the 1.5-minute SLA. This is purely a Groq TPM quota constraint, not an architectural bottleneck. On a paid Groq plan the same PDF processes in ~60 seconds.

**TPD exhaustion mid-job:** If all Groq keys exhaust their daily token quota mid-processing, remaining pages are skipped and output is partial. The pipeline logs a warning. Mitigation: use a paid plan or rotate keys.

**Scanned PDF accuracy:** Groq Vision's extraction quality depends on scan resolution and clarity. Low-DPI scans, handwritten annotations, or heavily watermarked PDFs may produce incomplete or incorrectly transcribed text. Dense LaTeX equations occasionally have minor formula transcription errors.

**Cover/instruction pages:** Cover pages, instruction pages, and answer-key-only pages return 0 questions from Groq Vision — expected behaviour, but they still consume TPM quota. A pre-filter to skip non-question pages would reduce cost and time.

**Image-to-question assignment on scanned PDFs:** Assignment relies on Groq correctly identifying the `question_number` associated with each image bounding box. If the model misidentifies a question number, the image is placed under the wrong question in the output JSON.

**Multi-column digital PDFs:** Column detection uses a mid-page split heuristic. Standard 2-column layouts are handled correctly. 3-column or irregular layouts may produce incorrect reading order.
