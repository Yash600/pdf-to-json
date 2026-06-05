"""
CLI Runner
----------
Run the pipeline directly on a PDF without starting the API.

Usage:
    python run.py path/to/exam.pdf

Output:
    output/<job_id>.json
    output/images/*.png  (or Cloudinary URLs in JSON)
"""

import sys
import uuid
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    job_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*50}")
    print(f"  PDF Pipeline — Job ID: {job_id}")
    print(f"  Input: {pdf_path}")
    print(f"{'='*50}\n")

    from app.pipeline import run_pipeline_and_save
    out_path = run_pipeline_and_save(pdf_path, job_id)

    # Print summary
    with open(out_path, encoding='utf-8') as f:
        result = json.load(f)

    print(f"\n{'='*50}")
    print(f"  ✓ Done in {result['elapsed_seconds']}s")
    print(f"  Questions extracted : {result['question_count']}")
    print(f"  Pages               : {result['page_count']}")
    print(f"  Scanned PDF         : {result['is_scanned']}")
    print(f"  Output JSON         : {out_path}")
    print(f"{'='*50}\n")

    # Print first question as preview
    if result["questions"]:
        q = result["questions"][0]
        print("── First question preview ──")
        print(f"  Type    : {q['questionType']}")
        print(f"  Text    : {q['questionText'][:120]}...")
        print(f"  Options : {len(q['options'])}")
        print(f"  hasImage: {q['hasImage']}")
        if q['hasImage']:
            for img in q['imageDetails']:
                print(f"  Stem image URL: {img['url']}")
        for opt in q.get('options', []):
            if opt.get('imageDetails'):
                print(f"  Option {opt['key']} image URL: {opt['imageDetails']['url']}")

if __name__ == '__main__':
    main()

