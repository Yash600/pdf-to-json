"""
Image Pipeline
--------------
1. Maps Groq vision bboxes (0-1000 grid) → actual pixel coords on rendered page
2. Crops images tightly using PIL
3. Uploads each crop to Cloudinary free tier
4. Returns ImageDetails-ready dicts with permanent URLs

Also handles direct embedded image extraction (from PyMuPDF) as a faster
path when the image is already extractable without cropping.
"""

import os
import io
import logging
from dataclasses import dataclass
from typing import Optional

import cloudinary
import cloudinary.uploader
from PIL import Image
from dotenv import load_dotenv

from .groq_vision import ImageLocation

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", ""),
    secure=True,
)

# Fallback: local file storage (if Cloudinary not configured)
LOCAL_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "images")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/images")


# ─────────────────────── data class ──────────────────────────────────────────

@dataclass
class StoredImage:
    url: str
    mapping_image_name: str   # stable id e.g. {{IMAGE:img_q3_opt_A}}
    alt_text: str
    image_type: str           # "question" or "option"
    figure_bbox: list[int]    # [x0,y0,x1,y1] on 0-1000 grid (dropped after assembly)
    local_path: Optional[str] = None


# ─────────────────────── upload helpers ──────────────────────────────────────

def _use_cloudinary() -> bool:
    return bool(
        os.getenv("CLOUDINARY_CLOUD_NAME") and
        os.getenv("CLOUDINARY_API_KEY") and
        os.getenv("CLOUDINARY_API_SECRET")
    )


def _upload_image(img_bytes: bytes, public_id: str) -> str:
    """Upload PNG bytes. Returns public URL. Falls back to local storage on failure."""

    if _use_cloudinary():
        try:
            result = cloudinary.uploader.upload(
                img_bytes,
                public_id=public_id,
                folder="exam_pipeline",
                resource_type="image",
                format="png",
                overwrite=True,
            )
            return result["secure_url"]
        except Exception as e:
            logger.warning(f"Cloudinary upload failed for {public_id}, falling back to local: {e}")

    # Local fallback (always used when Cloudinary is not configured or fails)
    os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
    filename = f"{public_id}.png"
    filepath = os.path.join(LOCAL_IMAGE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return f"{LOCAL_BASE_URL}/{filename}"


# ─────────────────────── bbox conversion ─────────────────────────────────────

def _bbox_0_1000_to_pixels(
    bbox: list[int],
    page_render_width: int,
    page_render_height: int,
    pad: int = 4,
) -> tuple[int, int, int, int]:
    """
    Convert bbox from 0-1000 normalised grid to pixel coordinates
    on the rendered page image.

    Args:
        pad: pixel padding added to each side for tight-but-safe crop
    """
    x0 = int(bbox[0] / 1000 * page_render_width) - pad
    y0 = int(bbox[1] / 1000 * page_render_height) - pad
    x1 = int(bbox[2] / 1000 * page_render_width) + pad
    y1 = int(bbox[3] / 1000 * page_render_height) + pad

    # Clamp to image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page_render_width, x1)
    y1 = min(page_render_height, y1)
    return x0, y0, x1, y1


# ─────────────────────── crop from page render ───────────────────────────────

def _trim_dark_borders(img: "Image.Image", dark_threshold: int = 30, min_trim_px: int = 5) -> "Image.Image":
    """
    Remove solid dark (near-black) strips from any edge of the image.
    NEET/JEE papers often have dark decorative borders that bleed into crops.

    Strategy: scan each edge inward until we find a row/col that is NOT
    predominantly dark. 'Predominantly dark' = >80% of pixels below threshold.
    """
    import numpy as np
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    def is_dark_strip(strip: "np.ndarray") -> bool:
        # strip shape: (pixels, 3)
        return float((strip.max(axis=1) < dark_threshold).mean()) > 0.80

    top, bottom, left, right = 0, h, 0, w

    # Top
    for i in range(h):
        if not is_dark_strip(arr[i]):
            top = i
            break

    # Bottom
    for i in range(h - 1, -1, -1):
        if not is_dark_strip(arr[i]):
            bottom = i + 1
            break

    # Left
    for j in range(w):
        if not is_dark_strip(arr[:, j]):
            left = j
            break

    # Right
    for j in range(w - 1, -1, -1):
        if not is_dark_strip(arr[:, j]):
            right = j + 1
            break

    # Only trim if we'd remove at least min_trim_px on any edge
    trimmed = top > min_trim_px or bottom < h - min_trim_px or \
              left > min_trim_px or right < w - min_trim_px
    if trimmed:
        img = img.crop((left, top, right, bottom))

    return img


def crop_image_from_render(
    render_bytes: bytes,
    bbox: list[int],
) -> bytes:
    """
    Crop a sub-image from a rendered page PNG using 0-1000 bbox.
    Automatically trims dark border strips (common in RCC/NEET papers).
    Returns PNG bytes of the crop.
    """
    page_img = Image.open(io.BytesIO(render_bytes)).convert("RGB")
    pw, ph = page_img.size

    x0, y0, x1, y1 = _bbox_0_1000_to_pixels(bbox, pw, ph)

    if x1 <= x0 or y1 <= y0:
        logger.warning(f"Invalid crop bbox after conversion: {bbox} → ({x0},{y0},{x1},{y1})")
        return render_bytes  # fallback: return full page

    crop = page_img.crop((x0, y0, x1, y1))
    crop = _trim_dark_borders(crop)
    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ─────────────────────── public functions ────────────────────────────────────

def process_image_location(
    loc: ImageLocation,
    render_bytes: bytes,
    job_id: str,
    image_index: int,
) -> StoredImage:
    """
    Crop the image from the page render and upload it.

    Args:
        loc: ImageLocation from Groq vision
        render_bytes: PNG bytes of the rendered page
        job_id: unique job identifier (used in naming)
        image_index: sequential index for this image in the document

    Returns:
        StoredImage with URL and metadata
    """
    # Build stable image name
    q_part = f"q{loc.question_number}" if loc.question_number else "qunk"
    opt_part = f"_opt{loc.option_key}" if loc.option_key else ""
    tbl_part = f"_tbl_r{loc.table_row}c{loc.table_col}" if loc.is_table_cell else ""
    mapping_name = f"{{{{IMAGE:img_{q_part}{opt_part}{tbl_part}_{image_index}}}}}"
    public_id = f"{job_id}_{q_part}{opt_part}{tbl_part}_{image_index}"

    # Determine image type
    image_type = "option" if loc.option_key else "question"

    # Crop
    try:
        img_bytes = crop_image_from_render(render_bytes, loc.bbox)
    except Exception as e:
        logger.error(f"Crop failed for {mapping_name}: {e}")
        img_bytes = render_bytes  # fallback

    # Upload
    try:
        url = _upload_image(img_bytes, public_id)
    except Exception as e:
        logger.error(f"Upload failed for {mapping_name}: {e}")
        url = ""

    # Alt text
    alt = loc.description or (
        f"Graph for option {loc.option_key}" if loc.option_key
        else f"Diagram for question {loc.question_number}"
    )

    return StoredImage(
        url=url,
        mapping_image_name=mapping_name,
        alt_text=alt,
        image_type=image_type,
        figure_bbox=loc.bbox,
        local_path=None,
    )


# ─────────────────────── public low-level helpers ────────────────────────────
# Exposed for schema_assembler's direct raster extraction path.
def points_to_0_1000(x0, y0, x1, y1, page_w, page_h) -> list[int]:
    """Convert page-point coordinates to 0-1000 normalised grid."""
    return _points_to_0_1000(x0, y0, x1, y1, page_w, page_h)


def upload_image(img_bytes: bytes, public_id: str) -> str:
    """Public wrapper around _upload_image."""
    return _upload_image(img_bytes, public_id)


# ─────────────────────── spatial grouping fallback ───────────────────────────

def _points_to_0_1000(x0, y0, x1, y1, page_w, page_h) -> list[int]:
    """Convert page-point coordinates to 0-1000 normalised grid."""
    return [
        int(x0 / page_w * 1000),
        int(y0 / page_h * 1000),
        int(x1 / page_w * 1000),
        int(y1 / page_h * 1000),
    ]


def cluster_image_rects(
    image_blocks: list,   # list of ImageBlock from pdf_parser
    page_width: float,
    page_height: float,
    gap_threshold: float = 60.0,  # points — rects within this distance are grouped
) -> list[list[int]]:
    """
    Group image fragment rects that are spatially close on the page,
    then return one merged bbox (0-1000 grid) per cluster.

    Algorithm: connected-components on a proximity graph.
    Two rects are "connected" if their bounding boxes are within
    gap_threshold points of each other in either x or y.

    Tool used: pure Python + PyMuPDF rect coordinates (no extra library needed).
    """
    if not image_blocks:
        return []

    rects = [(b.x0, b.y0, b.x1, b.y1) for b in image_blocks]
    n = len(rects)

    # Build adjacency: two rects are neighbours if they're within gap_threshold
    def close(r1, r2):
        # Expand each rect by gap_threshold and check overlap
        ax0, ay0, ax1, ay1 = r1[0]-gap_threshold, r1[1]-gap_threshold, r1[2]+gap_threshold, r1[3]+gap_threshold
        return not (r2[2] < ax0 or r2[0] > ax1 or r2[3] < ay0 or r2[1] > ay1)

    # Union-Find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if close(rects[i], rects[j]):
                union(i, j)

    # Collect clusters
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Merge each cluster into one bbox
    merged: list[list[int]] = []
    for indices in clusters.values():
        xs0 = min(rects[i][0] for i in indices)
        ys0 = min(rects[i][1] for i in indices)
        xs1 = max(rects[i][2] for i in indices)
        ys1 = max(rects[i][3] for i in indices)
        merged.append(_points_to_0_1000(xs0, ys0, xs1, ys1, page_width, page_height))

    return merged
