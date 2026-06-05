"""
Marker PDF Parser
-----------------
Converts PDFs to structured markdown + images using the marker-pdf library.
Supports both marker-pdf v0.2.x and v0.3.x APIs automatically.

Returns:
    markdown: str  — clean text with LaTeX in $...$ format
    images:   dict[str, bytes]  — {filename: png_bytes}
"""

import io
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Redirect HuggingFace model cache before any imports that trigger downloads
_hf_home = os.getenv("HF_HOME", r"D:\hf_cache")
os.environ["HF_HOME"] = _hf_home
os.environ["TRANSFORMERS_CACHE"] = _hf_home
os.environ["HF_HUB_CACHE"] = _hf_home
Path(_hf_home).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

# ── Model cache (loaded once at startup, reused per request) ──────────────────
_marker_models = None


def preload_models() -> None:
    """
    Pre-load marker-pdf models into memory.
    Call once at server/worker startup so the first PDF request
    doesn't pay the ~60s model loading penalty.
    """
    global _marker_models
    if _marker_models is not None:
        return
    logger.info("Pre-loading marker-pdf models...")
    try:
        _marker_models = _load_models_new_api()
        logger.info("marker-pdf models loaded (new API)")
    except (ImportError, AttributeError):
        try:
            _marker_models = _load_models_old_api()
            logger.info("marker-pdf models loaded (legacy API)")
        except ImportError:
            logger.warning("marker-pdf not installed — scanned PDFs will fall back")


def _load_models_new_api():
    from marker.models import create_model_dict
    return ("new", create_model_dict())


def _load_models_old_api():
    from marker.models import load_all_models
    return ("old", load_all_models())


def parse_with_marker(pdf_path: str) -> tuple[str, dict[str, bytes]]:
    """
    Convert a PDF to markdown + extracted images using marker-pdf.
    Uses pre-loaded models if available (fast path), otherwise loads on demand.

    Returns:
        (markdown_text, images_dict)
        images_dict: {filename: png_bytes}
    """
    global _marker_models

    # Use pre-loaded models if available
    if _marker_models is not None:
        api_type, models = _marker_models
        if api_type == "new":
            return _parse_new_api(pdf_path, models)
        else:
            return _parse_old_api(pdf_path, models)

    # Cold start: load models now (pays the ~60s cost once)
    logger.warning("marker-pdf models not pre-loaded — loading now (slow first run)")
    preload_models()
    return parse_with_marker(pdf_path)


# ── API helpers ───────────────────────────────────────────────────────────────

def _pil_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _parse_new_api(pdf_path: str, model_dict=None) -> tuple[str, dict[str, bytes]]:
    """marker-pdf >= 0.3.x"""
    from marker.converters.pdf import PdfConverter
    from marker.output import text_from_rendered
    if model_dict is None:
        from marker.models import create_model_dict
        model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)
    return text, {k: _pil_to_bytes(v) for k, v in images.items()}


def _parse_old_api(pdf_path: str, model_lst=None) -> tuple[str, dict[str, bytes]]:
    """marker-pdf < 0.3.x"""
    from marker.convert import convert_single_pdf
    if model_lst is None:
        from marker.models import load_all_models
        model_lst = load_all_models()
    full_text, images, _ = convert_single_pdf(pdf_path, model_lst)
    return full_text, {k: _pil_to_bytes(v) for k, v in images.items()}
