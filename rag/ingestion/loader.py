"""
Document loader using the Unstructured library.

Supports PDF (hi-res with table extraction), Word, HTML, plain text, and
any format Unstructured can auto-detect. Falls back to mimetypes when
python-magic / python-magic-bin is not installed.
"""

import base64
import mimetypes
import os
from typing import Any, Dict, List, Tuple

try:
    import magic as _magic
    _HAS_MAGIC = True
except ImportError:
    _magic = None  # type: ignore
    _HAS_MAGIC = False

from unstructured.documents.elements import (
    Image as UnstructuredImage,
    ListItem,
    NarrativeText,
    Table,
    Text,
    Title,
)
from unstructured.partition.auto import partition


def detect_mime(path: str) -> str:
    """Return MIME type using libmagic if available, else mimetypes fallback."""
    if _HAS_MAGIC and _magic is not None:
        try:
            return _magic.from_file(path, mime=True) or "application/octet-stream"
        except Exception:
            pass
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def partition_file(path: str) -> Tuple[List[Any], str]:
    """
    Partition a file into Unstructured elements.
    PDF files use hi-res strategy with table structure inference.
    """
    mime = detect_mime(path)
    kwargs: Dict[str, Any] = {}
    if mime == "application/pdf":
        kwargs = {
            "strategy": "hi_res",
            "infer_table_structure": True,
        }
    elements = partition(filename=path, **kwargs)
    return elements, mime


def extract_payload(elements: List[Any]) -> Dict[str, Any]:
    """
    Separate Unstructured elements into three buckets:
      - text  : plain paragraph / list / title text
      - tables: HTML strings of table elements
      - images: base64-encoded JPEG strings

    Returns:
        {
            "text": str,
            "tables": List[str],
            "images": List[str],
            "content_types": List[str],  # e.g. ["text", "table", "image"]
        }
    """
    texts: List[str] = []
    tables: List[str] = []
    images: List[str] = []
    content_types: set = set()

    for el in elements:
        if isinstance(el, (Title, NarrativeText, Text, ListItem)):
            t = (el.text or "").strip()
            if t:
                texts.append(t)
                content_types.add("text")

        elif isinstance(el, Table):
            content_types.add("table")
            html = getattr(el.metadata, "text_as_html", None) or (el.text or "").strip()
            if html:
                tables.append(html)

        elif isinstance(el, UnstructuredImage):
            content_types.add("image")
            b64 = _image_to_b64(el)
            if b64:
                images.append(b64)

    return {
        "text": "\n".join(texts).strip(),
        "tables": tables,
        "images": images,
        "content_types": sorted(content_types) or ["text"],
    }


def _image_to_b64(el: Any) -> str:
    """Extract base64-encoded image from an Unstructured Image element."""
    try:
        b64 = getattr(el.metadata, "image_base64", None)
        if b64:
            return b64
    except Exception:
        pass
    try:
        image_path = getattr(el.metadata, "image_path", None)
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass
    return ""
