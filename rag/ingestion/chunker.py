"""
Semantic chunking with telecom-domain metadata extraction.

Two document types are produced per ingested file:
  1. Semantic text chunks — split by SemanticChunker on embedding-space
     sentence boundaries for high retrieval precision.
  2. Multimodal summary chunk — one AI-generated dense summary per file
     that fuses text + table + image content (stored only when tables or
     images are present).

Telecom metadata (frequency bands, power specs, data rates, standards)
is auto-extracted via regex and stored as ChromaDB metadata fields for
structured filtering queries (e.g. `/where standards~5G`).
"""

import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from ..utils.language import detect_language

# ── Telecom regex patterns ─────────────────────────────────────────────────────
_RE_FREQ = re.compile(r"\b\d+(?:\.\d+)?\s*(?:MHz|GHz|kHz)\b", re.IGNORECASE)
_RE_POWER = re.compile(r"\b\d+(?:\.\d+)?\s*(?:dBm|dBi|dBd|dB|W|mW)\b", re.IGNORECASE)
_RE_RATE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:Gbps|Mbps|kbps|bps)\b", re.IGNORECASE)
_RE_STANDARDS = re.compile(
    r"\b(5G\s*NR|4G\s*LTE|LTE-A(?:dvanced)?|LTE|NR|WCDMA|UMTS|GSM|EDGE|GPRS"
    r"|CPRI|eCPRI|OBSAI|OTN|SDH|WDM|DWDM|PoE\+?|SFP\+?|QSFP\+?|MIMO|Massive\s*MIMO"
    r"|mmWave|Sub-6|FR1|FR2|O-RAN|FDD|TDD)\b",
    re.IGNORECASE,
)


def extract_telecom_metadata(text: str) -> Dict[str, Any]:
    """
    Extract telecom spec keywords from text and return as ChromaDB-safe metadata.
    All list values are JSON-serialised strings (ChromaDB only accepts scalar types).
    """
    freqs = list(dict.fromkeys(m.group(0) for m in _RE_FREQ.finditer(text)))[:6]
    powers = list(dict.fromkeys(m.group(0) for m in _RE_POWER.finditer(text)))[:6]
    rates = list(dict.fromkeys(m.group(0) for m in _RE_RATE.finditer(text)))[:6]
    standards = list(dict.fromkeys(m.group(1).upper() for m in _RE_STANDARDS.finditer(text)))[:10]
    return {
        "freq_bands": json.dumps(freqs),
        "power_specs": json.dumps(powers),
        "data_rates": json.dumps(rates),
        "standards": json.dumps(standards),
        "has_specs": bool(freqs or powers or rates or standards),
    }


class TelecomChunker:
    """Wraps SemanticChunker and enriches documents with telecom metadata."""

    def __init__(self, embeddings: OpenAIEmbeddings, config: Any) -> None:
        self._splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=config.breakpoint_threshold_type,
            breakpoint_threshold_amount=config.breakpoint_threshold_amount,
        )

    def build_documents(
        self,
        source_path: str,
        payload: Dict[str, Any],
        mime: str,
        force_lang: Optional[str] = None,
        ai_summary_fn: Optional[Callable[..., str]] = None,
    ) -> List[Document]:
        """
        Build a list of LangChain Documents ready for ChromaDB ingestion.

        Args:
            source_path: Absolute path to the original file (stored as metadata).
            payload: Output of loader.extract_payload().
            mime: MIME type string.
            force_lang: Override auto-detected language ('id' or 'en').
            ai_summary_fn: Callable(text, tables, images, lang) -> str.
                           When provided and tables/images exist, a multimodal
                           summary document is also produced.
        """
        raw_text = payload["text"]
        tables = payload["tables"]
        images = payload["images"]
        content_types = payload["content_types"]

        lang = force_lang or (detect_language(raw_text[:500]) if raw_text else "en")
        file_id = str(uuid.uuid4())

        base_meta: Dict[str, Any] = {
            "file_id": file_id,
            "source": source_path,
            "mime": mime,
            "lang": lang,
            "content_types": json.dumps(content_types),
        }

        docs: List[Document] = []

        # ── 1. Semantic text chunks ────────────────────────────────────────────
        for idx, chunk_text in enumerate(self._split(raw_text), start=1):
            telecom_meta = extract_telecom_metadata(chunk_text)
            docs.append(Document(
                page_content=chunk_text,
                metadata={
                    **base_meta,
                    **telecom_meta,
                    "chunk_index": idx,
                    "chunk_type": "semantic_text",
                    "is_summary": False,
                },
            ))

        # ── 2. Multimodal enriched summary ────────────────────────────────────
        if (tables or images) and ai_summary_fn is not None:
            try:
                summary = ai_summary_fn(raw_text, tables, images)
            except Exception:
                summary = (raw_text[:600] + " [...]") if raw_text else ""
            if summary:
                telecom_meta = extract_telecom_metadata(summary)
                docs.append(Document(
                    page_content=summary,
                    metadata={
                        **base_meta,
                        **telecom_meta,
                        "chunk_index": 0,
                        "chunk_type": "multimodal_summary",
                        "is_summary": True,
                        "tables_count": len(tables),
                        "images_count": len(images),
                    },
                ))

        return docs

    def _split(self, text: str) -> List[str]:
        """Semantic split with paragraph fallback."""
        if not text or not text.strip():
            return []
        try:
            raw_docs = self._splitter.create_documents([text])
            chunks = [d.page_content for d in raw_docs if (d.page_content or "").strip()]
            return chunks if chunks else [text]
        except Exception:
            # Fallback: split on double newlines
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            return paragraphs or [text]
