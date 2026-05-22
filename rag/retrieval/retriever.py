"""
Retrieval pipeline with token-aware filtering and sentence-level highlighting.

Pipeline per query
------------------
1. Fetch top-K candidates from ChromaDB (cosine similarity).
2. Drop chunks below `score_threshold` (relevance gate).
3. Deduplicate near-identical chunks (same leading content).
4. Limit to `max_context_docs` chunks.
5. For each surviving chunk, extract the most query-relevant sentences
   via keyword-overlap scoring — zero extra API calls.
6. Hard-cap each chunk at `max_chunk_chars` characters.

Token savings vs. naive full-chunk retrieval: typically 60–80 %.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ..ingestion.vectorstore import VectorStore
from ..utils.highlight import extract_relevant_sentences, trim_to_chars

# Each item: (original Document, cosine score, highlighted text)
RetrievalResult = Tuple[Document, float, str]


class TelecomRetriever:
    def __init__(self, vectorstore: VectorStore, config: Any) -> None:
        self._store = vectorstore
        self._cfg = config

    def retrieve(
        self,
        query: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Run the full retrieval + filtering + highlighting pipeline.

        Args:
            query: The (possibly reformulated) search query.
            where: Optional ChromaDB equality filter dict.

        Returns:
            List of (Document, score, highlighted_text) sorted by score desc.
        """
        docs, scores = self._store.similarity_search_with_scores(
            query, k=self._cfg.retrieval_k, where=where
        )

        # ── Score gate ─────────────────────────────────────────────────────────
        if scores:
            pairs: List[Tuple[Document, float]] = [
                (d, s) for d, s in zip(docs, scores)
                if s >= self._cfg.score_threshold
            ]
        else:
            # No scores available (fallback mode) — keep all
            pairs = [(d, 0.0) for d in docs]

        # ── Deduplication ──────────────────────────────────────────────────────
        seen: set = set()
        deduped: List[Tuple[Document, float]] = []
        for doc, score in pairs:
            fingerprint = (doc.page_content or "")[:100].strip().lower()
            if fingerprint not in seen:
                seen.add(fingerprint)
                deduped.append((doc, score))

        # ── Cap at max_context_docs ────────────────────────────────────────────
        limited = deduped[: self._cfg.max_context_docs]

        # ── Sentence-level highlighting (token minimisation) ───────────────────
        results: List[RetrievalResult] = []
        for doc, score in limited:
            highlighted = extract_relevant_sentences(
                doc.page_content,
                query,
                top_n=self._cfg.highlight_sentences,
            )
            highlighted = trim_to_chars(highlighted, self._cfg.max_chunk_chars)
            results.append((doc, score, highlighted))

        return results

    def eval_retrieve(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[str, float]]:
        """
        Lightweight retrieval for offline evaluation (returns source basenames).
        Bypasses the score gate so all k candidates are surfaced.
        """
        docs, scores = self._store.similarity_search_with_scores(query, k=k)
        return [
            (os.path.basename(d.metadata.get("source", "")), s)
            for d, s in zip(docs, scores or [0.0] * len(docs))
        ]
