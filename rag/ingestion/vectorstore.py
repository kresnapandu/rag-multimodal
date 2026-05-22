"""
ChromaDB vector store wrapper.

Provides a thin, type-safe interface over LangChain's Chroma client.
All retrieval returns (document, score) pairs so callers can apply
score thresholds without knowing ChromaDB internals.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    def __init__(self, config: Any, embeddings: OpenAIEmbeddings) -> None:
        self._config = config
        self._embeddings = embeddings
        self._db: Optional[Chroma] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        db = self._db_instance()
        db.add_documents(docs)
        try:
            db.persist()
        except Exception:
            pass  # Newer ChromaDB versions auto-persist

    def count(self) -> int:
        try:
            return self._db_instance()._collection.count()
        except Exception:
            return -1

    def similarity_search_with_scores(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Document], List[float]]:
        """
        Return (docs, scores) where scores are cosine similarity in [0, 1].
        Falls back to score-less search on any error.
        """
        db = self._db_instance()
        try:
            pairs = db.similarity_search_with_relevance_scores(
                query, k=k, filter=where or None
            )
            return [d for d, _ in pairs], [float(s) for _, s in pairs]
        except Exception:
            docs = db.similarity_search(query, k=k, filter=where or None)
            return docs, []

    # ── Private ────────────────────────────────────────────────────────────────

    def _db_instance(self) -> Chroma:
        if self._db is None:
            kwargs: Dict[str, Any] = dict(
                persist_directory=self._config.persist_dir,
                embedding_function=self._embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            if self._config.collection_name:
                kwargs["collection_name"] = self._config.collection_name
            self._db = Chroma(**kwargs)
        return self._db
