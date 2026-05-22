"""
Answer generator — the core RAG brain.

Flow per question
-----------------
1. Detect question language (stored as metadata only — responses are always English).
2. Reformulate query using chat history so it is self-contained
   (History-Aware Retrieval).
3. Retrieve highlighted chunks from TelecomRetriever.
4. Build a token-efficient context block.
5. Generate a grounded technical answer in English.
6. Update history and return a structured result dict.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .history import HistoryManager
from ..retrieval.retriever import TelecomRetriever, RetrievalResult
from ..utils.language import detect_language

_NOT_FOUND = (
    "I don't have enough information to answer that "
    "based on the provided documents."
)

_REFORMULATE = (
    "Given the chat history, rewrite the following question to be fully "
    "self-contained and optimised for searching telecommunications hardware "
    "datasheets. Return ONLY the rewritten question — no quotes, no markdown."
)


class RAGGenerator:
    def __init__(
        self,
        model: ChatOpenAI,
        retriever: TelecomRetriever,
        history: HistoryManager,
        config: Any,
    ) -> None:
        self._model = model
        self._retriever = retriever
        self._history = history
        self._cfg = config

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(
        self,
        user_question: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a single user question.

        Returns:
            {
                "question": str,
                "search_question": str,   # possibly reformulated
                "answer": str,
                "lang": str,              # detected language ('en' | 'id') — metadata only
                "sources": List[str],     # source file basenames
                "scores": List[float],    # cosine similarity per source
                "context_chars": int,     # chars sent to LLM (token proxy)
                "prompt_tokens": int | None,
                "completion_tokens": int | None,
            }
        """
        lang = detect_language(user_question)          # metadata / logging only
        search_question = self._reformulate(user_question)

        results = self._retriever.retrieve(search_question, where=where)

        if not results:
            self._history.add(user_question, _NOT_FOUND)
            return self._empty_result(user_question, search_question, lang, _NOT_FOUND)

        context_block = self._build_context(results)
        answer, usage = self._generate(user_question, context_block)
        self._history.add(user_question, answer)

        return {
            "question": user_question,
            "search_question": search_question,
            "answer": answer,
            "lang": lang,
            "sources": [
                os.path.basename(d.metadata.get("source", "")) for d, _, _ in results
            ],
            "scores": [round(s, 4) for _, s, _ in results],
            "context_chars": len(context_block),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    # ── Private ────────────────────────────────────────────────────────────────

    def _reformulate(self, question: str) -> str:
        """Rewrite the question to be self-contained using chat history."""
        if self._history.is_empty:
            return question

        messages = (
            [SystemMessage(content=_REFORMULATE)]
            + self._history.messages
            + [HumanMessage(content=f"New question: {question}")]
        )
        result = self._model.invoke(messages)
        return (result.content or "").strip().strip('"').strip("'")

    def _build_context(self, results: List[RetrievalResult]) -> str:
        """Assemble the context block from highlighted chunks."""
        blocks = []
        for i, (doc, score, highlighted) in enumerate(results, start=1):
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            chunk_type = doc.metadata.get("chunk_type", "")
            score_tag = f"score={score:.3f}" if score else ""
            header = " | ".join(filter(None, [f"Doc {i}", src, score_tag, chunk_type]))
            blocks.append(f"[{header}]\n{highlighted}")
        return "\n\n---\n\n".join(blocks)

    def _generate(
        self,
        question: str,
        context: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate the grounded answer and return (answer_text, token_usage)."""
        sys_content = (
            f"{self._cfg.domain_system_hint}\n\n"
            "Answer strictly using information from the provided documents. "
            "Quote exact spec values (dBm, MHz, Gbps, VSWR, etc.) where available. "
            f"If the answer is not found in the documents, respond exactly with: "
            f"\"{_NOT_FOUND}\""
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Documents:\n{context}\n\n"
            "Provide a precise technical answer using only the documents above."
        )

        messages = (
            [SystemMessage(content=sys_content)]
            + self._history.messages
            + [HumanMessage(content=user_content)]
        )

        result = self._model.invoke(messages)
        answer = (result.content or "").strip()

        usage: Dict[str, Any] = {}
        try:
            meta = getattr(result, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or meta.get("usage") or {}
        except Exception:
            pass

        return answer, usage

    @staticmethod
    def _empty_result(
        question: str,
        search_question: str,
        lang: str,
        answer: str,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "search_question": search_question,
            "answer": answer,
            "lang": lang,
            "sources": [],
            "scores": [],
            "context_chars": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
