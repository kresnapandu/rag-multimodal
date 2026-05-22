import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Paths ---
    persist_dir: str = "db/chroma_db"
    docs_dir: str = "docs"
    metrics_dir: str = "metrics"
    collection_name: str = "telecom_rag"

    # --- Models ---
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    temperature: float = 0.0

    # --- Retrieval ---
    retrieval_k: int = 8          # Candidates to fetch from vector DB
    score_threshold: float = 0.35  # Drop chunks below this cosine similarity
    max_context_docs: int = 5     # Max chunks sent to the LLM

    # --- Token optimisation ---
    max_history_turns: int = 6    # Compress history after this many Q/A pairs
    highlight_sentences: int = 5  # Sentences extracted per chunk (keyword overlap)
    max_chunk_chars: int = 1200   # Hard cap on each chunk after highlighting

    # --- Semantic chunker ---
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 70.0

    # --- Telecom domain system hint ---
    domain_system_hint: str = (
        "You are a technical assistant specialized in telecommunications hardware. "
        "You have deep knowledge of RF systems, antennas, base stations (BTS/NodeB/gNB), "
        "optical transceivers (SFP/QSFP), routers, switches, and telecom datasheet specs. "
        "When answering, reference exact spec values (dBm, MHz, Gbps, VSWR, etc.) "
        "from the provided documents. Be precise and concise."
    )

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            persist_dir=os.getenv("RAG_PERSIST_DIR", "db/chroma_db"),
            docs_dir=os.getenv("RAG_DOCS_DIR", "docs"),
            metrics_dir=os.getenv("RAG_METRICS_DIR", "metrics"),
            collection_name=os.getenv("RAG_COLLECTION", "telecom_rag"),
            embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o"),
            temperature=float(os.getenv("CHAT_TEMPERATURE", "0")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "8")),
            score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.35")),
            max_context_docs=int(os.getenv("MAX_CONTEXT_DOCS", "5")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "6")),
            highlight_sentences=int(os.getenv("HIGHLIGHT_SENTENCES", "5")),
            max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "1200")),
        )
