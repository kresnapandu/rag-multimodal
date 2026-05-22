"""
Token-minimisation via sentence-level relevance highlighting.

Strategy
--------
Instead of sending an entire 1 000-token chunk to the LLM, this module
extracts only the top-N sentences that overlap with the query keywords.

Typical savings: 60-80 % fewer context tokens — zero extra API calls.

Algorithm
---------
1. Split chunk into sentences on sentence-boundary punctuation.
2. Tokenise query into a keyword set (lowercase, stopwords removed).
3. Score each sentence by keyword overlap + a small positional bonus
   (first / last sentences tend to carry key facts in datasheets).
4. Select top-N sentences, restore original reading order.
5. Join with " [...] " to signal omissions.
"""

import re
from typing import List

# Common English stopwords — telecom terms (dBm, GHz, CPRI …) are intentionally excluded.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "be", "that", "this", "it", "its",
    "as", "by", "from", "not", "no", "if", "so", "do", "did", "has", "have",
    "had", "been", "will", "would", "can", "could", "may", "might", "shall",
    "should", "into", "than", "then", "when", "where", "which", "who", "how",
    "what", "such", "also", "any", "all", "each", "use", "used", "using",
})


def extract_relevant_sentences(text: str, query: str, top_n: int = 5) -> str:
    """
    Return the top_n most query-relevant sentences from *text*.

    Falls back to the full text when the sentence count is already ≤ top_n.
    No LLM call — runs in microseconds.
    """
    if not text or not text.strip():
        return ""

    sentences = _split_sentences(text)
    if len(sentences) <= top_n:
        return text

    query_tokens = _tokenize(query)
    if not query_tokens:
        return " ".join(sentences[:top_n])

    last_idx = len(sentences) - 1
    scored: List[tuple] = []
    for i, sent in enumerate(sentences):
        overlap = len(query_tokens & _tokenize(sent))
        # Positional bonus: first sentence often has product/model info,
        # last sentence often has a summary or key constraint.
        bonus = 0.5 if i == 0 else (0.3 if i == last_idx else 0.0)
        scored.append((overlap + bonus, i, sent))

    # Pick top_n by score, then sort back to document order.
    top = sorted(scored, key=lambda x: -x[0])[:top_n]
    top_ordered = sorted(top, key=lambda x: x[1])
    return " [...] ".join(s for _, _, s in top_ordered)


def trim_to_chars(text: str, max_chars: int) -> str:
    """Hard-cap text length, cutting at the last word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[: cut if cut > 0 else max_chars] + " [...]"


# ── Private helpers ────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split on '. ', '! ', '? ' or line breaks."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> frozenset:
    """Lowercase alphanum tokens, stopwords removed, min length 2."""
    words = re.findall(r"\b[a-z0-9]+(?:\.[0-9]+)?\b", (text or "").lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) >= 2)
