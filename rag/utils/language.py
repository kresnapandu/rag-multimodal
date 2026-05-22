"""
Lightweight heuristic language detector.

Returns 'id' (Indonesian) or 'en' (English/default).
Used for metadata logging only — all LLM responses are in English.
No external dependencies — instant, zero API cost.
"""

_INDO_WORDS = frozenset({
    "yang", "dan", "atau", "kenapa", "bagaimana", "gimana", "apa", "apakah",
    "berikan", "dengan", "untuk", "dari", "dalam", "ini", "itu", "saya", "aku",
    "kamu", "tolong", "jelaskan", "bikin", "buat", "pake", "nggak", "ga",
    "udah", "belum", "adalah", "akan", "juga", "tidak", "ada", "saat",
    "seperti", "lebih", "pada", "oleh", "telah", "sebuah", "jika", "maka",
    "karena", "ketika", "setelah", "sebelum", "antara", "bisa", "harus",
    "perlu", "sudah", "sedang", "masih", "bahwa", "namun", "tetapi",
})

_INDO_PARTICLES = frozenset({
    " ngg", " gak", " ga ", " dong", " deh", " kok", " sih", " yuk",
    " loh", " nih", " tuh", " gitu", " gini",
})


def detect_language(text: str) -> str:
    """Return 'id' if the text appears to be Indonesian, else 'en'."""
    t = (text or "").lower()
    padded = f" {t} "
    hits = sum(1 for w in _INDO_WORDS if f" {w} " in padded)
    if hits >= 2:
        return "id"
    if any(p in t for p in _INDO_PARTICLES):
        return "id"
    return "en"
