"""
In-memory metrics store with CSV export.
Tracks ingestion, query latency/tokens, and offline retrieval evaluation.
"""

import csv
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List


class MetricsStore:
    def __init__(self) -> None:
        self._ingestion: List[Dict[str, Any]] = []
        self._query: List[Dict[str, Any]] = []
        self._eval: List[Dict[str, Any]] = []

    def log_ingestion(self, **kwargs: Any) -> None:
        self._ingestion.append({"ts": _now_iso(), **kwargs})

    def log_query(self, **kwargs: Any) -> None:
        self._query.append({"ts": _now_iso(), **kwargs})

    def log_eval(self, **kwargs: Any) -> None:
        self._eval.append({"ts": _now_iso(), **kwargs})

    def export_csv(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for name, rows in [
            ("ingestion", self._ingestion),
            ("query", self._query),
            ("eval", self._eval),
        ]:
            path = os.path.join(out_dir, f"{name}.csv")
            if not rows:
                open(path, "w").close()
                continue
            keys = sorted({k for r in rows for k in r})
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"  Exported: {os.path.abspath(path)}")


class Timer:
    """Simple elapsed-time helper."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)


def new_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
