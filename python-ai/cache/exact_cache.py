"""
AcaRAG Pro — Level 1: Exact Query Cache
-----------------------------------------
Stores and retrieves responses by exact normalized query string.

Storage : JSON file on disk (persists across server restarts).
Key     : query lowercased + whitespace-normalized → deterministic hash.
Value   : { answer, sources, hit_count, cached_at, last_accessed }.

Latency benefit: ~0 ms (dict lookup + disk read on cold start vs ~2000 ms for full RAG).
Best for: Same question asked by multiple students (very common for exam dates,
          regulation queries, and calendar lookups).

Safety rules enforced here:
  - Out-of-scope responses are NOT stored.
  - Responses shorter than MIN_ANSWER_LENGTH are NOT stored.
  - Cache file is re-written atomically via a temp file to prevent corruption.
"""

import os
import json
import hashlib
import tempfile
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CACHE_DIR  = os.path.dirname(__file__)
CACHE_FILE  = os.path.join(_CACHE_DIR, "exact_store.json")

MIN_ANSWER_LENGTH   = 20    # chars — ignore trivially short responses
OUT_OF_SCOPE_PREFIX = "Information not available in provided academic documents"


class ExactCache:
    """
    In-process dict backed by a JSON file.
    Thread safety: adequate for single-worker uvicorn (default dev mode).
    For multi-worker production deployments, swap the JSON file for SQLite.
    """

    def __init__(self):
        self._store: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}   # Corrupt file — start fresh
        return {}

    def _save(self) -> None:
        """Atomic write: write to temp file, then rename."""
        tmp_fd, tmp_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, CACHE_FILE)   # atomic on POSIX + Windows (Py3.3+)
        except OSError:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(query: str, student_context: dict = None) -> str:
        """
        Hash of (context_key + normalised_query) — ensures two students
        with different branch/year/semester never share a cache entry.
        Name is excluded: it only affects greeting, not answer content.
        """
        ctx = student_context or {}
        ctx_key = "|".join([
            ctx.get("branch",   "").lower(),
            ctx.get("year",     "").lower(),
            ctx.get("semester", "").lower(),
        ])
        normalised = " ".join(query.lower().split())
        return hashlib.sha256(f"{ctx_key}:{normalised}".encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str, student_context: dict = None) -> dict | None:
        """
        Returns cached payload if an exact match exists, else None.
        Updates hit_count and last_accessed on every cache hit.
        """
        key = self._make_key(query, student_context)
        entry = self._store.get(key)
        if entry is None:
            return None

        # Update access metadata (best-effort — don't crash on save failure)
        entry["hit_count"] = entry.get("hit_count", 0) + 1
        entry["last_accessed"] = datetime.now().isoformat()
        self._save()

        return {
            "answer":  entry["answer"],
            "sources": entry["sources"],
        }

    def store(self, query: str, answer: str, sources: list,
              student_context: dict = None) -> bool:
        """
        Persists a query-answer pair.
        Returns True if stored, False if rejected by safety rules.
        """
        # Safety gate 1: too short
        if len(answer.strip()) < MIN_ANSWER_LENGTH:
            return False

        # Safety gate 2: out-of-scope answer — don't pollute cache
        if answer.strip().startswith(OUT_OF_SCOPE_PREFIX):
            return False

        key = self._make_key(query, student_context)
        self._store[key] = {
            "query":         query,          # human-readable original
            "answer":        answer,
            "sources":       sources,
            "hit_count":     0,
            "cached_at":     datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
        }
        self._save()
        return True

    def size(self) -> int:
        return len(self._store)

    def invalidate(self) -> None:
        """Clear all entries and delete the backing file."""
        self._store = {}
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
