"""
AcaRAG Pro — Cache Manager (Orchestrator)
-------------------------------------------
Single entry point for all cache operations.
Coordinates Level 1 (exact) → Level 2 (semantic) lookup order.
Exposes store(), invalidate_all(), and stats() to the rest of the system.

Execution flow on every /chat request:
  CacheManager.check(query)
    ├── ExactCache.get(query)       → hit? return immediately (0 ms overhead)
    └── SemanticCache.get(query)    → hit? return in ~80 ms (embedding only)
        └── [miss] → caller runs RAG → calls CacheManager.store(...)
              ├── ExactCache.store(...)
              └── SemanticCache.store(...)

Cache invalidation strategy:
  When /ingest is called (new documents added), call invalidate_all().
  This is safe because:
    - Cached answers are derived from the previous document set.
    - New documents may produce better or different answers.
    - Cache rebuilds automatically from real queries — no manual warmup needed.
"""

from cache.exact_cache    import ExactCache
from cache.semantic_cache import SemanticCache


class CacheManager:

    def __init__(self):
        self.exact    = ExactCache()
        self.semantic = SemanticCache()

        # In-process stats counters (reset on server restart, intentionally)
        self._stats = {
            "exact_hits":    0,
            "semantic_hits": 0,
            "misses":        0,
            "stores":        0,
            "rejected":      0,   # answers that failed safety gates
        }

    # ------------------------------------------------------------------
    # Lookup — called BEFORE RAG
    # ------------------------------------------------------------------

    def check(self, query: str, student_context: dict = None) -> dict | None:
        """
        Check all cache levels in priority order.
        student_context (branch/year/semester) is part of the cache key —
        two students with different profiles never share a cached answer.

        cache_hit values:
          "exact"    → Level 1 hit (identical query seen before)
          "semantic" → Level 2 hit (paraphrase of a seen query)
        """
        # Level 1 — exact match (fastest, no ML inference needed)
        result = self.exact.get(query, student_context)
        if result is not None:
            self._stats["exact_hits"] += 1
            return {**result, "cache_hit": "exact"}

        # Level 2 — semantic match (~80 ms embedding overhead)
        result = self.semantic.get(query, student_context)
        if result is not None:
            self._stats["semantic_hits"] += 1
            return {**result, "cache_hit": "semantic"}

        # Miss — caller must run full RAG pipeline
        self._stats["misses"] += 1
        return None

    # ------------------------------------------------------------------
    # Store — called AFTER RAG generates a response
    # ------------------------------------------------------------------

    def store(self, query: str, answer: str, sources: list,
              student_context: dict = None) -> None:
        """
        Persist a query-answer pair into both cache levels.
        Both layers enforce their own safety gates independently.
        """
        exact_stored    = self.exact.store(query, answer, sources, student_context)
        semantic_stored = self.semantic.store(query, answer, sources, student_context)

        if exact_stored or semantic_stored:
            self._stats["stores"] += 1
        else:
            self._stats["rejected"] += 1

    # ------------------------------------------------------------------
    # Invalidation — called after /ingest
    # ------------------------------------------------------------------

    def invalidate_all(self) -> None:
        """
        Wipe both cache levels.
        Must be called whenever the document corpus changes (re-ingestion).
        """
        self.exact.invalidate()
        self.semantic.invalidate()

        # Reset hit counters too — fresh dataset, fresh stats
        self._stats = {k: 0 for k in self._stats}

    # ------------------------------------------------------------------
    # Stats — exposed via /cache/stats endpoint
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Returns performance statistics for the current server session.
        Useful during demo to prove latency savings.
        """
        total_queries = (
            self._stats["exact_hits"] +
            self._stats["semantic_hits"] +
            self._stats["misses"]
        )

        hit_rate = 0.0
        if total_queries > 0:
            hits = self._stats["exact_hits"] + self._stats["semantic_hits"]
            hit_rate = round(hits / total_queries * 100, 1)

        return {
            **self._stats,
            "total_queries":      total_queries,
            "cache_hit_rate_pct": hit_rate,
            "exact_cache_size":   self.exact.size(),
            "semantic_cache_size": self.semantic.size(),
            "groq_calls_saved":   self._stats["exact_hits"] + self._stats["semantic_hits"],
        }
