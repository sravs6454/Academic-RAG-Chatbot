"""
AcaRAG Pro — Cache Manager
---------------------------
Lightweight cache manager for Render Free (512 MB).

Cache levels:
  Level 1: Exact Cache
  Level 2: Semantic Cache disabled
           (requires HuggingFace embeddings and adds unnecessary memory usage)

The main RAG pipeline uses:
  SQLite text index → Groq
"""

from cache.exact_cache import ExactCache


class CacheManager:

    def __init__(self):
        # Level 1 — lightweight exact cache
        self.exact = ExactCache()

        # Semantic cache intentionally disabled.
        # It requires HuggingFace embeddings, which are too heavy
        # for the Render Free 512 MB memory limit.

        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "stores": 0,
            "rejected": 0,
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def check(self, query: str, student_context: dict = None) -> dict | None:
        """
        Check the lightweight exact cache.

        Semantic cache is disabled to keep the Render backend
        within the 512 MB memory limit.
        """

        # Level 1 — exact match
        result = self.exact.get(query, student_context)

        if result is not None:
            self._stats["exact_hits"] += 1
            return {**result, "cache_hit": "exact"}

        # No semantic cache
        self._stats["misses"] += 1
        return None

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        query: str,
        answer: str,
        sources: list,
        student_context: dict = None,
    ) -> None:
        """
        Store the response only in the lightweight exact cache.
        """

        exact_stored = self.exact.store(
            query,
            answer,
            sources,
            student_context,
        )

        if exact_stored:
            self._stats["stores"] += 1
        else:
            self._stats["rejected"] += 1

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_all(self) -> None:
        """
        Clear the exact cache after document re-ingestion.
        """

        self.exact.invalidate()

        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "stores": 0,
            "rejected": 0,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return cache performance statistics.
        """

        total_queries = (
            self._stats["exact_hits"]
            + self._stats["misses"]
        )

        hit_rate = 0.0

        if total_queries > 0:
            hit_rate = round(
                self._stats["exact_hits"]
                / total_queries
                * 100,
                1,
            )

        return {
            **self._stats,
            "total_queries": total_queries,
            "cache_hit_rate_pct": hit_rate,
            "exact_cache_size": self.exact.size(),
            "semantic_cache_size": 0,
            "groq_calls_saved": self._stats["exact_hits"],
        }