"""
AcaRAG Pro — Level 2: Semantic Query Cache
--------------------------------------------
Embeds incoming queries and compares them against cached query embeddings
using cosine similarity. If similarity >= THRESHOLD, the cached response
is returned without touching ChromaDB or the GROQ API.

Embedding model : sentence-transformers/all-MiniLM-L6-v2  (same as RAG — no extra model)
Similarity metric : cosine similarity via numpy
Threshold         : 0.85  (empirically good for academic Q&A paraphrases)
Storage           : JSON file (embeddings stored as float lists)

Latency benefit:
  - Cache miss : ~80 ms  (embed query + scan N entries)
  - Cache hit  : ~80 ms  vs ~2200 ms full RAG+LLM pipeline
  - Net saving on hit : ~2100 ms and 1 GROQ API call

Design notes:
  - The embedder is loaded lazily and ONLY once (singleton pattern).
  - Embeddings are stored as plain Python lists in JSON — no extra binary
    format required. At 384 dimensions × 4 bytes each, 1000 cached entries
    ≈ 1.5 MB on disk — perfectly fine.
  - Linear scan is O(N). For N < 5000 entries this is fast enough; swap
    to FAISS indexing if the cache grows beyond that.
"""

import os
import re
import json
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np


def _exam_type(text: str) -> str:
    """Return a coarse exam-type label so semantically-similar queries
    about DIFFERENT exam types never share a cache entry."""
    t = text.lower()
    if re.search(r'\badvanced.?supplementary\b', t):
        return "advanced_supplementary"
    if re.search(r'\bsupplementary\b', t):
        return "supplementary"
    if re.search(r'\bregular\b', t):
        return "regular"
    return ""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CACHE_DIR    = os.path.dirname(__file__)
CACHE_FILE    = os.path.join(_CACHE_DIR, "semantic_store.json")

SIMILARITY_THRESHOLD = 0.85   # Tune lower (e.g. 0.80) to be more aggressive
MIN_ANSWER_LENGTH    = 20
OUT_OF_SCOPE_PREFIX  = "Information not available in provided academic documents"
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singleton so the model is only downloaded/loaded once per process
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        # Import here so the module loads fast even if sentence-transformers
        # is slow to initialise
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder


def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two equal-length float vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / denom)


class SemanticCache:
    """
    Stores (query_embedding, answer, sources) tuples.
    On lookup, embeds the incoming query and finds the closest stored entry.
    """

    def __init__(self):
        self._entries: list = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> list:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, CACHE_FILE)
        except OSError:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _make_context_key(student_context: dict = None) -> str:
        """branch|year|semester — uniquely identifies a student group."""
        ctx = student_context or {}
        return "|".join([
            ctx.get("branch",   "").lower(),
            ctx.get("year",     "").lower(),
            ctx.get("semester", "").lower(),
        ])

    def get(self, query: str, student_context: dict = None) -> Optional[dict]:
        """
        Embed the query and scan cached entries for a semantic match.
        Only compares entries that share the same student context (branch/year/semester).
        Returns the best match above SIMILARITY_THRESHOLD, or None.
        """
        if not self._entries:
            return None

        context_key = self._make_context_key(student_context)
        # Only consider entries from the same student context group
        relevant = [e for e in self._entries if e.get("context_key", "") == context_key]
        if not relevant:
            return None

        embedder = _get_embedder()
        query_vec = embedder.embed_query(query)

        best_score = -1.0
        best_entry = None

        for entry in relevant:
            score = _cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= SIMILARITY_THRESHOLD and best_entry is not None:
            # Reject if the queries are about different exam types
            # e.g. "supplementary fee" must not serve as a hit for "advanced supplementary fee"
            if _exam_type(query) != _exam_type(best_entry.get("query", "")):
                return None

            # Track usage metadata
            best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
            best_entry["last_accessed"] = datetime.now().isoformat()
            self._save()

            return {
                "answer":           best_entry["answer"],
                "sources":          best_entry["sources"],
                "similarity_score": round(best_score, 4),
            }

        return None

    def store(self, query: str, answer: str, sources: list,
              student_context: dict = None) -> bool:
        """
        Embed the query and persist the entry.
        Returns True if stored, False if rejected by safety rules.
        """
        if len(answer.strip()) < MIN_ANSWER_LENGTH:
            return False
        if answer.strip().startswith(OUT_OF_SCOPE_PREFIX):
            return False

        embedder = _get_embedder()
        embedding = embedder.embed_query(query)

        self._entries.append({
            "query":         query,
            "context_key":   self._make_context_key(student_context),
            "embedding":     embedding,     # list[float], 384 dims
            "answer":        answer,
            "sources":       sources,
            "hit_count":     0,
            "cached_at":     datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
        })

        self._save()
        return True

    def size(self) -> int:
        return len(self._entries)

    def invalidate(self) -> None:
        self._entries = []
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
