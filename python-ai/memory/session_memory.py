"""
AcaRAG Pro — Level 3: Conversation Cache / Enhanced Session Memory
-------------------------------------------------------------------
Provides per-session conversation history with:

  1. Turn storage   — keeps last MAX_TURNS Q&A pairs per session.
  2. Follow-up detection — identifies short or continuation queries.
  3. Query reformulation — prefixes follow-up queries with context so
     ChromaDB retrieval finds the right documents even for vague questions.
  4. Conversation summary — injects a compact history block into the
     RAG prompt, giving the LLM awareness of what was already discussed.
  5. Session metadata — tracks created_at, turn_count, last_active.

Relationship to existing memory.py:
  memory.py   → original 3-turn dict, unchanged (do not delete it).
  session_memory.py → extended version used by the upgraded app.py.

Both expose compatible interfaces: the session_id key, question/answer fields.

Storage: in-process Python dict (intentionally ephemeral — conversation
context does not need to survive server restarts for this use-case).
"""

import re
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_TURNS         = 5     # Keep last N turns per session
SUMMARY_TURNS     = 2     # How many recent turns to include in LLM prompt

# Keywords / patterns that signal a follow-up question
_FOLLOW_UP_TOKENS = {
    "what about", "and what", "tell me more", "elaborate",
    "explain further", "what if", "in that case", "also", "similarly",
    "what else", "how about", "continue", "can you explain", "more details",
    "what does that mean", "which one", "the same", "that rule", "those",
    "explain each", "describe each", "list each", "give details",
    "about that", "related to that", "for that", "about this", "related to this",
}

# Queries ending with a pronoun almost certainly reference prior context
# e.g. "Explain each topics related to it" → ends with "it"
_PRONOUN_ENDING = re.compile(r'\b(it|them|those|that|this)\s*\??\s*$')

# Queries this short (word count) are almost always follow-ups
SHORT_QUERY_WORDS = 3


class SessionMemory:
    """
    Enhanced conversation memory store.
    One instance shared across the FastAPI process (module-level singleton
    is created in app.py).
    """

    def __init__(self):
        # { session_id: { "history": [...], "created_at": str, "turn_count": int } }
        self._sessions: dict = {}

    # ------------------------------------------------------------------
    # History access
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list:
        """Returns full turn history for the session (empty list if new)."""
        return self._sessions.get(session_id, {}).get("history", [])

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """
        Append a Q&A turn to the session.
        Automatically prunes to MAX_TURNS.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "history":     [],
                "created_at":  datetime.now().isoformat(),
                "turn_count":  0,
                "last_active": datetime.now().isoformat(),
            }

        session = self._sessions[session_id]
        session["history"].append({
            "question":   question,
            "answer":     answer,
            "turn":       session["turn_count"] + 1,
            "timestamp":  datetime.now().isoformat(),
        })
        session["turn_count"] += 1
        session["last_active"] = datetime.now().isoformat()

        # Prune — keep only the last MAX_TURNS
        if len(session["history"]) > MAX_TURNS:
            session["history"] = session["history"][-MAX_TURNS:]

    # ------------------------------------------------------------------
    # Query reformulation (CAG — Conversation-Aware Generation)
    # ------------------------------------------------------------------

    def is_follow_up(self, session_id: str, query: str) -> bool:
        """
        Returns True if the query appears to be a follow-up to the
        ongoing conversation rather than a standalone question.
        """
        history = self.get_history(session_id)
        if not history:
            return False  # No prior context — cannot be a follow-up

        lower = query.lower().strip()

        if len(lower.split()) <= SHORT_QUERY_WORDS:
            return True

        if any(token in lower for token in _FOLLOW_UP_TOKENS):
            return True

        # "Explain each topics related to it" — pronoun at end references prior turn
        if _PRONOUN_ENDING.search(lower):
            return True

        return False

    def reformulate(self, session_id: str, query: str) -> str:
        """
        If the query is a follow-up, prepend the last question as context
        so RAG retrieval is anchored to the correct topic.

        Example:
          History Q: "What is the attendance regulation?"
          New query : "What happens if I fail?"
          Reformulated: "Regarding 'What is the attendance regulation?': What happens if I fail?"

        If not a follow-up, returns the query unchanged.
        """
        if not self.is_follow_up(session_id, query):
            return query

        history = self.get_history(session_id)
        last_question = history[-1]["question"]

        return f"Regarding '{last_question}': {query}"

    # ------------------------------------------------------------------
    # Conversation summary for LLM prompt injection
    # ------------------------------------------------------------------

    def get_summary(self, session_id: str) -> str:
        """
        Returns a compact 2-turn conversation summary string.
        Injected into the RAG prompt so the LLM knows what was discussed.
        Returns empty string if no history (no overhead for new sessions).
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        recent = history[-SUMMARY_TURNS:]
        lines = []
        for turn in recent:
            lines.append(f"Student: {turn['question']}")
            # Truncate long answers to avoid blowing up context window
            ans = turn["answer"]
            if len(ans) > 250:
                ans = ans[:247] + "..."
            lines.append(f"Assistant: {ans}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """Remove all history for a session (called on "New Chat")."""
        self._sessions.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_metadata(self, session_id: str) -> dict:
        """Returns session metadata without history (safe to expose in API)."""
        session = self._sessions.get(session_id, {})
        return {
            "session_id":  session_id,
            "turn_count":  session.get("turn_count", 0),
            "created_at":  session.get("created_at"),
            "last_active": session.get("last_active"),
        }

    def active_session_count(self) -> int:
        return len(self._sessions)
