"""
AcaRAG Pro — FastAPI Backend  v3.0 (CacheAG Edition)
------------------------------------------------------
Endpoints:
  GET  /health            → liveness check
  POST /chat              → CacheAG + RAG + memory pipeline
  POST /ingest            → re-index PDFs + invalidate all cache levels
  GET  /cache/stats       → real-time cache performance metrics
  POST /cache/invalidate  → manual cache wipe

Execution flow for /chat:
  1. Validate request, assign session_id
  2. Level 1 — Exact cache check      (~0 ms if hit)
  3. Level 2 — Semantic cache check   (~80 ms if hit)
  4. Level 3 — Load conversation history (SessionMemory)
  5. Reformulate query if follow-up detected
  6. Run RAG pipeline (ChromaDB + GROQ)  ← only reached on cache miss
  7. Store result into cache
  8. Update session memory
  9. Return response

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import shutil
import threading
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Core RAG engine — called only on cache miss
from rag import answer_question, reset_vectorstore

# CacheAG — multi-level cache system
from cache.cache_manager import CacheManager

# Enhanced session memory (Level 3 — Conversation Cache)
from memory.session_memory import SessionMemory

# Domain-aware query spell correction
from autocorrect import correct_query

FRONTEND_DIR  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset")
)

# ---------------------------------------------------------------------------
# Module-level singletons (one instance for the lifetime of the server)
# ---------------------------------------------------------------------------
cache_manager  = CacheManager()
session_memory = SessionMemory()

app = FastAPI(
    title="AcaRAG Pro",
    description=(
        "Context-Aware Academic Assistant powered by GROQ LLM + "
        "3-Level Cache Augmented Generation (CacheAG)"
    ),
    version="3.0.0",
)

# Allow the local HTML frontend to call the API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None   # Client sends this on follow-ups
    # Student profile — sent by frontend after the profile setup step
    year:     Optional[str] = None   # "1" | "2" | "3" | "4"
    semester: Optional[str] = None   # "1" | "2"
    branch:   Optional[str] = None   # "CSE" | "ECE" | "EEE" | "ME" | "CIVIL" | "IT"
    name:     Optional[str] = None   # student first name (personalisation only)


class SourceCitation(BaseModel):
    document: str
    page_number: int | str


class ChatResponse(BaseModel):
    answer:           str
    sources:          list
    session_id:       str
    cache_hit:        Optional[str] = None   # "exact" | "semantic" | null (RAG path)
    corrected_query:  Optional[str] = None   # non-null if autocorrect changed the query


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check. Frontend polls this to confirm the server is up."""
    return {"status": "ok", "service": "AcaRAG Pro", "version": "3.0.0 (CacheAG)"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main CacheAG + RAG pipeline endpoint.

    Priority order:
      Cache hit (exact)    → ~0 ms   — identical query answered before
      Cache hit (semantic) → ~80 ms  — paraphrase of answered query
      RAG pipeline         → ~2200 ms — full retrieval + GROQ generation
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 0: Auto-correct obvious typos in the query
    corrected, was_corrected = correct_query(question)
    corrected_query_out = corrected if was_corrected else None
    if was_corrected:
        question = corrected   # use corrected question downstream

    # Step 1: Session management
    session_id = req.session_id or str(uuid.uuid4())

    # Build student context early — it is part of the cache key
    student_context = {
        "year":     req.year     or "",
        "semester": req.semester or "",
        "branch":   req.branch   or "",
        "name":     req.name     or "",
    }

    # ------------------------------------------------------------------
    # Steps 2+3: Cache check — Levels 1 (exact) and 2 (semantic)
    # Short-circuit: if cached, skip ChromaDB and GROQ entirely.
    # Cache key includes branch/year/semester so different student
    # profiles never share each other's cached answers.
    # ------------------------------------------------------------------
    cached = cache_manager.check(question, student_context)

    if cached is not None:
        # Cache HIT — update session memory so conversation stays coherent
        session_memory.add_turn(session_id, question, cached["answer"])
        return ChatResponse(
            answer=cached["answer"],
            sources=cached.get("sources", []),
            session_id=session_id,
            cache_hit=cached.get("cache_hit"),   # "exact" or "semantic"
            corrected_query=corrected_query_out,
        )

    # ------------------------------------------------------------------
    # Step 4: Load conversation history — Level 3 (SessionMemory)
    # ------------------------------------------------------------------
    history = session_memory.get_history(session_id)

    # ------------------------------------------------------------------
    # Step 5: Follow-up detection + query reformulation
    # If the query is a vague follow-up (e.g., "what about that rule?"),
    # SessionMemory prepends last question for anchored retrieval.
    # rag.py's own reformulation also runs inside answer_question().
    # ------------------------------------------------------------------
    effective_question = session_memory.reformulate(session_id, question)

    # ------------------------------------------------------------------
    # Step 6: Full RAG pipeline (only on cache miss)
    # ------------------------------------------------------------------
    result  = answer_question(effective_question, history, student_context)
    answer  = result.get("answer", "")
    sources = result.get("sources", [])

    # ------------------------------------------------------------------
    # Step 7: Store into cache (safety gates enforced inside store())
    # Store original question — not reformulated — as the cache key,
    # so future identical raw queries still get a hit.
    # Do NOT cache LLM errors (rate limits, timeouts) — the student
    # should get a real answer on retry, not a cached error forever.
    # ------------------------------------------------------------------
    if not answer.startswith("LLM error:"):
        cache_manager.store(question, answer, sources, student_context)

    # ------------------------------------------------------------------
    # Step 8: Update session memory with original question
    # ------------------------------------------------------------------
    session_memory.add_turn(session_id, question, answer)

    # ------------------------------------------------------------------
    # Step 9: Return (cache_hit=None signals this was a live RAG response)
    # ------------------------------------------------------------------
    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session_id,
        cache_hit=None,
        corrected_query=corrected_query_out,
    )


def _run_ingest_background():
    """Background thread: run ingestion, reload vectorstore, clear cache."""
    try:
        import ingest as _ingest_mod
        _ingest_mod.run_ingestion()
        reset_vectorstore()          # reload from newly built DB on next query
        cache_manager.invalidate_all()
    except Exception as e:
        import ingest as _ingest_mod
        _ingest_mod.ingest_progress.update({
            "running": False, "stage": "error", "error": str(e),
            "message": f"Ingestion failed: {e}",
        })


@app.post("/ingest")
def ingest():
    """
    Start a background re-index of all PDFs.
    Returns immediately (202) — poll GET /admin/ingest-status for progress.
    """
    import ingest as _ingest_mod
    if _ingest_mod.ingest_progress.get("running"):
        return {"status": "already_running", "message": "Re-index already in progress."}
    t = threading.Thread(target=_run_ingest_background, daemon=True)
    t.start()
    return {"status": "started", "message": "Re-indexing started in background."}


@app.get("/admin/ingest-status")
def ingest_status():
    """Real-time ingestion progress — polled by admin UI every second."""
    import ingest as _ingest_mod
    return _ingest_mod.ingest_progress


@app.get("/cache/stats")
def cache_stats():
    """
    Real-time CacheAG performance metrics.
    Use during demo to prove GROQ calls saved and latency improvement.
    """
    stats = cache_manager.get_stats()
    stats["active_sessions"] = session_memory.active_session_count()
    return stats


@app.post("/cache/invalidate")
def cache_invalidate():
    """
    Manually wipe all cache levels without re-ingesting documents.
    Useful during testing or demo resets.
    """
    cache_manager.invalidate_all()
    return {"status": "success", "message": "All cache levels cleared."}


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.get("/admin/documents")
def list_documents():
    """List all PDF files in the dataset folder with their sizes."""
    if not os.path.isdir(DATASET_DIR):
        return {"documents": []}
    docs = []
    for fname in sorted(os.listdir(DATASET_DIR)):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(DATASET_DIR, fname)
            docs.append({
                "filename": fname,
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
            })
    return {"documents": docs}


@app.post("/admin/upload")
async def upload_document(
    file: UploadFile = File(...),
    year: str     = Form("all"),
    semester: str = Form("all"),
    branch: str   = Form("all"),
    doc_type: str = Form("syllabus"),
):
    """
    Upload a PDF to the dataset folder and add its metadata hint to ingest.py's
    _MANUAL_METADATA dict, then trigger a full re-index.

    NOTE: Re-indexing rebuilds the entire ChromaDB from scratch and takes
    several minutes on large datasets. All cache is invalidated automatically.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(DATASET_DIR, exist_ok=True)
    dest_path = os.path.join(DATASET_DIR, file.filename)

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Persist metadata hint so future re-ingests tag this file correctly
    _persist_manual_metadata(file.filename, year, semester, branch, doc_type)

    # Trigger full re-index in background (returns immediately)
    import ingest as _ingest_mod
    if not _ingest_mod.ingest_progress.get("running"):
        t = threading.Thread(target=_run_ingest_background, daemon=True)
        t.start()

    return {
        "status":   "success",
        "message":  f"{file.filename} uploaded. Re-indexing started — poll /admin/ingest-status for progress.",
        "filename": file.filename,
    }


@app.delete("/admin/documents/{filename}")
def delete_document(filename: str):
    """Remove a PDF from the dataset and trigger re-index."""
    fpath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        os.remove(fpath)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not delete: {e}")

    import ingest as _ingest_mod
    if not _ingest_mod.ingest_progress.get("running"):
        t = threading.Thread(target=_run_ingest_background, daemon=True)
        t.start()
    return {"status": "success", "message": f"{filename} deleted. Re-indexing started — poll /admin/ingest-status."}


def _persist_manual_metadata(filename: str, year: str, semester: str,
                              branch: str, doc_type: str) -> None:
    """
    Append (or update) an entry in ingest.py's _MANUAL_METADATA dict so the
    uploaded file is correctly tagged on every future re-ingest.
    Uses a simple text-insertion approach — only edits the specific dict block.
    """
    ingest_path = os.path.join(os.path.dirname(__file__), "ingest.py")
    try:
        with open(ingest_path, "r", encoding="utf-8") as f:
            src = f.read()

        entry = (
            f'    "{filename}": '
            f'{{"year": "{year}", "semester": "{semester}", '
            f'"branch": "{branch}", "doc_type": "{doc_type}"}},\n'
        )

        # If file already has an entry, replace it
        import re as _re
        existing_pattern = _re.compile(
            rf'    "{_re.escape(filename)}":\s*\{{[^}}]*\}},?\n'
        )
        if existing_pattern.search(src):
            src = existing_pattern.sub(entry, src)
        else:
            # Insert before the closing brace of _MANUAL_METADATA
            # Find the dict end marker: a line with just "}" after the dict
            marker = "_MANUAL_METADATA: dict = {"
            idx = src.find(marker)
            if idx != -1:
                # Find the closing } of this dict
                brace_start = src.index("{", idx)
                depth, pos = 0, brace_start
                while pos < len(src):
                    if src[pos] == "{":
                        depth += 1
                    elif src[pos] == "}":
                        depth -= 1
                        if depth == 0:
                            src = src[:pos] + entry + src[pos:]
                            break
                    pos += 1

        with open(ingest_path, "w", encoding="utf-8") as f:
            f.write(src)
    except Exception:
        pass  # Non-critical: metadata will be auto-detected by ingest.py heuristics


# ---------------------------------------------------------------------------
# Serve frontend from the same FastAPI server
# ---------------------------------------------------------------------------
# This MUST be declared AFTER all API routes so API endpoints take priority.
# Students visit http://localhost:8000 in their browser — no separate server needed.
# StaticFiles with html=True serves index.html for "/" automatically.
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
