"""
AcaRAG Pro — Document Ingestion Pipeline
-----------------------------------------
Creates a lightweight NumPy-based vector index.

What changed:
  - ChromaDB storage removed from ingestion.
  - Embeddings are stored in embeddings.npy.
  - Document text + metadata are stored in documents.json.
  - Uses the existing paraphrase-MiniLM-L3-v2 embedding model.
  - Existing metadata extraction and OCR logic are preserved.

Run independently:
    python ingest.py
"""

import os
import re
import shutil
import json
import sqlite3

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset")

# We keep the existing folder name so the rest of the project can continue
# using the same path.
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


# ---------------------------------------------------------------------------
# Live progress state — updated during run_ingestion(), polled by app.py
# ---------------------------------------------------------------------------

ingest_progress: dict = {
    "running": False,
    "stage": "idle",
    "files_done": 0,
    "files_total": 0,
    "current_file": "",
    "chunks_total": 0,
    "message": "",
    "error": "",
}


OCR_THRESHOLD = 80


# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    import fitz

    FITZ_AVAILABLE = True

except ImportError:
    FITZ_AVAILABLE = False


try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True

except ImportError:
    try:
        from PyPDF2 import PdfReader

        PYPDF_AVAILABLE = True

    except ImportError:
        PYPDF_AVAILABLE = False


try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True

except ImportError:
    PDFPLUMBER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Filename metadata extraction
# ---------------------------------------------------------------------------

def _extract_branch(filename: str) -> str:
    """
    Detect the branch/department from the filename.

    Returns:
        CSE, ECE, EEE, IT, ME, CIVIL, AI&ML, AIDS, CYBER
        or 'all' for cross-branch documents.
    """

    name = (
        filename.lower()
        .replace("b.tech", "")
        .replace("btech", "")
        .replace("_", "-")
    )

    # Specific CSE specialisations before generic "cse"
    if re.search(r'csecs|cyber.?security|cyber', name):
        return "CYBER"

    if re.search(r'cseaiml|ai.?ml', name):
        return "AI&ML"

    if re.search(r'ai.?ds|adsr|cseds', name):
        return "AIDS"

    if "ece" in name:
        return "ECE"

    if "eee" in name:
        return "EEE"

    if "civil" in name or re.search(r'(?<![a-z])ce(?![a-z])', name):
        return "CIVIL"

    if re.search(r'(?<![a-z])it(?![a-z])', name):
        return "IT"

    if re.search(r'(?<![a-z])me(?![a-z])|mech', name):
        return "ME"

    if "cse" in name:
        return "CSE"

    return "all"


# ---------------------------------------------------------------------------
# Manual metadata overrides
# ---------------------------------------------------------------------------

_MANUAL_METADATA: dict = {

    # ECE Year 1
    "ECE IB TechR23Syllabus.pdf": {
        "year": "1",
        "branch": "ECE",
        "doc_type": "syllabus",
    },

    # ECE Year 2
    "R23_ECE_II-Year-I-Sem-and-II-Sem-Syllabus.pdf": {
        "year": "2",
        "branch": "ECE",
        "doc_type": "syllabus",
    },

    # ME Year 4
    "IV-YR-COURSE-STRUCTURE-SYLLABUS.pdf": {
        "year": "4",
        "branch": "ME",
        "doc_type": "syllabus",
    },

    # ME Year 2
    "MEIISyllabus.pdf.pdf": {
        "year": "2",
        "branch": "ME",
        "doc_type": "syllabus",
    },

    # ME Year 1
    "ME I Syllabus.pdf": {
        "year": "1",
        "branch": "ME",
        "doc_type": "syllabus",
    },

    # EEE Year 1 R23
    "R23 EEE Syllabus.pdf": {
        "year": "1",
        "branch": "EEE",
        "doc_type": "syllabus",
    },
}


# ---------------------------------------------------------------------------
# Files ingested for more than one branch
# ---------------------------------------------------------------------------

_MULTI_BRANCH_FILES: dict = {

    # Year 1 common syllabus shared by AI&Data Science and AI&ML
    "ADSR23IYr.pdf": ["AIDS", "AI&ML"],
}


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Parse SVECW naming convention to extract:

        year
        semester
        branch
        doc_type
    """

    name = filename.lower()

    meta = {
        "source": filename,
        "year": "all",
        "semester": "all",
        "branch": "all",
        "doc_type": "general",
    }

    # Manual override
    if filename in _MANUAL_METADATA:
        meta.update(_MANUAL_METADATA[filename])
        return meta

    # Pattern:
    # 11_R23_Reg_Jan26.pdf
    # 21_Reg_Nov25.pdf
    m = re.match(r'^(\d)(\d)[_\-A-Za-z]', filename)

    if m and m.group(2) in ("1", "2"):
        meta["year"] = m.group(1)
        meta["semester"] = m.group(2)
        meta["doc_type"] = "exam_schedule"

        return meta

    # Academic calendars
    m = re.match(
        r'^2025-26-(I{1,3}V?|IV)-',
        filename,
        re.IGNORECASE,
    )

    if m:
        roman = m.group(1).upper()

        year_map = {
            "I": "1",
            "II": "2",
            "III": "3",
            "IV": "4",
        }

        meta["year"] = year_map.get(roman, "all")
        meta["doc_type"] = "calendar"

        return meta

    # Normalize Roman numeral search
    norm = re.sub(r'[\s_]+', '-', filename)

    # IYr / IIYr / IIIYr / IVYr
    m = re.search(
        r'(IV|III|II|I)Yr',
        norm,
        re.IGNORECASE,
    )

    if m:
        roman = m.group(1).upper()

        year_map = {
            "I": "1",
            "II": "2",
            "III": "3",
            "IV": "4",
        }

        meta["year"] = year_map.get(roman, "all")
        meta["branch"] = _extract_branch(filename)
        meta["doc_type"] = "syllabus"

        return meta

    # IB Tech = Year 1
    if (
        re.search(r'\bIB[\s\-]', norm, re.IGNORECASE)
        or re.search(r'\bI-B\b', norm, re.IGNORECASE)
    ):
        meta["year"] = "1"
        meta["branch"] = _extract_branch(filename)
        meta["doc_type"] = "syllabus"

        return meta

    # Digit-based year
    # Example:
    # 4-1-CSE.pdf
    m = re.match(r'^(\d)-(\d)-', filename)

    if m:
        meta["year"] = m.group(1)
        meta["semester"] = m.group(2)
        meta["branch"] = _extract_branch(filename)
        meta["doc_type"] = "syllabus"

        return meta

    # Roman numeral syllabus
    m = re.search(
        r'\b(IV|III|II|I)\b',
        norm,
        re.IGNORECASE,
    )

    if m:
        roman = m.group(1).upper()

        year_map = {
            "I": "1",
            "II": "2",
            "III": "3",
            "IV": "4",
        }

        meta["year"] = year_map.get(roman, "all")
        meta["branch"] = _extract_branch(filename)
        meta["doc_type"] = "syllabus"

        return meta

    # Regulations
    if "regulation" in name or name.startswith("re"):
        meta["doc_type"] = "regulation"
        return meta

    # Syllabus fallback
    if "syllabus" in name or "syllabi" in name:
        meta["branch"] = _extract_branch(filename)
        meta["doc_type"] = "syllabus"

        return meta

    # Utilities
    if (
        "holiday" in name
        or "library" in name
        or "timing" in name
    ):
        meta["doc_type"] = "general"

        return meta

    return meta


# ---------------------------------------------------------------------------
# OCR initialisation
# ---------------------------------------------------------------------------

def _init_ocr():
    """
    Returns True if pytesseract + fitz are available.
    """

    if not FITZ_AVAILABLE:
        print(
            "WARNING: PyMuPDF not installed — OCR disabled "
            "(pip install pymupdf)"
        )

        return None

    try:
        import pytesseract

        # Windows Tesseract locations
        if os.name == "nt":

            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(
                    r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
                ),
            ]

            for path in candidates:

                if os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

        pytesseract.get_tesseract_version()

        print(
            "pytesseract + PyMuPDF available — "
            "scanned PDFs will be OCR-processed."
        )

        return True

    except Exception as e:

        print(
            f"WARNING: Tesseract OCR unavailable "
            f"({type(e).__name__}: {e})"
        )

        print(
            "         Install Tesseract: "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        )

        print(
            "         Scanned PDFs will be skipped. "
            "Text-based PDFs will still be indexed."
        )

        return None


# ---------------------------------------------------------------------------
# Text extraction using pdfplumber
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(
    pdf_path: str,
    filename: str,
    base_meta: dict,
    ocr_available: bool = False,
) -> list:

    """
    Primary extractor using pdfplumber.

    Preserves table layout and column order.
    Falls back to pypdf if needed.
    """

    documents = []

    try:

        with pdfplumber.open(pdf_path) as pdf:

            native_count = 0
            skipped_count = 0

            for page_num, page in enumerate(pdf.pages):

                text = (page.extract_text() or "").strip()

                if len(text) >= OCR_THRESHOLD:

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                **base_meta,
                                "page_number": page_num + 1,
                                "page": page_num + 1,
                            },
                        )
                    )

                    native_count += 1

                else:
                    skipped_count += 1

            tag = f"{native_count} pages"

            if skipped_count:

                hint = (
                    "will be OCR-processed"
                    if ocr_available
                    else "install Tesseract for OCR"
                )

                tag += (
                    f" ({skipped_count} scanned/blank — {hint})"
                )

            print(f"  -> {tag} | {filename}")

            return documents

    except Exception as e:

        print(
            f"  pdfplumber failed for {filename} "
            f"({e}), falling back to pypdf"
        )

        return _extract_with_pypdf(
            pdf_path,
            filename,
            base_meta,
        )


# ---------------------------------------------------------------------------
# Text extraction using pypdf
# ---------------------------------------------------------------------------

def _extract_with_pypdf(
    pdf_path: str,
    filename: str,
    base_meta: dict,
) -> list:

    """
    Fallback extractor using pypdf.
    """

    if not PYPDF_AVAILABLE:

        print(
            f"  ERROR: pypdf not available — "
            f"cannot process {filename}"
        )

        return []

    documents = []

    try:

        reader = PdfReader(pdf_path)

    except Exception as e:

        print(
            f"  ERROR reading {filename}: {e}"
        )

        return []

    native_count = 0
    skipped_count = 0

    for page_num, page in enumerate(reader.pages):

        text = (page.extract_text() or "").strip()

        if len(text) >= OCR_THRESHOLD:

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_meta,
                        "page_number": page_num + 1,
                        "page": page_num + 1,
                    },
                )
            )

            native_count += 1

        else:
            skipped_count += 1

    tag = f"{native_count} pages"

    if skipped_count:

        tag += (
            f" ({skipped_count} scanned/blank skipped — "
            f"install Tesseract for OCR)"
        )

    print(f"  -> {tag} | {filename}")

    return documents


# ---------------------------------------------------------------------------
# OCR extraction
# ---------------------------------------------------------------------------

def _extract_with_fitz_and_ocr(
    pdf_path: str,
    filename: str,
    base_meta: dict,
    ocr_ready,
) -> list:

    """
    Use PyMuPDF to render pages + pytesseract OCR.
    """

    import io
    import pytesseract
    from PIL import Image

    documents = []

    native_cnt = 0
    ocr_cnt = 0

    try:

        pdf = fitz.open(pdf_path)

    except Exception as e:

        print(
            f"  ERROR (fitz) opening {filename}: {e}"
        )

        return _extract_with_pypdf(
            pdf_path,
            filename,
            base_meta,
        )

    for page_num in range(len(pdf)):

        page = pdf[page_num]

        native_text = page.get_text().strip()

        if len(native_text) >= OCR_THRESHOLD:

            text = native_text
            native_cnt += 1

        else:

            try:

                pix = page.get_pixmap(dpi=300)

                img = Image.open(
                    io.BytesIO(
                        pix.tobytes("png")
                    )
                ).convert("RGB")

                text = pytesseract.image_to_string(
                    img,
                    lang="eng",
                ).strip()

                if not text:
                    continue

                ocr_cnt += 1

            except Exception as e:

                print(
                    f"  OCR failed page {page_num + 1} "
                    f"of {filename}: {e}"
                )

                continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    **base_meta,
                    "page_number": page_num + 1,
                    "page": page_num + 1,
                },
            )
        )

    pdf.close()

    print(
        f"  -> {native_cnt} native + "
        f"{ocr_cnt} OCR pages | {filename}"
    )

    return documents


# ---------------------------------------------------------------------------
# Extraction dispatcher
# ---------------------------------------------------------------------------

def extract_documents_from_pdf(
    pdf_path: str,
    filename: str,
    ocr_reader,
) -> list:

    """
    Extraction priority:

      1. pdfplumber
      2. fitz + OCR
      3. pypdf
    """

    base_meta = extract_metadata_from_filename(
        filename
    )

    if PDFPLUMBER_AVAILABLE:

        ocr_available = (
            ocr_reader is not None
            and FITZ_AVAILABLE
        )

        docs = _extract_with_pdfplumber(
            pdf_path,
            filename,
            base_meta,
            ocr_available,
        )

        if docs:
            return docs

        if ocr_available:

            return _extract_with_fitz_and_ocr(
                pdf_path,
                filename,
                base_meta,
                ocr_reader,
            )

        return docs

    if (
        ocr_reader is not None
        and FITZ_AVAILABLE
    ):

        return _extract_with_fitz_and_ocr(
            pdf_path,
            filename,
            base_meta,
            ocr_reader,
        )

    return _extract_with_pypdf(
        pdf_path,
        filename,
        base_meta,
    )


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def run_ingestion():

    global ingest_progress

    ingest_progress = {
        "running": True,
        "stage": "starting",
        "files_done": 0,
        "files_total": 0,
        "current_file": "",
        "chunks_total": 0,
        "message": "Starting ingestion…",
        "error": "",
    }

    print("\n" + "=" * 65)
    print("AcaRAG Pro — Lightweight Document Ingestion Pipeline")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Dependency information
    # -----------------------------------------------------------------------

    if PDFPLUMBER_AVAILABLE:

        print(
            "pdfplumber : available "
            "(primary extractor — preserves table layout)"
        )

    else:

        print(
            "pdfplumber : NOT installed "
            "(pip install pdfplumber) — using pypdf"
        )

    if FITZ_AVAILABLE:

        print(
            "PyMuPDF    : available "
            "(used for OCR rendering)"
        )

    else:

        print(
            "PyMuPDF    : NOT installed "
            "(pip install pymupdf) — OCR disabled"
        )

    # -----------------------------------------------------------------------
    # Dataset validation
    # -----------------------------------------------------------------------

    if not os.path.exists(DATASET_PATH):

        print(
            f"ERROR: Dataset folder not found at "
            f"{DATASET_PATH}"
        )

        return

    pdf_files = sorted(
        [
            f
            for f in os.listdir(DATASET_PATH)
            if f.lower().endswith(".pdf")
        ]
    )

    if not pdf_files:

        print("No PDF files found in dataset folder.")

        ingest_progress.update(
            {
                "running": False,
                "stage": "error",
                "error": "No PDF files found.",
            }
        )

        return

    print(
        f"\nFound {len(pdf_files)} PDF files."
    )

    ingest_progress.update(
        {
            "stage": "extracting",
            "files_total": len(pdf_files),
            "message": (
                f"Extracting text from "
                f"{len(pdf_files)} PDFs…"
            ),
        }
    )

    # -----------------------------------------------------------------------
    # OCR
    # -----------------------------------------------------------------------

    ocr_reader = _init_ocr()

    if ocr_reader is None:

        print(
            "OCR     : disabled "
            "(install Tesseract + "
            "pip install pytesseract pymupdf)\n"
        )

    else:

        print()

    # -----------------------------------------------------------------------
    # Extract all documents
    # -----------------------------------------------------------------------

    all_documents = []

    for filename in pdf_files:

        ingest_progress.update(
            {
                "current_file": filename,
                "message": (
                    f"Reading {filename}…"
                ),
            }
        )

        pdf_path = os.path.join(
            DATASET_PATH,
            filename,
        )

        docs = extract_documents_from_pdf(
            pdf_path,
            filename,
            ocr_reader,
        )

        # Multi-branch files
        if filename in _MULTI_BRANCH_FILES:

            branches = _MULTI_BRANCH_FILES[
                filename
            ]

            print(
                f"  [multi-branch] {filename} "
                f"-> branches: {branches}"
            )

            for branch in branches:

                for doc in docs:

                    import copy

                    d = copy.deepcopy(doc)

                    d.metadata["branch"] = branch

                    all_documents.append(d)

        else:

            all_documents.extend(docs)

        ingest_progress[
            "files_done"
        ] += 1

    # -----------------------------------------------------------------------
    # Validate extraction
    # -----------------------------------------------------------------------

    if not all_documents:

        print(
            "\nERROR: No text extracted. "
            "Check your PDF files."
        )

        ingest_progress.update(
            {
                "running": False,
                "stage": "error",
                "error": "No text extracted.",
            }
        )

        return

    print(
        f"\nTotal pages extracted : "
        f"{len(all_documents)}"
    )

    # -----------------------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------------------

    ingest_progress.update(
        {
            "stage": "chunking",
            "message": (
                f"Splitting "
                f"{len(all_documents)} pages "
                f"into chunks…"
            ),
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140,
        length_function=len,
    )

    chunks = splitter.split_documents(
        all_documents
    )

    print(
        f"Total chunks          : "
        f"{len(chunks)}"
    )

    ingest_progress.update(
        {
            "chunks_total": len(chunks),
            "stage": "indexing",
            "message": (
                f"Building lightweight text index for "
                f"{len(chunks)} chunks…"
            ),
        }
    )

    # -----------------------------------------------------------------------
    # Temporary index directory
    # -----------------------------------------------------------------------

    CHROMA_NEW = CHROMA_PATH + "_new"

    if os.path.exists(CHROMA_NEW):
        shutil.rmtree(CHROMA_NEW)

    os.makedirs(CHROMA_NEW, exist_ok=True)

    # -----------------------------------------------------------------------
    # Lightweight SQLite text index
    # -----------------------------------------------------------------------
    #
    # No local embedding model is used here.
    # SQLite is part of Python's standard library and keeps the Render
    # runtime extremely small compared with Chroma + Sentence Transformers.
    #

    index_db = os.path.join(CHROMA_NEW, "index.db")

    print("Building lightweight SQLite text index...")

    conn = sqlite3.connect(index_db)

    try:
        conn.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                page_content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                source TEXT,
                year TEXT,
                semester TEXT,
                branch TEXT,
                doc_type TEXT,
                page_number INTEGER
            )
            """
        )

        conn.execute(
            "CREATE INDEX idx_documents_source "
            "ON documents(source)"
        )
        conn.execute(
            "CREATE INDEX idx_documents_year "
            "ON documents(year)"
        )
        conn.execute(
            "CREATE INDEX idx_documents_branch "
            "ON documents(branch)"
        )
        conn.execute(
            "CREATE INDEX idx_documents_doc_type "
            "ON documents(doc_type)"
        )
        conn.execute(
            "CREATE INDEX idx_documents_page "
            "ON documents(page_number)"
        )

        rows = []

        for chunk in chunks:
            metadata = dict(chunk.metadata or {})

            page_number = metadata.get("page_number")
            try:
                page_number = int(page_number)
            except (TypeError, ValueError):
                page_number = None

            rows.append(
                (
                    chunk.page_content,
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    ),
                    str(metadata.get("source", "")),
                    str(metadata.get("year", "all")),
                    str(metadata.get("semester", "all")),
                    str(metadata.get("branch", "all")),
                    str(metadata.get("doc_type", "general")),
                    page_number,
                )
            )

            if len(rows) >= 500:
                conn.executemany(
                    """
                    INSERT INTO documents (
                        page_content,
                        metadata_json,
                        source,
                        year,
                        semester,
                        branch,
                        doc_type,
                        page_number
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                rows.clear()

        if rows:
            conn.executemany(
                """
                INSERT INTO documents (
                    page_content,
                    metadata_json,
                    source,
                    year,
                    semester,
                    branch,
                    doc_type,
                    page_number
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        conn.execute("VACUUM")
        conn.commit()

    finally:
        conn.close()

    # Save basic index information for diagnostics.
    index_info_path = os.path.join(
        CHROMA_NEW,
        "index_info.json",
    )

    with open(
        index_info_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "index_type": "sqlite_text",
                "document_count": len(chunks),
                "embedding_model": None,
                "embedding_dimension": None,
            },
            f,
            indent=2,
        )

    print(
        f"[OK] Saved {len(chunks)} documents to SQLite index."
    )

    print(
        f"[OK] Index database: {index_db}"
    )

    # -----------------------------------------------------------------------
    # Swap new index into place
    # -----------------------------------------------------------------------

    ingest_progress.update(
        {
            "stage": "finalizing",
            "message": (
                "Swapping in new lightweight index…"
            ),
        }
    )

    try:

        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        os.rename(
            CHROMA_NEW,
            CHROMA_PATH,
        )

        print(
            f"\n[OK] Ingestion complete! "
            f"{len(chunks)} chunks indexed."
        )

        print(
            "[OK] Lightweight SQLite index created:"
        )

        print(
            "     - index.db"
        )

        print(
            "     - index_info.json"
        )

        ingest_progress.update(
            {
                "running": False,
                "stage": "done",
                "message": (
                    f"Done! {len(chunks)} chunks "
                    f"indexed and ready."
                ),
            }
        )

    except PermissionError:

        print(
            f"\n[OK] Lightweight index written "
            f"to chroma_db_new "
            f"({len(chunks)} chunks)."
        )

        print(
            "     The server is holding "
            "the old index open."
        )

        print(
            "     Stop the server, then rename "
            "chroma_db_new to chroma_db."
        )

        ingest_progress.update(
            {
                "running": False,
                "stage": "done",
                "message": (
                    f"Done! {len(chunks)} chunks "
                    f"in chroma_db_new. "
                    f"Restart server to activate."
                ),
            }
        )

    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()
