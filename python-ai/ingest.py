"""
AcaRAG Pro — Document Ingestion Pipeline (v2 — fixed for installed packages)
------------------------------------------------------------------------------
What changed from v1:
  - Replaced `fitz` (NOT installed) with `pypdf` (IS installed) as the primary extractor.
  - fitz/PyMuPDF is used ONLY when available (optional, for OCR rendering).
  - Extracts year, semester, doc_type metadata from filename naming convention:
      11_  → year=1, semester=1   (Year 1, Sem 1 exam schedules)
      21_  → year=2, semester=1   etc.
      2025-26-I-B.Tech → academic calendar (year=all)
      regulations, syllabus files → year=all, semester=all
  - This metadata enables ChromaDB filtered retrieval by student's year/semester.
  - 24 scanned PDFs are handled by OCR if EasyOCR + PyMuPDF are installed.
    Without them, 51 text-based PDFs are indexed (full coverage for most queries).

Run independently:
    python ingest.py
"""

import os
import re
import shutil
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset")
CHROMA_PATH  = os.path.join(os.path.dirname(__file__), "chroma_db")

# ---------------------------------------------------------------------------
# Live progress state — updated during run_ingestion(), polled by app.py
# ---------------------------------------------------------------------------
ingest_progress: dict = {
    "running":      False,
    "stage":        "idle",        # "extracting" | "embedding" | "finalizing" | "done" | "error"
    "files_done":   0,
    "files_total":  0,
    "current_file": "",
    "chunks_total": 0,
    "message":      "",
    "error":        "",
}
OCR_THRESHOLD = 80   # chars below this → page is likely scanned

# ---------------------------------------------------------------------------
# Optional dependency checks (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF — needed to render pages for OCR
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
    Checks from most-specific to least-specific to avoid false matches.
    Returns a branch code (CSE, ECE, EEE, IT, ME, CIVIL, AI&ML, AIDS, CYBER)
    or 'all' for cross-branch documents (regulations, calendars, faculty lists).
    """
    # Normalise: lowercase, remove "B.Tech"/"BTech", treat underscores as hyphens
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
    # "IT" — require non-letter boundaries to avoid matching "width", "bit" etc.
    if re.search(r'(?<![a-z])it(?![a-z])', name):
        return "IT"
    # "ME" — mechanical
    if re.search(r'(?<![a-z])me(?![a-z])|mech', name):
        return "ME"
    if "cse" in name:
        return "CSE"
    return "all"


# ---------------------------------------------------------------------------
# Manual metadata overrides
# Files whose names don't follow the standard pattern need explicit tagging.
# Keys are exact filenames (case-sensitive, as they appear in the dataset folder).
# ---------------------------------------------------------------------------
_MANUAL_METADATA: dict = {
    # ECE Year 1 — "IB" (I B.Tech) not parsed as Roman numeral I
    "ECE IB TechR23Syllabus.pdf":                  {"year": "1", "branch": "ECE",   "doc_type": "syllabus"},
    # ECE Year 2 — regex was picking "I" from "I-Sem" instead of "II" from "II-Year"
    "R23_ECE_II-Year-I-Sem-and-II-Sem-Syllabus.pdf": {"year": "2", "branch": "ECE",   "doc_type": "syllabus"},
    # ME Year 4 — no "ME/Mechanical" keyword in filename
    "IV-YR-COURSE-STRUCTURE-SYLLABUS.pdf":          {"year": "4", "branch": "ME",    "doc_type": "syllabus"},
    # ME Year 2 — "MEII" word-boundary issue + double .pdf extension
    "MEIISyllabus.pdf.pdf":                         {"year": "2", "branch": "ME",    "doc_type": "syllabus"},
    # ME Year 1 — verify ME branch (no "mech" keyword)
    "ME I Syllabus.pdf":                            {"year": "1", "branch": "ME",    "doc_type": "syllabus"},
    # EEE Year 1 R23 — no year digit in name; separate R22/R23 II and III files exist
    "R23 EEE Syllabus.pdf":                         {"year": "1", "branch": "EEE",   "doc_type": "syllabus"},
}

# Files that should be ingested for MORE THAN ONE branch.
# Each entry maps filename → list of branch codes to tag.
# The file is ingested once per branch with identical content.
_MULTI_BRANCH_FILES: dict = {
    # Year 1 common syllabus shared by AI&Data Science and AI&ML streams
    "ADSR23IYr.pdf": ["AIDS", "AI&ML"],
}


def extract_metadata_from_filename(filename: str) -> dict:
    """
    Parse the SVECW naming convention to extract year, semester, branch, and doc_type.

    Convention observed in dataset:
      YS_<regulation>_<type>_<date>.pdf  where Y=year(1-4), S=semester(1-2)
      Examples:
        11_R23_Reg_Jan26.pdf              → year=1, sem=1, branch=all,   doc_type=exam_schedule
        21_Reg_Nov25.pdf                  → year=2, sem=1, branch=all,   doc_type=exam_schedule
        2025-26-I-B.Tech-Academic-Calendar.pdf → year=1, sem=all, branch=all, doc_type=calendar
        regulations R22.pdf               → year=all, sem=all, branch=all,  doc_type=regulation
        II-CSE-SYLLABUS.pdf               → year=2, sem=all, branch=CSE,  doc_type=syllabus
        R22-IV-B.Tech-IT-Syllabus.pdf     → year=4, sem=all, branch=IT,   doc_type=syllabus
        CYBER-III-R23.pdf                 → year=3, sem=all, branch=CYBER, doc_type=syllabus
    """
    name = filename.lower()
    meta = {
        "source":    filename,
        "year":      "all",
        "semester":  "all",
        "branch":    "all",
        "doc_type":  "general",
    }

    # Manual override takes highest priority
    if filename in _MANUAL_METADATA:
        meta.update(_MANUAL_METADATA[filename])
        return meta

    # Pattern: starts with two digits (year + semester) — exam schedules
    # Handles both "41_AS_March26.pdf" (underscore) and "41Reg_R23_Oct25.pdf" (letter)
    m = re.match(r'^(\d)(\d)[_\-A-Za-z]', filename)
    if m and m.group(2) in ("1", "2"):   # semester must be 1 or 2
        meta["year"]     = m.group(1)
        meta["semester"] = m.group(2)
        meta["doc_type"] = "exam_schedule"
        # Exam schedules are cross-branch so branch stays "all"
        return meta

    # Academic calendars: "2025-26-I-B.Tech..." or "2025-26-II-..."
    m = re.match(r'^2025-26-(I{1,3}V?|IV)-', filename, re.IGNORECASE)
    if m:
        roman = m.group(1).upper()
        year_map = {"I": "1", "II": "2", "III": "3", "IV": "4"}
        meta["year"]     = year_map.get(roman, "all")
        meta["doc_type"] = "calendar"
        # Calendars are cross-branch
        return meta

    # Normalise for Roman-numeral search: replace spaces and underscores with hyphens
    # so that "ME_II_Syllabus" and "ME II Syllabus" both yield "-II-" with clean boundaries.
    norm = re.sub(r'[\s_]+', '-', filename)

    # Year encoded as "IYr", "IIYr", "IIIYr", "IVYr" — e.g. "IT R23 IYr.pdf"
    # Check this BEFORE generic Roman numeral search to avoid false matches on "I" in other words
    m = re.search(r'(IV|III|II|I)Yr', norm, re.IGNORECASE)
    if m:
        roman = m.group(1).upper()
        year_map = {"I": "1", "II": "2", "III": "3", "IV": "4"}
        meta["year"]     = year_map.get(roman, "all")
        meta["branch"]   = _extract_branch(filename)
        meta["doc_type"] = "syllabus"
        return meta

    # "IB Tech" / "IB-Tech" = I B.Tech = Year 1
    if re.search(r'\bIB[\s\-]', norm, re.IGNORECASE) or re.search(r'\bI-B\b', norm, re.IGNORECASE):
        meta["year"]     = "1"
        meta["branch"]   = _extract_branch(filename)
        meta["doc_type"] = "syllabus"
        return meta

    # Digit-based year in filename: "4-1-CSE.pdf" → year=4, sem=1
    m = re.match(r'^(\d)-(\d)-', filename)
    if m:
        meta["year"]     = m.group(1)
        meta["semester"] = m.group(2)
        meta["branch"]   = _extract_branch(filename)
        meta["doc_type"] = "syllabus"
        return meta

    # Syllabus files with roman numerals — use normalised name so underscores/spaces
    # don't break word boundaries ("R23_ECE_II-Year" normalises to "R23-ECE-II-Year")
    m = re.search(r'\b(IV|III|II|I)\b', norm, re.IGNORECASE)
    if m:
        roman = m.group(1).upper()
        year_map = {"I": "1", "II": "2", "III": "3", "IV": "4"}
        meta["year"]     = year_map.get(roman, "all")
        meta["branch"]   = _extract_branch(filename)
        meta["doc_type"] = "syllabus"
        return meta

    # Regulations — cross-branch
    if "regulation" in name or name.startswith("re"):
        meta["doc_type"] = "regulation"
        return meta

    # Syllabi by name pattern (fallback)
    if "syllabus" in name or "syllabi" in name:
        meta["branch"]   = _extract_branch(filename)
        meta["doc_type"] = "syllabus"
        return meta

    # Utilities: holidays, library
    if "holiday" in name or "library" in name or "timing" in name:
        meta["doc_type"] = "general"
        return meta

    return meta


# ---------------------------------------------------------------------------
# OCR initialisation (optional)
# ---------------------------------------------------------------------------

def _init_ocr():
    """
    Returns True if pytesseract + fitz are both available, else None.
    pytesseract requires the Tesseract binary to be installed separately:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      Linux:   sudo apt install tesseract-ocr
    """
    if not FITZ_AVAILABLE:
        print("WARNING: PyMuPDF not installed — OCR disabled (pip install pymupdf)")
        return None
    try:
        import pytesseract

        # On Windows, Tesseract is often not on PATH — try common install locations
        if os.name == "nt":
            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

        # Probe for the tesseract binary — raises if not found
        pytesseract.get_tesseract_version()
        print("pytesseract + PyMuPDF available — scanned PDFs will be OCR-processed.")
        return True   # sentinel: OCR is ready
    except Exception as e:
        print(f"WARNING: Tesseract OCR unavailable ({type(e).__name__}: {e})")
        print("         Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("         Scanned PDFs will be skipped. Text-based PDFs will still be indexed.")
        return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(pdf_path: str, filename: str, base_meta: dict,
                              ocr_available: bool = False) -> list:
    """
    Primary extractor using pdfplumber — preserves table layout and column order.
    Much better than pypdf for academic calendars, timetables, and structured tables.
    Falls back to pypdf if pdfplumber fails or yields no text.
    """
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            native_count  = 0
            skipped_count = 0
            for page_num, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if len(text) >= OCR_THRESHOLD:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            **base_meta,
                            "page_number": page_num + 1,
                            "page":        page_num + 1,
                        }
                    ))
                    native_count += 1
                else:
                    skipped_count += 1

            tag = f"{native_count} pages"
            if skipped_count:
                hint = "will be OCR-processed" if ocr_available else "install Tesseract for OCR"
                tag += f" ({skipped_count} scanned/blank — {hint})"
            print(f"  -> {tag} | {filename}")
            return documents
    except Exception as e:
        print(f"  pdfplumber failed for {filename} ({e}), falling back to pypdf")
        return _extract_with_pypdf(pdf_path, filename, base_meta)


def _extract_with_pypdf(pdf_path: str, filename: str, base_meta: dict) -> list:
    """Fallback extractor using pypdf."""
    if not PYPDF_AVAILABLE:
        print(f"  ERROR: pypdf not available — cannot process {filename}")
        return []

    documents = []
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"  ERROR reading {filename}: {e}")
        return []

    native_count  = 0
    skipped_count = 0

    for page_num, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) >= OCR_THRESHOLD:
            documents.append(Document(
                page_content=text,
                metadata={
                    **base_meta,
                    "page_number": page_num + 1,
                    "page":        page_num + 1,
                }
            ))
            native_count += 1
        else:
            skipped_count += 1

    tag = f"{native_count} pages"
    if skipped_count:
        tag += f" ({skipped_count} scanned/blank skipped — install Tesseract for OCR)"
    print(f"  -> {tag} | {filename}")
    return documents


def _extract_with_fitz_and_ocr(pdf_path: str, filename: str, base_meta: dict,
                                 ocr_ready) -> list:
    """
    Use PyMuPDF to render pages + pytesseract for scanned content.
    ocr_ready is True (sentinel from _init_ocr) when pytesseract is available.
    """
    import io
    import pytesseract
    from PIL import Image

    documents  = []
    native_cnt = 0
    ocr_cnt    = 0

    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ERROR (fitz) opening {filename}: {e}")
        return _extract_with_pypdf(pdf_path, filename, base_meta)

    for page_num in range(len(pdf)):
        page        = pdf[page_num]
        native_text = page.get_text().strip()

        if len(native_text) >= OCR_THRESHOLD:
            text = native_text
            native_cnt += 1
        else:
            # Render page to image and run Tesseract OCR
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                text = pytesseract.image_to_string(img, lang="eng").strip()
                if not text:
                    continue
                ocr_cnt += 1
            except Exception as e:
                print(f"  OCR failed page {page_num+1} of {filename}: {e}")
                continue

        documents.append(Document(
            page_content=text,
            metadata={
                **base_meta,
                "page_number": page_num + 1,
                "page":        page_num + 1,
            }
        ))

    pdf.close()
    print(f"  -> {native_cnt} native + {ocr_cnt} OCR pages | {filename}")
    return documents


def extract_documents_from_pdf(pdf_path: str, filename: str, ocr_reader) -> list:
    """
    Dispatch to the best available extraction method.

    Priority:
      1. pdfplumber (best text layout, preserves table structure) — always tried first
      2. fitz + EasyOCR (for scanned pages that pdfplumber couldn't extract)
      3. pypdf (fallback if pdfplumber unavailable)
    """
    base_meta = extract_metadata_from_filename(filename)

    if PDFPLUMBER_AVAILABLE:
        ocr_available = ocr_reader is not None and FITZ_AVAILABLE
        docs = _extract_with_pdfplumber(pdf_path, filename, base_meta, ocr_available)
        # If pdfplumber got text, use it; otherwise try OCR on the scanned pages
        if docs:
            return docs
        # Zero pages extracted — file is likely scanned; try OCR if available
        if ocr_available:
            return _extract_with_fitz_and_ocr(pdf_path, filename, base_meta, ocr_reader)
        return docs  # empty — scanned without OCR support

    # pdfplumber not available: use fitz+OCR or pypdf
    if ocr_reader is not None and FITZ_AVAILABLE:
        return _extract_with_fitz_and_ocr(pdf_path, filename, base_meta, ocr_reader)
    return _extract_with_pypdf(pdf_path, filename, base_meta)


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def run_ingestion():
    global ingest_progress
    ingest_progress = {
        "running": True, "stage": "starting", "files_done": 0,
        "files_total": 0, "current_file": "", "chunks_total": 0,
        "message": "Starting ingestion…", "error": "",
    }
    print("\n" + "=" * 65)
    print("AcaRAG Pro — Document Ingestion Pipeline")
    print("=" * 65)

    if PDFPLUMBER_AVAILABLE:
        print("pdfplumber : available (primary extractor — preserves table layout)")
    else:
        print("pdfplumber : NOT installed  (pip install pdfplumber)  — using pypdf")
    if FITZ_AVAILABLE:
        print("PyMuPDF    : available (used for OCR rendering)")
    else:
        print("PyMuPDF    : NOT installed  (pip install pymupdf)  — OCR disabled")

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset folder not found at {DATASET_PATH}")
        return

    pdf_files = sorted([f for f in os.listdir(DATASET_PATH) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("No PDF files found in dataset folder.")
        ingest_progress.update({"running": False, "stage": "error", "error": "No PDF files found."})
        return

    print(f"\nFound {len(pdf_files)} PDF files.")
    ingest_progress.update({"stage": "extracting", "files_total": len(pdf_files),
                            "message": f"Extracting text from {len(pdf_files)} PDFs…"})
    ocr_reader = _init_ocr()
    if ocr_reader is None:
        print("OCR     : disabled  (install Tesseract + pip install pytesseract pymupdf)\n")
    else:
        print()

    all_documents = []
    for filename in pdf_files:
        ingest_progress.update({"current_file": filename,
                                "message": f"Reading {filename}…"})
        pdf_path = os.path.join(DATASET_PATH, filename)
        docs = extract_documents_from_pdf(pdf_path, filename, ocr_reader)

        # Multi-branch files: ingest the same content under each branch separately
        # so students of every listed branch can retrieve it via their branch-specific lane.
        if filename in _MULTI_BRANCH_FILES:
            branches = _MULTI_BRANCH_FILES[filename]
            print(f"  [multi-branch] {filename} -> branches: {branches}")
            for branch in branches:
                for doc in docs:
                    import copy
                    d = copy.deepcopy(doc)
                    d.metadata["branch"] = branch
                    all_documents.append(d)
        else:
            all_documents.extend(docs)
        ingest_progress["files_done"] += 1

    if not all_documents:
        print("\nERROR: No text extracted. Check your PDF files.")
        ingest_progress.update({"running": False, "stage": "error", "error": "No text extracted."})
        return

    print(f"\nTotal pages extracted : {len(all_documents)}")
    ingest_progress.update({"stage": "chunking", "message": f"Splitting {len(all_documents)} pages into chunks…"})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140,
        length_function=len,
    )
    chunks = splitter.split_documents(all_documents)
    print(f"Total chunks          : {len(chunks)}")
    ingest_progress.update({"chunks_total": len(chunks),
                            "stage": "embedding",
                            "message": f"Generating embeddings for {len(chunks)} chunks (2–5 min)…"})

    # Write to a temp path first so the server can keep running during embedding
    CHROMA_NEW = CHROMA_PATH + "_new"
    if os.path.exists(CHROMA_NEW):
        shutil.rmtree(CHROMA_NEW)

    print("Generating embeddings (this may take a few minutes)...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device.upper()}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (no GPU found)"))
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": device},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_NEW,
    )

    # Swap new DB in — requires server to not hold old DB open
    ingest_progress.update({"stage": "finalizing", "message": "Swapping in new index…"})
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        os.rename(CHROMA_NEW, CHROMA_PATH)
        print(f"\n[OK] Ingestion complete! {len(chunks)} chunks indexed.")
        ingest_progress.update({
            "running": False, "stage": "done",
            "message": f"Done! {len(chunks)} chunks indexed and ready.",
        })
    except PermissionError:
        print(f"\n[OK] Embeddings written to chroma_db_new ({len(chunks)} chunks).")
        print("     The server is holding chroma_db open — stop it, then run:")
        print("       rmdir /s /q chroma_db && move chroma_db_new chroma_db")
        print("     Then restart the server.")
        ingest_progress.update({
            "running": False, "stage": "done",
            "message": f"Done! {len(chunks)} chunks in chroma_db_new. Restart server to activate.",
        })
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_ingestion()
