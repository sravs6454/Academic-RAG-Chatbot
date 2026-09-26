"""
AcaRAG — Domain-aware query spell correction
----------------------------------------------
Corrects obvious typos in student queries before RAG retrieval.

Design choices:
- Uses difflib.get_close_matches against a domain vocabulary
- All-caps tokens (acronyms like CSE, ECE, GPA) are never corrected
- Short tokens ≤3 chars are never corrected (too risky)
- Only corrects if a single match scores ≥ 0.82 similarity (high threshold)
- Returns (corrected_query, was_corrected, original_query)
"""

import re
import difflib

# Academic terms that are always treated as correctly spelled
_DOMAIN_VOCAB = {
    # Departments / specialisations
    "syllabus", "semester", "supplementary", "regulation", "regulations",
    "elective", "electives", "timetable", "attendance", "backlog",
    "examination", "examinations", "calendar", "academic", "university",
    "autonomous", "college", "department", "faculty", "staff", "professor",
    "lecturer", "principal", "campus", "hostel", "library",
    # Exam / fee terms
    "registration", "notification", "fee", "deadline", "payment", "schedule",
    "marks", "result", "grade", "internal", "external", "midterm",
    # Academic programme terms
    "engineering", "technology", "science", "mathematics", "physics",
    "chemistry", "english", "communication", "professional", "open",
    "honours", "minor", "course", "structure", "curriculum", "units",
    "unit", "credits", "prerequisite", "laboratory", "practical",
    # Regulation / policy terms
    "detained", "promoted", "arrears", "supplementary", "advanced",
    "regular", "lateral", "admission", "prescribed", "candidate",
    # Common question words in academic context
    "subjects", "subject", "topics", "topic", "notes", "study",
    "holidays", "holiday", "classes", "class", "timing", "timings",
    "working", "hours", "available", "allowed", "permitted",
    "difference", "between", "what", "when", "where", "which", "how",
    "many", "much", "does", "contain", "have", "last", "date", "pay",
}

# Extra correction overrides — for domain terms whose difflib score
# isn't high enough but we know the user means a specific word.
_FORCE_CORRECTIONS = {
    "syllbus":       "syllabus",
    "sylabous":      "syllabus",
    "slylabus":      "syllabus",
    "sylabus":       "syllabus",
    "syllabs":       "syllabus",
    "suplimentary":  "supplementary",
    "supllementary": "supplementary",
    "suplementary":  "supplementary",
    "suplemental":   "supplementary",
    "atendance":     "attendance",
    "attendence":    "attendance",
    "attendanc":     "attendance",
    "timetabel":     "timetable",
    "timetble":      "timetable",
    "timtable":      "timetable",
    "examinaton":    "examination",
    "elecitve":      "elective",
    "eelective":     "elective",
    "registartion":  "registration",
    "registraton":   "registration",
    "calander":      "calendar",
    "calandar":      "calendar",
    "semeseter":     "semester",
    "semster":       "semester",
    "regulashion":   "regulation",
    "regulaton":     "regulation",
    "regulaion":     "regulation",
    "holday":        "holiday",
    "holliday":      "holiday",
    "librery":       "library",
    "libary":        "library",
    "subects":       "subjects",
    "subejcts":      "subjects",
    "subjets":       "subjects",
    "faculity":      "faculty",
    "faculity":      "faculty",
    "professer":     "professor",
    "proffesor":     "professor",
    "lecutrer":      "lecturer",
    "backlogg":      "backlog",
    "backlogs":      "backlogs",
    "scholership":   "scholarship",
    "scholarhip":    "scholarship",
    "detaied":       "detained",
    "promted":       "promoted",
}

_VOCAB_LIST = sorted(_DOMAIN_VOCAB)
_CLOSE_CUTOFF = 0.82   # must be a very close match to auto-correct
_CLOSE_N      = 1      # only accept if there is exactly one best suggestion


def _correct_token(token: str) -> str:
    """Return corrected version of a single token, or the original if unsure."""
    lower = token.lower()

    # Never touch all-caps (acronyms: CSE, ECE, GPA, CGPA, SVECW…)
    if token.isupper() and len(token) >= 2:
        return token

    # Never touch very short tokens
    if len(lower) <= 3:
        return token

    # Already a known domain term → no change
    if lower in _DOMAIN_VOCAB:
        return token

    # Hard-coded override
    if lower in _FORCE_CORRECTIONS:
        corrected = _FORCE_CORRECTIONS[lower]
        # Preserve original casing style
        if token[0].isupper():
            corrected = corrected.capitalize()
        return corrected

    # difflib fuzzy match against domain vocab
    matches = difflib.get_close_matches(lower, _VOCAB_LIST, n=_CLOSE_N, cutoff=_CLOSE_CUTOFF)
    if len(matches) == 1:
        corrected = matches[0]
        if token[0].isupper():
            corrected = corrected.capitalize()
        return corrected

    return token


def correct_query(query: str) -> tuple[str, bool]:
    """
    Correct obvious typos in query.

    Returns:
        (corrected_query, was_changed)
        was_changed is True only if at least one token was actually corrected.
    """
    # Split on spaces but preserve punctuation attachment
    tokens = re.split(r'(\s+)', query)
    corrected_tokens = []
    changed = False

    for tok in tokens:
        if not tok.strip():
            corrected_tokens.append(tok)  # preserve whitespace
            continue
        # Extract leading/trailing punctuation
        m = re.fullmatch(r'([^\w]*)(\w+)([^\w]*)', tok)
        if m:
            prefix, word, suffix = m.group(1), m.group(2), m.group(3)
            fixed = _correct_token(word)
            if fixed.lower() != word.lower():
                changed = True
            corrected_tokens.append(prefix + fixed + suffix)
        else:
            corrected_tokens.append(tok)

    corrected = "".join(corrected_tokens)
    return corrected, changed
