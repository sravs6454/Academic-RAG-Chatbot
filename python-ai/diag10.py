import warnings; warnings.filterwarnings("ignore")
from langchain_groq import ChatGroq
from rag import _retrieve_docs, _check_confidence, GROUNDED_PROMPT, OUT_OF_SCOPE, _get_llm
import os; from dotenv import load_dotenv; load_dotenv()

question = "What is the academic schedule for fourth year students?"
ctx = {"year": "4", "semester": "1", "branch": "CSE", "name": "Priya"}

# Step 1: retrieve
docs = _retrieve_docs(question, ctx)
print(f"Retrieved: {len(docs)} docs")
for i, d in enumerate(docs[:5]):
    print(f"  [{i+1}] {d.metadata.get('source')} type={d.metadata.get('doc_type')} chars={len(d.page_content)}")

# Step 2: check confidence
ok = _check_confidence(docs)
print(f"_check_confidence: {ok}")

if not ok:
    print("BLOCKED by _check_confidence -> OUT_OF_SCOPE")
else:
    # Step 3: sort and build context
    _TYPE_PRIORITY = {"calendar": 0, "exam_schedule": 1, "syllabus": 2, "regulation": 3}
    docs_sorted = sorted(docs, key=lambda d: _TYPE_PRIORITY.get(d.metadata.get("doc_type", ""), 4))

    has_ts = any(d.metadata.get("doc_type", "") in ("calendar", "exam_schedule") for d in docs_sorted)
    type_limits = {"calendar": 3, "exam_schedule": 3, "syllabus": 1, "regulation": 1} if has_ts else {}

    context_parts = []
    total_chars = 0
    type_counts = {}
    for doc in docs_sorted:
        content = doc.page_content.strip()
        dt = doc.metadata.get("doc_type", "")
        limit = type_limits.get(dt, 3)
        if type_counts.get(dt, 0) >= limit:
            continue
        if total_chars + len(content) > 20000:
            break
        label = f"[Document: {doc.metadata.get('source')} | Page: {doc.metadata.get('page_number')} | Type: {dt}]"
        context_parts.append(f"{label}\n{content}")
        total_chars += len(content)
        type_counts[dt] = type_counts.get(dt, 0) + 1

    print(f"Context: {len(context_parts)} docs, {total_chars} chars, types={type_counts}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = GROUNDED_PROMPT.format(
        student_profile="Priya, 4th Year, Semester 1, CSE",
        context=context,
        question=question
    )

    llm = _get_llm()
    try:
        resp = llm.invoke(prompt)
        print(f"LLM ANSWER: {resp.content[:400]}")
    except Exception as e:
        print(f"LLM ERROR: {e}")
