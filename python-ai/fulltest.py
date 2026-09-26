"""
AcaRAG Full Pipeline Test — tests retrieval + LLM answer generation end-to-end.
Run with: python fulltest.py
"""
import warnings, sys
warnings.filterwarnings("ignore")

from rag import answer_question

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []

def test(label, question, ctx, expect_answer=True):
    """
    Run a single question through the full RAG pipeline and report result.
    expect_answer=True  → answer should NOT be "Information not available"
    expect_answer=False → answer IS expected to be out-of-scope
    """
    try:
        r = answer_question(question, [], ctx)
        answer  = r.get("answer", "")
        sources = r.get("sources", [])
        is_oos  = answer.startswith("Information not available")

        if expect_answer:
            ok = not is_oos and len(answer.strip()) > 30
        else:
            ok = is_oos

        status = PASS if ok else FAIL
        src_str = ", ".join(s.get("document","?") for s in sources[:2]) if sources else "none"
        snippet = answer[:120].replace("\n", " ")

        print(f"{status} [{label}]")
        print(f"       Q: {question}")
        print(f"       A: {snippet}...")
        print(f"       Sources: {src_str}")
        print()
        results.append((status, label))
    except Exception as e:
        print(f"{FAIL} [{label}] EXCEPTION: {e}\n")
        results.append((FAIL, label))

# -
print("=" * 72)
print("AcaRAG Pro — Full Pipeline Test")
print("=" * 72 + "\n")

# -- 1. SYLLABUS / SUBJECTS -
print("-- SYLLABUS / SUBJECTS -\n")

test("CSE Y2S1 subjects",
     "What subjects do I have this semester?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("CSE Y2S2 subjects",
     "What are the subjects in second semester?",
     {"year":"2","semester":"2","branch":"CSE","name":"Priya"})

test("CSE Y3 subjects",
     "List all subjects for 3rd year CSE",
     {"year":"3","semester":"1","branch":"CSE","name":"Priya"})

test("ECE Y1 subjects",
     "What subjects do I study in first year?",
     {"year":"1","semester":"1","branch":"ECE","name":"Anu"})

test("ECE Y3 subjects",
     "What are the subjects for ECE third year?",
     {"year":"3","semester":"1","branch":"ECE","name":"Anu"})

test("EEE Y2 subjects",
     "What subjects does second year EEE have?",
     {"year":"2","semester":"1","branch":"EEE","name":"Sita"})

test("IT Y2 subjects",
     "List subjects for IT second year",
     {"year":"2","semester":"1","branch":"IT","name":"Ravi"})

test("CIVIL Y2 subjects",
     "What are my Civil Engineering subjects?",
     {"year":"2","semester":"1","branch":"CIVIL","name":"Meena"})

test("ME Y1 subjects",
     "What subjects do mechanical engineering first year students study?",
     {"year":"1","semester":"1","branch":"ME","name":"Kiran"})

test("AIML Y2 subjects",
     "What subjects are there for CSE AI&ML second year?",
     {"year":"2","semester":"1","branch":"AI&ML","name":"Divya"})

test("AIDS Y2 subjects",
     "List subjects for AI and Data Science second year",
     {"year":"2","semester":"1","branch":"AIDS","name":"Latha"})

test("CYBER Y2 subjects",
     "What subjects does cyber security second year have?",
     {"year":"2","semester":"1","branch":"CYBER","name":"Pooja"})

test("CSE Y4 subjects",
     "What are the fourth year CSE subjects?",
     {"year":"4","semester":"1","branch":"CSE","name":"Priya"})

# -- 2. SYLLABUS CONTENT -
print("-- SYLLABUS CONTENT -\n")

test("CSE DBMS syllabus",
     "What are the units covered in Database Management Systems?",
     {"year":"2","semester":"2","branch":"CSE","name":"Priya"})

test("CSE OS syllabus",
     "What topics are covered in Operating Systems subject?",
     {"year":"2","semester":"2","branch":"CSE","name":"Priya"})

test("ECE course outcomes",
     "What are the course outcomes for Machine Learning in ECE?",
     {"year":"3","semester":"2","branch":"ECE","name":"Anu"})

# -- 3. ACADEMIC CALENDAR -
print("-- ACADEMIC CALENDAR -\n")

test("Y1 calendar dates",
     "When does the first semester start and end for first year?",
     {"year":"1","semester":"1","branch":"CSE","name":"Priya"})

test("Y2 calendar dates",
     "What are the mid exam dates for second year students?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("Y3 calendar dates",
     "When are the end examinations for third year?",
     {"year":"3","semester":"1","branch":"ECE","name":"Anu"})

test("Y4 calendar dates",
     "What is the academic schedule for fourth year students?",
     {"year":"4","semester":"1","branch":"CSE","name":"Priya"})

# -- 4. REGULATIONS -
print("-- REGULATIONS -\n")

test("Attendance minimum",
     "What is the minimum attendance percentage required?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("Attendance condonation",
     "What happens if attendance is below 65 percent?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("GPA grading",
     "How is GPA calculated? What are the grade points?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("Backlog rules",
     "How many backlogs am I allowed to have?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("Supplementary exam rules",
     "What are the rules for supplementary examinations?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("Internal marks",
     "How are internal marks calculated?",
     {"year":"1","semester":"1","branch":"CSE","name":"Priya"})

# -- 5. EXAM SCHEDULES -
print("-- EXAM SCHEDULES -\n")

test("Y1S1 exam schedule",
     "What is the exam schedule for first year first semester?",
     {"year":"1","semester":"1","branch":"CSE","name":"Priya"})

test("Y1S2 exam schedule",
     "Show me the supplementary exam timetable for first year second semester",
     {"year":"1","semester":"2","branch":"CSE","name":"Priya"})

test("Y2S1 exam schedule",
     "When are the semester exams for second year?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

# -- 6. GENERAL COLLEGE INFO -
print("-- GENERAL COLLEGE INFO -\n")

test("Library timings",
     "What are the library working hours?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

test("Holidays list",
     "What are the holidays this academic year?",
     {"year":"1","semester":"1","branch":"CSE","name":"Priya"})

test("Faculty details",
     "Who are the faculty members in the CSE department?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"})

# -- 7. OUT-OF-SCOPE (should return "Information not available") -
print("-- OUT-OF-SCOPE (should refuse) -\n")

test("Weather query (OOS)",
     "What is the weather today in Bhimavaram?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"},
     expect_answer=False)

test("Food menu (OOS)",
     "What is the canteen menu today?",
     {"year":"2","semester":"1","branch":"CSE","name":"Priya"},
     expect_answer=False)

# -
# Summary
# -
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
total  = len(results)

print("=" * 72)
print(f"RESULTS: {passed}/{total} passed  |  {failed} failed")
print("=" * 72)

if failed:
    print("\nFailed tests:")
    for s, label in results:
        if s == FAIL:
            print(f"  - {label}")
