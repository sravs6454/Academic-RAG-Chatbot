import warnings; warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vs = Chroma(persist_directory="chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Check Y4 calendar
print("=== Y4 Calendar docs ===")
try:
    docs = vs.similarity_search("fourth year academic schedule exam dates", k=5,
        filter={"$and": [{"year": {"$eq": "4"}}, {"branch": {"$eq": "all"}}, {"doc_type": {"$eq": "calendar"}}]})
    print(f"{len(docs)} docs found")
    for d in docs:
        print(f"  {d.metadata.get('source')} p{d.metadata.get('page_number')}")
        print(f"    {d.page_content[:200]}")
except Exception as e:
    print(f"Error: {e}")

# Check backlogs in regulations
print()
print("=== Backlog rules in regulations ===")
try:
    docs = vs.similarity_search("backlog arrears allowed maximum ATKT", k=8,
        filter={"year": {"$eq": "all"}})
    for d in docs:
        if "backlog" in d.page_content.lower() or "arrear" in d.page_content.lower() or "atkt" in d.page_content.lower():
            print(f"  [{d.metadata.get('source')} p{d.metadata.get('page_number')}]")
            print(f"  {d.page_content[:400]}")
            print()
except Exception as e:
    print(f"Error: {e}")

# Check Y1 exam schedule content
print("=== Y1 Exam Schedule chunks ===")
try:
    docs = vs.similarity_search("first year first semester exam schedule timetable", k=8,
        filter={"$and": [{"year": {"$eq": "1"}}, {"branch": {"$eq": "all"}}, {"doc_type": {"$eq": "exam_schedule"}}]})
    print(f"{len(docs)} docs found")
    for d in docs:
        print(f"  {d.metadata.get('source')} p{d.metadata.get('page_number')}")
        print(f"  {d.page_content[:200]}")
        print()
except Exception as e:
    print(f"Error: {e}")
