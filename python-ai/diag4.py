import warnings; warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vs = Chroma(persist_directory="chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Check EEE Y2 docs
print("=== EEE Year 2 docs ===")
try:
    docs = vs.similarity_search("EEE second year subjects", k=5,
        filter={"$and": [{"year": {"$eq": "2"}}, {"branch": {"$eq": "EEE"}}]})
    print(f"Exact year=2, branch=EEE: {len(docs)} docs")
    for d in docs:
        print(f"  {d.metadata.get('source')} p{d.metadata.get('page_number')}")
except Exception as e:
    print(f"Error: {e}")

print()
print("=== All EEE docs ===")
try:
    docs = vs.similarity_search("EEE subjects", k=10, filter={"branch": {"$eq": "EEE"}})
    print(f"branch=EEE: {len(docs)} docs")
    for d in docs:
        print(f"  {d.metadata.get('source')} year={d.metadata.get('year')} p{d.metadata.get('page_number')}")
except Exception as e:
    print(f"Error: {e}")

# Check Y2 calendar chunks with doc_type filter
print()
print("=== Y2 Calendar chunks ===")
try:
    docs = vs.similarity_search("mid examinations second year 2025 2026", k=10,
        filter={"$and": [{"year": {"$eq": "2"}}, {"branch": {"$eq": "all"}}, {"doc_type": {"$eq": "calendar"}}]})
    print(f"year=2, calendar: {len(docs)} docs")
    for d in docs:
        print(f"  {d.metadata.get('source')} p{d.metadata.get('page_number')}")
        print(f"    {d.page_content[:200]}")
except Exception as e:
    print(f"Error: {e}")

# Check Y2 mid exam dates - raw similarity
print()
print("=== Y2 mid exam raw search ===")
docs = vs.similarity_search("CSE 2nd Year What are the mid exam dates", k=5,
    filter={"$and": [{"year": {"$eq": "2"}}, {"branch": {"$eq": "all"}}, {"doc_type": {"$eq": "calendar"}}]})
print(f"Found: {len(docs)}")
for d in docs:
    print(f"  [{d.metadata.get('source')}]")
    print(f"  {d.page_content[:300]}")
    print()
