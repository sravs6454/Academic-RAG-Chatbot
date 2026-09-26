import warnings; warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vs = Chroma(persist_directory="chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Check metadata for 4-1-CSE.pdf
print("=== 4-1-CSE.pdf metadata ===")
docs = vs.get(where={"source": {"$eq": "4-1-CSE.pdf"}})
print(f"Total chunks: {len(docs['ids'])}")
if docs['metadatas']:
    m = docs['metadatas'][0]
    print(f"year={m.get('year')} branch={m.get('branch')} doc_type={m.get('doc_type')}")

# Lane A test for Y4 CSE
print()
print("=== Lane A for year=4, branch=CSE ===")
try:
    results = vs.similarity_search("CSE 4th Year academic schedule", k=6,
        filter={"$and": [{"year": {"$eq": "4"}}, {"branch": {"$eq": "CSE"}}]})
    print(f"Results: {len(results)}")
    for d in results:
        print(f"  {d.metadata.get('source')} year={d.metadata.get('year')} branch={d.metadata.get('branch')}")
except Exception as e:
    print(f"Error: {e}")

# Lane C test for Y4 all
print()
print("=== Lane C for year=4, branch=all ===")
try:
    results = vs.similarity_search("CSE 4th Year academic schedule", k=8,
        filter={"$and": [{"year": {"$eq": "4"}}, {"branch": {"$eq": "all"}}]})
    print(f"Results: {len(results)}")
    for d in results:
        print(f"  {d.metadata.get('source')} year={d.metadata.get('year')} branch={d.metadata.get('branch')} type={d.metadata.get('doc_type')}")
except Exception as e:
    print(f"Error: {e}")

# Check all year=4 files
print()
print("=== All year=4 files ===")
try:
    results = vs.similarity_search("fourth year", k=20,
        filter={"year": {"$eq": "4"}})
    seen = set()
    for d in results:
        src = d.metadata.get('source')
        if src not in seen:
            seen.add(src)
            print(f"  {src} year={d.metadata.get('year')} branch={d.metadata.get('branch')} type={d.metadata.get('doc_type')}")
except Exception as e:
    print(f"Error: {e}")
