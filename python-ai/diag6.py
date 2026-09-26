import warnings; warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vs = Chroma(persist_directory="chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Get ALL chunks from Y4 calendar PDF using a broad search
print("=== ALL Y4 calendar chunks (broad search) ===")
queries = [
    "fourth year instruction period semester",
    "IV B.Tech mid examination calendar 2025 2026",
    "4th year unit instructions semester 1 2",
]
seen = set()
for q in queries:
    try:
        docs = vs.similarity_search(q, k=10,
            filter={"$and": [{"year": {"$eq": "4"}}, {"branch": {"$eq": "all"}}]})
        for d in docs:
            key = d.page_content[:50]
            if key not in seen:
                seen.add(key)
                m = d.metadata
                print(f"[{m.get('source')} p{m.get('page_number')} type={m.get('doc_type')}]")
                print(f"  {d.page_content[:300]}")
                print()
    except Exception as e:
        print(f"Error: {e}")
