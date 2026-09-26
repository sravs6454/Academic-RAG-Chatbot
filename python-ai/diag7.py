import warnings; warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vs = Chroma(persist_directory="chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Get ALL unique chunks from the Y4 calendar PDF
print("=== ALL chunks from 2025-26-IV-B.Tech-Academic-Calendar (1).pdf ===")
try:
    # Use a broad query and look for all calendar chunks for this specific file
    docs = vs.get(where={"source": {"$eq": "2025-26-IV-B.Tech-Academic-Calendar (1).pdf"}})
    print(f"Total chunks in vectorstore for Y4 calendar: {len(docs['ids'])}")
    for i, (doc_id, doc, meta) in enumerate(zip(docs['ids'], docs['documents'], docs['metadatas'])):
        print(f"\n[{i+1}] page={meta.get('page_number')}")
        print(f"  {doc[:400]}")
except Exception as e:
    print(f"Error with .get(): {e}")
    # Fallback: broad similarity search
    for q in ["fourth year instruction mid examination schedule 2025", "IV B.Tech semester 1 2025 academic calendar unit"]:
        docs = vs.similarity_search(q, k=10, filter={"source": {"$eq": "2025-26-IV-B.Tech-Academic-Calendar (1).pdf"}})
        print(f"\nQuery: '{q}' -> {len(docs)} docs")
        for d in docs:
            print(f"  p{d.metadata.get('page_number')}: {d.page_content[:300]}")
            print()
