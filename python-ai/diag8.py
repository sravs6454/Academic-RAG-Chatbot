import warnings; warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vs = Chroma(persist_directory="chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Get the FULL content of all Y4 calendar chunks
docs = vs.get(where={"source": {"$eq": "2025-26-IV-B.Tech-Academic-Calendar (1).pdf"}})
for i, (doc_id, doc, meta) in enumerate(zip(docs['ids'], docs['documents'], docs['metadatas'])):
    print(f"\n=== Chunk {i+1} (full content) ===")
    print(repr(doc))
