import warnings; warnings.filterwarnings("ignore")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

vs = Chroma(persist_directory="chroma_db",
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

def lane_c_test(query, year, branch):
    search_query = f"{branch} {year}nd Year {query}"
    print(f"\nLane C test for: '{query}' (year={year}, branch={branch})")
    print(f"Search query: '{search_query}'")
    try:
        docs = vs.similarity_search(
            search_query, k=8,
            filter={"$and": [{"year": {"$eq": year}}, {"branch": {"$eq": "all"}}]}
        )
        print(f"Results ({len(docs)}):")
        for d in docs:
            m = d.metadata
            print(f"  [{m.get('source')} p{m.get('page_number')} type={m.get('doc_type')}] {d.page_content[:100]}")
    except Exception as e:
        print(f"  Error: {e}")

def reg_test(query, year, branch):
    search_query = f"{branch} {year}nd Year {query}"
    print(f"\nLane B (regulations) for: '{query}'")
    try:
        docs = vs.similarity_search(search_query, k=8, filter={"year": {"$eq": "all"}})
        print(f"Results ({len(docs)}):")
        for d in docs:
            m = d.metadata
            print(f"  [{m.get('source')} p{m.get('page_number')}] {d.page_content[:150]}")
    except Exception as e:
        print(f"  Error: {e}")

# Calendar retrieval tests
lane_c_test("What are the mid exam dates for second year students?", "2", "CSE")
lane_c_test("When are the end examinations for third year?", "3", "ECE")
lane_c_test("What is the exam schedule for first year first semester?", "1", "CSE")

# Regulations retrieval tests
reg_test("What is the minimum attendance percentage required?", "2", "CSE")
reg_test("How many backlogs am I allowed to have?", "2", "CSE")
