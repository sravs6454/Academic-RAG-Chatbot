from rag import load_rag_chain

qa = load_rag_chain()

print("Academic RAG Chatbot is ready.")
print("Ask any academic question. Type 'exit' to quit.")

while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break

    response = qa.invoke(query)

    print("\nAnswer:")
    print(response["result"])
