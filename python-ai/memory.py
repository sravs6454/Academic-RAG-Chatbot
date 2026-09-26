SESSION_MEMORY = {}

def get_memory(session_id):
    return SESSION_MEMORY.get(session_id, [])

def update_memory(session_id, question, answer):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    SESSION_MEMORY[session_id].append({
        "question": question,
        "answer": answer
    })

    # keep last 3 Q&A only
    SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-3:]
