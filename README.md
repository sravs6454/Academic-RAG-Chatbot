# 🎓 Academic RAG Chatbot

An AI-powered Academic Assistant that answers student queries using **Retrieval-Augmented Generation (RAG)**, **semantic search**, and **multi-level caching**. The chatbot retrieves information only from official academic documents, ensuring accurate, context-aware responses while minimizing hallucinations.

---

## 📌 Features

- 📄 Answers questions from official academic PDFs
- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 Semantic search using vector embeddings
- ⚡ Three-Level Cache System
  - Exact Cache
  - Semantic Cache
  - Session Memory
- 💬 Interactive chatbot interface
- 📚 Supports multiple academic documents
- 🚀 Faster responses through intelligent caching

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### AI & Machine Learning
- LangChain
- Hugging Face Embeddings
- Sentence Transformers
- ChromaDB
- Retrieval-Augmented Generation (RAG)

### Libraries
- PyPDF
- NumPy
- Scikit-learn

---

## 📂 Project Structure

```
Academic-RAG-Chatbot
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── python-ai/
│   ├── app.py
│   ├── ingest.py
│   ├── rag.py
│   ├── retrieve.py
│   ├── cache/
│   ├── memory/
│   └── requirements.txt
│
├── dataset/
│
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

1. Upload academic PDF documents.
2. Documents are split into smaller chunks.
3. Chunks are converted into embeddings.
4. Embeddings are stored in ChromaDB.
5. User asks a question.
6. Relevant document chunks are retrieved.
7. The LLM generates an answer using retrieved context.
8. Responses are cached for faster future queries.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/sravs6454/Academic-RAG-Chatbot.git
```

```bash
cd Academic-RAG-Chatbot
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r python-ai/requirements.txt
```

---

### Add Academic PDFs

Place your academic documents inside the **dataset/** folder.

---

### Generate Vector Database

```bash
python python-ai/ingest.py
```

---

### Run the Backend

```bash
python python-ai/app.py
```

---

### Open Frontend

Open

```
frontend/index.html
```

or

Run using Live Server in VS Code.

---

## 💡 Example Questions

- What are the subjects in 3rd Year AIML?
- What are the library timings?
- Show the academic calendar.
- What are the R23 regulations?
- When are the semester holidays?

---


---

## 🔮 Future Enhancements

- Voice-based interaction
- Student login
- Multi-language support
- Cloud deployment
- OCR for scanned PDFs
- Knowledge Graph integration
- Support for multiple universities

---

## 👩‍💻 Author

**Ganapavarapu Sravya Jyothi**

- B.Tech – Artificial Intelligence & Machine Learning
- Shri Vishnu Engineering College for Women

GitHub:
https://github.com/sravs6454


---

## ⭐ If you found this project useful, consider giving it a star!
