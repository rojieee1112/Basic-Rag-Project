# 🤖 AI PDF Chatbot using RAG (Retrieval-Augmented Generation)

An AI-powered chatbot that allows users to ask questions about a PDF document using **Retrieval-Augmented Generation (RAG)**. The application retrieves the most relevant information from the uploaded document and generates accurate, context-aware responses using an open-source Large Language Model.

## 🚀 Live Demo

🔗 **Hugging Face Spaces:**  https://rojieee-basic-rag-project.hf.space/ 

---

## 📌 Project Overview

This project demonstrates how Retrieval-Augmented Generation (RAG) can be used to build an intelligent document-based question-answering system.

Instead of relying only on the language model's knowledge, the chatbot first retrieves relevant information from the uploaded PDF and then generates answers based on that context, resulting in more accurate and grounded responses.

The current implementation uses a **9-page PDF containing basic Artificial Intelligence concepts**, but the pipeline can be adapted for other documents with minimal changes.

---

## ✨ Features

* 📄 Upload and process PDF documents
* 💬 Ask natural language questions about the document
* 🔍 Semantic search using vector embeddings
* 🧠 Context-aware answer generation using RAG
* ⚡ Runs completely on open-source models
* 🌐 Deployed on Hugging Face Spaces
* 💸 No paid APIs required

---

## 🛠️ Tech Stack

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Framework            | Streamlit                                |
| RAG Framework        | LangChain                                |
| Embedding Model      | all-MiniLM-L6-v2 (Sentence Transformers) |
| Vector Database      | ChromaDB                                 |
| Language Model       | TinyLlama-1.1B-Chat                      |
| Programming Language | Python                                   |
| Deployment           | Hugging Face Spaces                      |

---

## ⚙️ How It Works

1. Upload the PDF document.
2. Extract text from the PDF.
3. Split the text into smaller chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in ChromaDB.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant chunks.
8. Provide the retrieved context to TinyLlama.
9. Generate a context-aware response.

---

## 📂 Project Structure

```
.
├── app.py              # Streamlit application
├── ingest.py           # Creates embeddings and stores them in ChromaDB
├── Data/
│   └── Sample.pdf
├── chroma_db/          # Vector database
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 💡 What I Learned

* Understanding the complete RAG pipeline
* Working with document embeddings
* Semantic search using vector databases
* Using LangChain to orchestrate retrieval and generation
* Deploying AI applications on Hugging Face Spaces
* Running fully open-source LLMs without paid APIs
* Handling deployment and inference limitations on the free tier

---

## ⚠️ Note

This project uses **TinyLlama-1.1B-Chat**, a lightweight open-source language model.

Since it runs on the free tier of Hugging Face Spaces, generating responses may take a few seconds. Although it is not as powerful as commercial models such as ChatGPT or Claude, it effectively demonstrates the complete Retrieval-Augmented Generation workflow using entirely free and open-source tools.

---

## 🔮 Future Improvements

* Support multiple PDF uploads
* Chat history and conversation memory
* Source citations with page numbers
* Better chunking strategies
* Hybrid search (semantic + keyword)
* Support for larger open-source LLMs
* User authentication
* PDF upload through the UI

---

## 🖥️ Installation

Clone the repository:

```bash
git clone https://github.com/rojieee1112/your-repository.git
cd your-repository
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is intended for learning and educational purposes.
