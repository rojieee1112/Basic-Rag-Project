from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import ollama

# ----------------------------
# 1. Embedding Model Wrapper
# ----------------------------

class SBERTEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


# ----------------------------
# 2. Load Vector Database
# ----------------------------

print("\nLoading vector database...")

embedding_model = SBERTEmbeddings()

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

print("Vector DB loaded successfully!")


# ----------------------------
# 3. Chat Loop (RAG)
# ----------------------------
print("\nRAG Chat is ready! Type 'exit' to stop.\n")

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Step 1: Retrieve relevant chunks
    docs = vectordb.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    print("\n--- Retrieved Context ---")
    print(context)

    # Step 2: Build prompt
    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    # Step 3: Call local LLM (Ollama)
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Step 4: Output answer
    print("\n--- Answer ---")
    print(response["message"]["content"])
    print("\n" + "-" * 50 + "\n")
