from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


# ----------------------------
# Embeddings
# ----------------------------
class SBERTEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


print("Loading PDF...")

# CHANGE THIS FILE NAME
loader = PyPDFLoader("data/AI_note.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# ----------------------------
# Chunking
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# ----------------------------
# Embeddings
# ----------------------------
embedding_model = SBERTEmbeddings()

print("Creating vector database...")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Vector DB created successfully!")