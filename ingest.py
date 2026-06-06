from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


# Wrapper for SentenceTransformer
class SBERTEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


print("Loading document...")

loader = TextLoader("data/sample.txt")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")

# Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

docs = splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# Create embeddings
embedding_model = SBERTEmbeddings()

print("Creating vector database...")

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Vector database created successfully!")