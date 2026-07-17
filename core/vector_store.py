import os
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "video_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


_client: chromadb.ClientAPI | None = None

def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


# Function: Initialize HuggingFace embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

# Delete the transcript collection
def clear_vector_store():
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🧹 Cleared previous transcript collection.")
    except Exception as e:
        print(f"No existing collection to clear ({e}).")


'''
Function: Build a new Chroma vector store from transcript
 - Clears any previous transcript data first 
 - Splits transcript into chunks (500 chars, 50 overlap)
 - Wraps each chunk in a Document with metadata
 - Embeds chunks and persists them in Chroma DB
'''
def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store")

    # Ensure we start from a clean collection every time a new video is analyzed, since the collection name is fixed.
    clear_vector_store()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=get_client(),
    )

    return vector_store


# Function: Load an existing Chroma vector store
def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=get_client(),
    )

    return vector_store


# Function: Get k most relevant chunks from vector store
def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )