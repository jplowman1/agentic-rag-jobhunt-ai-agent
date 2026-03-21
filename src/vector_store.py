"""
Vector store abstraction — wraps LangChain's Chroma integration.
Keeps the low-level chunk storage API compatible with build_index.py
while exposing a LangChain VectorStore for retrieval.
"""
from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import get_embeddings

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "rag_chunks"


def get_vector_store(collection_name: str = COLLECTION_NAME) -> Chroma:
    """Return a LangChain Chroma VectorStore backed by the local chroma_db."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )


def store_chunks(chunks: list[dict], collection_name: str = COLLECTION_NAME) -> Chroma:
    """
    Store pre-embedded chunks in Chroma.

    Each chunk dict must have: text, chunk_id, source, embedding.
    Returns the LangChain VectorStore instance.
    """
    if not chunks:
        print("No chunks to store.")
        return get_vector_store(collection_name)

    texts = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
    vs._collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Stored {len(chunks)} chunks in Chroma ({collection_name}).")
    return vs


def reset_collection(collection_name: str = COLLECTION_NAME) -> Chroma:
    """Delete and recreate the named collection. Returns fresh VectorStore."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
        print(f"Deleted collection '{collection_name}'.")
    except Exception:
        pass
    # Re-initialise via LangChain wrapper
    return get_vector_store(collection_name)


def collection_count(collection_name: str = COLLECTION_NAME) -> int:
    vs = get_vector_store(collection_name)
    return vs._collection.count()
