"""
Retrieval — returns a LangChain retriever or raw similarity hits.
"""
from langchain_core.vectorstores import VectorStoreRetriever

from src.vector_store import get_vector_store


def get_retriever(k: int = 20, collection_name: str = "rag_chunks") -> VectorStoreRetriever:
    """
    Return a LangChain retriever suitable for use inside LCEL chains.
    k controls how many chunks are returned per query.
    """
    vs = get_vector_store(collection_name)
    return vs.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int = 10, collection_name: str = "rag_chunks") -> list[dict]:
    """
    Convenience function — returns a list of dicts with text, source,
    chunk_id, and similarity score. Useful outside of LangChain chains.
    """
    vs = get_vector_store(collection_name)
    results = vs.similarity_search_with_score(query, k=k)

    hits = []
    for doc, score in results:
        hits.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "score": score,
        })
    return hits


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "RAG pipeline agentic AI experience"
    print(f"Query: {query}\n")
    for i, hit in enumerate(retrieve(query, k=5), 1):
        print(f"[{i}] {hit['source']}  (score: {hit['score']:.4f})")
        print(f"    {hit['text'][:200]}\n")
