"""
Embedding abstraction — supports Bedrock (Titan) and OpenAI.
Provider and model are controlled by config/preferences.yaml.
"""
import os

from dotenv import load_dotenv

from src.config_loader import load_config

load_dotenv()


def get_embeddings():
    """
    Return a LangChain Embeddings instance based on preferences.yaml.
    Drop-in replacement anywhere LangChain expects an Embeddings object.
    """
    cfg = load_config()
    provider = cfg["embeddings"]["provider"]
    model_id = cfg["embeddings"]["model_id"]
    region = cfg["embeddings"].get("region", "us-east-1")

    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(
            model_id=model_id,
            region_name=region,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        return OpenAIEmbeddings(model=model_id, api_key=api_key)

    raise ValueError(f"Unknown embeddings provider: {provider!r}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns a list of float vectors."""
    return get_embeddings().embed_documents(texts)


def embed_text(text: str) -> list[float]:
    """Embed a single string."""
    return get_embeddings().embed_query(text)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Take a list of chunk dicts (with a 'text' key) and return the same
    list enriched with an 'embedding' key on each chunk.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count mismatch.")
    return [{**c, "embedding": v} for c, v in zip(chunks, vectors)]
