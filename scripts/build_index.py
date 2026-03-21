"""
Ingestion pipeline — load profile documents, chunk, embed, store in ChromaDB.

Run this whenever you update files in data/resumes/ or data/profile/.

Usage:
    python -m scripts.build_index [--reset]

  --reset   Drop and recreate the Chroma collection before indexing.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from src.chunking import chunk_documents
from src.embeddings import embed_chunks
from src.ingest import load_documents
from src.vector_store import collection_count, reset_collection, store_chunks


def main():
    parser = argparse.ArgumentParser(description="Build the ChromaDB profile index.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection first.")
    args = parser.parse_args()

    if args.reset:
        print("Resetting collection...")
        reset_collection()

    print("Loading documents...")
    documents = load_documents()
    print(f"  Loaded {len(documents)} documents")

    print("Chunking...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Embedding (this calls Bedrock — may take a moment)...")
    embedded = embed_chunks(chunks)
    print(f"  Generated {len(embedded)} embeddings")

    print("Storing in ChromaDB...")
    store_chunks(embedded)

    total = collection_count()
    print(f"\nDone. Collection now contains {total} chunks.")


if __name__ == "__main__":
    main()
