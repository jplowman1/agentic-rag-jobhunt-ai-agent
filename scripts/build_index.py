import sys
import os

# allow scripts/ to import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import load_documents
from src.chunking import chunk_documents


def main():
    print("Loading documents...")

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("\nChunking documents...")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks\n")

    # show a few example chunks
    for chunk in chunks[:3]:
        print("SOURCE:", chunk["source"])
        print("CHUNK_ID:", chunk["chunk_id"])
        print("TEXT PREVIEW:", chunk["text"][:200])
        print("-" * 60)


if __name__ == "__main__":
    main()
    