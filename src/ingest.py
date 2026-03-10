import os

DATA_DIR = "data"

def load_documents():
    documents = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith((".txt", ".md")):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append({"text": text, "source": path})
    return documents

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(doc["source"])