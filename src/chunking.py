def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_documents(documents, chunk_size=800, overlap=100):
    all_chunks = []
    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": f'{doc["source"]}::chunk_{i}'
            })
    return all_chunks