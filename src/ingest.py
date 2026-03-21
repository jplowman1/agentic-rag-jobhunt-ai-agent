"""
Document ingestion — loads .txt, .md, .pdf, and .docx files from data/.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


_READERS = {
    ".txt": _read_txt,
    ".md":  _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Recursively load all supported documents under data_dir.
    Returns list of dicts: {text, source}
    Skips files in data/output/ (those are generated artifacts).
    """
    documents = []
    for root, dirs, files in os.walk(data_dir):
        # Don't index generated output
        dirs[:] = [d for d in dirs if d != "output"]
        for file in sorted(files):
            path = Path(root) / file
            reader = _READERS.get(path.suffix.lower())
            if reader is None:
                continue
            try:
                text = reader(path)
                if text.strip():
                    documents.append({"text": text, "source": str(path)})
            except Exception as e:
                print(f"  Warning: could not read {path}: {e}")
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  {doc['source']}  ({len(doc['text'])} chars)")
