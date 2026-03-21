"""
Resume version manager.

Responsibilities:
  - Load all base resume variants from data/resumes/
  - Save tailored resume versions to data/output/resumes/
  - Maintain a JSON index (data/output/resume_index.json) that tracks
    each tailored version: which base resume, which job, when generated.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

_RESUME_DIR = Path(__file__).parent.parent / "data" / "resumes"
_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output" / "resumes"
_INDEX_FILE = Path(__file__).parent.parent / "data" / "output" / "resume_index.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_resume_texts(resume_dir: Path = _RESUME_DIR) -> list[dict]:
    """
    Load all resume files (.txt, .md, .pdf) from resume_dir.
    Returns list of dicts: {path, name, text}
    """
    resumes = []
    for p in sorted(resume_dir.iterdir()):
        if p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            resumes.append({"path": str(p), "name": p.stem, "text": text})
        elif p.suffix.lower() == ".pdf":
            text = _extract_pdf_text(p)
            resumes.append({"path": str(p), "name": p.stem, "text": text})
    return resumes


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_best_resume(resumes: list[dict], jd_text: str) -> dict:
    """
    Simple heuristic: pick the resume with the most keyword overlap with the JD.
    Returns the resume dict with the highest overlap score.
    """
    if not resumes:
        raise ValueError("No resumes loaded.")
    if len(resumes) == 1:
        return resumes[0]

    jd_tokens = set(re.findall(r"\b\w+\b", jd_text.lower()))

    def overlap(r):
        tokens = set(re.findall(r"\b\w+\b", r["text"].lower()))
        return len(tokens & jd_tokens)

    return max(resumes, key=overlap)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_tailored_resume(
    tailored_text: str,
    base_resume_name: str,
    jd_label: str,
    mlflow_run_id: str | None = None,
) -> Path:
    """
    Save a tailored resume to data/output/resumes/ and update the index.
    Returns the path of the saved file.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^\w\-]", "_", jd_label)
    filename = f"{safe_label}_{timestamp}.txt"
    out_path = _OUTPUT_DIR / filename
    out_path.write_text(tailored_text, encoding="utf-8")

    # Update index
    index = _load_index()
    index.append({
        "file": filename,
        "base_resume": base_resume_name,
        "jd_label": jd_label,
        "created_at": datetime.now().isoformat(),
        "mlflow_run_id": mlflow_run_id,
    })
    _save_index(index)

    print(f"Saved tailored resume: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _load_index() -> list[dict]:
    if _INDEX_FILE.exists():
        return json.loads(_INDEX_FILE.read_text(encoding="utf-8"))
    return []


def _save_index(index: list[dict]):
    _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def list_tailored_resumes() -> list[dict]:
    """Return the full index of generated resume versions."""
    return _load_index()
