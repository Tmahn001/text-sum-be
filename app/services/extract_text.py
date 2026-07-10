"""Extract plain text from uploaded documents (.txt, .pdf, .docx)."""
from __future__ import annotations

import io

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _from_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(filename: str, data: bytes) -> str:
    """Dispatch on file extension and return extracted text.

    Raises ValueError for unsupported types so the API can return a clean 400.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _from_pdf(data)
    if name.endswith(".docx"):
        return _from_docx(data)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise ValueError(
        f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED))}"
    )
