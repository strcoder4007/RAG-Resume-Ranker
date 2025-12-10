from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from PyPDF2 import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file, concatenating all pages."""
    text_parts: list[str] = []
    with open(file_path, "rb") as file_handle:
        reader = PdfReader(file_handle)
        for page in reader.pages:
            content = page.extract_text() or ""
            if content:
                text_parts.append(content)
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file by joining all paragraphs."""
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    return "\n".join(paragraphs).strip()


def process_document(file_path: str) -> Optional[Dict[str, str]]:
    """
    Process a single document and return structured data.

    Returns:
        A dictionary containing filename, content, and file_type keys, or None if processing fails.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning("File does not exist: %s", file_path)
        return None

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported file type for %s", path.name)
        return None

    try:
        if suffix == ".pdf":
            content = extract_text_from_pdf(str(path))
        else:
            content = extract_text_from_docx(str(path))
    except Exception as exc:
        logger.warning("Skipping %s: failed to process (%s)", path.name, exc)
        return None

    if not content:
        logger.warning("Skipping %s: extracted content is empty", path.name)
        return None

    return {
        "filename": path.name,
        "content": content,
        "file_type": suffix.lstrip("."),
    }


def load_job_description(jd_path: str = "./jd.txt") -> str:
    """Load and validate the job description from a file."""
    path = Path(jd_path)
    if not path.exists():
        raise FileNotFoundError(f"Job description file not found at {jd_path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("Job description file is empty.")

    return content
