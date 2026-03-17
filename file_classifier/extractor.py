"""File content extraction utilities."""

from pathlib import Path


def extract_preview(filepath: Path, max_chars: int = 1500) -> str:
    """Extract text preview from various file types.

    Args:
        filepath: Path to the file
        max_chars: Maximum characters to extract

    Returns:
        Extracted text preview or error message
    """
    ext = filepath.suffix.lower()

    try:
        if ext == '.pdf':
            return _extract_pdf(filepath, max_chars)
        elif ext == '.docx':
            return _extract_docx(filepath, max_chars)
        elif ext in ['.txt', '.md', '.csv']:
            return _extract_text(filepath, max_chars)
        elif ext in ['.xlsx', '.xls']:
            return _extract_excel(filepath, max_chars)
        else:
            # Try to read as text first, fallback to binary message
            try:
                return _extract_text(filepath, max_chars)
            except:
                return f"[Binary file: {ext} - no text preview available]"
    except Exception:
        # If specific extractor fails, try reading as plain text
        # (handles files with wrong extensions)
        try:
            return _extract_text(filepath, max_chars)
        except Exception as e2:
            return f"[Error reading file: {e2}]"


def _extract_pdf(filepath: Path, max_chars: int) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) > max_chars:
            break
    doc.close()
    return text[:max_chars]


def _extract_docx(filepath: Path, max_chars: int) -> str:
    """Extract text from Word document."""
    import docx
    doc = docx.Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs[:20]]
    text = "\n".join(paragraphs)
    return text[:max_chars]


def _extract_text(filepath: Path, max_chars: int) -> str:
    """Extract text from plain text files."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read(max_chars)


def _extract_excel(filepath: Path, max_chars: int) -> str:
    """Extract preview from Excel files."""
    import pandas as pd
    df = pd.read_excel(filepath, nrows=10)
    preview = df.head().to_string()
    return preview[:max_chars]
