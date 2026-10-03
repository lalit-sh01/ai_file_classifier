"""Turn a file or folder into a short text preview.

Office formats (docx/xlsx/pptx/odt/ods/odp) are zip archives of XML, so they
are read with the standard library. PDF uses pypdf or PyMuPDF if installed,
then falls back to the `pdftotext` CLI.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".xml",
    ".html", ".htm", ".log", ".ini", ".tex", ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".sh", ".sql", ".ipynb", ".srt", ".vtt", ".eml", ".ics", ".vcf",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass
class Preview:
    text: str                     # what the model reads
    image: bytes | None = None    # raw image bytes for vision models
    readable: bool = True         # False when we only know name/size/type


def preview_file(path: Path, max_chars: int = 2000, want_image: bool = False) -> Preview:
    ext = path.suffix.lower()
    try:
        if ext in IMAGE_EXTS:
            if want_image and path.stat().st_size <= MAX_IMAGE_BYTES:
                return Preview(_meta(path), image=path.read_bytes())
            return Preview(_meta(path), readable=False)
        text = _extract(path, ext)
    except Exception:
        text = ""
    text = _squash(text)
    if not text:
        return Preview(_meta(path), readable=False)
    return Preview(text[:max_chars])


def preview_folder(path: Path, max_chars: int = 2000) -> Preview:
    """Summarise a folder: counts, extension mix, sample names and a peek at one document."""
    files, dirs = [], []
    for p in sorted(path.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(path).parts):
            continue
        (dirs if p.is_dir() else files).append(p)
        if len(files) > 500:
            break
    exts: dict[str, int] = {}
    for f in files:
        exts[f.suffix.lower() or "(none)"] = exts.get(f.suffix.lower() or "(none)", 0) + 1
    mix = ", ".join(f"{e}×{n}" for e, n in sorted(exts.items(), key=lambda kv: -kv[1])[:8])
    names = ", ".join(str(f.relative_to(path)) for f in files[:15])
    lines = [f"Folder with {len(files)} files and {len(dirs)} subfolders.", f"File types: {mix or 'none'}",
             f"Sample files: {names or 'none'}"]
    budget = max_chars - sum(len(l) for l in lines)
    for f in files[:20]:
        if f.suffix.lower() not in IMAGE_EXTS and budget > 200:
            p = preview_file(f, max_chars=min(600, budget))
            if p.readable:
                lines.append(f"Excerpt from {f.name}: {p.text}")
                break
    return Preview("\n".join(lines)[:max_chars])


def pdf_support() -> str | None:
    """Name of the PDF backend that will be used, or None."""
    for mod in ("pypdf", "fitz"):
        try:
            __import__(mod)
            return mod
        except ImportError:
            pass
    return "pdftotext" if shutil.which("pdftotext") else None


# ── extractors ──────────────────────────────────────────────────────────────

def _extract(path: Path, ext: str) -> str:
    if ext == ".pdf":
        return _pdf(path)
    if ext in (".docx", ".dotx"):
        return _zip_xml(path, ["word/document.xml"], para_tag="p")
    if ext in (".pptx",):
        return _zip_xml(path, _members(path, r"ppt/slides/slide\d+\.xml"), para_tag="p")
    if ext in (".xlsx", ".xlsm"):
        return _xlsx(path)
    if ext in (".odt", ".ods", ".odp"):
        return _zip_xml(path, ["content.xml"], para_tag="p")
    if ext in TEXT_EXTS or ext == "":
        return _text(path)
    # Unknown extension: accept it only if it looks like text.
    head = path.read_bytes()[:4096]
    if b"\x00" in head:
        return ""
    return _text(path)


def _text(path: Path, limit: int = 20000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(limit)


def _pdf(path: Path, pages: int = 3) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:pages])
    except ImportError:
        pass
    try:
        import fitz

        with fitz.open(path) as doc:
            return "\n".join(doc[i].get_text() for i in range(min(pages, len(doc))))
    except ImportError:
        pass
    if shutil.which("pdftotext"):
        out = subprocess.run(["pdftotext", "-l", str(pages), str(path), "-"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout
    return ""


def _members(path: Path, pattern: str) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.fullmatch(pattern, n)]
    return sorted(names, key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[-1]).group(1)))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _zip_xml(path: Path, members: list[str], para_tag: str) -> str:
    out: list[str] = []
    with zipfile.ZipFile(path) as z:
        for m in members:
            root = ElementTree.fromstring(z.read(m))
            for el in root.iter():
                if _local(el.tag) == para_tag:
                    line = "".join(t for t in el.itertext()).strip()
                    if line:
                        out.append(line)
    return "\n".join(out)


def _xlsx(path: Path, max_rows: int = 15) -> str:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")):
                shared.append("".join(si.itertext()))
        sheets = sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if not sheets:
            return ""
        root = ElementTree.fromstring(z.read(sheets[0]))
    rows = []
    for row in root.iter():
        if _local(row.tag) != "row":
            continue
        cells = []
        for c in row:
            if _local(c.tag) != "c":
                continue
            v = next((x.text for x in c if _local(x.tag) == "v"), None)
            if c.get("t") == "s" and v is not None:
                v = shared[int(v)]
            elif c.get("t") == "inlineStr":
                v = "".join(c.itertext())
            cells.append(v or "")
        rows.append(", ".join(cells))
        if len(rows) >= max_rows:
            break
    return "\n".join(rows)


def _meta(path: Path) -> str:
    size = path.stat().st_size
    human = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.0f} KB"
    return f"[No readable text. File type: {path.suffix.lower() or 'unknown'}, size: {human}]"


def _squash(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()
