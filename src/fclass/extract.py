"""Turn any file or folder into something a model can judge.

Files are identified by their content (magic bytes), not their extension, so
a PDF called `download` or a photo called `file.bin` is still read properly.
Every file gets a preview: text where there is text, an image for the vision
model where there is a picture (photos, screenshots, scanned PDF pages), and
at minimum a description (type, size, date, tags, archive contents) so even
an unreadable file can be sorted or asked about.

Only pypdf is needed beyond the standard library; Office, ebook, email,
archive and audio-tag formats are parsed with the standard library.
"""

from __future__ import annotations

import email
import email.policy
import html
import json
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from . import native

MAX_IMAGE_BYTES = 8 * 1024 * 1024
VISION_FORMATS = ("png", "jpeg", "gif", "webp")


@dataclass
class Preview:
    text: str                     # what the model reads (always includes a description line)
    image: bytes | None = None    # a picture for the vision model, when one exists and is wanted
    readable: bool = True         # False when we only know metadata, not content
    kind: str = "file"            # human description, e.g. "scanned PDF, 2 pages"
    has_picture: bool = False     # a picture exists (even if vision is off)


# ── public entry points ─────────────────────────────────────────────────────

def preview_file(path: Path, max_chars: int = 2000, want_image: bool = False) -> Preview:
    try:
        head = _head(path)
    except OSError as e:
        return Preview(f"[Could not open: {e}]", readable=False, kind="unreadable file")
    try:
        p = _dispatch(path, head, want_image)
    except Exception as e:  # a broken file must never stop a run
        p = Preview("", readable=False, kind=f"{_ext(path)} file (could not parse: {type(e).__name__})")
    header = f"Type: {p.kind}. {_size_and_date(path)}"
    origin = native.describe_origin(native.download_origin(path))
    if origin:
        header += f" Downloaded from: {origin}."
    body = _squash(p.text)
    if not body and not p.image:
        p.readable = False
    p.text = (header + ("\n" + body if body else ""))[:max_chars]
    return p


def preview_folder(path: Path, max_chars: int = 2000) -> Preview:
    """Summarise a folder: counts, type mix, sample names and a peek at one document."""
    files, dirs = [], []
    for p in sorted(path.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(path).parts):
            continue
        (dirs if p.is_dir() else files).append(p)
        if len(files) > 500:
            break
    exts: dict[str, int] = {}
    for f in files:
        key = f.suffix.lower() or "(none)"
        exts[key] = exts.get(key, 0) + 1
    mix = ", ".join(f"{e}×{n}" for e, n in sorted(exts.items(), key=lambda kv: -kv[1])[:8])
    names = ", ".join(str(f.relative_to(path)) for f in files[:15])
    lines = [f"Type: folder with {len(files)} files and {len(dirs)} subfolders.", f"File types: {mix or 'none'}",
             f"Sample files: {names or 'none'}"]
    budget = max_chars - sum(len(l) for l in lines)
    for f in files[:20]:
        if budget < 200:
            break
        p = preview_file(f, max_chars=min(600, budget))
        if p.readable and not p.has_picture:
            lines.append(f"Excerpt from {f.name}: {p.text}")
            break
    return Preview("\n".join(lines)[:max_chars], kind="folder")


def pdf_support() -> str | None:
    """Name of the PDF text backend that will be used, or None."""
    for mod in ("pypdf", "fitz"):
        try:
            __import__(mod)
            return mod
        except ImportError:
            pass
    return "pdftotext" if shutil.which("pdftotext") else None


# ── identification ──────────────────────────────────────────────────────────

def _head(path: Path, n: int = 8192) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _ext(path: Path) -> str:
    return path.suffix.lower().lstrip(".").upper() or "extensionless"


def sniff(head: bytes, path: Path) -> str:
    """Identify a file from its first bytes. Falls back to the extension, then to 'text' or 'binary'."""
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"mif1", b"msf1", b"hevc", b"avif"):
            return "heic"
        if brand in (b"M4A ", b"M4B ", b"M4P "):
            return "audio"
        return "video"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") or head.startswith(b"fLaC") \
            or head.startswith(b"OggS") or (head[:4] == b"RIFF" and head[8:12] == b"WAVE"):
        return "audio"
    if head.startswith(b"\x1a\x45\xdf\xa3") or (head[:4] == b"RIFF" and head[8:12] == b"AVI "):
        return "video"
    if head.startswith(b"\x1f\x8b") or head.startswith(b"BZh") or head.startswith(b"\xfd7zXZ\x00") \
            or head[257:262] == b"ustar":
        return "tar"
    if head.startswith(b"7z\xbc\xaf\x27\x1c") or head.startswith(b"Rar!"):
        return "archive"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head.startswith(b"SQLite format 3\x00"):
        return "sqlite"
    if head.startswith(b"MZ"):
        return "windows-program"
    if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
        return "mac-program"
    if head.startswith(b"\x7fELF"):
        return "linux-program"
    if head.startswith(b"xar!"):
        return "mac-installer"
    if head.startswith(b"!<arch>") or head.startswith(b"\xed\xab\xee\xdb"):
        return "linux-package"
    if head[:4] in (b"OTTO", b"\x00\x01\x00\x00", b"wOFF", b"wOF2", b"true"):
        return "font"
    if _is_dmg(path):
        return "mac-disk-image"
    if b"\x00" not in head[:4096]:
        text = head.decode("utf-8", errors="ignore").lstrip("\ufeff")
        if re.match(r"(?is)\s*(<!doctype html|<html)", text):
            return "html"
        if re.match(r"(?m)^(Return-Path|Received|Delivered-To|MIME-Version|From|Message-ID):", text) \
                and re.search(r"(?mi)^(Subject|Date):", text):
            return "email"
        if path.suffix.lower() == ".ipynb" and text.lstrip().startswith("{"):
            return "notebook"
        return "text"
    return "binary"


def _is_dmg(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(-512, 2)
            return f.read(4) == b"koly"
    except OSError:
        return False


# ── dispatch ────────────────────────────────────────────────────────────────

LABELS = {
    "windows-program": "Windows program or installer", "mac-program": "macOS program",
    "linux-program": "Linux program", "mac-installer": "macOS installer package",
    "linux-package": "Linux software package", "mac-disk-image": "macOS disk image (installer)",
    "font": "font file", "sqlite": "SQLite database", "archive": "compressed archive",
}


def _dispatch(path: Path, head: bytes, want_image: bool) -> Preview:
    t = sniff(head, path)
    if t == "pdf":
        return _pdf(path, want_image)
    if t == "zip":
        return _zip_family(path)
    if t in VISION_FORMATS:
        return _image(path, t, want_image)
    if t == "heic":
        if native.can_convert():  # macOS: sips turns iPhone photos into JPEG
            jpeg = native.sips_jpeg(path) if want_image else None
            return Preview("", image=jpeg, readable=bool(jpeg), kind="HEIC photo", has_picture=True)
        return Preview("", readable=False, kind="HEIC/AVIF photo (only readable on macOS)", has_picture=True)
    if t == "audio":
        return _audio(path, head)
    if t == "video":
        return Preview("", readable=False, kind=f"video ({_ext(path)})")
    if t == "tar":
        return _tar(path)
    if t == "rtf":
        return Preview(_rtf(path), kind="RTF document")
    if t == "email":
        return _email(path)
    if t == "html":
        return _html(path)
    if t == "notebook":
        return _notebook(path)
    if t == "text":
        return Preview(_text(path), kind=f"text ({_ext(path)})")
    if t in LABELS:
        return Preview(_strings(path, 300), readable=False, kind=LABELS[t])
    return Preview(_strings(path), readable=False, kind=f"binary {_ext(path)} file")


# ── PDF ─────────────────────────────────────────────────────────────────────

def _pdf(path: Path, want_image: bool, pages: int = 3) -> Preview:
    text, n_pages, title, page_image = "", 0, "", None
    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return Preview("", readable=False, kind="password-protected PDF")
        n_pages = len(reader.pages)
        title = str((reader.metadata or {}).get("/Title") or "").strip()
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:pages])
        if len(text.strip()) < 40 and n_pages:
            page_image = _pdf_page_image(reader.pages[0])
    except ImportError:
        if shutil.which("pdftotext"):
            text = subprocess.run(["pdftotext", "-l", str(pages), str(path), "-"],
                                  capture_output=True, text=True, timeout=30).stdout
    pages_label = f"{n_pages} page{'s' if n_pages != 1 else ''}" if n_pages else "PDF"
    if len(text.strip()) >= 40:
        lead = f"Title: {title}\n" if title and title.lower() not in text[:200].lower() else ""
        return Preview(lead + text, kind=f"PDF, {pages_label}")
    # A scan or a photo saved as PDF: no text layer, so show the page to the vision model.
    if page_image is None:
        page_image = _render_page(path)
    kind = f"scanned PDF without a text layer, {pages_label}"
    if page_image and want_image:
        return Preview(text, image=page_image, kind=kind, has_picture=True)
    return Preview(text, readable=False, kind=kind, has_picture=bool(page_image))


def _pdf_page_image(page) -> bytes | None:
    """Largest image on the page as JPEG or PNG bytes, using only pypdf and zlib."""
    try:
        xobjects = page["/Resources"]["/XObject"]
    except (KeyError, TypeError):
        return None
    best, best_area = None, 0
    for key in xobjects:
        obj = xobjects[key].get_object()
        if obj.get("/Subtype") != "/Image":
            continue
        area = int(obj.get("/Width", 0)) * int(obj.get("/Height", 0))
        if area > best_area:
            best, best_area = obj, area
    if best is None:
        return None
    filters = best.get("/Filter")
    filters = [str(f) for f in filters] if isinstance(filters, list) else [str(filters)]
    if filters[-1] == "/DCTDecode":
        return best._data  # the stream *is* a JPEG file
    if filters == ["/FlateDecode"] and int(best.get("/BitsPerComponent", 8)) == 8:
        cs = str(best.get("/ColorSpace"))
        channels = {"/DeviceRGB": 3, "/DeviceGray": 1}.get(cs)
        if channels:
            return _png(int(best["/Width"]), int(best["/Height"]), best.get_data(), channels)
    return None  # JBIG2, CCITT fax, JPX ... rendered by pdftoppm instead, when available


def _render_page(path: Path) -> bytes | None:
    if not shutil.which("pdftoppm"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "page"
        subprocess.run(["pdftoppm", "-png", "-r", "110", "-f", "1", "-l", "1", "-singlefile", str(path), str(out)],
                       capture_output=True, timeout=60)
        png = out.with_suffix(".png")
        return png.read_bytes() if png.exists() else None


def _png(width: int, height: int, raw: bytes, channels: int) -> bytes | None:
    stride = width * channels
    if len(raw) < stride * height:
        return None
    rows = b"".join(b"\x00" + raw[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    color_type = 2 if channels == 3 else 0
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b"")


# ── images ──────────────────────────────────────────────────────────────────

def _image(path: Path, fmt: str, want_image: bool) -> Preview:
    dims = _dimensions(path, fmt)
    kind = f"{fmt.upper()} image" + (f", {dims[0]}×{dims[1]}" if dims else "")
    if dims and _looks_like_screenshot(path.name, dims):
        kind += ", probably a screenshot"
    size = path.stat().st_size
    if want_image:
        big = size > 1_500_000 or (dims is not None and max(dims) > 2000)
        if big and native.can_convert():  # a smaller picture reads several times faster
            small = native.sips_jpeg(path)
            if small:
                return Preview("", image=small, kind=kind, has_picture=True)
        if size <= MAX_IMAGE_BYTES:
            return Preview("", image=path.read_bytes(), kind=kind, has_picture=True)
    return Preview("", readable=False, kind=kind, has_picture=True)


def _dimensions(path: Path, fmt: str) -> tuple[int, int] | None:
    with open(path, "rb") as f:
        data = f.read(65536)
    try:
        if fmt == "png":
            return struct.unpack(">II", data[16:24])
        if fmt == "gif":
            return struct.unpack("<HH", data[6:10])
        if fmt == "jpeg":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    except struct.error:
        pass
    return None


def _looks_like_screenshot(name: str, dims: tuple[int, int]) -> bool:
    if re.search(r"(?i)screen ?shot|screen[_ -]?capture|scr_?\d", name):
        return True
    return dims in {(1170, 2532), (1179, 2556), (1290, 2796), (1080, 2400), (2880, 1800), (2560, 1600),
                    (1920, 1080), (3024, 1964), (3456, 2234)}


# ── zip family: Office, OpenDocument, EPUB, plain archives ──────────────────

def _zip_family(path: Path) -> Preview:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        mimetype = z.read("mimetype").decode(errors="ignore").strip() if "mimetype" in names else ""
        if "word/document.xml" in names:
            return Preview(_xml_text(z, ["word/document.xml"], "p"), kind="Word document")
        if "ppt/presentation.xml" in names:
            slides = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                            key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
            return Preview(_xml_text(z, slides[:12], "p"), kind=f"PowerPoint deck, {len(slides)} slides")
        if "xl/workbook.xml" in names:
            return Preview(_xlsx(z, names), kind="Excel spreadsheet")
        if mimetype.startswith("application/vnd.oasis.opendocument"):
            label = {"text": "OpenDocument text", "spreadsheet": "OpenDocument spreadsheet",
                     "presentation": "OpenDocument presentation"}.get(mimetype.rsplit(".", 1)[-1], "OpenDocument file")
            return Preview(_xml_text(z, ["content.xml"], "p"), kind=label)
        if mimetype == "application/epub+zip":
            return _epub(z, names)
        if "AndroidManifest.xml" in names:
            return Preview("", readable=False, kind="Android app package")
    files = [n for n in names if not n.endswith("/") and "__MACOSX" not in n]
    return Preview(_listing(files), kind=f"ZIP archive, {len(files)} files")


def _xml_text(z: zipfile.ZipFile, members: list[str], para_tag: str) -> str:
    out: list[str] = []
    for m in members:
        root = ElementTree.fromstring(z.read(m))
        for el in root.iter():
            if el.tag.rsplit("}", 1)[-1] == para_tag:
                line = "".join(el.itertext()).strip()
                if line:
                    out.append(line)
        if sum(len(l) for l in out) > 6000:
            break
    return "\n".join(out)


def _xlsx(z: zipfile.ZipFile, names: list[str], max_rows: int = 15) -> str:
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(si.itertext()))
    sheets = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    if not sheets:
        return ""
    rows = []
    for row in ElementTree.fromstring(z.read(sheets[0])).iter():
        if row.tag.rsplit("}", 1)[-1] != "row":
            continue
        cells = []
        for c in row:
            if c.tag.rsplit("}", 1)[-1] != "c":
                continue
            v = next((x.text for x in c if x.tag.rsplit("}", 1)[-1] == "v"), None)
            if c.get("t") == "s" and v is not None:
                v = shared[int(v)]
            elif c.get("t") == "inlineStr":
                v = "".join(c.itertext())
            cells.append(v or "")
        rows.append(", ".join(cells))
        if len(rows) >= max_rows:
            break
    return "\n".join(rows)


def _epub(z: zipfile.ZipFile, names: list[str]) -> Preview:
    title = author = ""
    opf = next((n for n in names if n.endswith(".opf")), None)
    if opf:
        meta = z.read(opf).decode("utf-8", errors="ignore")
        title = _first(r"<dc:title[^>]*>(.*?)</dc:title>", meta)
        author = _first(r"<dc:creator[^>]*>(.*?)</dc:creator>", meta)
    pages = sorted(n for n in names if n.endswith((".xhtml", ".html", ".htm")))
    body = ""
    for n in pages[:6]:
        body += " " + _strip_tags(z.read(n).decode("utf-8", errors="ignore"))
        if len(body) > 3000:
            break
    lead = "\n".join(x for x in (f"Title: {title}" if title else "", f"Author: {author}" if author else "") if x)
    return Preview(f"{lead}\n{body}".strip(), kind="EPUB ebook")


def _tar(path: Path) -> Preview:
    try:
        with tarfile.open(path) as t:
            files = [m.name for m in t.getmembers() if m.isfile()]
        return Preview(_listing(files), kind=f"compressed archive, {len(files)} files")
    except tarfile.TarError:
        return Preview("", readable=False, kind="compressed file")


def _listing(files: list[str]) -> str:
    exts: dict[str, int] = {}
    for f in files:
        e = Path(f).suffix.lower() or "(none)"
        exts[e] = exts.get(e, 0) + 1
    mix = ", ".join(f"{e}×{n}" for e, n in sorted(exts.items(), key=lambda kv: -kv[1])[:8])
    return f"Contains: {mix}\nFiles: " + ", ".join(files[:25])


# ── text-like formats ───────────────────────────────────────────────────────

def _text(path: Path, limit: int = 20000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(limit)


def _html(path: Path) -> Preview:
    raw = _text(path, 200000)
    title = _first(r"<title[^>]*>(.*?)</title>", raw)
    body = _strip_tags(raw)
    return Preview((f"Title: {title}\n" if title else "") + body, kind="web page (HTML)")


def _strip_tags(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw)
    return html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))


def _email(path: Path) -> Preview:
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    lines = [f"{h}: {msg[h]}" for h in ("Subject", "From", "To", "Date") if msg[h]]
    body = msg.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body else ""
    if body and body.get_content_type() == "text/html":
        text = _strip_tags(text)
    attachments = [a.get_filename() for a in msg.iter_attachments() if a.get_filename()]
    if attachments:
        lines.append("Attachments: " + ", ".join(attachments[:10]))
    return Preview("\n".join(lines) + "\n" + text, kind="email message")


def _notebook(path: Path) -> Preview:
    nb = json.loads(_text(path, 2_000_000))
    cells = ["".join(c.get("source", "")) for c in nb.get("cells", [])[:30]]
    return Preview("\n".join(cells), kind="Jupyter notebook")


def _rtf(path: Path) -> str:
    raw = _text(path, 200000)
    raw = re.sub(r"\\'[0-9a-f]{2}", " ", raw)
    raw = re.sub(r"\\[a-z]+-?\d* ?|[{}]", " ", raw)
    return raw


# ── media and binaries ─────────────────────────────────────────────────────

ID3_FRAMES = {"TIT2": "Title", "TPE1": "Artist", "TALB": "Album", "TCON": "Genre", "TYER": "Year", "TDRC": "Year"}


def _audio(path: Path, head: bytes) -> Preview:
    tags = _id3(path) if head.startswith(b"ID3") else {}
    text = "\n".join(f"{k}: {v}" for k, v in tags.items())
    return Preview(text, readable=bool(tags), kind=f"audio ({_ext(path)})")


def _id3(path: Path) -> dict[str, str]:
    with open(path, "rb") as f:
        header = f.read(10)
        size = (header[6] << 21) | (header[7] << 14) | (header[8] << 7) | header[9]
        data = f.read(min(size, 256 * 1024))
    version = header[3]
    out: dict[str, str] = {}
    i = 0
    while i + 10 <= len(data):
        fid = data[i:i + 4].decode("latin-1", errors="ignore")
        if not fid.strip("\x00"):
            break
        n = struct.unpack(">I", data[i + 4:i + 8])[0]
        if version == 4:  # syncsafe sizes
            n = (n & 0x7F) | ((n >> 8) & 0x7F) << 7 | ((n >> 16) & 0x7F) << 14 | ((n >> 24) & 0x7F) << 21
        payload = data[i + 10:i + 10 + n]
        if fid in ID3_FRAMES and payload:
            enc, body = payload[0], payload[1:]
            codec = {0: "latin-1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}.get(enc, "latin-1")
            value = body.decode(codec, errors="ignore").strip("\x00 ").strip()
            if value:
                out.setdefault(ID3_FRAMES[fid], value)
        i += 10 + n
    return out


def _strings(path: Path, limit: int = 600) -> str:
    """Printable runs from the start of a binary file: often names, titles or vendors."""
    with open(path, "rb") as f:
        data = f.read(65536)
    runs = [r.decode("ascii") for r in re.findall(rb"[ -~]{6,}", data)]
    words = [r for r in runs if re.search(r"[A-Za-z]{4,}", r)]
    out = " | ".join(dict.fromkeys(words))
    return f"Text found inside: {out[:limit]}" if out else ""


# ── helpers ─────────────────────────────────────────────────────────────────

def _first(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.S | re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip() if m else ""


def _size_and_date(path: Path) -> str:
    st = path.stat()
    size = st.st_size
    human = f"{size / 1e9:.1f} GB" if size > 1e9 else f"{size / 1e6:.1f} MB" if size > 1e6 else f"{max(size, 1) / 1e3:.0f} KB"
    return f"Size: {human}. Modified: {time.strftime('%Y-%m-%d', time.localtime(st.st_mtime))}."


def _squash(text: str) -> str:
    text = re.sub(r"[ \t\u00a0]+", " ", text or "")
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()
