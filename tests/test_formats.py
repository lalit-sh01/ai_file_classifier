"""Any file type gets a useful preview, identified by content rather than extension."""

import struct
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest

from fclass.extract import preview_file, sniff


# ── a tiny PDF writer, so tests need no PDF library ─────────────────────────

def make_pdf(path: Path, text: str | None = None, image: tuple[str, int, int, bytes] | None = None):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"]
    resources, content = "", b""
    extra: list[bytes] = []
    if text:
        resources = "/Font << /F1 5 0 R >>"
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    if image:
        filt, w, h, data = image
        resources += " /XObject << /Im1 5 0 R >>" if not text else ""
        content = b"q 200 0 0 200 72 500 cm /Im1 Do Q"
        cs = "/DeviceRGB"
        extra.append(f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace {cs} "
                     f"/BitsPerComponent 8 /Filter {filt} /Length {len(data)} >>\nstream\n".encode() + data
                     + b"\nendstream")
    objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << {resources} >> /Contents 4 0 R >>")
    objs_bytes = [o.encode() for o in objs]
    objs_bytes.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
    if text:
        objs_bytes.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs_bytes += extra
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs_bytes, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs_bytes) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs_bytes) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def test_pdf_text_layer_is_read(tmp_path):
    make_pdf(tmp_path / "doc", text="Form W-2 Wage and Tax Statement 2024 federal income tax withheld")
    p = preview_file(tmp_path / "doc")  # no extension at all
    assert "W-2 Wage and Tax Statement" in p.text and p.kind.startswith("PDF") and p.readable


def test_scanned_pdf_jpeg_page_goes_to_vision(tmp_path):
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"
    make_pdf(tmp_path / "scan.pdf", image=("/DCTDecode", 8, 8, fake_jpeg))
    p = preview_file(tmp_path / "scan.pdf", want_image=True)
    assert "scanned PDF" in p.kind and p.image == fake_jpeg
    blind = preview_file(tmp_path / "scan.pdf", want_image=False)
    assert blind.image is None and not blind.readable and blind.has_picture


def test_scanned_pdf_raw_pixels_become_png(tmp_path):
    w, h = 4, 3
    pixels = bytes([200, 30, 30] * w * h)
    make_pdf(tmp_path / "scan.pdf", image=("/FlateDecode", w, h, zlib.compress(pixels)))
    png = preview_file(tmp_path / "scan.pdf", want_image=True).image
    assert png.startswith(b"\x89PNG") and struct.unpack(">II", png[16:24]) == (w, h)


# ── everything else ─────────────────────────────────────────────────────────

def test_wrong_extension_is_ignored(tmp_path):
    (tmp_path / "photo.txt").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1170, 2532) + b"\0" * 20)
    p = preview_file(tmp_path / "photo.txt")
    assert p.kind.startswith("PNG image, 1170×2532") and "screenshot" in p.kind and p.has_picture


def test_archive_lists_its_contents(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        z.writestr("invoices/2024-03.pdf", "x")
        z.writestr("invoices/2024-04.pdf", "x")
    with tarfile.open(tmp_path / "b.tar.gz", "w:gz") as t:
        f = tmp_path / "notes.md"
        f.write_text("x")
        t.add(f, arcname="course/notes.md")
    assert "invoices/2024-03.pdf" in preview_file(tmp_path / "a.zip").text
    assert "course/notes.md" in preview_file(tmp_path / "b.tar.gz").text


def test_epub_email_html_rtf(tmp_path):
    with zipfile.ZipFile(tmp_path / "book.epub", "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("OEBPS/content.opf", "<package><dc:title>Dune</dc:title><dc:creator>Frank Herbert</dc:creator></package>")
        z.writestr("OEBPS/ch1.xhtml", "<html><body><p>A beginning is the time</p></body></html>")
    (tmp_path / "mail").write_text("From: airline@example.com\nTo: me@example.com\nSubject: Your e-ticket BOM-EWR\n"
                                   "Date: Mon, 2 Jun 2025 10:00:00 +0000\nMIME-Version: 1.0\nContent-Type: text/plain\n\n"
                                   "Booking ref Q7XK2P, seat 34A.\n")
    (tmp_path / "page.html").write_text("<!doctype html><title>Sourdough guide</title><script>x()</script><p>Autolyse 1 hr</p>")
    (tmp_path / "letter.rtf").write_text(r"{\rtf1\ansi{\fonttbl\f0 Helvetica;}\f0\fs24 Residential lease agreement}")
    book = preview_file(tmp_path / "book.epub")
    assert book.kind == "EPUB ebook" and "Frank Herbert" in book.text
    mail = preview_file(tmp_path / "mail")
    assert mail.kind == "email message" and "e-ticket" in mail.text and "seat 34A" in mail.text
    page = preview_file(tmp_path / "page.html")
    assert "Sourdough guide" in page.text and "x()" not in page.text
    assert "lease agreement" in preview_file(tmp_path / "letter.rtf").text


def test_audio_tags_and_binaries(tmp_path):
    def frame(fid, value):
        body = b"\x03" + value.encode()
        return fid.encode() + struct.pack(">I", len(body)) + b"\0\0" + body
    frames = frame("TIT2", "Kesariya") + frame("TPE1", "Arijit Singh")
    (tmp_path / "track").write_bytes(b"ID3\x03\x00\x00" + bytes([0, 0, 0, len(frames)]) + frames + b"\xff\xfb" + b"\0" * 50)
    song = preview_file(tmp_path / "track")
    assert song.kind.startswith("audio") and "Arijit Singh" in song.text and song.readable

    (tmp_path / "setup.exe").write_bytes(b"MZ\x90\x00" + b"\0" * 60 + b"Zoom Video Communications Installer\0" * 3)
    exe = preview_file(tmp_path / "setup.exe")
    assert exe.kind == "Windows program or installer" and "Zoom Video" in exe.text and not exe.readable

    (tmp_path / "blob").write_bytes(bytes(range(256)) * 4)
    assert not preview_file(tmp_path / "blob").readable


@pytest.mark.parametrize("head,kind", [
    (b"%PDF-1.7", "pdf"), (b"\xff\xd8\xff\xe0", "jpeg"), (b"\x00\x00\x00\x18ftypheic", "heic"),
    (b"\x00\x00\x00\x18ftypisom", "video"), (b"7z\xbc\xaf\x27\x1c", "archive"), (b"SQLite format 3\x00", "sqlite"),
    (b"{\\rtf1", "rtf"), (b"hello world", "text"),
])
def test_sniff(head, kind, tmp_path):
    f = tmp_path / "x"
    f.write_bytes(head + b"\0" * 600 if kind not in ("text", "rtf") else head)
    assert sniff(head + (b"\0" * 600 if kind not in ("text", "rtf") else b""), f) == kind


def test_every_preview_says_what_it_is(tmp_path):
    (tmp_path / "empty").write_bytes(b"")
    p = preview_file(tmp_path / "empty")
    assert p.text.startswith("Type:") and "Size:" in p.text and not p.readable
