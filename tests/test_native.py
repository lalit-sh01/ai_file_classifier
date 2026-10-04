"""Download origin, picture conversion and native dialogs.

Tests marked `mac` run against the real sips, xattr and AppleScript compiler
on macOS (the CI matrix includes a macOS runner); elsewhere they are skipped
and the portable tests cover the same logic with fakes.
"""

import os
import platform
import plistlib
import shutil
import stat
import subprocess
import sys

import pytest

from fclass import native, questions
from fclass.extract import _png, preview_file
from fclass.plan import Item

mac = pytest.mark.skipif(platform.system() != "Darwin", reason="needs macOS")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


# ── download origin ─────────────────────────────────────────────────────────

def test_describe_origin_keeps_host_and_path_drops_secrets():
    urls = ["https://netbanking.hdfcbank.com/statements/download.pdf?session=SECRET&t=1",
            "https://www.hdfcbank.com/personal/"]
    assert native.describe_origin(urls) == "netbanking.hdfcbank.com/statements/download.pdf (via hdfcbank.com/personal)"
    assert native.describe_origin(["blob:https://x", "data:,abc"]) == ""
    assert native.describe_origin(["https://a.com/x", "https://a.com/y"]) == "a.com/x"


def test_mac_where_froms_plist_is_parsed(tmp_path, monkeypatch):
    f = tmp_path / "download.pdf"
    f.write_text("x")
    blob = plistlib.dumps(["https://eportal.incometax.gov.in/form16.pdf", "https://eportal.incometax.gov.in/"],
                          fmt=plistlib.FMT_BINARY)
    monkeypatch.setattr(native, "IS_MAC", True)
    monkeypatch.setattr(native, "_mac_getxattr", lambda p, n: blob if n == native.WHERE_FROMS else None)
    assert native.download_origin(f)[0] == "https://eportal.incometax.gov.in/form16.pdf"
    assert "Downloaded from: eportal.incometax.gov.in/form16.pdf" in preview_file(f).text


def test_linux_xdg_origin(tmp_path):
    f = tmp_path / "x.pdf"
    f.write_text("x")
    if native.IS_MAC or not hasattr(os, "setxattr"):
        pytest.skip("Linux xattrs only")
    try:
        os.setxattr(f, native.XDG_ORIGIN, b"https://www.airindia.com/boarding-pass.pdf?pnr=Q7XK2P")
    except OSError:
        pytest.skip("filesystem without user xattrs")
    assert native.download_origin(f) == ["https://www.airindia.com/boarding-pass.pdf?pnr=Q7XK2P"]
    assert "Downloaded from: airindia.com/boarding-pass.pdf." in preview_file(f).text


@mac
def test_real_mac_where_froms(tmp_path):
    f = tmp_path / "download.pdf"
    f.write_text("x")
    blob = plistlib.dumps(["https://www.irs.gov/pub/irs-pdf/fw2.pdf"], fmt=plistlib.FMT_BINARY)
    subprocess.run(["xattr", "-wx", native.WHERE_FROMS, blob.hex(), str(f)], check=True)
    assert native.download_origin(f) == ["https://www.irs.gov/pub/irs-pdf/fw2.pdf"]
    assert "irs.gov/pub/irs-pdf/fw2.pdf" in preview_file(f).text


# ── pictures through sips ───────────────────────────────────────────────────

def fake_sips(tmp_path):
    """A stand-in for macOS sips: writes a tiny JPEG to the --out path."""
    script = tmp_path / "sips"
    script.write_text(f"#!{sys.executable}\nimport sys\nout = sys.argv[sys.argv.index('--out') + 1]\n"
                      "open(out, 'wb').write(b'\\xff\\xd8\\xff\\xe0fake-jpeg\\xff\\xd9')\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_heic_becomes_jpeg_when_sips_is_available(tmp_path, monkeypatch):
    heic = tmp_path / "IMG_0001.HEIC"
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\0" * 64)
    sips = fake_sips(tmp_path)
    monkeypatch.setattr(native, "can_convert", lambda: True)
    real = native.sips_jpeg
    monkeypatch.setattr(native, "sips_jpeg", lambda p, max_side=1600: real(p, max_side, sips=sips))
    p = preview_file(heic, want_image=True)
    assert p.kind == "HEIC photo" and p.image.startswith(b"\xff\xd8") and p.readable


def test_heic_without_sips_is_metadata_and_a_question(tmp_path, monkeypatch):
    heic = tmp_path / "IMG_0001.HEIC"
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\0" * 64)
    monkeypatch.setattr(native, "can_convert", lambda: False)
    p = preview_file(heic, want_image=True)
    assert p.image is None and not p.readable and p.has_picture


@mac
def test_real_sips_reads_an_iphone_photo(tmp_path):
    png = tmp_path / "photo.png"
    png.write_bytes(_png(64, 48, bytes([90, 140, 200] * 64 * 48), 3))
    heic = tmp_path / "IMG_0001.HEIC"
    made = subprocess.run(["sips", "-s", "format", "heic", str(png), "--out", str(heic)], capture_output=True)
    if made.returncode != 0 or not heic.exists():
        pytest.skip("this macOS cannot write HEIC")
    p = preview_file(heic, want_image=True)
    assert p.kind == "HEIC photo" and p.image[:3] == b"\xff\xd8\xff"


@mac
def test_real_sips_shrinks_big_pictures(tmp_path):
    big = tmp_path / "Screenshot 2025-06-14.png"
    big.write_bytes(_png(2400, 40, bytes([255] * 2400 * 40 * 3), 3))
    p = preview_file(big, want_image=True)
    assert p.image[:3] == b"\xff\xd8\xff"  # converted to a smaller JPEG


# ── native dialogs ──────────────────────────────────────────────────────────

CATS = ["Finance/Taxes", "Keep/Important", "Recreation/Travel", "Study/Notes"]


def item():
    return Item("/x/Form \"16\".pdf", "file", "Finance/Taxes", 0.3, "a TDS certificate", "llm",
                ["Keep/Important"], True, "", "PDF, 1 page")


def runner(*replies):
    seen, it = [], iter(replies)

    def run(script, timeout=None):
        seen.append(script)
        return next(it)
    return run, seen


def test_dialog_options_put_guesses_first():
    opts = questions.dialog_options(item(), CATS)
    assert opts[0] == "Finance/Taxes   (best guess)" and opts[1] == "Keep/Important"
    assert opts[-2:] == [questions.NEW, questions.LEAVE] and "Study/Notes" in opts


def test_dialog_answers():
    run, seen = runner("Finance/Taxes   (best guess)")
    assert questions.ask_dialog(item(), CATS, run).category == "Finance/Taxes"
    assert '\\"16\\"' in seen[0]  # quotes in file names are escaped for AppleScript
    run, _ = runner("Study/Notes")
    assert questions.ask_dialog(item(), CATS, run).category == "Study/Notes"
    run, _ = runner(questions.SEPARATOR, questions.LEAVE)
    assert questions.ask_dialog(item(), CATS, run).action == "leave"
    run, _ = runner("false")
    assert questions.ask_dialog(item(), CATS, run).action == "quit"      # "Not now"
    run, _ = runner(None)
    assert questions.ask_dialog(item(), CATS, run).action == "quit"      # timed out
    run, _ = runner(questions.NEW, "Work/Payslips", "Salary slips")
    a = questions.ask_dialog(item(), CATS, run)
    assert (a.action, a.category, a.description) == ("new", "Work/Payslips", "Salary slips")


@mac
def test_real_applescript_compiles(tmp_path):
    for script in (questions.choose_script(item(), questions.dialog_options(item(), CATS)),
                   questions.text_script('Folder "path":')):
        out = subprocess.run(["osacompile", "-o", str(tmp_path / "q.scpt"), "-e", script], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
