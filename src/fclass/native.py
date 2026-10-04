"""What the operating system already knows about a file. No dependencies.

- Where a download came from. macOS records the source URL of every download
  in the `com.apple.metadata:kMDItemWhereFroms` attribute (a binary plist);
  Chrome and Firefox on Linux write `user.xdg.origin.url`. A PDF from
  netbanking.hdfcbank.com is a bank document whatever its name is.
- Pictures the vision model can read. macOS ships `sips`, which converts
  HEIC/AVIF photos to JPEG and shrinks large images (faster to read).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

IS_MAC = platform.system() == "Darwin"
WHERE_FROMS = "com.apple.metadata:kMDItemWhereFroms"
XDG_ORIGIN = "user.xdg.origin.url"
XDG_REFERRER = "user.xdg.referrer.url"


# ── download origin ─────────────────────────────────────────────────────────

def _mac_getxattr(path: Path, name: str) -> bytes | None:
    """getxattr(2) through libc: Python's os.getxattr does not exist on macOS."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        fn = libc.getxattr
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
        fn.restype = ctypes.c_ssize_t
        p, n = os.fsencode(path), name.encode()
        size = fn(p, n, None, 0, 0, 0)
        if size <= 0:
            return None
        buf = ctypes.create_string_buffer(size)
        got = fn(p, n, buf, size, 0, 0)
        return buf.raw[:got] if got > 0 else None
    except (OSError, AttributeError, TypeError):
        pass
    if shutil.which("xattr"):  # the built-in tool, as a fallback
        out = subprocess.run(["xattr", "-px", name, str(path)], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return bytes.fromhex("".join(out.stdout.split()))
    return None


def _linux_getxattr(path: Path, name: str) -> bytes | None:
    try:
        return os.getxattr(path, name)
    except (OSError, AttributeError):
        return None


def download_origin(path: Path) -> list[str]:
    """URLs a downloaded file came from (download URL first, then the page it was on). Empty if unknown."""
    urls: list[str] = []
    if IS_MAC:
        raw = _mac_getxattr(path, WHERE_FROMS)
        if raw:
            try:
                value = plistlib.loads(raw)
                urls = [u for u in (value if isinstance(value, list) else [value]) if isinstance(u, str)]
            except Exception:
                urls = []
    elif hasattr(os, "getxattr"):
        for name in (XDG_ORIGIN, XDG_REFERRER):
            raw = _linux_getxattr(path, name)
            if raw:
                urls.append(raw.decode("utf-8", errors="ignore").strip("\x00 "))
    return [u for u in urls if u]


def describe_origin(urls: list[str]) -> str:
    """'netbanking.hdfcbank.com/statements/download' — host and path only.
    Query strings are dropped: they carry tokens and session ids, not meaning."""
    seen, parts = set(), []
    for u in urls:
        s = urlsplit(u)
        if s.scheme not in ("http", "https", "ftp") or not s.hostname:
            continue
        host = s.hostname.removeprefix("www.")
        label = (host + s.path.rstrip("/"))[:100]
        if host not in seen:
            seen.add(host)
            parts.append(label)
    return " (via ".join(parts[:2]) + (")" if len(parts) > 1 else "")


# ── pictures through sips ───────────────────────────────────────────────────

def can_convert() -> bool:
    return IS_MAC and shutil.which("sips") is not None


def sips_jpeg(path: Path, max_side: int = 1600, sips: str = "sips") -> bytes | None:
    """A JPEG of any picture macOS can open (HEIC, AVIF, TIFF, large PNGs…), at most max_side pixels."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "picture.jpg"
        try:
            subprocess.run([sips, "-s", "format", "jpeg", "-s", "formatOptions", "80", "-Z", str(max_side),
                            str(path), "--out", str(out)], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.read_bytes() if out.exists() and out.stat().st_size else None
