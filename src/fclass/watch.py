"""`fclass watch`: sort new arrivals as they land.

Polls the watched folders (no extra dependencies, works the same on macOS,
Linux and Windows). A new item is handled only after it has stopped changing
for `settle_seconds`, so half-finished downloads are never touched; browser
partial files (*.crdownload, *.part, *.download) are ignored outright.

Confident items are filed and journaled (`fclass undo --last 1` reverts the
latest). Unsure items stay where they are and become questions for
`fclass ask`, with a desktop notification.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backends import BackendError
from .classify import Classifier
from .config import Config
from .plan import Item, journal_dir, move
from .planner import classify_one, scan
from .native import IS_MAC
from .questions import enqueue


@dataclass
class Seen:
    signature: tuple
    since: float


def signature(path: Path) -> tuple | None:
    """Changes while a file or folder is still being written."""
    try:
        if path.is_dir():
            total, newest, count = 0, 0.0, 0
            for p in path.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    total, newest, count = total + st.st_size, max(newest, st.st_mtime), count + 1
                    if count > 2000:
                        break
            return ("dir", count, total, newest)
        st = path.stat()
        return ("file", st.st_size, st.st_mtime)
    except OSError:
        return None


class Watcher:
    def __init__(self, cfg: Config, classifier: Classifier, folders: list[Path],
                 log: Callable[[str, Item | None, str], None], include_existing: bool = False,
                 notify: Callable[[str], None] | None = None, clock: Callable[[], float] = time.time,
                 ask_now: Callable[[Item], str | None] | None = None):
        self.cfg = cfg
        self.classifier = classifier
        self.folders = [f.expanduser().resolve() for f in folders]
        self.log = log
        self.notify = notify
        self.clock = clock
        self.ask_now = ask_now  # returns a category, "" to leave the file, or None to save the question
        self.seen: dict[Path, Seen] = {}
        self.handled: set[Path] = set()
        if not include_existing:
            for folder in self.folders:
                self.handled.update(scan(cfg, folder))  # only what arrives from now on

    def journal(self) -> Path:
        return journal_dir() / f"watch-{time.strftime('%Y%m%d')}.jsonl"

    def tick(self) -> int:
        """One pass over the watched folders. Returns how many items were handled."""
        now, done = self.clock(), 0
        present: set[Path] = set()
        for folder in self.folders:
            if not folder.is_dir():
                continue
            for entry in scan(self.cfg, folder):
                present.add(entry)
                if entry in self.handled:
                    continue
                sig = signature(entry)
                if sig is None:
                    continue
                prev = self.seen.get(entry)
                if prev is None or prev.signature != sig:
                    self.seen[entry] = Seen(sig, now)  # new or still changing: wait
                    continue
                if now - prev.since < self.cfg.watch.settle_seconds:
                    continue
                self._safely(entry)
                self.handled.add(entry)
                self.seen.pop(entry, None)
                done += 1
        # forget things that went away, so a re-download of the same name is handled again
        self.handled &= present
        for gone in set(self.seen) - present:
            self.seen.pop(gone)
        return done

    def sweep(self) -> int:
        """Handle everything present now, except items changed within settle_seconds (still arriving)."""
        now, done = self.clock(), 0
        for folder in self.folders:
            for entry in scan(self.cfg, folder) if folder.is_dir() else []:
                sig = signature(entry)
                if entry in self.handled or sig is None:
                    continue
                if now - sig[-1] < self.cfg.watch.settle_seconds:
                    self.log("left", Item(str(entry), "folder" if entry.is_dir() else "file", None),
                             "still changing; try again shortly")
                    continue
                self._safely(entry)
                self.handled.add(entry)
                done += 1
        return done

    def _safely(self, entry: Path) -> None:
        """A file that can't be read or moved is reported, never fatal; only a down model server stops watch."""
        try:
            self.handle(entry)
        except BackendError:
            raise
        except OSError as e:
            self.log("left", Item(str(entry), "folder" if entry.is_dir() else "file", None), f"could not move: {e}")

    def handle(self, entry: Path) -> None:
        item = classify_one(self.cfg, self.classifier, entry)
        self.classifier.save()
        if item.category is None:
            self.log("left", item, item.reason)
            return
        dest = self.cfg.destination.expanduser()
        if not item.review:
            dst = move(entry, dest / item.category / entry.name, self.journal())
            self.log("filed", item, str(dst.parent))
            return
        if self.cfg.when_unsure == "review_folder":
            dst = move(entry, dest / self.cfg.review_folder / entry.name, self.journal())
            self.log("review", item, str(dst.parent))
        elif self.cfg.when_unsure == "ask":
            answer = self.ask_now(item) if self.ask_now else None
            if answer:
                dst = move(entry, dest / answer / entry.name, self.journal())
                item.category, item.via = answer, "user"
                self.log("filed", item, str(dst.parent))
                return
            if answer == "":
                self.log("left", item, "you chose to leave it")
                return
            if enqueue([item], dest) and self.notify:
                cmd = "fclass ask --dialog" if IS_MAC else "fclass ask"
                self.notify(f"Not sure where {entry.name} goes. Run `{cmd}` to decide.")
            self.log("asked", item, "")
        else:
            self.log("left", item, "not sure; left in place")

    def run(self, stop: Callable[[], bool] = lambda: False) -> None:
        while not stop():
            self.tick()
            time.sleep(self.cfg.watch.interval)


# ── desktop notifications (best effort, never required) ─────────────────────

def desktop_notify(message: str) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            safe = message.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "fclass"'],
                           capture_output=True, timeout=5)
        elif system == "Linux" and shutil.which("notify-send"):
            subprocess.run(["notify-send", "fclass", message], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


# ── run at login ────────────────────────────────────────────────────────────

def service_file() -> tuple[str, str]:
    """(where to save it, contents) for running `fclass watch` at login."""
    exe = sys.executable
    if platform.system() == "Darwin":
        where = "~/Library/LaunchAgents/dev.fclass.watch.plist  (then: launchctl load -w <that file>)"
        body = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.fclass.watch</string>
  <key>ProgramArguments</key><array><string>{exe}</string><string>-m</string><string>fclass</string><string>watch</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/fclass-watch.log</string>
  <key>StandardErrorPath</key><string>/tmp/fclass-watch.log</string>
</dict></plist>
"""
        return where, body
    where = "~/.config/systemd/user/fclass-watch.service  (then: systemctl --user enable --now fclass-watch)"
    body = f"""[Unit]
Description=fclass: sort new downloads on this computer

[Service]
ExecStart={exe} -m fclass watch
Restart=on-failure

[Install]
WantedBy=default.target
"""
    return where, body
