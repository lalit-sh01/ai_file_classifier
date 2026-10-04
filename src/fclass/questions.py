"""When fclass is not sure, it asks you instead of guessing.

Questions are asked right away when you are at the terminal (`fclass sort`),
or queued when you are not (`fclass watch`, `sort -y`), and answered later
with `fclass ask`. Until you answer, the file stays exactly where it is.
Every answer is remembered as an example, so the same doubt comes up less.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import state_dir, valid_category_path
from .plan import Item

# ── the queue ───────────────────────────────────────────────────────────────


def _queue_path() -> Path:
    return state_dir() / "questions.json"


def load_queue() -> list[dict]:
    try:
        return json.loads(_queue_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(queue: list[dict]) -> None:
    _queue_path().write_text(json.dumps(queue, indent=1, ensure_ascii=False), encoding="utf-8")


def enqueue(items: list[Item], destination: str | Path) -> int:
    """Add unsure items to the queue (once each), remembering where they were going. Returns how many are new."""
    queue = load_queue()
    known = {q["src"] for q in queue}
    added = 0
    for it in items:
        if it.src in known:
            continue
        queue.append({"src": it.src, "kind": it.kind, "category": it.category, "alternatives": it.alternatives,
                      "reason": it.reason, "snippet": it.snippet, "about": it.about, "dest": str(destination),
                      "asked": time.time()})
        known.add(it.src)
        added += 1
    _save(queue)
    return added


def pending() -> list[Item]:
    """Queued questions whose file is still where it was; stale ones are dropped."""
    queue = [q for q in load_queue() if Path(q["src"]).exists()]
    _save(queue)
    return [Item(q["src"], q["kind"], q["category"], 0.0, q.get("reason", ""), "queued",
                 q.get("alternatives", []), True, q.get("snippet", ""), q.get("about", ""), q.get("dest", ""))
            for q in queue]


def resolve(src: str) -> None:
    _save([q for q in load_queue() if q["src"] != src])


# ── asking ──────────────────────────────────────────────────────────────────

@dataclass
class Answer:
    action: str                  # "file" | "new" | "leave" | "quit"
    category: str | None = None
    description: str = ""


def guesses(item: Item, categories: list[str]) -> list[str]:
    out = [c for c in [item.category, *item.alternatives] if c in categories]
    return list(dict.fromkeys(out))[:3] or categories[:3]


def ask(item: Item, categories: list[str], ask_fn: Callable[[str], str] = input,
        say: Callable[[str], None] = print, style=lambda s, *_: s) -> Answer:
    """One question about one file. `style(text, role)` lets the CLI add colour."""
    options = guesses(item, categories)
    name = item.name + ("/" if item.kind == "folder" else "")
    say("")
    say(f"  {style('?', 'q')} {style(name, 'b')}  {style('· ' + item.about, 'd') if item.about else ''}")
    hint = item.reason if item.reason and not item.reason.startswith(("closest match", "error")) else ""
    hint = hint or _first_line(item.snippet)
    if hint:
        say(f"    {style(_clip(hint, 90), 'd')}")
    showing_all = False
    while True:
        for i, c in enumerate(options, 1):
            tag = style("  ← best guess", "d") if i == 1 and not showing_all else ""
            say(f"    {style(str(i), 'b')}  {c}{tag}")
        extra = "" if showing_all else f"{style('a', 'b')} all categories   "
        say(f"    {extra}{style('n', 'b')} new category   {style('s', 'b')} leave it here   {style('q', 'b')} stop asking")
        raw = ask_fn("  › ").strip().lower()
        if raw == "" and options:
            return Answer("file", options[0])
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return Answer("file", options[int(raw) - 1])
        if raw == "a" and not showing_all:
            options, showing_all = list(categories), True
            continue
        if raw == "s":
            return Answer("leave")
        if raw == "q":
            return Answer("quit")
        if raw == "n":
            new = _new_category(ask_fn, say, style, categories)
            if new:
                return new
            continue
        say(f"    {style('Type a number, or a, n, s or q.', 'd')}")


def _new_category(ask_fn, say, style, categories) -> Answer | None:
    path = ask_fn("    Folder path, e.g. Work/Payslips: ").strip().strip("/")
    if not path:
        return None
    if not valid_category_path(path):
        say(f"    {style('That is not a usable folder path.', 'd')}")
        return None
    if path in categories:
        return Answer("file", path)
    desc = ask_fn("    What belongs there, in a few words: ").strip()
    return Answer("new", path, desc or path.replace("/", " ").lower())


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.startswith(("Type:", "Size:")):
            return line
    return ""


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ── native macOS dialogs (no dependencies: AppleScript via osascript) ───────

SEPARATOR = "──────────"
NEW = "New category…"
LEAVE = "Leave it where it is"


def applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _osascript(script: str, timeout: float | None = None) -> str | None:
    """Run AppleScript; None when cancelled, timed out or unavailable."""
    import subprocess

    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def choose_script(item: Item, options: list[str], prompt_extra: str = "") -> str:
    name = item.name + ("/" if item.kind == "folder" else "")
    hint = _clip(item.reason if item.reason and not item.reason.startswith(("closest match", "error")) else
                 _first_line(item.snippet), 160)
    prompt = f"Where should “{name}” go?" + (f"\n{item.about}" if item.about else "") + (f"\n\n{hint}" if hint else "")
    items = "{" + ", ".join(applescript_string(o) for o in options) + "}"
    return (f"choose from list {items} with title \"fclass\" with prompt {applescript_string(prompt + prompt_extra)} "
            f"default items {{{applescript_string(options[0])}}} OK button name \"Move\" cancel button name \"Not now\"")


def text_script(prompt: str, default: str = "") -> str:
    return (f"text returned of (display dialog {applescript_string(prompt)} default answer {applescript_string(default)} "
            f"with title \"fclass\" buttons {{\"Cancel\", \"OK\"}} default button \"OK\")")


def dialog_options(item: Item, categories: list[str]) -> list[str]:
    best = guesses(item, categories)
    labelled = [f"{best[0]}   (best guess)"] + best[1:]
    rest = [c for c in categories if c not in best]
    return labelled + [SEPARATOR] + rest + [SEPARATOR, NEW, LEAVE]


def ask_dialog(item: Item, categories: list[str], run=_osascript, timeout: float | None = None) -> Answer:
    """The same question as `ask`, as a native macOS picker. Cancel or no answer in time = ask again later."""
    options = dialog_options(item, categories)
    while True:
        picked = run(choose_script(item, options), timeout)
        if picked is None or picked == "false":
            return Answer("quit")
        picked = picked.removesuffix("   (best guess)")
        if picked == SEPARATOR:
            continue
        if picked == LEAVE:
            return Answer("leave")
        if picked == NEW:
            path = (run(text_script("Folder for the new category, e.g. Work/Payslips:"), timeout) or "").strip().strip("/")
            if not path or not valid_category_path(path):
                continue
            if path in categories:
                return Answer("file", path)
            desc = (run(text_script(f"What belongs in {path}? (a few words)"), timeout) or "").strip()
            return Answer("new", path, desc or path.replace("/", " ").lower())
        if picked in categories:
            return Answer("file", picked)
