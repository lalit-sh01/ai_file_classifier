"""Plans, applying them, and undoing them.

Nothing moves without a plan. Every move is appended to a journal *before*
the next one starts, so `fclass undo` works even after a crash mid-run.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import state_dir


@dataclass
class Item:
    src: str
    kind: str                     # "file" | "folder"
    category: str | None          # chosen category path, None when skipped
    confidence: float = 0.0
    reason: str = ""
    via: str = ""                 # "rule" | "embed" | "llm" | "cache" | "user" | "skip"
    alternatives: list[str] = field(default_factory=list)
    review: bool = False          # not sure: ask, use the review folder, or leave (see when_unsure)
    snippet: str = ""             # short excerpt, used when the user teaches a correction
    about: str = ""               # detected file type, e.g. "scanned PDF without a text layer, 1 page"
    dest: str = ""                # for saved questions: the destination the question was asked for

    @property
    def name(self) -> str:
        return Path(self.src).name


@dataclass
class Plan:
    source: str
    destination: str
    review_folder: str
    model: str
    items: list[Item]
    created: float = field(default_factory=time.time)
    when_unsure: str = "ask"

    def target(self, item: Item) -> Path | None:
        """Where the item goes, or None if it stays where it is."""
        if item.category is None:
            return None
        if item.review:
            if self.when_unsure != "review_folder":
                return None  # unsure items are asked about, never filed silently
            return Path(self.destination) / self.review_folder / item.name
        return Path(self.destination) / item.category / item.name

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        data = json.loads(text)
        data["items"] = [Item(**i) for i in data["items"]]
        return cls(**data)

    def save(self, path: Path | None = None) -> Path:
        path = path or plans_dir() / f"plan-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def plans_dir() -> Path:
    p = state_dir() / "plans"
    p.mkdir(parents=True, exist_ok=True)
    return p


def journal_dir() -> Path:
    p = state_dir() / "journal"
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_plan() -> Path | None:
    plans = sorted(plans_dir().glob("plan-*.json"))
    return plans[-1] if plans else None


def _free_path(path: Path) -> Path:
    """Never overwrite: report.pdf -> report (2).pdf -> report (3).pdf ..."""
    if not path.exists():
        return path
    stem, suffix = (path.name, "") if path.is_dir() else (path.stem, path.suffix)
    n = 2
    while True:
        candidate = path.with_name(f"{stem} ({n}){suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def new_journal(prefix: str = "run") -> Path:
    return journal_dir() / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def move(src: Path, target: Path, journal: Path) -> Path:
    """Move one file or folder without overwriting, and record it before returning."""
    target.parent.mkdir(parents=True, exist_ok=True)
    dst = _free_path(target)
    shutil.move(str(src), str(dst))
    with open(journal, "a", encoding="utf-8") as log:
        log.write(json.dumps({"src": str(src), "dst": str(dst), "at": time.time()}) + "\n")
        log.flush()
    return dst


def apply(plan: Plan, on_move=None, journal: Path | None = None) -> tuple[Path, int, list[str]]:
    """Execute a plan. Returns (journal path, moved count, error messages)."""
    journal = journal or new_journal()
    moved, errors = 0, []
    for item in plan.items:
        target = plan.target(item)
        if target is None:
            continue
        src = Path(item.src)
        if not src.exists():
            errors.append(f"{item.name}: no longer exists")
            continue
        try:
            dst = move(src, target, journal)
            moved += 1
            if on_move:
                on_move(item, dst)
        except OSError as e:
            errors.append(f"{item.name}: {e}")
    return journal, moved, errors


def latest_journal() -> Path | None:
    """The journal with the most recent move (a sort run, `fclass ask` or `fclass watch`)."""
    runs = [p for p in journal_dir().glob("*.jsonl") if p.stat().st_size]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def undo(journal: Path, last: int | None = None) -> tuple[int, list[str]]:
    """Reverse moves, newest first: all of them, or only the `last` N. Skips anything that changed since."""
    entries = [json.loads(l) for l in journal.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep, todo = (entries[:-last], entries[-last:]) if last else ([], entries)
    restored, problems = 0, []
    for e in reversed(todo):
        src, dst = Path(e["src"]), Path(e["dst"])
        if not dst.exists():
            problems.append(f"{dst.name}: no longer at {dst.parent}")
            continue
        if src.exists():
            problems.append(f"{src.name}: something already exists at the original location")
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(src))
        restored += 1
    if keep:
        journal.write_text("".join(json.dumps(e) + "\n" for e in keep), encoding="utf-8")
    else:
        journal.rename(journal.with_suffix(f".undone-{int(time.time())}"))
    return restored, problems
