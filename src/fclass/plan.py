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
    review: bool = False          # routed to the review folder
    snippet: str = ""             # short excerpt, used when the user teaches a correction

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

    def target(self, item: Item) -> Path | None:
        if item.category is None:
            return None
        folder = self.review_folder if item.review else item.category
        return Path(self.destination) / folder / item.name

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


def apply(plan: Plan, on_move=None) -> tuple[Path, int, list[str]]:
    """Execute a plan. Returns (journal path, moved count, error messages)."""
    journal = journal_dir() / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    moved, errors = 0, []
    with open(journal, "a", encoding="utf-8") as log:
        for item in plan.items:
            target = plan.target(item)
            if target is None:
                continue
            src = Path(item.src)
            if not src.exists():
                errors.append(f"{item.name}: no longer exists")
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                dst = _free_path(target)
                shutil.move(str(src), str(dst))
                log.write(json.dumps({"src": str(src), "dst": str(dst)}) + "\n")
                log.flush()
                moved += 1
                if on_move:
                    on_move(item, dst)
            except OSError as e:
                errors.append(f"{item.name}: {e}")
    if moved == 0:
        journal.unlink(missing_ok=True)
    return journal, moved, errors


def latest_journal() -> Path | None:
    runs = sorted(journal_dir().glob("run-*.jsonl"))
    return runs[-1] if runs else None


def undo(journal: Path) -> tuple[int, list[str]]:
    """Reverse a run, newest move first. Skips anything that changed since."""
    entries = [json.loads(l) for l in journal.read_text(encoding="utf-8").splitlines() if l.strip()]
    restored, problems = 0, []
    for e in reversed(entries):
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
    journal.rename(journal.with_suffix(".undone"))
    return restored, problems
