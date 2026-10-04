"""Configuration: the taxonomy, routing rules and model settings live in one TOML file."""

from __future__ import annotations

import fnmatch
import hashlib
import ipaddress
import json
import os
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_CONFIG_TOML = """\
# fclass configuration
# Everything runs on this computer. Edit freely: categories, descriptions and
# rules are all yours. `fclass categories add/remove` and `fclass discover`
# edit this file for you.

[model]
# backend: "ollama" (default) or "openai" for any OpenAI-compatible server on
# this machine (LM Studio, llama.cpp, vLLM, Jan, mlx-lm, ...).
backend = "ollama"
name = "qwen3:4b"
url = "http://localhost:11434"
# Photos, screenshots and scanned PDFs are read by a vision model.
# "auto" uses vision_model when it is installed; true requires it; false never.
vision = "auto"
vision_model = "gemma3:4b"
# fclass refuses model servers that are not on this computer or your local
# network, so your files never leave your devices. Set true to override.
allow_remote = false
temperature = 0.0
timeout = 180

[strategy]
# "hybrid": a small embedding model decides clear cases in ~0.1 s and hands
#           close calls to the LLM (recommended)
# "llm":    the LLM reads every file (slower, no embedding model needed)
# "embed":  embeddings only (fastest, least accurate on subtle cases)
mode = "hybrid"
embed_model = "embeddinggemma"
# How far ahead the best category must be for embeddings to decide alone.
margin = 0.04
# Close calls: how many top candidates the LLM sees (0 = all, ordered by
# likelihood). Benchmarked: all = 91.7%, top-3 = 86.7%.
shortlist = 0

[organize]
# Where category folders are created.
destination = "~"
# When fclass is not sure where something goes:
#   "ask"           ask you (now, or later via `fclass ask`); the file stays put
#   "review_folder" move it into review_folder for you to look at
#   "leave"         leave it where it is without asking
when_unsure = "ask"
review_folder = "_Review"
min_confidence = 0.55
# Treat top-level folders as single units (classified and moved whole).
folders_as_units = true
max_preview_chars = 2000
# Never touched: hidden files, Office lock files, unfinished downloads.
ignore = [".*", "~$*", "*.part", "*.partial", "*.crdownload", "*.download", "*.opdownload", "*.tmp",
          "desktop.ini", "Thumbs.db"]

[watch]
# `fclass watch` sorts new arrivals in these folders.
folders = ["~/Downloads"]
# Seconds between checks, and how long a file must stay unchanged before it
# counts as fully downloaded.
interval = 5
settle_seconds = 8
# Desktop notification when fclass has a question for you.
notify = true
# How watch asks when it is unsure: "notification" (answer later with
# `fclass ask`) or "dialog" (macOS: a native picker right away; unanswered
# within two minutes, it becomes a saved question).
ask_with = "notification"

# ── Categories ───────────────────────────────────────────────────────────────
# path:        folder path under destination (use "/" for nesting)
# description: what belongs here, in plain words. This is what the model reads.

[[category]]
path = "Finance/Statements"
description = "Bank statements, credit card bills, utility bills, invoices, payment receipts"

[[category]]
path = "Finance/Taxes"
description = "Tax returns, W-2, 1099, Form 16, tax receipts and deduction proofs"

[[category]]
path = "Finance/Investments"
description = "Brokerage and portfolio statements, stock trades, retirement and pension accounts"

[[category]]
path = "Study/Courses"
description = "Syllabi, assignments, course materials, exam papers, certificates of completion"

[[category]]
path = "Study/Notes"
description = "Personal notes, lecture notes, highlights, study guides, journals"

[[category]]
path = "Study/References"
description = "Research papers, technical documentation, textbooks, reference material"

[[category]]
path = "Recreation/Hobbies"
description = "Hobby projects, recipes, patterns, collections, DIY guides"

[[category]]
path = "Recreation/Entertainment"
description = "Ebooks, articles, saved web content, games, movies, music and playlists"

[[category]]
path = "Recreation/Travel"
description = "Itineraries, flight and hotel bookings, tickets, visas, trip plans and trip photos"

[[category]]
path = "Keep/Important"
description = "IDs, passports, birth certificates, wills, contracts, leases, insurance policies, legal documents. These go here even when they mention money."

[[category]]
path = "Keep/Archives"
description = "Old or completed projects and historical records worth preserving"

[[category]]
path = "Keep/Manuals"
description = "Product manuals, warranties, setup and instruction guides"

# ── Rules ────────────────────────────────────────────────────────────────────
# Optional. Rules run before any model, are free and instant. First match wins.
# match:    filename glob (case-insensitive)
# category: send matches straight to this category, or
# action:   "skip" to leave matches untouched.
#
# [[rule]]
# match = ["*.dmg", "*.pkg", "*.exe", "*.msi"]
# action = "skip"
#
# [[rule]]
# match = ["*.epub", "*.mobi"]
# category = "Recreation/Entertainment"
"""

WHEN_UNSURE = ("ask", "review_folder", "leave")


@dataclass
class Category:
    path: str
    description: str


@dataclass
class Rule:
    patterns: list[str]
    category: str | None = None
    action: str = "move"  # "move" | "skip"

    def matches(self, name: str) -> bool:
        lowered = name.lower()
        return any(fnmatch.fnmatchcase(lowered, p.lower()) for p in self.patterns)


@dataclass
class ModelSettings:
    backend: str = "ollama"
    name: str = "qwen3:4b"
    url: str = "http://localhost:11434"
    vision: str = "auto"           # "auto" | "on" | "off"
    vision_model: str = "gemma3:4b"
    allow_remote: bool = False
    temperature: float = 0.0
    timeout: float = 180
    api_key: str | None = None


@dataclass
class Strategy:
    mode: str = "hybrid"          # "hybrid" | "llm" | "embed"
    embed_model: str = "embeddinggemma"
    margin: float = 0.04
    shortlist: int = 0


@dataclass
class WatchSettings:
    folders: list[Path] = field(default_factory=lambda: [Path("~/Downloads").expanduser()])
    interval: float = 5
    settle_seconds: float = 8
    notify: bool = True
    ask_with: str = "notification"  # "notification" | "dialog"


@dataclass
class Config:
    model: ModelSettings
    strategy: Strategy
    categories: list[Category]
    rules: list[Rule]
    destination: Path
    when_unsure: str = "ask"
    review_folder: str = "_Review"
    min_confidence: float = 0.55
    folders_as_units: bool = True
    max_preview_chars: int = 2000
    ignore: list[str] = field(default_factory=list)
    watch: WatchSettings = field(default_factory=WatchSettings)
    source_path: Path | None = None

    @property
    def category_paths(self) -> list[str]:
        return [c.path for c in self.categories]

    @property
    def top_level_names(self) -> set[str]:
        """Top-level folder names the organizer owns (never treated as inputs)."""
        names = {c.path.split("/")[0] for c in self.categories}
        names.add(self.review_folder)
        for r in self.rules:
            if r.category:
                names.add(r.category.split("/")[0])
        return names

    def is_ignored(self, name: str) -> bool:
        lowered = name.lower()
        return any(fnmatch.fnmatchcase(lowered, p.lower()) for p in self.ignore)

    def match_rule(self, name: str) -> Rule | None:
        return next((r for r in self.rules if r.matches(name)), None)

    def fingerprint(self) -> str:
        """Changes whenever anything that affects classification changes."""
        from . import __version__

        payload = json.dumps(
            {
                "version": __version__,
                "model": [self.model.backend, self.model.name, self.model.vision, self.model.vision_model],
                "strategy": [self.strategy.mode, self.strategy.embed_model, self.strategy.margin,
                             self.strategy.shortlist],
                "categories": [[c.path, c.description] for c in self.categories],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── locations ───────────────────────────────────────────────────────────────

def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "fclass"


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    path = Path(base) / "fclass"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_config_path() -> Path:
    env = os.environ.get("FCLASS_CONFIG")
    return Path(env).expanduser() if env else config_dir() / "config.toml"


def write_default_config(path: Path, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return True


# ── offline guard ───────────────────────────────────────────────────────────

def locality(url: str) -> str:
    """'device' (this computer), 'network' (your LAN) or 'remote' (the internet)."""
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "") or host.endswith(".localhost"):
        return "device"
    if host.endswith(".local") or host.endswith(".lan") or host.endswith(".home.arpa"):
        return "network"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "remote"
    if ip.is_loopback:
        return "device"
    if ip.is_private or ip.is_link_local:
        return "network"
    return "remote"


class OfflineError(ValueError):
    pass


def check_offline(settings: ModelSettings) -> None:
    if not settings.allow_remote and locality(settings.url) == "remote":
        raise OfflineError(
            f"{settings.url} is not on this computer or your local network, so files would leave "
            "your devices. Point [model] url at a local server, or set allow_remote = true to override."
        )


# ── loading ─────────────────────────────────────────────────────────────────

def load_config(path: Path | None = None) -> Config:
    """Load the user's config, falling back to built-in defaults."""
    path = path or default_config_path()
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        source = path
    else:
        data = tomllib.loads(DEFAULT_CONFIG_TOML)
        source = None
    return parse_config(data, source)


def _vision_mode(value) -> str:
    if value is True or str(value).lower() in ("true", "on", "yes"):
        return "on"
    if value is False or str(value).lower() in ("false", "off", "no"):
        return "off"
    return "auto"


def parse_config(data: dict, source: Path | None = None) -> Config:
    m = data.get("model", {})
    backend = m.get("backend", "ollama")
    if backend not in ("ollama", "openai"):
        raise ValueError(f"[model] backend must be 'ollama' or 'openai', got {backend!r}")
    model = ModelSettings(
        backend=backend,
        name=m.get("name", "qwen3:4b"),
        url=m.get("url", "http://localhost:11434" if backend == "ollama" else "http://localhost:1234/v1"),
        vision=_vision_mode(m.get("vision", "auto")),
        vision_model=m.get("vision_model", "gemma3:4b"),
        allow_remote=bool(m.get("allow_remote", False)),
        temperature=float(m.get("temperature", 0.0)),
        timeout=float(m.get("timeout", 180)),
        api_key=m.get("api_key"),
    )

    st = data.get("strategy", {})
    strategy = Strategy(
        mode=st.get("mode", "hybrid"),
        embed_model=st.get("embed_model", "embeddinggemma"),
        margin=float(st.get("margin", 0.04)),
        shortlist=int(st.get("shortlist", 0)),
    )
    if strategy.shortlist == 1 or strategy.shortlist < 0:
        raise ValueError("strategy.shortlist must be 0 (all) or at least 2")
    if strategy.mode not in ("hybrid", "llm", "embed"):
        raise ValueError(f"strategy.mode must be hybrid, llm or embed, got {strategy.mode!r}")

    categories = [Category(c["path"].strip("/"), c.get("description", "")) for c in data.get("category", [])]
    if not categories:
        raise ValueError("Config defines no [[category]] entries")
    seen = set()
    for c in categories:
        if c.path in seen:
            raise ValueError(f"Duplicate category: {c.path}")
        seen.add(c.path)

    rules = []
    for r in data.get("rule", []):
        patterns = r["match"] if isinstance(r["match"], list) else [r["match"]]
        action = r.get("action", "move")
        category = r.get("category")
        if action not in ("move", "skip"):
            raise ValueError(f"Rule action must be 'move' or 'skip', got {action!r}")
        if action == "move" and not category:
            raise ValueError(f"Rule {patterns} needs a category or action = 'skip'")
        rules.append(Rule(patterns, category.strip("/") if category else None, action))

    o = data.get("organize", {})
    when_unsure = o.get("when_unsure", "ask")
    if when_unsure not in WHEN_UNSURE:
        raise ValueError(f"organize.when_unsure must be one of {', '.join(WHEN_UNSURE)}")

    w = data.get("watch", {})
    watch = WatchSettings(
        folders=[Path(f).expanduser() for f in w.get("folders", ["~/Downloads"])],
        interval=float(w.get("interval", 5)),
        settle_seconds=float(w.get("settle_seconds", 8)),
        notify=bool(w.get("notify", True)),
        ask_with=w.get("ask_with", "notification"),
    )
    if watch.ask_with not in ("notification", "dialog"):
        raise ValueError('[watch] ask_with must be "notification" or "dialog"')

    return Config(
        model=model,
        strategy=strategy,
        categories=categories,
        rules=rules,
        destination=Path(o.get("destination", "~")).expanduser(),
        when_unsure=when_unsure,
        review_folder=o.get("review_folder", "_Review"),
        min_confidence=float(o.get("min_confidence", 0.55)),
        folders_as_units=bool(o.get("folders_as_units", True)),
        max_preview_chars=int(o.get("max_preview_chars", 2000)),
        ignore=list(o.get("ignore", [])),
        watch=watch,
        source_path=source,
    )


# ── editing the category list in place ──────────────────────────────────────
# The file is the user's: comments and layout are kept. Only [[category]]
# blocks are added or removed, and every write is re-parsed before it lands.

_PATH_RE = re.compile(r"^[^/\\\x00]+(/[^/\\\x00]+)*$")
# A category block: its header, its `key = value` lines, then any blank lines.
# Comments that follow (e.g. the Rules section heading) are not part of it.
_BLOCK_RE = re.compile(r"^\[\[category\]\][ \t]*\n(?:[ \t]*[A-Za-z0-9_\"-]+[ \t]*=.*(?:\n|$))*(?:[ \t]*\n)*", re.M)


def valid_category_path(path: str) -> bool:
    path = path.strip().strip("/")
    return bool(path) and bool(_PATH_RE.match(path)) and ".." not in path.split("/")


def _block(path: str, description: str) -> str:
    return f"[[category]]\npath = {json.dumps(path)}\ndescription = {json.dumps(description, ensure_ascii=False)}\n\n"


def _editable(path: Path | None) -> Path:
    path = path or default_config_path()
    write_default_config(path)  # materialise defaults so the user owns them
    return path


def _commit(path: Path, text: str) -> None:
    parse_config(tomllib.loads(text), path)  # refuse to write a config that won't load
    shutil.copy2(path, path.with_suffix(".toml.bak"))
    path.write_text(text, encoding="utf-8")


def _insert_after_categories(text: str, blocks: str) -> str:
    matches = list(_BLOCK_RE.finditer(text))
    at = matches[-1].end() if matches else len(text)
    if at and not text[:at].endswith("\n\n"):
        blocks = "\n" + blocks
    return text[:at] + blocks + text[at:]


def add_categories(new: list[Category], path: Path | None = None) -> Path:
    path = _editable(path)
    text = path.read_text(encoding="utf-8")
    existing = {c.path for c in parse_config(tomllib.loads(text)).categories}
    blocks = ""
    for c in new:
        p = c.path.strip().strip("/")
        if not valid_category_path(p):
            raise ValueError(f"Not a usable folder path: {c.path!r}")
        if p in existing:
            raise ValueError(f"Category already exists: {p}")
        existing.add(p)
        blocks += _block(p, c.description.strip())
    _commit(path, _insert_after_categories(text, blocks))
    return path


def remove_category(cat_path: str, path: Path | None = None) -> Path:
    path = _editable(path)
    text = path.read_text(encoding="utf-8")
    target = cat_path.strip().strip("/")
    out, found = text, False
    for m in reversed(list(_BLOCK_RE.finditer(text))):
        data = tomllib.loads(m.group(0).replace("[[category]]", "", 1))
        if data.get("path", "").strip("/") == target:
            out, found = out[:m.start()] + out[m.end():], True
    if not found:
        raise ValueError(f"No category {target!r}")
    _commit(path, out)
    return path


def replace_categories(new: list[Category], path: Path | None = None) -> Path:
    path = _editable(path)
    text = path.read_text(encoding="utf-8")
    matches = list(_BLOCK_RE.finditer(text))
    first = matches[0].start() if matches else len(text)
    stripped = _BLOCK_RE.sub("", text)
    blocks = "".join(_block(c.path.strip().strip("/"), c.description.strip()) for c in new)
    out = stripped[:first] + blocks + stripped[first:]
    _commit(path, out)
    return path
