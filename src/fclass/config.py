"""Configuration: the taxonomy, routing rules and model settings live in one TOML file."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

UNSURE = "Unsure"

DEFAULT_CONFIG_TOML = """\
# fclass configuration
# Edit freely: categories, descriptions and rules are all yours.

[model]
# backend: "ollama" (default), "openai" (any OpenAI-compatible local server:
#          LM Studio, llama.cpp, vLLM, Jan, ...) or "anthropic" (cloud, opt-in)
backend = "ollama"
name = "qwen3:4b"
url = "http://localhost:11434"
# Send images (screenshots, scanned receipts) to the model. Needs a vision
# model such as gemma3:4b or qwen2.5vl:3b.
vision = false
temperature = 0.0
timeout = 180

[strategy]
# "hybrid": a small embedding model decides clear cases in ~0.1 s and hands
#           close calls to the LLM with a shortlist (recommended)
# "llm":    the LLM reads every file (slower, no embedding model needed)
# "embed":  embeddings only (fastest, least accurate on subtle cases)
mode = "hybrid"
embed_model = "embeddinggemma"
# How far ahead the best category must be for embeddings to decide alone.
margin = 0.04
# How many top candidates the LLM chooses between on a close call.
shortlist = 3

[organize]
# Where category folders are created.
destination = "~"
# Low-confidence or "Unsure" items land here for you to look at.
review_folder = "_Review"
min_confidence = 0.55
# Treat top-level folders as single units (classified and moved whole).
folders_as_units = true
max_preview_chars = 2000
ignore = [".*", "~$*", "*.part", "*.crdownload", "*.tmp", "desktop.ini", "Thumbs.db"]

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
description = "Ebooks, articles, saved web content, games, movies and music lists"

[[category]]
path = "Recreation/Travel"
description = "Itineraries, flight and hotel bookings, tickets, visas, trip plans"

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
# Rules run before the model, are free and instant. First match wins.
# match:    filename glob (case-insensitive)
# category: send matches straight to this category, or
# action:   "skip" to leave matches untouched.

[[rule]]
match = ["*.dmg", "*.pkg", "*.exe", "*.msi", "*.deb", "*.rpm", "*.appimage", "*.iso"]
action = "skip"

[[rule]]
match = ["*.zip", "*.tar", "*.gz", "*.tgz", "*.rar", "*.7z"]
action = "skip"

[[rule]]
match = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.mp3", "*.wav", "*.aac", "*.flac", "*.m4a"]
action = "skip"
"""


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
    vision: bool = False
    temperature: float = 0.0
    timeout: float = 180
    api_key: str | None = None


@dataclass
class Strategy:
    mode: str = "hybrid"          # "hybrid" | "llm" | "embed"
    embed_model: str = "embeddinggemma"
    margin: float = 0.04
    shortlist: int = 3


@dataclass
class Config:
    model: ModelSettings
    strategy: Strategy
    categories: list[Category]
    rules: list[Rule]
    destination: Path
    review_folder: str = "_Review"
    min_confidence: float = 0.55
    folders_as_units: bool = True
    max_preview_chars: int = 2000
    ignore: list[str] = field(default_factory=list)
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
        payload = json.dumps(
            {
                "model": [self.model.backend, self.model.name, self.model.vision],
                "strategy": [self.strategy.mode, self.strategy.embed_model, self.strategy.margin,
                             self.strategy.shortlist],
                "categories": [[c.path, c.description] for c in self.categories],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


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


def parse_config(data: dict, source: Path | None = None) -> Config:
    m = data.get("model", {})
    backend = m.get("backend", "ollama")
    default_url = {
        "ollama": "http://localhost:11434",
        "openai": "http://localhost:1234/v1",
        "anthropic": "https://api.anthropic.com",
    }.get(backend, "http://localhost:11434")
    model = ModelSettings(
        backend=backend,
        name=m.get("name", "qwen3:4b"),
        url=m.get("url", default_url),
        vision=bool(m.get("vision", False)),
        temperature=float(m.get("temperature", 0.0)),
        timeout=float(m.get("timeout", 180)),
        api_key=m.get("api_key"),
    )

    st = data.get("strategy", {})
    strategy = Strategy(
        mode=st.get("mode", "hybrid"),
        embed_model=st.get("embed_model", "embeddinggemma"),
        margin=float(st.get("margin", 0.04)),
        shortlist=max(2, int(st.get("shortlist", 3))),
    )
    if strategy.mode not in ("hybrid", "llm", "embed"):
        raise ValueError(f"strategy.mode must be hybrid, llm or embed, got {strategy.mode!r}")
    if backend == "anthropic" and strategy.mode != "llm":
        strategy.mode = "llm"  # no embeddings endpoint

    categories = [Category(c["path"].strip("/"), c.get("description", "")) for c in data.get("category", [])]
    if not categories:
        raise ValueError("Config defines no [[category]] entries")
    seen = set()
    for c in categories:
        if c.path in seen:
            raise ValueError(f"Duplicate category: {c.path}")
        if c.path == UNSURE:
            raise ValueError(f"'{UNSURE}' is reserved")
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
    return Config(
        model=model,
        strategy=strategy,
        categories=categories,
        rules=rules,
        destination=Path(o.get("destination", "~")).expanduser(),
        review_folder=o.get("review_folder", "_Review"),
        min_confidence=float(o.get("min_confidence", 0.55)),
        folders_as_units=bool(o.get("folders_as_units", True)),
        max_preview_chars=int(o.get("max_preview_chars", 2000)),
        ignore=list(o.get("ignore", [])),
        source_path=source,
    )
