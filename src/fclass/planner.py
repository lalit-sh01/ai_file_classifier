"""Scan a folder and turn it into a Plan."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .classify import Classifier
from .config import Config
from .extract import preview_file, preview_folder
from .plan import Item, Plan


def scan(cfg: Config, source: Path) -> list[Path]:
    """Top-level entries worth looking at, files first."""
    source = source.expanduser().resolve()
    dest = cfg.destination.expanduser().resolve()
    owned = cfg.top_level_names if source == dest else set()
    entries = []
    for p in sorted(source.iterdir(), key=lambda p: p.name.lower()):
        if cfg.is_ignored(p.name) or p.name in owned or p.is_symlink():
            continue
        if p.is_dir() and not cfg.folders_as_units:
            continue
        entries.append(p)
    return sorted(entries, key=lambda p: p.is_dir())


def build_plan(cfg: Config, source: Path, classifier: Classifier,
               progress: Callable[[int, int, Item], None] | None = None) -> Plan:
    entries = scan(cfg, source)
    items: list[Item] = []
    for i, path in enumerate(entries, 1):
        items.append(classify_one(cfg, classifier, path))
        if progress:
            progress(i, len(entries), items[-1])
        if i % 10 == 0:
            classifier.save()
    classifier.save()
    return Plan(
        source=str(source.expanduser().resolve()),
        destination=str(cfg.destination.expanduser().resolve()),
        review_folder=cfg.review_folder,
        model=_model_label(cfg),
        items=items,
    )


def classify_one(cfg: Config, classifier: Classifier, path: Path) -> Item:
    kind = "folder" if path.is_dir() else "file"
    rule = cfg.match_rule(path.name) if kind == "file" else None
    if rule:
        classifier.stats["rule"] += 1
        if rule.action == "skip":
            return Item(str(path), kind, None, 1.0, f"rule: {', '.join(rule.patterns[:3])}", "skip")
        return Item(str(path), kind, rule.category, 1.0, f"rule: {', '.join(rule.patterns[:3])}", "rule")

    preview = (preview_folder(path, cfg.max_preview_chars) if kind == "folder"
               else preview_file(path, cfg.max_preview_chars, want_image=cfg.model.vision))
    try:
        v = classifier.classify(path.name, kind, preview)
    except Exception as e:  # one bad file must not sink the run
        return Item(str(path), kind, cfg.category_paths[0], 0.0, f"error: {e}", "error",
                    review=True, snippet=preview.text[:400])
    category = v.category or (v.alternatives[0] if v.alternatives else cfg.category_paths[0])
    review = v.category is None or v.confidence < cfg.min_confidence
    return Item(str(path), kind, category, round(v.confidence, 2), v.reason, v.via,
                v.alternatives, review, preview.text[:400])


def _model_label(cfg: Config) -> str:
    s = cfg.strategy
    if s.mode == "embed":
        return f"{s.embed_model} (embed)"
    if s.mode == "llm":
        return f"{cfg.model.name} ({cfg.model.backend})"
    return f"{s.embed_model} + {cfg.model.name} ({cfg.model.backend})"
