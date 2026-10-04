"""The classification pipeline: rules → cache → embeddings → LLM (only when needed).

    ┌───────┐  match   ┌──────────────┐
    │ rules │─────────▶│ done (free)  │
    └───┬───┘          └──────────────┘
        │ no match
    ┌───▼───┐  hit
    │ cache │─────────▶ done (instant)
    └───┬───┘
    ┌───▼──────────────┐ clear winner
    │ embeddings rank  │─────────────▶ done (~0.1 s)
    │ all categories   │
    └───┬──────────────┘
        │ close call
    ┌───▼──────────────┐
    │ LLM picks; all   │─────────────▶ done (seconds)
    │ categories, best │
    │ guesses first    │
    └──────────────────┘

Corrections you make are stored as examples. They feed both stages: as
extra evidence for the embedding ranker (only for closely similar files), and as few-shot lines in the
LLM prompt (the most similar ones only).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .backends import Backend
from .config import Config, state_dir
from .extract import Preview


@dataclass
class Verdict:
    category: str | None
    confidence: float
    reason: str
    via: str
    alternatives: list[str]


# ── persistence ────────────────────────────────────────────────────────────

class JsonStore:
    """A small JSON dict persisted to the state directory."""

    def __init__(self, name: str):
        self.path = state_dir() / name
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
        self.dirty = False

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.dirty = True

    def save(self):
        if self.dirty:
            self.path.write_text(json.dumps(self.data), encoding="utf-8")
            self.dirty = False


def examples_path() -> Path:
    return state_dir() / "examples.jsonl"


def load_examples() -> list[dict]:
    try:
        lines = examples_path().read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(l) for l in lines if l.strip()]


def add_example(name: str, snippet: str, category: str) -> None:
    with open(examples_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "snippet": snippet[:600], "category": category}, ensure_ascii=False) + "\n")


# ── maths ──────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# Cosine above which two documents are very likely the same *type*. Tuned in
# benchmarks/ so that teaching one category never lowers accuracy elsewhere.
SIMILAR = {"embeddinggemma": 0.65, "nomic-embed-text": 0.70}


def similar_threshold(model: str) -> float:
    return next((v for k, v in SIMILAR.items() if k in model.lower()), 0.62)


def score_categories(doc_vec: list[float], desc_vecs: dict[str, list[float]],
                     examples: list[tuple[str, list[float]]], tau: float,
                     weight: float = 0.5) -> list[tuple[str, float]]:
    """Rank categories best-first.

    Base score: cosine to the category description. Your corrections add
    evidence only when a document closely resembles one (cosine above tau),
    so a single example can't tilt unrelated files, and scores stay on the
    plain cosine scale that the margin threshold is tuned for.
    """
    scores = {p: cosine(doc_vec, v) for p, v in desc_vecs.items()}
    boost: dict[str, float] = {}
    for cat, vec in examples:
        if cat in scores:
            boost[cat] = max(boost.get(cat, 0.0), cosine(doc_vec, vec) - tau)
    for cat, b in boost.items():
        if b > 0:
            scores[cat] += weight * b
    return sorted(scores.items(), key=lambda kv: -kv[1])


def embed_prefixes(model: str) -> tuple[str, str]:
    """(document-to-classify prefix, category-description prefix) recommended per model."""
    m = model.lower()
    if "nomic" in m:
        return "classification: ", "classification: "
    if "embeddinggemma" in m:
        return "task: classification | query: ", "title: none | text: "
    return "", ""


# ── classifier ─────────────────────────────────────────────────────────────

SYSTEM = """You sort a person's files into folders. Read the content and pick exactly one category.
Judge by what the document IS, not by words it happens to contain: a lease or insurance policy
that mentions money is still a legal document. Filenames are often meaningless; trust content.

Categories:
{categories}
{examples}"""


class Classifier:
    def __init__(self, cfg: Config, backend: Backend):
        self.cfg = cfg
        self.backend = backend
        self.cache = JsonStore("cache.json")
        self.vectors = JsonStore("vectors.json")
        self.examples = [e for e in load_examples() if e["category"] in cfg.category_paths]
        self.doc_prefix, self.desc_prefix = embed_prefixes(cfg.strategy.embed_model)
        self._desc_vecs: dict[str, list[float]] | None = None
        self._example_vecs: list[list[float]] | None = None
        self.stats = {"rule": 0, "cache": 0, "embed": 0, "llm": 0, "vision": 0}
        self._vision: str | None | bool = False  # False = not resolved yet
        self._warm: set[str] = set()

    def ready_model(self, model: str) -> str:
        """Load a model once, with a long allowance, so per-file timeouts measure reading, not loading."""
        if model not in self._warm:
            self.backend.warm(model)
            self._warm.add(model)
        return model

    @property
    def vision_model(self) -> str | None:
        """The model that reads pictures, or None. "auto" checks once whether it is installed."""
        if self._vision is False:
            m = self.cfg.model
            if m.vision == "off":
                self._vision = None
            elif m.vision == "on":
                self._vision = m.vision_model
            else:
                try:
                    self._vision = m.vision_model if self.backend.has_model(m.vision_model) else None
                except Exception:
                    self._vision = None
        return self._vision

    # -- embeddings with a persistent cache keyed by text
    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self.cfg.strategy.embed_model
        keys = [hashlib.sha256(f"{model}\0{t}".encode()).hexdigest()[:24] for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if self.vectors.get(k) is None]
        for i in range(0, len(missing), 32):
            chunk = missing[i:i + 32]
            for (k, _), v in zip(chunk, self.backend.embed([t for _, t in chunk])):
                self.vectors.put(k, [round(x, 5) for x in v])
        return [self.vectors.get(k) for k in keys]

    def _prepare(self):
        if self._desc_vecs is not None:
            return
        cats = self.cfg.categories
        texts = [f"{self.desc_prefix}{c.path.replace('/', ' / ')}: {c.description}" for c in cats]
        self._desc_vecs = dict(zip([c.path for c in cats], self._embed(texts)))
        self._example_vecs = self._embed([self.doc_prefix + f"{e['name']}\n{e['snippet']}" for e in self.examples])

    def rank(self, doc_vec: list[float]) -> list[tuple[str, float]]:
        """Categories ordered best-first, nudged by your past corrections."""
        self._prepare()
        examples = [(e["category"], v) for e, v in zip(self.examples, self._example_vecs)]
        return score_categories(doc_vec, self._desc_vecs, examples, similar_threshold(self.cfg.strategy.embed_model))

    def taught_match(self, doc_vec: list[float] | None) -> tuple[str, str] | None:
        """(category, example name) of a file you sorted before that closely resembles this one."""
        if doc_vec is None or not self.examples:
            return None
        self._prepare()
        tau = similar_threshold(self.cfg.strategy.embed_model)
        best = max(zip(self.examples, self._example_vecs), key=lambda ev: cosine(doc_vec, ev[1]))
        return (best[0]["category"], best[0]["name"]) if cosine(doc_vec, best[1]) >= tau else None

    def _nearest_examples(self, doc_vec: list[float], n: int = 4) -> str:
        if not self.examples:
            return ""
        self._prepare()
        scored = sorted(zip(self.examples, self._example_vecs), key=lambda ev: -cosine(doc_vec, ev[1]))[:n]
        lines = [f'- "{e["name"]}" ({e["snippet"][:120].replace(chr(10), " ")}…) → {e["category"]}' for e, _ in scored]
        return "\nThe user corrected similar files before. Follow their preferences:\n" + "\n".join(lines)

    def classify(self, name: str, kind: str, preview: Preview) -> Verdict:
        s = self.cfg.strategy
        # 1. cache
        key_src = json.dumps([self.cfg.fingerprint(), len(self.examples), kind, name, preview.text,
                              hashlib.sha256(preview.image).hexdigest() if preview.image else None])
        key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
        if (hit := self.cache.get(key)) is not None:
            self.stats["cache"] += 1
            return Verdict(**{**hit, "via": "cache"})

        verdict = self._decide(name, kind, preview, s)
        self.cache.put(key, {k: v for k, v in verdict.__dict__.items() if k != "via"} | {"via": verdict.via})
        return verdict

    def _decide(self, name, kind, preview, s) -> Verdict:
        doc = f"{'Folder' if kind == 'folder' else 'File'}: {name}\n{preview.text}"
        paths = self.cfg.category_paths

        ranked: list[tuple[str, float]] = []
        doc_vec = None
        taught = None
        if s.mode in ("hybrid", "embed"):
            doc_vec = self._embed([self.doc_prefix + doc])[0]
            ranked = self.rank(doc_vec)
            margin = ranked[0][1] - ranked[1][1]
            taught = self.taught_match(doc_vec)
            clear = margin >= s.margin and preview.readable and preview.image is None
            if s.mode == "embed" or clear:
                self.stats["embed"] += 1
                conf = 0.9 if clear else 0.5
                if not preview.readable and preview.image is None:
                    conf = 0.2
                return Verdict(ranked[0][0], conf, f"closest match (margin {margin:.3f})", "embed",
                               [p for p, _ in ranked[1:4]])

        # LLM stage. Hybrid offers categories ordered by embedding rank (optionally cut to a shortlist).
        choices = [p for p, _ in ranked] if ranked else paths
        if s.shortlist and preview.image is None:
            choices = choices[:s.shortlist]
        system = SYSTEM.format(
            categories="\n".join(f"- {c.path}: {c.description}" for c in self.cfg.categories if c.path in choices),
            examples=self._nearest_examples(doc_vec) if doc_vec is not None else "",
        )
        user = f"{doc}\n\nChoose one of: {', '.join(choices)}"
        if preview.image:
            # The picture is the content: a photo, screenshot or scanned page. Only a vision model can read it.
            user = "Look at the attached image; it is the file's content.\n" + user
            result = self.backend.choose(system, user, choices, image=preview.image,
                                         model=self.ready_model(self.vision_model))
            self.stats["vision"] += 1
            via = "vision"
        else:
            self.ready_model(self.cfg.model.name)
            result = self.backend.choose(system, user, choices)
            self.stats["llm"] += 1
            via = "llm"
        cat = result["category"]
        if cat is None:
            return Verdict(None, 0.0, "model gave no valid category", via, choices[:3])
        alternatives = [p for p in choices if p != cat][:3]
        if not preview.readable and preview.image is None:
            if taught and taught[0] == cat:
                # Only name and type to go on, but it closely resembles a file you sorted, and the model agrees.
                conf, result["reason"] = 0.8, f"like {taught[1]}, which you put here; {result['reason']}"
            else:
                conf = 0.25  # guessed from the name and file type alone
        elif preview.image is not None:
            conf = 0.7   # read by the vision model; the text ranking can't see the picture
        elif ranked:
            conf = 0.8 if cat == ranked[0][0] else 0.6  # agreement between two independent signals
        else:
            conf = 0.7
        return Verdict(cat, conf, result["reason"], via, alternatives)

    def save(self):
        self.cache.save()
        self.vectors.save()
