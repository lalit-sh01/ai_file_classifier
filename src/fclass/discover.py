"""`fclass discover`: propose categories from a folder you already have.

1. Read up to `limit` files (recursively) and embed each one.
2. Split them into many small groups of near-identical files (k-medoids).
3. Ask the LLM to name each group: either one of your existing categories,
   or a new folder path with a one-line description. Groups that get the
   same name merge, so the number of categories comes from meaning, not a
   statistic.

Nothing changes until you accept: accepted categories are written to your
config, and the files closest to each group's centre are kept as examples
so the classifier starts out knowing your taste.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .classify import Classifier, cosine
from .config import Config, valid_category_path
from .extract import preview_file


@dataclass
class Group:
    files: list[Path]
    snippets: list[str]
    centrality: list[float]           # similarity of each file to the group centre
    path: str = ""
    description: str = ""
    existing: str | None = None       # set when the group fits a category you already have
    reason: str = ""

    @property
    def is_new(self) -> bool:
        return self.existing is None

    def central(self, n: int = 3) -> list[tuple[Path, str]]:
        order = sorted(range(len(self.files)), key=lambda i: -self.centrality[i])
        return [(self.files[i], self.snippets[i]) for i in order[:n]]


@dataclass
class Proposal:
    root: str
    groups: list[Group]
    ungrouped: list[Path] = field(default_factory=list)
    k: int = 0


# ── gathering ───────────────────────────────────────────────────────────────

def gather(cfg: Config, root: Path, limit: int = 300, depth: int = 4) -> list[Path]:
    root = root.expanduser().resolve()
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).parts
        if len(rel) > depth or any(part.startswith(".") for part in rel) or cfg.is_ignored(p.name):
            continue
        if p.is_file() and not p.is_symlink():
            files.append(p)
    if len(files) > limit:  # an even sample across the whole tree, not just the first folders
        step = len(files) / limit
        files = [files[int(i * step)] for i in range(limit)]
    return files


# ── clustering (pure Python; a few hundred files take seconds) ──────────────

def _normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _mean(vs: list[list[float]]) -> list[float]:
    return _normalise([sum(col) / len(vs) for col in zip(*vs)])


def similarity_matrix(vecs: list[list[float]]) -> list[list[float]]:
    vecs = [_normalise(v) for v in vecs]
    n = len(vecs)
    sim = [[1.0] * n for _ in range(n)]
    for i in range(n):
        a = vecs[i]
        for j in range(i + 1, n):
            sim[i][j] = sim[j][i] = sum(map(float.__mul__, a, vecs[j]))
    return sim


def kmedoids(sim: list[list[float]], k: int, seed: int = 0, iters: int = 20) -> list[int]:
    """k-medoids on a similarity matrix: every step is a lookup, so it stays fast in pure Python."""
    n = len(sim)
    rng = random.Random(seed)
    medoids = [rng.randrange(n)]
    while len(medoids) < k:  # k-means++ style seeding
        d = [min(1 - sim[i][m] for m in medoids) ** 2 for i in range(n)]
        r, acc = rng.random() * (sum(d) or 1.0), 0.0
        for i, w in enumerate(d):
            acc += w
            if acc >= r and i not in medoids:
                medoids.append(i)
                break
        else:
            medoids.append(next(i for i in range(n) if i not in medoids))
    labels: list[int] = []
    for _ in range(iters):
        labels = [max(range(k), key=lambda j: sim[i][medoids[j]]) for i in range(n)]
        changed = False
        for j in range(k):
            members = [i for i in range(n) if labels[i] == j]
            if not members:
                continue
            best = max(members, key=lambda c: sum(sim[c][m] for m in members))
            if best != medoids[j]:
                medoids[j], changed = best, True
        if not changed:
            break
    return labels


def objective(labels: list[int], sim: list[list[float]], medoid_of: dict[int, int] | None = None) -> float:
    """Total similarity of each file to the most central member of its group."""
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    total = 0.0
    for members in groups.values():
        total += max(sum(sim[c][m] for m in members) for c in members)
    return total


def over_split(vecs: list[list[float]], per_group: int = 3, k_min: int = 3, k_max: int = 30) -> tuple[int, list[int]]:
    """Deliberately many small, tight groups. The LLM names them, and groups given the
    same name are merged, so the final number of categories is decided by meaning.
    (Choosing k by silhouette picked 4 groups for 12 real categories in our benchmark; ~3 files
    per group gave the purest groups before naming: 75% vs 57% at 4 and 53% at 5.)"""
    n = len(vecs)
    if n < 4:
        return 1, [0] * n
    k = max(k_min, min(k_max, round(n / per_group), n // 2))
    sim = similarity_matrix(vecs)
    runs = [kmedoids(sim, k, seed) for seed in (0, 1, 2)]
    return k, max(runs, key=lambda labels: objective(labels, sim))


# ── naming ──────────────────────────────────────────────────────────────────

NAMING_SYSTEM = """You organise a person's files. You are shown a group of similar files from one of their
folders. Propose ONE folder for the whole group.

If one of the person's existing categories clearly fits the whole group, put it in "fits_existing".
Otherwise set "fits_existing" to "none" and propose a new folder:
- "path": at most two levels, "Area/Topic", in Title Case, e.g. "Work/Payslips" or "Home/Utilities".
  Reuse an existing top-level area when it makes sense.
- "description": one plain line saying what belongs there, general enough for future files of this kind.

Existing categories:
{existing}"""


def naming_schema(existing: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": 200},
            "fits_existing": {"type": "string", "enum": [*existing, "none"]},
            "path": {"type": "string", "maxLength": 60},
            "description": {"type": "string", "maxLength": 160},
        },
        "required": ["reason", "fits_existing", "path", "description"],
    }


def clean_path(path: str) -> str:
    parts = [re.sub(r"[^\w &+-]", "", p).strip() for p in path.replace("\\", "/").split("/")]
    parts = [" ".join(w[:1].upper() + w[1:] for w in p.split()) for p in parts if p][:2]
    return "/".join(parts)


def name_group(cfg: Config, classifier: Classifier, group: Group, fresh: bool = False) -> None:
    existing = [] if fresh else cfg.category_paths
    listing = "\n".join(f"- {c.path}: {c.description}" for c in cfg.categories) if existing else "(none yet)"
    samples = "\n".join(f"- {p.name}: {' '.join(s.split())[:220]}" for p, s in group.central(8))
    user = f"{len(group.files)} files in this group. The most typical ones:\n{samples}"
    classifier.ready_model(cfg.model.name)
    out = classifier.backend.generate(NAMING_SYSTEM.format(existing=listing), user, naming_schema(existing))
    fit = out.get("fits_existing", "none")
    group.reason = str(out.get("reason", ""))[:200]
    if fit in existing:
        desc = next(c.description for c in cfg.categories if c.path == fit)
        group.existing, group.path, group.description = fit, fit, desc
        return
    group.path = clean_path(str(out.get("path", ""))) or "Unsorted/Group"
    group.description = str(out.get("description", "")).strip() or f"Files like {group.files[0].name}"


# ── the whole flow ──────────────────────────────────────────────────────────

def discover(cfg: Config, classifier: Classifier, root: Path, limit: int = 300, fresh: bool = False,
             progress: Callable[[str, int, int], None] | None = None) -> Proposal:
    files = gather(cfg, root, limit)
    snippets: list[str] = []
    for i, f in enumerate(files, 1):
        snippets.append(preview_file(f, 800).text)
        if progress:
            progress("reading", i, len(files))
    if progress:
        progress("grouping", 0, len(files))
    docs = [classifier.doc_prefix + f"File: {f.name}\n{s}" for f, s in zip(files, snippets)]
    vecs = classifier._embed(docs) if docs else []
    classifier.save()
    k, labels = over_split(vecs)

    groups, ungrouped = [], []
    for j in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == j]
        if len(idx) < 2:
            ungrouped += [files[i] for i in idx]
            continue
        centre = _mean([_normalise(vecs[i]) for i in idx])
        groups.append(Group([files[i] for i in idx], [snippets[i] for i in idx],
                            [cosine(vecs[i], centre) for i in idx]))
    groups.sort(key=lambda g: -len(g.files))
    for n, g in enumerate(groups, 1):
        if progress:
            progress("naming", n, len(groups))
        name_group(cfg, classifier, g, fresh)
    return Proposal(str(root), _merge(groups), ungrouped, k)


def _merge(groups: list[Group]) -> list[Group]:
    """Two groups given the same folder become one."""
    by_path: dict[str, Group] = {}
    for g in groups:
        if not valid_category_path(g.path):
            continue
        if g.path in by_path:
            m = by_path[g.path]
            m.files += g.files
            m.snippets += g.snippets
            m.centrality += g.centrality
        else:
            by_path[g.path] = g
    return sorted(by_path.values(), key=lambda g: (-len(g.files)))
