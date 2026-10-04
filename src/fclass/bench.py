"""`fclass bench`: accuracy and speed on *your* machine, with *your* models.

Runs the shipped 60-document benchmark (or 24 with --quick) in a throwaway
state folder, so your cache and learned examples are never touched or used.
Every number is measured end to end, model warm-up excluded.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from .backends import make_backend
from .benchdata import CATEGORIES, DATASET
from .classify import Classifier
from .config import Category, Config
from .extract import Preview


@dataclass
class Result:
    mode: str
    models: str
    correct: int
    total: int
    seconds: float
    llm_calls: int
    misses: list[str]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def per_file(self) -> float:
        return self.seconds / self.total


@contextmanager
def _throwaway_state():
    old = os.environ.get("XDG_STATE_HOME")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["XDG_STATE_HOME"] = tmp
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("XDG_STATE_HOME", None)
            else:
                os.environ["XDG_STATE_HOME"] = old


def items(quick: bool) -> list[tuple[str, str, str]]:
    if not quick:
        return list(DATASET)
    out, seen = [], {}
    for name, label, text in DATASET:
        if seen.get(label, 0) < 2:
            out.append((name, label, text))
            seen[label] = seen.get(label, 0) + 1
    return out


def bench_config(cfg: Config, mode: str) -> Config:
    c = copy.deepcopy(cfg)
    c.categories = [Category(p, d) for p, d in CATEGORIES.items()]
    c.strategy.mode = mode
    c.rules = []
    return c


def run_mode(cfg: Config, mode: str, docs, progress: Callable[[str, int, int], None] | None = None) -> Result:
    c = bench_config(cfg, mode)
    with _throwaway_state():
        clf = Classifier(c, make_backend(c.model, c.strategy.embed_model))
        clf.classify("warmup.txt", "file", Preview("Type: text. warm-up"))  # load models; not timed
        clf.cache.data.clear()
        clf.stats = {k: 0 for k in clf.stats}
        correct, misses = 0, []
        started = time.perf_counter()
        for i, (name, label, text) in enumerate(docs, 1):
            v = clf.classify(name, "file", Preview(f"Type: text. {text}"))
            if v.category == label:
                correct += 1
            else:
                misses.append(f"{name}: {label} → {v.category}")
            if progress:
                progress(mode, i, len(docs))
        seconds = time.perf_counter() - started
    models = {"embed": c.strategy.embed_model, "llm": c.model.name,
              "hybrid": f"{c.strategy.embed_model} + {c.model.name}"}[mode]
    return Result(mode, models, correct, len(docs), seconds, clf.stats["llm"], misses)


def machine() -> dict:
    info = {"os": platform.platform(terse=True), "arch": platform.machine(), "python": platform.python_version()}
    mac = platform.mac_ver()[0]
    if mac:
        info["os"] = f"macOS {mac}"
        try:
            info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                          text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9)
    except (ValueError, OSError, AttributeError):
        pass
    return info


def to_json(results: list[Result], quick: bool) -> dict:
    return {"machine": machine(), "quick": quick, "when": time.strftime("%Y-%m-%d %H:%M"),
            "results": [{"mode": r.mode, "models": r.models, "accuracy": round(r.accuracy, 3), "correct": r.correct,
                         "total": r.total, "sec_per_file": round(r.per_file, 2), "llm_calls": r.llm_calls,
                         "misses": r.misses} for r in results]}


def markdown(report: dict) -> str:
    m = report["machine"]
    where = " · ".join(str(v) for k, v in m.items() if k in ("chip", "os", "arch")) + \
        (f" · {m['ram_gb']} GB" if "ram_gb" in m else "")
    rows = "\n".join(f"| {r['mode']} | {r['models']} | {r['accuracy']:.1%} ({r['correct']}/{r['total']}) | "
                     f"{r['sec_per_file']:.2f} s | {r['llm_calls']} |" for r in report["results"])
    return (f"**fclass bench** on {where}{' (quick)' if report['quick'] else ''}\n\n"
            f"| Mode | Models | Accuracy | Per file | LLM calls |\n|---|---|---|---|---|\n{rows}\n")
