"""Head-to-head benchmark of classification strategies on a local Ollama server.

    python benchmarks/run.py [--llms qwen3:1.7b,qwen3:4b] [--embedders embeddinggemma]

Strategies
  v1-batch     v1's design: 10 files per prompt, free-form JSON, index mapping
  llm-single   one file per call, category constrained to an enum via JSON schema
  embed-zero   cosine similarity between file and category descriptions
  embed-fewshot  same, plus k corrected examples per category (simulates learning)
  hybrid       embeddings shortlist top-3; LLM decides only when the margin is thin
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DATASET  # noqa: E402

URL = "http://localhost:11434"

CATEGORIES = {
    "Finance/Statements": "Bank statements, credit card bills, utility bills, invoices, payment receipts",
    "Finance/Taxes": "Tax returns, W-2, 1099, Form 16, tax receipts and deduction proofs",
    "Finance/Investments": "Brokerage and portfolio statements, stock trades, retirement and pension accounts",
    "Study/Courses": "Syllabi, assignments, course materials, exam papers, certificates of completion",
    "Study/Notes": "Personal notes, lecture notes, highlights, study guides, journals",
    "Study/References": "Research papers, technical documentation, textbooks, reference material",
    "Recreation/Hobbies": "Hobby projects, recipes, patterns, collections, DIY guides",
    "Recreation/Entertainment": "Ebooks, articles, saved web content, games, movies and music lists",
    "Recreation/Travel": "Itineraries, flight and hotel bookings, tickets, visas, trip plans",
    "Keep/Important": "IDs, passports, birth certificates, wills, contracts, leases, insurance policies, legal documents. These go here even when they mention money.",
    "Keep/Archives": "Old or completed projects and historical records worth preserving",
    "Keep/Manuals": "Product manuals, warranties, setup and instruction guides",
}
PATHS = list(CATEGORIES)


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def category_block() -> str:
    return "\n".join(f"- {p}: {d}" for p, d in CATEGORIES.items())


# ── LLM strategies ─────────────────────────────────────────────────────────

SYSTEM = (
    "You sort a person's files into folders. Read the content and pick exactly one category.\n"
    "Judge by what the document IS, not by words it happens to contain. Filenames are often meaningless.\n\n"
    f"Categories:\n{category_block()}"
)


def llm_single(model: str, name: str, text: str, choices: list[str] | None = None) -> str:
    choices = choices or PATHS
    schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "category": {"type": "string", "enum": choices},
        },
        "required": ["reason", "category"],
    }
    user = f"Filename: {name}\nContent preview:\n{text}"
    if choices is not PATHS:
        user += "\n\nChoose among: " + ", ".join(choices)
    r = post("/api/chat", {
        "model": model, "stream": False, "think": False, "format": schema,
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    })
    try:
        return json.loads(r["message"]["content"])["category"]
    except Exception:
        return "INVALID"


def v1_batch(model: str, items: list[tuple[str, str, str]]) -> list[str]:
    """Faithful reproduction of v1's ollama_classifier prompt (batch + format=json)."""
    preds: list[str] = []
    for start in range(0, len(items), 10):
        batch = items[start:start + 10]
        files_info = [{"index": i, "filename": n, "preview": t[:1000]} for i, (n, _, t) in enumerate(batch)]
        prompt = f"""Classify each of these {len(files_info)} files into one category.

IMPORTANT: Classify based on the FILE CONTENT (preview text), NOT the filename or file extension.
Filenames may be random or misleading - always use the content preview to determine the category.

Categories:
{category_block()}

Respond ONLY with valid JSON in this exact format:
{{"classifications": [{{"index": 0, "category": "Finance/Taxes", "reason": "brief explanation"}}]}}

Files to classify (focus on the 'preview' content, ignore 'filename'):
{json.dumps(files_info, indent=2)}

JSON response:"""
        r = post("/api/generate", {
            "model": model, "prompt": prompt, "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0.1, "num_predict": 2000, "num_ctx": 8192},
        })
        out = ["INVALID"] * len(batch)
        try:
            for c in json.loads(r["response"]).get("classifications", []):
                idx = c.get("index")
                if isinstance(idx, int) and 0 <= idx < len(batch):
                    out[idx] = c.get("category", "INVALID")
        except Exception:
            pass
        preds.extend(out)
    return preds


# ── Embedding strategies ───────────────────────────────────────────────────

def embed(model: str, texts: list[str]) -> list[list[float]]:
    return post("/api/embed", {"model": model, "input": texts})["embeddings"]


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def prefixes(model: str) -> tuple[str, str]:
    if "nomic" in model:
        return "classification: ", "classification: "
    if "embeddinggemma" in model:
        return "task: classification | query: ", "title: none | text: "
    return "", ""


def doc_text(name: str, text: str) -> str:
    return f"{name}\n{text}"


def _z(scores: dict[str, float]) -> dict[str, float]:
    vals = list(scores.values())
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
    return {k: (v - mu) / sd for k, v in scores.items()}


def rank(vec, desc: dict[str, list[float]], shots: dict[str, list[list[float]]] | None = None,
         shot_weight: float = 1.0) -> list[tuple[str, float]]:
    """Score = z(similarity to description) + w * z(mean similarity to corrected examples).

    The two signals live on different scales (doc-doc similarity runs much
    higher than doc-description), so each is z-normalised across categories
    before blending. Returned scores are raw description cosines when there
    are no shots, so the margin threshold stays interpretable.
    """
    desc_s = {p: cos(vec, v) for p, v in desc.items()}
    if not shots or not any(shots.values()):
        return sorted(desc_s.items(), key=lambda kv: -kv[1])
    shot_s = {p: (sum(cos(vec, v) for v in vs) / len(vs)) if vs else min(desc_s.values()) for p, vs in shots.items()}
    zd, zs = _z(desc_s), _z(shot_s)
    blended = {p: (zd[p] + shot_weight * zs[p]) / (1 + shot_weight) for p in desc}
    return sorted(blended.items(), key=lambda kv: -kv[1])


def bench_embed(model: str, k_shot: int):
    qp, dp = prefixes(model)
    t0 = time.time()
    doc_vecs = embed(model, [qp + doc_text(n, t) for n, _, t in DATASET])
    per_item = (time.time() - t0) / len(DATASET)
    desc_vecs = dict(zip(PATHS, embed(model, [dp + f"{p.replace('/', ' / ')}: {d}" for p, d in CATEGORIES.items()])))

    correct = 0
    ranked_all = []
    for i, (_, label, _) in enumerate(DATASET):
        shots = {}
        if k_shot:
            # Leave-one-out: the item never sees itself, only other "corrected" files.
            for p in PATHS:
                idx = [j for j, (_, l, _) in enumerate(DATASET) if l == p and j != i][:k_shot]
                shots[p] = [doc_vecs[j] for j in idx]
        ranked = rank(doc_vecs[i], desc_vecs, shots)
        ranked_all.append(ranked)
        correct += ranked[0][0] == label
    return correct, per_item, ranked_all


# ── Driver ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llms", default="qwen3:1.7b,qwen3:4b,gemma3:4b")
    ap.add_argument("--embedders", default="embeddinggemma,nomic-embed-text")
    ap.add_argument("--hybrid-llm", default="qwen3:4b")
    ap.add_argument("--hybrid-embedder", default="embeddinggemma")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    args = ap.parse_args()
    n = len(DATASET)
    labels = [l for _, l, _ in DATASET]
    results = []

    def record(name, model, correct, secs, invalid=0, misses=None):
        row = {"strategy": name, "model": model, "accuracy": round(correct / n, 3),
               "correct": correct, "n": n, "sec_per_file": round(secs, 3), "invalid": invalid,
               "misses": misses or []}
        results.append(row)
        print(f"{name:<15} {model:<22} acc {correct:>2}/{n} = {correct / n:5.1%}   "
              f"{secs:6.2f}s/file   invalid {invalid}", flush=True)

    # warm every model once so load time doesn't pollute latency
    for m in [x for x in args.llms.split(",") if x]:
        llm_single(m, "warmup.txt", "hello")
    for m in [x for x in args.embedders.split(",") if x]:
        embed(m, ["warmup"])

    for m in [x for x in args.embedders.split(",") if x]:
        for k in (0, 1, 2, 3):
            correct, secs, ranked = bench_embed(m, k)
            misses = [f"{DATASET[i][0]}: {labels[i]} -> {r[0][0]}" for i, r in enumerate(ranked) if r[0][0] != labels[i]]
            record("embed-zero" if k == 0 else f"embed-{k}shot", m, correct, secs, misses=misses)

    for m in [x for x in args.llms.split(",") if x]:
        t0 = time.time()
        preds = v1_batch(m, DATASET)
        secs = (time.time() - t0) / n
        invalid = sum(p not in CATEGORIES for p in preds)
        correct = sum(p == l for p, l in zip(preds, labels))
        misses = [f"{DATASET[i][0]}: {labels[i]} -> {p}" for i, p in enumerate(preds) if p != labels[i]]
        record("v1-batch", m, correct, secs, invalid, misses)

        t0 = time.time()
        preds = [llm_single(m, nm, t) for nm, _, t in DATASET]
        secs = (time.time() - t0) / n
        invalid = sum(p not in CATEGORIES for p in preds)
        correct = sum(p == l for p, l in zip(preds, labels))
        misses = [f"{DATASET[i][0]}: {labels[i]} -> {p}" for i, p in enumerate(preds) if p != labels[i]]
        record("llm-single", m, correct, secs, invalid, misses)

    # Hybrid: embeddings decide when confident, otherwise LLM picks from top-3.
    if args.hybrid_llm and args.hybrid_embedder:
        _, embed_secs, ranked = bench_embed(args.hybrid_embedder, 0)
        t0 = time.time()
        correct = escalated = 0
        misses = []
        for i, (nm, label, t) in enumerate(DATASET):
            top = ranked[i]
            if top[0][1] - top[1][1] >= args.margin:
                pred = top[0][0]
            else:
                escalated += 1
                pred = llm_single(args.hybrid_llm, nm, t, choices=[p for p, _ in top[:3]])
            correct += pred == label
            if pred != label:
                misses.append(f"{nm}: {label} -> {pred}")
        secs = embed_secs + (time.time() - t0) / n
        record(f"hybrid({escalated} esc)", f"{args.hybrid_embedder}+{args.hybrid_llm}", correct, secs, misses=misses)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
