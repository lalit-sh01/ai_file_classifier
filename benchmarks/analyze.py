"""Derive hybrid-with-full-choice results from results.json without new LLM calls.

hybrid-full(m): embeddings decide when margin >= m; otherwise use the LLM's
answer when it chose freely among all categories (the llm-single run).
Latency = embed cost + (escalated share x LLM cost per file).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DATASET  # noqa: E402
from run import bench_embed  # noqa: E402

here = Path(__file__).parent
results = json.loads((here / "results.json").read_text())
labels = [l for _, l, _ in DATASET]
names = [n for n, _, _ in DATASET]
embedder = sys.argv[1] if len(sys.argv) > 1 else "embeddinggemma"
_, embed_secs, ranked = bench_embed(embedder, 0)

derived = []
for r in [r for r in results if r["strategy"] == "llm-single"]:
    wrong = {m.split(": ")[0]: m.split(" -> ")[1] for m in r["misses"]}
    llm_pred = [wrong.get(n, l) for n, l in zip(names, labels)]
    for m in (0.03, 0.04, 0.05, 0.06):
        esc = [i for i, rk in enumerate(ranked) if rk[0][1] - rk[1][1] < m]
        preds = [llm_pred[i] if i in esc else ranked[i][0][0] for i in range(len(labels))]
        correct = sum(p == l for p, l in zip(preds, labels))
        secs = embed_secs + len(esc) / len(labels) * r["sec_per_file"]
        row = {"strategy": f"hybrid-full m={m}", "model": f"{embedder}+{r['model']}", "correct": correct,
               "n": len(labels), "accuracy": round(correct / len(labels), 3), "sec_per_file": round(secs, 2),
               "escalated": len(esc), "misses": [f"{names[i]}: {labels[i]} -> {preds[i]}"
                                                 for i in range(len(labels)) if preds[i] != labels[i]]}
        derived.append(row)

(here / "results_derived.json").write_text(json.dumps(derived, indent=2))
for row in sorted(results + derived, key=lambda r: -r["accuracy"]):
    esc = f"  escalated {row['escalated']}" if "escalated" in row else ""
    print(f"{row['strategy']:<22} {row['model']:<28} {row['accuracy']:6.1%}  {row['sec_per_file']:6.2f}s/file"
          f"  invalid {row.get('invalid', 0)}{esc}")
