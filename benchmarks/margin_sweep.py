"""How does the embedding margin threshold trade accuracy against LLM calls?

For each threshold: what share of files embeddings decide alone, how accurate
those decisions are, and whether the true label is in the top-k shortlist the
LLM would see for the rest (the hybrid's ceiling).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DATASET  # noqa: E402
from run import bench_embed  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "embeddinggemma"
_, _, ranked = bench_embed(model, 0)
labels = [l for _, l, _ in DATASET]
rows = []
print(f"{'margin':>7} {'decided':>8} {'acc-when-decided':>17} {'escalated':>10} {'top2-recall':>12} {'top3-recall':>12}")
for m in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
    dec = [i for i, r in enumerate(ranked) if r[0][1] - r[1][1] >= m]
    esc = [i for i in range(len(ranked)) if i not in dec]
    acc = sum(ranked[i][0][0] == labels[i] for i in dec) / max(len(dec), 1)
    t2 = sum(labels[i] in [p for p, _ in ranked[i][:2]] for i in esc) / max(len(esc), 1)
    t3 = sum(labels[i] in [p for p, _ in ranked[i][:3]] for i in esc) / max(len(esc), 1)
    rows.append({"margin": m, "decided": len(dec), "acc_decided": round(acc, 3), "escalated": len(esc),
                 "top2_recall": round(t2, 3), "top3_recall": round(t3, 3)})
    print(f"{m:>7.2f} {len(dec):>8} {acc:>17.1%} {len(esc):>10} {t2:>12.1%} {t3:>12.1%}")
Path(__file__).with_name(f"margin_{model}.json").write_text(json.dumps(rows, indent=2))
