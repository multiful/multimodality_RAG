"""Compute PRD 5.1-② metrics from eval_logo_vlm.py JSONL results.

Usage:
    python scripts/eval_metrics.py results/qwen.jsonl results/llava.jsonl
"""

import json
import statistics
import sys
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix


def load(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def report(path):
    recs = load(path)
    ok = [r for r in recs if "error" not in r]
    errs = [r for r in recs if "error" in r]

    y_true = [r["label"] for r in ok]
    y_top1 = [(r["pred"][0] if r["pred"] else "NONE") for r in ok]
    top1 = sum(t == p for t, p in zip(y_true, y_top1)) / len(ok)
    top3 = sum(r["label"] in r["pred"] for r in ok) / len(ok)
    no_pred = sum(1 for r in ok if not r["pred"]) / len(ok)
    lat = [r["ms"] for r in ok]

    rep = classification_report(
        y_true, y_top1, output_dict=True, zero_division=0
    )
    print(f"\n===== {path} =====")
    print(f"samples={len(recs)} evaluated={len(ok)} load_errors={len(errs)}")
    print(f"Top-1 Accuracy : {top1:.4f}")
    print(f"Top-3 Accuracy : {top3:.4f}")
    print(f"Macro Precision: {rep['macro avg']['precision']:.4f}")
    print(f"Macro Recall   : {rep['macro avg']['recall']:.4f}")
    print(f"Macro F1       : {rep['macro avg']['f1-score']:.4f}")
    print(f"No-entity rate : {no_pred:.4f}  (answer had no linkable company)")
    print(f"Latency ms/img : mean={statistics.mean(lat):.0f} median={statistics.median(lat):.0f} p95={sorted(lat)[int(len(lat) * 0.95)]:.0f}")

    per_class = {
        k: v for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg", "NONE")
    }
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["f1-score"])[:10]
    print("Worst-10 classes by F1:")
    for k, v in worst:
        print(f"  {k:6s} f1={v['f1-score']:.2f} recall={v['recall']:.2f} support={int(v['support'])}")

    top_conf = Counter(
        (t, p) for t, p in zip(y_true, y_top1) if t != p
    ).most_common(10)
    print("Top-10 confusions (true -> predicted):")
    for (t, p), c in top_conf:
        print(f"  {t:6s} -> {p:6s} x{c}")

    return {
        "path": path, "n": len(ok), "top1": top1, "top3": top3,
        "macro_f1": rep["macro avg"]["f1-score"],
        "macro_p": rep["macro avg"]["precision"],
        "macro_r": rep["macro avg"]["recall"],
        "no_pred": no_pred,
        "lat_mean": statistics.mean(lat), "lat_median": statistics.median(lat),
    }


def save_confusion(path):
    recs = [r for r in load(path) if "error" not in r]
    y_true = [r["label"] for r in recs]
    y_pred = [(r["pred"][0] if r["pred"] else "NONE") for r in recs]
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = path.rsplit(".", 1)[0] + "_confusion.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "matrix": cm.tolist()}, f)
    print(f"confusion matrix -> {out}")


if __name__ == "__main__":
    summaries = [report(p) for p in sys.argv[1:]]
    for p in sys.argv[1:]:
        save_confusion(p)
    if len(summaries) > 1:
        print("\n===== Comparison =====")
        hdr = f"{'model':32s} {'Top-1':>7s} {'Top-3':>7s} {'MacroF1':>8s} {'NoEnt':>7s} {'ms/img':>8s}"
        print(hdr)
        for s in summaries:
            name = s["path"].split("/")[-1].split("\\")[-1]
            print(f"{name:32s} {s['top1']:7.4f} {s['top3']:7.4f} {s['macro_f1']:8.4f} {s['no_pred']:7.4f} {s['lat_median']:8.0f}")
