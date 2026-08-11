"""
Combines metrics.json outputs from multiple `python -m src.evaluate` runs
into one side-by-side comparison table -- drop straight into the "final
model selection reasoning" section of the report.

Usage:
    python scripts/compare_models.py \
        --results tfidf_lr:eval_results/tfidf_lr/metrics.json \
                  fasttext:eval_results/fasttext/metrics.json \
                  cnn:eval_results/cnn/metrics.json \
                  bilstm:eval_results/bilstm/metrics.json \
        --out eval_results/comparison.csv
"""
import argparse
import csv
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True, help="name:path_to_metrics.json pairs")
    p.add_argument("--out", default="eval_results/comparison.csv")
    args = p.parse_args()

    rows = []
    for item in args.results:
        name, path = item.split(":", 1)
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "model": name,
            "n_evaluated": m["n_evaluated"],
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            "precision_weighted": m["precision_weighted"],
            "recall_weighted": m["recall_weighted"],
            "f1_weighted": m["f1_weighted"],
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':<12}{'accuracy':>10}{'f1_macro':>10}{'f1_weighted':>13}")
    for r in rows:
        print(f"{r['model']:<12}{r['accuracy']:>10.4f}{r['f1_macro']:>10.4f}{r['f1_weighted']:>13.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
