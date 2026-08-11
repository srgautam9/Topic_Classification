"""
Full evaluation + error analysis on the deterministic validation split (the
same split used during training: every 20th row, by streaming order -- see
TRAIN_VAL_MOD in src/data.py).

Streams the dataset once (does not require holding it in memory), evaluates
one model, and writes report-ready artifacts to --out-dir:

    metrics.json               overall + per-class accuracy/precision/recall/F1
    per_class_metrics.csv      per-class table
    confusion_matrix.csv       raw counts
    confusion_matrix.png       heatmap (row-normalized), skipped if matplotlib missing
    top_confused_pairs.csv     the (true, predicted) pairs the model confuses most
    misclassified_samples.csv  a few example texts per confused pair, for qualitative analysis
    length_error_analysis.csv  error rate bucketed by input text length
    confidence_analysis.csv    predicted-class confidence, correct vs incorrect predictions

Usage
-----
    python -m src.evaluate --approach fasttext --model final_models/fasttext_final.pt \
        --data /path/to/dataset_10.parquet --out-dir eval_results/fasttext

    python -m src.evaluate --approach tfidf_lr --model final_models/tfidf_lr.joblib \
        --data /path/to/dataset_10.parquet --out-dir eval_results/tfidf_lr

Run once per trained model, then use scripts/compare_models.py to combine
the resulting metrics.json files into one side-by-side comparison table.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

from .data import TRAIN_VAL_MOD
from .model import build_model
from .utils import clean_text, hash_token, hashed_ngram_ids, stream_parquet_batches, tokenize


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_deep(model_path, approach):
    ckpt = torch.load(model_path, map_location="cpu")
    label2id = ckpt["label2id"]
    saved_args = ckpt.get("args", {})
    model = build_model(
        approach, saved_args.get("num_buckets", 1_000_000), len(label2id),
        embed_dim=saved_args.get("embed_dim", 100), hidden_dim=saved_args.get("hidden_dim", 128),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, label2id, saved_args


def predict_batch_deep(model, texts, approach, num_buckets, max_len, device):
    """Returns (pred_ids, confidences) for a batch of raw texts."""
    with torch.no_grad():
        if approach == "fasttext":
            ids_list = [hashed_ngram_ids(clean_text(t), num_buckets, (1, 2)) or [0] for t in texts]
            lengths = [len(x) for x in ids_list]
            offsets = torch.tensor([0] + lengths[:-1]).cumsum(0).to(device)
            flat = torch.tensor([i for ids in ids_list for i in ids], dtype=torch.long).to(device)
            logits = model(flat, offsets)
        else:
            seqs = [[hash_token(t, num_buckets) + 1 for t in tokenize(clean_text(x))[:max_len]] or [0]
                    for x in texts]
            lengths = torch.tensor([len(s) for s in seqs])
            mx = max(lengths).item()
            padded = torch.zeros(len(seqs), mx, dtype=torch.long)
            for j, s in enumerate(seqs):
                padded[j, : len(s)] = torch.tensor(s)
            padded, lengths = padded.to(device), lengths.to(device)
            logits = model(padded, lengths) if approach == "bilstm" else model(padded)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    return pred.cpu().tolist(), conf.cpu().tolist()


def predict_batch_classical(bundle, texts):
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(**bundle["vectorizer_params"])
    clf = bundle["clf"]
    X = vectorizer.transform([clean_text(t) for t in texts])
    preds = clf.predict(X)
    try:
        probs = clf.predict_proba(X)
        conf = probs.max(axis=1).tolist()
    except Exception:
        conf = [None] * len(texts)
    return list(preds), conf


def length_bucket(n_words):
    if n_words <= 5:
        return "1-5 words"
    if n_words <= 15:
        return "6-15 words"
    if n_words <= 30:
        return "16-30 words"
    return "31+ words"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approach", required=True, choices=["tfidf_lr", "fasttext", "cnn", "bilstm"])
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--text-col", default="DATA")
    p.add_argument("--label-col", default="TOPIC")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-val-rows", type=int, default=None,
                    help="cap number of validation rows evaluated (default: evaluate all)")
    p.add_argument("--num-buckets", type=int, default=1_000_000, help="fallback if not in checkpoint")
    p.add_argument("--max-len", type=int, default=128, help="fallback if not in checkpoint")
    p.add_argument("--samples-per-pair", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    if args.approach == "tfidf_lr":
        import joblib
        bundle = joblib.load(args.model)
        label2id = bundle["label2id"]
        num_buckets, max_len = None, None
    else:
        model, label2id, saved_args = load_deep(args.model, args.approach)
        model.to(device)
        num_buckets = saved_args.get("num_buckets", args.num_buckets)
        max_len = saved_args.get("max_len", args.max_len)

    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    misclassified_examples = defaultdict(list)  # (true,pred) -> [text,...]
    length_buckets = defaultdict(lambda: [0, 0])  # bucket -> [n_correct, n_total]
    conf_correct, conf_incorrect = [], []

    buf_texts, buf_labels = [], []
    n_evaluated = 0

    def flush():
        nonlocal n_evaluated
        if not buf_texts:
            return
        if args.approach == "tfidf_lr":
            preds, confs = predict_batch_classical(bundle, buf_texts)
        else:
            preds, confs = predict_batch_deep(model, buf_texts, args.approach, num_buckets, max_len, device)
        for text, true_id, pred_id, conf in zip(buf_texts, buf_labels, preds, confs):
            confusion[true_id, pred_id] += 1
            n_words = len(tokenize(clean_text(text)))
            b = length_bucket(n_words)
            length_buckets[b][1] += 1
            if true_id == pred_id:
                length_buckets[b][0] += 1
                if conf is not None:
                    conf_correct.append(conf)
            else:
                if conf is not None:
                    conf_incorrect.append(conf)
                key = (id2label[true_id], id2label[pred_id])
                if len(misclassified_examples[key]) < args.samples_per_pair:
                    misclassified_examples[key].append(text)
        n_evaluated += len(buf_texts)
        buf_texts.clear()
        buf_labels.clear()

    row_idx = 0
    stop = False
    for batch in stream_parquet_batches(args.data, [args.text_col, args.label_col], 100_000):
        for text, label in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            is_val = (row_idx % TRAIN_VAL_MOD == 0)
            row_idx += 1
            if not is_val or label not in label2id:
                continue
            buf_texts.append(text)
            buf_labels.append(label2id[label])
            if len(buf_texts) >= args.batch_size:
                flush()
            if args.max_val_rows and n_evaluated >= args.max_val_rows:
                stop = True
                break
        if stop:
            break
    flush()

    print(f"Evaluated {n_evaluated:,} validation rows.")

    # ---- overall + per-class metrics, derived from the confusion matrix ----
    tp = np.diag(confusion)
    support = confusion.sum(axis=1)
    pred_totals = confusion.sum(axis=0)
    accuracy = float(tp.sum() / confusion.sum())

    per_class = {}
    for i, label in id2label.items():
        precision = float(tp[i] / pred_totals[i]) if pred_totals[i] > 0 else 0.0
        recall = float(tp[i] / support[i]) if support[i] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support[i])}

    macro_p = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_r = float(np.mean([v["recall"] for v in per_class.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    w = np.array([v["support"] for v in per_class.values()], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else w
    weighted_p = float(np.sum(w * [v["precision"] for v in per_class.values()]))
    weighted_r = float(np.sum(w * [v["recall"] for v in per_class.values()]))
    weighted_f1 = float(np.sum(w * [v["f1"] for v in per_class.values()]))

    metrics = {
        "n_evaluated": int(n_evaluated),
        "accuracy": accuracy,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f1,
        "per_class": per_class,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))

    with open(os.path.join(args.out_dir, "per_class_metrics.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["label", "precision", "recall", "f1", "support"])
        for label, v in sorted(per_class.items(), key=lambda x: -x[1]["support"]):
            wcsv.writerow([label, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}", v["support"]])

    labels_sorted = [id2label[i] for i in range(num_classes)]
    with open(os.path.join(args.out_dir, "confusion_matrix.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["true\\pred"] + labels_sorted)
        for i, row in enumerate(confusion):
            wcsv.writerow([labels_sorted[i]] + row.tolist())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        norm_conf = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), max(5, num_classes * 0.6)))
        im = ax.imshow(norm_conf, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(labels_sorted)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (row-normalized) -- {args.approach}")
        for i in range(num_classes):
            for j in range(num_classes):
                if confusion[i, j] > 0:
                    ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                             fontsize=7, color="white" if norm_conf[i, j] > 0.5 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed -- skipping confusion_matrix.png (CSV still written)")

    pair_counts = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                pair_counts.append((int(confusion[i, j]), labels_sorted[i], labels_sorted[j]))
    pair_counts.sort(reverse=True)
    with open(os.path.join(args.out_dir, "top_confused_pairs.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["true_label", "predicted_label", "count"])
        for count, t, pr in pair_counts[:30]:
            wcsv.writerow([t, pr, count])

    with open(os.path.join(args.out_dir, "misclassified_samples.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["true_label", "predicted_label", "text"])
        for (t, pr), texts in misclassified_examples.items():
            for text in texts:
                wcsv.writerow([t, pr, text])

    with open(os.path.join(args.out_dir, "length_error_analysis.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["length_bucket", "n_total", "n_correct", "error_rate"])
        for bucket in ["1-5 words", "6-15 words", "16-30 words", "31+ words"]:
            n_correct, n_total = length_buckets.get(bucket, [0, 0])
            err = (1 - n_correct / n_total) if n_total else None
            wcsv.writerow([bucket, n_total, n_correct, f"{err:.4f}" if err is not None else "n/a"])

    if conf_correct or conf_incorrect:
        with open(os.path.join(args.out_dir, "confidence_analysis.csv"), "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["group", "n", "mean_confidence", "median_confidence"])
            for name, arr in [("correct", conf_correct), ("incorrect", conf_incorrect)]:
                if arr:
                    wcsv.writerow([name, len(arr), f"{np.mean(arr):.4f}", f"{np.median(arr):.4f}"])

    print(f"\nWrote evaluation artifacts to {args.out_dir}/")
    print("  metrics.json, per_class_metrics.csv, confusion_matrix.{csv,png}, "
          "top_confused_pairs.csv, misclassified_samples.csv, "
          "length_error_analysis.csv, confidence_analysis.csv")


if __name__ == "__main__":
    main()
