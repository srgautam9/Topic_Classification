"""
Inference script - works for both the classical model (joblib) and the deep
learning models (.pt checkpoints).

Examples
--------
Single string:
    python -m src.inference --model final_models/fasttext_final.pt --approach fasttext \
        --input "some text to classify"

Batch from CSV (must contain the text column, default name DATA):
    python -m src.inference --model final_models/fasttext_final.pt --approach fasttext \
        --input-file texts.csv --text-col DATA --output-file preds.csv

Classical model:
    python -m src.inference --model final_models/tfidf_lr.joblib --approach tfidf_lr \
        --input "some text to classify"
"""
import argparse
import csv

import torch

from .utils import clean_text, hashed_ngram_ids, tokenize, hash_token
from .model import build_model


def load_deep_model(path, approach, num_buckets, embed_dim, hidden_dim):
    ckpt = torch.load(path, map_location="cpu")
    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    saved_args = ckpt.get("args", {})
    model = build_model(approach, saved_args.get("num_buckets", num_buckets),
                         len(label2id), embed_dim=saved_args.get("embed_dim", embed_dim),
                         hidden_dim=saved_args.get("hidden_dim", hidden_dim))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, label2id, id2label, saved_args


def predict_deep(model, texts, approach, num_buckets, max_len, id2label):
    with torch.no_grad():
        if approach == "fasttext":
            ids_list = [hashed_ngram_ids(clean_text(t), num_buckets, (1, 2)) or [0] for t in texts]
            lengths = [len(x) for x in ids_list]
            offsets = torch.tensor([0] + lengths[:-1]).cumsum(0)
            flat = torch.tensor([i for ids in ids_list for i in ids], dtype=torch.long)
            logits = model(flat, offsets)
        else:
            seqs = [[hash_token(t, num_buckets) + 1 for t in tokenize(clean_text(x))[:max_len]] or [0]
                    for x in texts]
            lengths = torch.tensor([len(s) for s in seqs])
            mx = max(lengths).item()
            padded = torch.zeros(len(seqs), mx, dtype=torch.long)
            for j, s in enumerate(seqs):
                padded[j, :len(s)] = torch.tensor(s)
            logits = model(padded, lengths) if approach == "bilstm" else model(padded)
        pred_ids = logits.argmax(1).tolist()
    return [id2label[i] for i in pred_ids]


def predict_classical(bundle_path, texts):
    import joblib
    from sklearn.feature_extraction.text import HashingVectorizer
    bundle = joblib.load(bundle_path)
    vectorizer = HashingVectorizer(**bundle["vectorizer_params"])
    clf = bundle["clf"]
    id2label = {v: k for k, v in bundle["label2id"].items()}
    X = vectorizer.transform([clean_text(t) for t in texts])
    pred_ids = clf.predict(X)
    return [id2label[i] for i in pred_ids]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--approach", required=True,
                    choices=["tfidf_lr", "fasttext", "cnn", "bilstm"])
    p.add_argument("--input", help="single text string to classify")
    p.add_argument("--input-file", help="CSV file with a text column")
    p.add_argument("--text-col", default="DATA")
    p.add_argument("--output-file", help="where to write predictions CSV")
    p.add_argument("--num-buckets", type=int, default=1_000_000)
    p.add_argument("--embed-dim", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()

    if args.input_file:
        with open(args.input_file) as f:
            rows = list(csv.DictReader(f))
        texts = [r[args.text_col] for r in rows]
    elif args.input:
        texts = [args.input]
        rows = [{args.text_col: args.input}]
    else:
        raise ValueError("Provide --input or --input-file")

    if args.approach == "tfidf_lr":
        preds = predict_classical(args.model, texts)
    else:
        model, label2id, id2label, saved_args = load_deep_model(
            args.model, args.approach, args.num_buckets, args.embed_dim, args.hidden_dim)
        preds = predict_deep(model, texts, args.approach,
                              saved_args.get("num_buckets", args.num_buckets),
                              saved_args.get("max_len", args.max_len), id2label)

    for r, pred in zip(rows, preds):
        r["predicted_topic"] = pred

    if args.output_file:
        with open(args.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote predictions to {args.output_file}")
    else:
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
