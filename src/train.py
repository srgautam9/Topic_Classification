"""
Unified training entrypoint for all approaches.

Examples
--------
Classical (out-of-core TF-IDF-hashing + Logistic Regression):
    python -m src.train --approach tfidf_lr --data /path/to/dataset_10.parquet

FastText-style (hashed n-gram embeddings + linear head):
    python -m src.train --approach fasttext --data /path/to/dataset_10.parquet \
        --epochs 3 --batch-size 512 --num-buckets 2000000 --embed-dim 100

TextCNN:
    python -m src.train --approach cnn --data /path/to/dataset_10.parquet

BiLSTM:
    python -m src.train --approach bilstm --data /path/to/dataset_10.parquet
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import set_seed, build_label_vocab, compute_metrics, clean_text, \
    tokenize, hash_token, hashed_ngram_ids, stream_parquet_batches, count_rows
from .model import build_model, BiLSTMScratch
from .data import HashedBagDataset, SequenceDataset, bag_collate_fn, seq_collate_fn, TRAIN_VAL_MOD


def get_device():
    if torch.cuda.is_available():  # also True for ROCm builds of torch
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Classical, out-of-core: HashingVectorizer (stateless -> no fit pass needed,
# streaming-friendly for 10M rows) + SGDClassifier with partial_fit (log loss
# => online Logistic Regression). This is the "TF-IDF / Logistic Regression /
# FastText"-family classical baseline.
# ---------------------------------------------------------------------------
def train_tfidf_lr(args, label2id):
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    import joblib

    vectorizer = HashingVectorizer(
        n_features=args.num_buckets, alternate_sign=False,
        ngram_range=(1, 2), norm="l2",
    )
    clf = SGDClassifier(loss="log_loss", random_state=args.seed, learning_rate="optimal")
    classes = np.array(sorted(label2id.values()))

    val_texts, val_labels = [], []
    n_seen, t0 = 0, time.time()
    for epoch in range(args.epochs):
        row_idx = 0
        for batch in stream_parquet_batches(args.data, [args.text_col, args.label_col], args.batch_size):
            texts = batch.column(0).to_pylist()
            labels = batch.column(1).to_pylist()
            train_texts, train_labels = [], []
            for text, label in zip(texts, labels):
                if label not in label2id:
                    row_idx += 1
                    continue
                is_val = (row_idx % TRAIN_VAL_MOD == 0)
                cleaned = clean_text(text)
                if is_val:
                    if epoch == 0:
                        val_texts.append(cleaned)
                        val_labels.append(label2id[label])
                else:
                    train_texts.append(cleaned)
                    train_labels.append(label2id[label])
                row_idx += 1
            if train_texts:
                X = vectorizer.transform(train_texts)
                y = np.array(train_labels)
                clf.partial_fit(X, y, classes=classes)
                n_seen += len(train_texts)
            if n_seen and (n_seen // args.batch_size) % 20 == 0:
                print(f"[tfidf_lr] epoch {epoch} seen {n_seen} rows "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)
        if val_texts:
            Xv = vectorizer.transform(val_texts)
            preds = clf.predict(Xv)
            m = compute_metrics(val_labels, preds)
            print(f"[tfidf_lr] epoch {epoch} val: {m}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "tfidf_lr.joblib")
    joblib.dump({"vectorizer_params": vectorizer.get_params(), "clf": clf,
                 "label2id": label2id}, out_path)
    print(f"Saved classical model to {out_path}")


# ---------------------------------------------------------------------------
# Deep learning approaches (fasttext / cnn / bilstm), all trained from scratch.
# ---------------------------------------------------------------------------
def make_val_holdout(args, label2id, max_rows=20_000):
    """Materializes a small, fixed validation set in memory (cheap: capped at
    max_rows) by taking every TRAIN_VAL_MOD-th row during a single stream."""
    texts, labels = [], []
    row_idx = 0
    for batch in stream_parquet_batches(args.data, [args.text_col, args.label_col], args.batch_size):
        for text, label in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if row_idx % TRAIN_VAL_MOD == 0 and label in label2id:
                texts.append(text)
                labels.append(label2id[label])
                if len(texts) >= max_rows:
                    return texts, labels
            row_idx += 1
    return texts, labels


def evaluate_bag(model, texts, labels, num_buckets, ngram_range, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            chunk = texts[i:i + 512]
            ids_list = [hashed_ngram_ids(clean_text(t), num_buckets, ngram_range) or [0] for t in chunk]
            lengths = [len(x) for x in ids_list]
            offsets = torch.tensor([0] + lengths[:-1]).cumsum(0).to(device)
            flat = torch.tensor([i for ids in ids_list for i in ids], dtype=torch.long).to(device)
            logits = model(flat, offsets)
            preds.extend(logits.argmax(1).cpu().tolist())
    return compute_metrics(labels, preds)


def evaluate_seq(model, texts, labels, num_buckets, max_len, device, approach):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            chunk = texts[i:i + 512]
            seqs = [[hash_token(t, num_buckets) + 1 for t in tokenize(clean_text(x))[:max_len]] or [0]
                    for x in chunk]
            lengths = torch.tensor([len(s) for s in seqs])
            mx = max(lengths).item()
            padded = torch.zeros(len(seqs), mx, dtype=torch.long)
            for j, s in enumerate(seqs):
                padded[j, :len(s)] = torch.tensor(s)
            padded, lengths = padded.to(device), lengths.to(device)
            logits = model(padded, lengths) if approach == "bilstm" else model(padded)
            preds.extend(logits.argmax(1).cpu().tolist())
    return compute_metrics(labels, preds)


def train_deep(args, label2id):
    device = get_device()
    print(f"Using device: {device}", flush=True)
    num_classes = len(label2id)

    if args.approach == "fasttext":
        model = build_model("fasttext", args.num_buckets, num_classes, embed_dim=args.embed_dim)
        ds = HashedBagDataset(args.data, label2id, args.num_buckets, batch_size=args.batch_size)
        collate = bag_collate_fn
    else:
        model = build_model(args.approach, args.num_buckets, num_classes,
                             embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
        ds = SequenceDataset(args.data, label2id, args.num_buckets, max_len=args.max_len,
                              batch_size=args.batch_size)
        collate = seq_collate_fn

    n_params = model.num_parameters()
    print(f"Model parameter count: {n_params:,}", flush=True)
    assert n_params <= 5_000_000_000, "Exceeds 5B parameter constraint!"

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()

    val_texts, val_labels = make_val_holdout(args, label2id)

    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate,
                         num_workers=args.num_workers)

    step, t0 = 0, time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if args.approach == "fasttext":
                    flat_ids, offsets, labels = batch
                    flat_ids, offsets, labels = flat_ids.to(device), offsets.to(device), labels.to(device)
                    logits = model(flat_ids, offsets)
                elif args.approach == "cnn":
                    ids, lengths, labels = batch
                    ids, labels = ids.to(device), labels.to(device)
                    logits = model(ids)
                else:  # bilstm
                    ids, lengths, labels = batch
                    ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
                    logits = model(ids, lengths)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"[{args.approach}] epoch {epoch} step {step} "
                      f"loss {running_loss / args.log_every:.4f} "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)
                running_loss = 0.0

        if args.approach == "fasttext":
            m = evaluate_bag(model, val_texts, val_labels, args.num_buckets, (1, 2), device)
        else:
            m = evaluate_seq(model, val_texts, val_labels, args.num_buckets, args.max_len,
                              device, args.approach)
        print(f"[{args.approach}] epoch {epoch} val metrics: {m}", flush=True)

        ckpt_path = os.path.join(args.output_dir, f"{args.approach}_epoch{epoch}.pt")
        torch.save({"model_state": model.state_dict(), "label2id": label2id,
                    "args": vars(args)}, ckpt_path)

    final_path = os.path.join(args.output_dir, f"{args.approach}_final.pt")
    torch.save({"model_state": model.state_dict(), "label2id": label2id,
                "args": vars(args)}, final_path)
    print(f"Saved final model to {final_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--approach", required=True,
                    choices=["tfidf_lr", "fasttext", "cnn", "bilstm"])
    p.add_argument("--data", required=True, help="path to dataset_10.parquet")
    p.add_argument("--text-col", default="DATA")
    p.add_argument("--label-col", default="TOPIC")
    p.add_argument("--output-dir", default="final_models")
    p.add_argument("--num-buckets", type=int, default=1_000_000)
    p.add_argument("--embed-dim", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-cache", default="experiments/label2id.json")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"Dataset rows: {count_rows(args.data):,}")
    label2id = build_label_vocab(args.data, args.label_col, cache_path=args.label_cache)
    print(f"Num classes: {len(label2id)}")

    if args.approach == "tfidf_lr":
        train_tfidf_lr(args, label2id)
    else:
        train_deep(args, label2id)


if __name__ == "__main__":
    main()
