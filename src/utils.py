"""
Utility functions: data streaming, text cleaning, feature hashing, label vocab,
metrics, seeding.

Design note on scale: the dataset is 10M rows / 4GB. Everything here is written
to work in a single streaming pass over the parquet file using
pyarrow.parquet.ParquetFile.iter_batches, so the full file never needs to be
materialized in memory.
"""
import os
import re
import zlib
import random
import json

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, strip URLs/HTML/punctuation, collapse whitespace."""
    if text is None:
        return ""
    t = str(text).lower()
    t = _URL_RE.sub(" ", t)
    t = _HTML_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str):
    return text.split()


def hash_token(token: str, num_buckets: int, seed: int = 0) -> int:
    """Deterministic hash -> bucket id. Uses crc32 (not builtin hash(), which
    is randomized per-process unless PYTHONHASHSEED is fixed) so that the same
    token always maps to the same bucket across training and inference runs."""
    h = zlib.crc32((str(seed) + "_" + token).encode("utf-8"))
    return h % num_buckets


def hashed_ngram_ids(text: str, num_buckets: int, ngram_range=(1, 2), seed: int = 0):
    """FastText-style hashing trick: word unigrams (+ bigrams) hashed into a
    fixed bucket space of size num_buckets.

    Why hashing instead of an explicit vocabulary: with 10M rows the number of
    unique tokens/bigrams can run into the tens of millions, which is
    expensive to build and store as a dict. Hashing gives O(1) memory feature
    extraction with a single streaming pass and no separate vocab-building
    pass over the data -- important for the "efficient" requirement at this
    scale.
    """
    tokens = tokenize(text)
    ids = []
    if ngram_range[0] <= 1 <= ngram_range[1]:
        ids.extend(hash_token(t, num_buckets, seed) for t in tokens)
    if ngram_range[1] >= 2 and len(tokens) > 1:
        ids.extend(
            hash_token(tokens[i] + "_" + tokens[i + 1], num_buckets, seed)
            for i in range(len(tokens) - 1)
        )
    return ids


def build_label_vocab(path, label_col="TOPIC", cache_path=None):
    """Single cheap pass reading ONLY the label column (columnar parquet read,
    so this does not touch the much larger text column) to build a sorted
    label -> id mapping. Cached to disk so repeated runs / restarts don't
    re-scan all 10M rows just for this."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    pf = pq.ParquetFile(path)
    labels = set()
    for batch in pf.iter_batches(columns=[label_col], batch_size=200_000):
        col = batch.column(0)
        labels.update(x.as_py() for x in col if x.is_valid)
    label2id = {lbl: i for i, lbl in enumerate(sorted(labels))}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(label2id, f)
    return label2id


def stream_parquet_batches(path, columns, batch_size=50_000):
    """Yields pyarrow RecordBatches without loading the full file into memory."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        yield batch


def compute_metrics(y_true, y_pred, average_list=("macro", "weighted")):
    acc = accuracy_score(y_true, y_pred)
    out = {"accuracy": acc}
    for avg in average_list:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        out[f"precision_{avg}"] = p
        out[f"recall_{avg}"] = r
        out[f"f1_{avg}"] = f1
    return out


def count_rows(path):
    return pq.ParquetFile(path).metadata.num_rows
