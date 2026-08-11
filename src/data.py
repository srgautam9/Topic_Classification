"""
Streaming datasets built on pyarrow batch iteration, so the full 10M-row /
4GB file is never materialized in memory at once. Every 20th row (by
streaming order) is reserved as validation (see TRAIN_VAL_MOD) so the split
is deterministic and spread across the whole file without needing a shuffle
pass over all 10M rows.
"""
import torch
from torch.utils.data import IterableDataset

from .utils import clean_text, hashed_ngram_ids, tokenize, hash_token, stream_parquet_batches

TRAIN_VAL_MOD = 20  # 1/20 = 5% held out for validation


class HashedBagDataset(IterableDataset):
    """For EmbeddingBag-based models (FastText). Yields (list_of_ids, label_id)."""

    def __init__(self, path, label2id, num_buckets, ngram_range=(1, 2),
                 text_col="DATA", label_col="TOPIC", batch_size=50_000, split="train"):
        self.path = path
        self.label2id = label2id
        self.num_buckets = num_buckets
        self.ngram_range = ngram_range
        self.text_col = text_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.split = split

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid, nworkers = (0, 1)
        if worker_info is not None:
            wid, nworkers = worker_info.id, worker_info.num_workers
        row_idx = 0
        for batch in stream_parquet_batches(self.path, [self.text_col, self.label_col], self.batch_size):
            texts = batch.column(0).to_pylist()
            labels = batch.column(1).to_pylist()
            for text, label in zip(texts, labels):
                is_val = (row_idx % TRAIN_VAL_MOD == 0)
                keep_split = is_val if self.split == "val" else not is_val
                if keep_split and (row_idx % nworkers == wid) and label in self.label2id:
                    clean = clean_text(text)
                    ids = hashed_ngram_ids(clean, self.num_buckets, self.ngram_range)
                    if ids:
                        yield ids, self.label2id[label]
                row_idx += 1


class SequenceDataset(IterableDataset):
    """For CNN/BiLSTM. Yields (list_of_token_ids[<=max_len], label_id).
    Token id 0 is reserved as the padding id (see hash_token(...) + 1 below)."""

    def __init__(self, path, label2id, num_buckets, max_len=128,
                 text_col="DATA", label_col="TOPIC", batch_size=50_000, split="train"):
        self.path = path
        self.label2id = label2id
        self.num_buckets = num_buckets
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.split = split

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid, nworkers = (0, 1)
        if worker_info is not None:
            wid, nworkers = worker_info.id, worker_info.num_workers
        row_idx = 0
        for batch in stream_parquet_batches(self.path, [self.text_col, self.label_col], self.batch_size):
            texts = batch.column(0).to_pylist()
            labels = batch.column(1).to_pylist()
            for text, label in zip(texts, labels):
                is_val = (row_idx % TRAIN_VAL_MOD == 0)
                keep_split = is_val if self.split == "val" else not is_val
                if keep_split and (row_idx % nworkers == wid) and label in self.label2id:
                    clean = clean_text(text)
                    toks = tokenize(clean)[: self.max_len]
                    ids = [hash_token(t, self.num_buckets) + 1 for t in toks]
                    if ids:
                        yield ids, self.label2id[label]
                row_idx += 1


def bag_collate_fn(batch):
    """Builds the (flat_ids, offsets, labels) format nn.EmbeddingBag expects."""
    ids_list, labels = zip(*batch)
    lengths = [len(x) for x in ids_list]
    offsets = torch.tensor([0] + lengths[:-1]).cumsum(0)
    flat_ids = torch.tensor([i for ids in ids_list for i in ids], dtype=torch.long)
    return flat_ids, offsets, torch.tensor(labels, dtype=torch.long)


def seq_collate_fn(batch):
    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
    max_len = max(lengths).item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)
