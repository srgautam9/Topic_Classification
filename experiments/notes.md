# Experiment Notes (working log)

This file is for logs/configs/observations as runs happen against the real
`dataset_10.parquet`. Not the final report (report.pdf comes later per the
task instructions) — just scratch notes to make report-writing easier.

## Planned experiment sequence

1. **Baseline: `tfidf_lr`** (HashingVectorizer + SGDClassifier/log-loss,
   out-of-core via `partial_fit`). Cheapest to run, gives a fast sanity-check
   on label distribution, class balance, and roughly how separable the
   topics are lexically. Sets the bar the deep models need to beat.

2. **`fasttext`** (hashed n-gram EmbeddingBag + linear head). Expected to be
   close to or better than the linear baseline, much cheaper than CNN/BiLSTM,
   good candidate for the "efficient" requirement given 10M rows.

3. **`cnn`** (TextCNN over hashed token embeddings). Should pick up local
   n-gram/phrase patterns TF-IDF/FastText miss (word order within a window).

4. **`bilstm`**. Captures longer-range dependencies; most expensive to train
   at this scale, included to see whether topic classification (likely
   dominated by keyword presence rather than long-range structure) actually
   benefits from it, or whether it just costs more compute for similar
   accuracy vs. fasttext/cnn — a useful negative result either way.

## Things to record per run (fill in as experiments happen)

- `num_buckets`, `embed_dim`, `hidden_dim`, `max_len`, `batch_size`, `lr`, `epochs`
- Parameter count (must be < 5B, see model.num_parameters())
- Wall-clock time per epoch, hardware used
- Val accuracy / precision / recall / F1 (macro + weighted) per epoch
- Loss curve behavior (divergence, plateaus, overfitting)
- Any bucket-collision effects observed at smaller `num_buckets` (hashing
  trick trades memory for a small amount of collision noise -- worth an
  ablation over num_buckets to quantify)

## Known efficiency / scale decisions baked into the code (src/utils.py, src/data.py)

- Streaming parquet reads via `pyarrow.parquet.ParquetFile.iter_batches` --
  never loads the full 4GB file into memory.
- Label vocabulary built via a single columnar pass over just the TOPIC
  column (cached to `experiments/label2id.json` so repeated runs don't
  re-scan).
- Feature hashing (crc32-based, deterministic) instead of an explicit
  vocabulary for both the classical (HashingVectorizer) and deep models --
  avoids building/storing a vocab dict that could have tens of millions of
  entries at this scale, and makes inference vocab-independent (no OOV
  handling needed).
- Deterministic train/val split via `row_idx % 20 == 0` computed during the
  same streaming pass, so no separate shuffle/split pass over 10M rows is
  needed.
- Mixed precision (`torch.amp`) enabled automatically when a CUDA/ROCm
  device is available.
