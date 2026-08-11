# Topic Classification

Topic classifier that predicts TOPIC given the DATA, built entirely from scratch (no pretrained models, no finetuning), under the 5B parameter constraint, designed to scale to a 10M-row / 4GB parquet dataset without loading it fully into memory.

## Repository structure

```
project/
├── src/
│   ├── train.py       # unified training entrypoint for all approaches
│   ├── evaluate.py     # evaluation + error analysis on the val split
│   ├── inference.py   # unified inference entrypoint
│   ├── model.py        # from-scratch architectures (FastText/CNN/BiLSTM)
│   ├── data.py         # streaming IterableDatasets for the DL models
│   └── utils.py        # streaming I/O, text cleaning, feature hashing, metrics
├── scripts/
│   ├── make_synthetic_data.py   # generates a tiny synthetic parquet for smoke-testing
│   └── compare_models.py        # combines multiple metrics.json into one comparison table
├── experiments/
│   └── notes.md         # working experiment log (configs/observations)
├── eval_results/         # evaluation artifacts land here (metrics, confusion matrices, etc.)
├── final_models/        # trained checkpoints land here (.pt / .joblib)
├── report.pdf           
├── requirements.txt
└── README.md
```

## Approaches implemented

| approach   | family                        | notes |
|------------|--------------------------------|-------|
| `tfidf_lr` | Classical                     | `HashingVectorizer` (TF-IDF-style hashing) + `SGDClassifier(loss="log_loss")`, trained out-of-core with `partial_fit` over streamed chunks |
| `fasttext` | Deep learning, from scratch   | FastText-style hashed n-gram `EmbeddingBag` (mean pool) + linear head, reimplemented in PyTorch (not the `fasttext` library) |
| `cnn`      | Deep learning, from scratch   | Kim (2014)-style TextCNN over hashed token embeddings |
| `bilstm`   | Deep learning, from scratch   | Bidirectional LSTM classifier over hashed token embeddings |

All models consume **hashed** tokens/n-grams (feature-hashing trick) rather
than an explicit vocabulary. At 10M rows, the number of unique tokens/bigrams
can run into the tens of millions — hashing avoids building/storing that
vocabulary, needs only a single streaming pass, and keeps inference
vocab-independent (no OOV handling required). This trade-off (small chance of
hash collisions vs. large memory savings) is documented further in
`experiments/notes.md` and should be quantified with a `num_buckets` ablation
in the final report.

## a. Setup Instructions

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Works on CPU, NVIDIA CUDA, and AMD ROCm builds of PyTorch — device selection
in `src/train.py` uses `torch.cuda.is_available()`, which resolves correctly
for ROCm-built PyTorch as well (ROCm exposes itself through the same `cuda`
device namespace).

### Sanity-check the pipeline before the real run

The real dataset is 4GB / 10M rows; before pointing everything at it, verify
the whole pipeline works end-to-end on a tiny synthetic file:

```bash
python scripts/make_synthetic_data.py --n-rows 20000 --out synthetic_dataset.parquet
python -m src.train --approach fasttext --data synthetic_dataset.parquet --epochs 1
```

## b. Training Instructions

```bash
# Classical baseline (fast, good sanity check on class separability)
python -m src.train --approach tfidf_lr --data /path/to/dataset_10.parquet

# FastText-style (recommended first deep-learning run: cheapest, streams well)
python -m src.train --approach fasttext --data /path/to/dataset_10.parquet \
    --epochs 3 --batch-size 512 --num-buckets 2000000 --embed-dim 100

# TextCNN
python -m src.train --approach cnn --data /path/to/dataset_10.parquet \
    --epochs 3 --batch-size 256 --num-buckets 2000000 --embed-dim 128 --max-len 128

# BiLSTM
python -m src.train --approach bilstm --data /path/to/dataset_10.parquet \
    --epochs 3 --batch-size 256 --num-buckets 2000000 --embed-dim 128 --hidden-dim 128
```

Key flags (see `python -m src.train --help` for the full list):

- `--num-buckets`: size of the hashing feature space. Larger = fewer
  collisions but more parameters (`num_buckets * embed_dim` for the
  embedding table) — this is the main lever for the 5B parameter budget.
- `--batch-size`, `--epochs`, `--lr`, `--num-workers`
- `--output-dir` (default `final_models/`): checkpoints are saved every
  epoch (`{approach}_epoch{N}.pt`) plus a final one (`{approach}_final.pt`).
- `--label-cache` (default `experiments/label2id.json`): the label
  vocabulary is scanned once from the TOPIC column and cached so repeated
  runs don't re-scan all 10M rows.

Validation is a deterministic 5% holdout (every 20th row, by streaming
order) computed during the same pass — no separate shuffle/split pass over
the full dataset is needed. Random seeds are fixed via `--seed` (default 42)
across `random`, `numpy`, and `torch`.

Training prints running loss every `--log-every` steps and full
accuracy/precision/recall/F1 (macro + weighted) on the validation holdout
after every epoch.

## c. Inference Instructions

```bash
# Single string
python -m src.inference --model final_models/fasttext_final.pt --approach fasttext \
    --input "some text to classify"

# Batch from CSV
python -m src.inference --model final_models/fasttext_final.pt --approach fasttext \
    --input-file texts.csv --text-col DATA --output-file preds.csv

# Classical model
python -m src.inference --model final_models/tfidf_lr.joblib --approach tfidf_lr \
    --input "some text to classify"
```

`--num-buckets`/`--embed-dim`/`--hidden-dim`/`--max-len` passed at inference
time are only fallbacks — the actual values used at training time are saved
inside the checkpoint (`args`) and take precedence automatically.

## e. Evaluation and Error Analysis

`src/evaluate.py` streams the same deterministic validation split used
during training (every 20th row, `row_idx % 20 == 0`) and writes report-ready
artifacts for one model at a time:

```bash
python -m src.evaluate --approach fasttext --model final_models/fasttext_final.pt \
    --data /path/to/dataset_10.parquet --out-dir eval_results/fasttext

python -m src.evaluate --approach tfidf_lr --model final_models/tfidf_lr.joblib \
    --data /path/to/dataset_10.parquet --out-dir eval_results/tfidf_lr

python -m src.evaluate --approach cnn --model final_models/cnn_final.pt \
    --data /path/to/dataset_10.parquet --out-dir eval_results/cnn

python -m src.evaluate --approach bilstm --model final_models/bilstm_final.pt \
    --data /path/to/dataset_10.parquet --out-dir eval_results/bilstm
```

Each run writes to `--out-dir`:

- `metrics.json` — overall accuracy, precision/recall/F1 (macro + weighted),
  and a full per-class breakdown. This is the source for the **Evaluation
  Metrics** section of the report.
- `per_class_metrics.csv` — the same per-class numbers as a table, ready to
  paste in.
- `confusion_matrix.csv` / `confusion_matrix.png` — raw counts and a
  row-normalized heatmap.
- `top_confused_pairs.csv` — the (true, predicted) label pairs the model
  confuses most often, sorted by count — the starting point for "patterns in
  misclassification".
- `misclassified_samples.csv` — a handful of actual misclassified input
  texts per confused pair, for qualitative error analysis (what kind of text
  is being confused, and why).
- `length_error_analysis.csv` — error rate bucketed by input length in
  words, to check whether the model struggles more on very short or very
  long inputs.
- `confidence_analysis.csv` — mean/median predicted-class confidence, split
  by correct vs. incorrect predictions (deep models: softmax probability;
  classical model: `predict_proba` where available). A model that's
  similarly confident when wrong as when right is a useful thing to flag in
  the report.

`--max-val-rows` caps how many validation rows are evaluated, useful for a
quick check before committing to the full ~500k-row validation pass on the
real dataset.

Once you've run `evaluate.py` for each trained model, combine them into one
comparison table for the report's "final model selection reasoning" section:

```bash
python scripts/compare_models.py \
    --results tfidf_lr:eval_results/tfidf_lr/metrics.json \
              fasttext:eval_results/fasttext/metrics.json \
              cnn:eval_results/cnn/metrics.json \
              bilstm:eval_results/bilstm/metrics.json \
    --out eval_results/comparison.csv
```

## f. Input / Output Schema

**Training input:** a parquet file with (at least) two columns:

| column | type   | description        |
|--------|--------|---------------------|
| `DATA`  | string | input text          |
| `TOPIC` | string | target topic label  |

**Inference input:** either a single `--input "text"` string, or a
`--input-file` CSV with a text column (name configurable via `--text-col`,
default `DATA`).

**Inference output:** for `--input`, prints a dict with the original text and
`predicted_topic`. For `--input-file` with `--output-file`, writes a CSV that
is the input file plus an added `predicted_topic` column.

## Reproducibility

- All random seeds (`random`, `numpy`, `torch`, `PYTHONHASHSEED`) are fixed
  via `set_seed()` in `src/utils.py`, called at the start of `src/train.py`.
- Token/n-gram hashing uses a deterministic CRC32-based hash (not Python's
  built-in `hash()`, which is randomized per-process), so the same text
  always maps to the same features across training and inference runs/processes.
- Train/val split is a deterministic function of row order (`row_idx % 20`),
  not a random shuffle, so it's identical across runs given the same file.
- The full pipeline runs end-to-end with a single `python -m src.train ...`
  command and no manual intervention beyond pointing `--data` at the parquet
  file.

## Parameter budget

Every model exposes `.num_parameters()` and `src/train.py` asserts the count
is ≤ 5,000,000,000 before training starts. The dominant cost is the
embedding table (`num_buckets * embed_dim`); with the defaults above
(`num_buckets=2,000,000`, `embed_dim=100-128`) all four architectures are
comfortably in the low hundreds-of-millions range at most, leaving headroom
to scale `num_buckets` up if collision-related accuracy loss turns out to be
significant.
