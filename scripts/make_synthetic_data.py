"""
Generates a small synthetic dataset_10-like parquet file (DATA, TOPIC columns)
purely for smoke-testing the pipeline end to end before running on the real
4GB / 10M-row file. Not part of the graded deliverable -- a dev/testing aid.
"""
import argparse
import random

import pyarrow as pa
import pyarrow.parquet as pq

TOPIC_VOCAB = {
    "sports": ["match", "goal", "team", "player", "score", "tournament", "coach", "stadium"],
    "politics": ["election", "government", "minister", "policy", "vote", "parliament", "law"],
    "technology": ["software", "computer", "algorithm", "internet", "device", "startup", "app"],
    "business": ["market", "stock", "company", "revenue", "investment", "trade", "profit"],
    "health": ["hospital", "doctor", "disease", "treatment", "medicine", "patient", "vaccine"],
}

FILLER = ["the", "a", "is", "was", "and", "in", "of", "to", "for", "on", "with", "this", "that"]


def make_row(rng):
    topic = rng.choice(list(TOPIC_VOCAB.keys()))
    words = rng.choices(TOPIC_VOCAB[topic], k=rng.randint(4, 8)) + rng.choices(FILLER, k=rng.randint(3, 6))
    rng.shuffle(words)
    text = " ".join(words)
    return text, topic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=20000)
    p.add_argument("--out", default="synthetic_dataset.parquet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    data, topics = [], []
    for _ in range(args.n_rows):
        t, topic = make_row(rng)
        data.append(t)
        topics.append(topic)

    table = pa.table({"DATA": data, "TOPIC": topics})
    pq.write_table(table, args.out)
    print(f"Wrote {args.n_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
