"""
download_data.py
-----------------
Small helper to fetch dataset_10.parquet from Google Drive.

Google Drive requires the "confirm" token dance for large files (>~100MB),
which is why we use the `gdown` package instead of a raw requests GET.

Usage:
    pip install gdown
    python src/download_data.py --file_id 1iib_mYxLcN6pNVHMpANeUCpe2qum0n-K \
        --out data/dataset_10.parquet

The file_id below is parsed directly from the shared link:
    https://drive.google.com/file/d/1iib_mYxLcN6pNVHMpANeUCpe2qum0n-K/view
"""

import argparse
import os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file_id", type=str, default="1iib_mYxLcN6pNVHMpANeUCpe2qum0n-K",
                    help="Google Drive file id (from the /d/<FILE_ID>/view URL).")
    p.add_argument("--out", type=str, default="data/dataset_10.parquet")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    try:
        import gdown
    except ImportError:
        raise SystemExit(
            "gdown is required for downloading from Google Drive.\n"
            "Install it with: pip install gdown"
        )

    url = f"https://drive.google.com/uc?id={args.file_id}"
    print(f"[download] fetching {url} -> {args.out}")
    gdown.download(url, args.out, quiet=False)
    print("[download] done.")


if __name__ == "__main__":
    main()
