"""CLI wrapper around find_similar_products.

  make find-similar ID=<uniq_id> N=5
  python -m scripts.find_similar --id <uniq_id> --n 5

With no id, it uses the first product as a demo so the command works out of the box.
"""
import argparse

import pandas as pd

from src.pipeline.config import CLEAN_PATH, VECTORS_PATH
from src.similarity.engine import find_similar_products


def main() -> None:
    ap = argparse.ArgumentParser(description="Find products similar to a given uniq_id.")
    ap.add_argument("--id", default="", help="product uniq_id (default: first product)")
    ap.add_argument("--n", type=int, default=10, help="number of similar products")
    args = ap.parse_args()

    if not (CLEAN_PATH.exists() and VECTORS_PATH.exists()):
        raise SystemExit("artifacts missing — run `make etl embed` first")

    df = pd.read_parquet(CLEAN_PATH).set_index("uniq_id")
    pid = args.id or df.index[0]
    if not args.id:
        print(f"(no ID given — demoing with the first product: {pid})")
    if pid not in df.index:
        raise SystemExit(f"unknown product_id: {pid}")

    q = df.loc[pid]
    print(f"\nQUERY: {q['product_name']}")
    print(f"       {q['brand']} | {q['child_category']} | ₹{q['sales_price']:.0f} | rating {q['rating']}\n")
    for rank, rid in enumerate(find_similar_products(pid, args.n), 1):
        s = df.loc[rid]
        print(f"{rank:2d}. {s['product_name'][:70]}")
        print(f"     {rid} | {s['brand']} | {s['child_category']} | ₹{s['sales_price']:.0f}")


if __name__ == "__main__":
    main()
