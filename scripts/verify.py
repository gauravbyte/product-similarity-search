"""Smoke-test the similarity engine on the real artifacts.

Run:  make verify        (or: python -m scripts.verify)

Checks that find_similar_products returns sensible, self-excluding, unique
results and errors correctly on an unknown id. Exits non-zero on any failure.
"""
import sys

import pandas as pd

from src.pipeline.config import CLEAN_PATH, VECTORS_PATH
from src.similarity.engine import find_similar_products


def main() -> None:
    if not (CLEAN_PATH.exists() and VECTORS_PATH.exists()):
        sys.exit("artifacts missing — run `make etl embed` first")

    df = pd.read_parquet(CLEAN_PATH).set_index("uniq_id")
    pid = df.index[0]
    n = 5

    results = find_similar_products(pid, n)

    q = df.loc[pid]
    print(f"QUERY: {q['product_name'][:70]}  [{q['child_category']}, ₹{q['sales_price']:.0f}]")
    for r in results:
        s = df.loc[r]
        print(f"  -> {s['product_name'][:66]}  [{s['child_category']}, ₹{s['sales_price']:.0f}]")

    assert len(results) == n, f"expected {n} results, got {len(results)}"
    assert pid not in results, "query product returned itself"
    assert len(set(results)) == n, "duplicate ids in results"
    try:
        find_similar_products("does-not-exist", n)
        sys.exit("FAIL: expected KeyError for unknown id")
    except KeyError:
        pass

    print(f"\nPASS — {n} unique, self-excluded results; KeyError on unknown id.")


if __name__ == "__main__":
    main()
