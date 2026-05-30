"""find_similar_products — the Part 1 deliverable (+ Part 3 FAISS bonus).

Loads the pre-built hybrid vectors (see src/pipeline/embed.py) and returns the
most similar products by cosine similarity. Vectors are unit-norm, so cosine is a
dot product. If a FAISS index is present we use it for sub-linear ANN search
(Part 3); otherwise we fall back to an exact brute-force scan.

    from src.similarity.engine import find_similar_products
    find_similar_products("26d41bdc1495de290bc8e6062d927729", 5)
"""
import json
from typing import List

import numpy as np

from ..pipeline.config import VECTORS_PATH, ID_MAP_PATH


class SimilarityIndex:
    """In-memory similarity index over the hybrid vectors (loaded once).

    `records` holds the per-product display fields (name, brand, price, image)
    so the API can serve details without touching pandas/parquet at runtime.
    Uses FAISS when its index exists, else exact brute-force cosine.
    """

    def __init__(self):
        self.vectors = np.load(VECTORS_PATH)
        self.records = json.loads(ID_MAP_PATH.read_text())
        self.position = {row["uniq_id"]: i for i, row in enumerate(self.records)}
        self.ann = self._load_ann()
        self.backend = "faiss" if self.ann is not None else "brute-force"

    @staticmethod
    def _load_ann():
        """Load the FAISS index, or None if faiss/index is unavailable."""
        try:
            from .ann import load_index
            return load_index()
        except Exception:
            return None

    def query(self, product_id: str, num_similar: int) -> List[str]:
        if product_id not in self.position:
            raise KeyError(f"product_id not found: {product_id}")
        i = self.position[product_id]
        return self._query_ann(i, num_similar) if self.ann is not None \
            else self._query_bruteforce(i, num_similar)

    def _query_ann(self, i: int, num_similar: int) -> List[str]:
        # ask for one extra so we can drop the product itself
        k = min(num_similar + 1, len(self.records))
        _, idx = self.ann.search(self.vectors[i:i + 1], k)
        hits = [j for j in idx[0] if j != i and j != -1]
        return [self.records[j]["uniq_id"] for j in hits[:num_similar]]

    def _query_bruteforce(self, i: int, num_similar: int) -> List[str]:
        scores = self.vectors @ self.vectors[i]   # cosine against every product
        scores[i] = -np.inf                        # never return the product itself
        n = min(num_similar, len(scores) - 1)
        top = np.argpartition(-scores, n)[:n]
        top = top[np.argsort(-scores[top])]
        return [self.records[j]["uniq_id"] for j in top]

    def record(self, product_id: str) -> "dict | None":
        """Display fields for one product (None if unknown)."""
        i = self.position.get(product_id)
        return self.records[i] if i is not None else None

    def search(self, text: str, limit: int) -> List[dict]:
        """Substring match on product_name — lets a demo discover product_ids."""
        t = text.lower()
        hits = [r for r in self.records if t in r["product_name"].lower()]
        return hits[:limit]


_index: "SimilarityIndex | None" = None


def _get_index() -> SimilarityIndex:
    """Lazily load the index on first use, then reuse it."""
    global _index
    if _index is None:
        _index = SimilarityIndex()
    return _index


def find_similar_products(product_id: str, num_similar: int = 10) -> List[str]:
    """Return up to `num_similar` uniq_ids most similar to `product_id`."""
    return _get_index().query(product_id, num_similar)
