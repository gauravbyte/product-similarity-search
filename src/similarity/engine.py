"""find_similar_products — the Part 1 deliverable.

Loads the pre-built hybrid vectors (see src/pipeline/embed.py) and returns the
most similar products by cosine similarity. Vectors are unit-norm, so cosine is
just a dot product: one matrix-vector multiply scores the whole catalogue.

    from src.similarity.engine import find_similar_products
    find_similar_products("26d41bdc1495de290bc8e6062d927729", 5)
"""
import json
from typing import List

import numpy as np

from ..pipeline.config import VECTORS_PATH, ID_MAP_PATH


class SimilarityIndex:
    """In-memory cosine index over the hybrid vectors (loaded once)."""

    def __init__(self):
        self.vectors = np.load(VECTORS_PATH)
        self.id_map = json.loads(ID_MAP_PATH.read_text())
        self.position = {row["uniq_id"]: i for i, row in enumerate(self.id_map)}

    def query(self, product_id: str, num_similar: int) -> List[str]:
        if product_id not in self.position:
            raise KeyError(f"product_id not found: {product_id}")

        i = self.position[product_id]
        scores = self.vectors @ self.vectors[i]   # cosine against every product
        scores[i] = -np.inf                        # never return the product itself

        # top-N by score (partial sort, then order the small slice)
        n = min(num_similar, len(scores) - 1)
        top = np.argpartition(-scores, n)[:n]
        top = top[np.argsort(-scores[top])]
        return [self.id_map[j]["uniq_id"] for j in top]


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
