"""Product similarity engine — loads pre-built vectors, queries by cosine.
Uses FAISS if available, falls back to brute-force.
"""
import json
from typing import List

import numpy as np

from ..pipeline.config import (
    VECTORS_PATH, ID_MAP_PATH, SBERT_MODEL, STRUCTURED_FEATURES,
)


class SimilarityIndex:
    """In-memory cosine index over the hybrid vectors (loaded once).

    `records` holds the per-product display fields (name, brand, price, image)
    so the API can serve details without touching pandas/parquet at runtime.
    """

    def __init__(self):
        self.vectors = np.load(VECTORS_PATH)
        self.records = json.loads(ID_MAP_PATH.read_text())
        self.position = {row["uniq_id"]: i for i, row in enumerate(self.records)}
        self.ann = self._load_ann()
        self.backend = "faiss" if self.ann is not None else "brute-force"
        self._semantic_model = None

    @staticmethod
    def _load_ann():
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

    def query_records(self, product_id: str, num_similar: int) -> List[dict]:
        if product_id not in self.position:
            raise KeyError(f"product_id not found: {product_id}")
        i = self.position[product_id]
        pairs = self._query_ann_pairs(i, num_similar) if self.ann is not None \
            else self._query_bruteforce_pairs(i, num_similar)
        return [self._record_with_score(j, score) for j, score in pairs]

    def _query_ann(self, i: int, num_similar: int) -> List[str]:
        return [self.records[j]["uniq_id"] for j, _ in self._query_ann_pairs(i, num_similar)]

    def _query_bruteforce(self, i: int, num_similar: int) -> List[str]:
        return [self.records[j]["uniq_id"] for j, _ in self._query_bruteforce_pairs(i, num_similar)]

    def _query_ann_pairs(self, i: int, num_similar: int) -> List[tuple[int, float]]:
        k = min(num_similar + 1, len(self.records))
        scores, idx = self.ann.search(self.vectors[i:i + 1], k)
        hits = [(int(j), float(s)) for s, j in zip(scores[0], idx[0]) if j != i and j != -1]
        return hits[:num_similar]

    def _query_bruteforce_pairs(self, i: int, num_similar: int) -> List[tuple[int, float]]:
        scores = self.vectors @ self.vectors[i]   # cosine against every product
        scores[i] = -np.inf                        # never return the product itself
        n = min(num_similar, len(scores) - 1)
        top = np.argpartition(-scores, n)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(j), float(scores[j])) for j in top]

    def semantic_search(self, text: str, limit: int) -> List[dict]:
        q = self._semantic_vector(text)
        if self.ann is not None:
            k = min(limit, len(self.records))
            scores, idx = self.ann.search(q, k)
            pairs = [(int(j), float(s)) for s, j in zip(scores[0], idx[0]) if j != -1]
        else:
            scores = self.vectors @ q[0]
            k = min(limit, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            pairs = [(int(j), float(scores[j])) for j in top]
        return [self._record_with_score(j, score) for j, score in pairs[:limit]]

    def _semantic_vector(self, text: str) -> np.ndarray:
        if self._semantic_model is None:
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer(SBERT_MODEL)
        embedding = self._semantic_model.encode([text], convert_to_numpy=True)
        embedding = self._l2_normalise(embedding.astype("float32"))
        q = np.zeros((1, self.vectors.shape[1]), dtype="float32")
        text_dims = self.vectors.shape[1] - len(STRUCTURED_FEATURES)
        q[:, :text_dims] = embedding[:, :text_dims]
        return np.ascontiguousarray(self._l2_normalise(q), dtype="float32")

    @staticmethod
    def _l2_normalise(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.clip(norms, 1e-9, None)

    def _record_with_score(self, i: int, score: float) -> dict:
        row = dict(self.records[i])
        row["score"] = round(score, 6)
        return row

    def record(self, product_id: str) -> "dict | None":
        """Display fields for one product (None if unknown)."""
        i = self.position.get(product_id)
        return self.records[i] if i is not None else None

    def search(self, text: str, limit: int) -> List[dict]:
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
