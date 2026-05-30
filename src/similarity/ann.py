"""FAISS approximate-nearest-neighbour index (Part 3 bonus).

Brute-force cosine is exact but scans every row (O(N) per query). For larger
catalogues we want sub-linear search, so we build an IVF (inverted-file) index:
k-means partitions the vectors into FAISS_NLIST cells, and a query only scans the
FAISS_NPROBE nearest cells instead of all N — trading a little recall for speed.

Vectors are unit-norm, so inner product == cosine and we use METRIC_INNER_PRODUCT.

Reference: Douze et al., "The Faiss library", 2024 (arXiv:2401.08281) — frames
vector search as a speed/recall/memory trade-off; IVF buys speed by probing
fewer cells, and IVF+PQ adds compression when memory is the bottleneck at scale.
"""
import faiss
import numpy as np

from ..pipeline.config import FAISS_INDEX_PATH, FAISS_NLIST, FAISS_NPROBE


def build_index(vectors: np.ndarray):
    """Train + populate an IVFFlat index over the (unit-norm) hybrid vectors."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    d = vectors.shape[1]
    # nlist can't exceed the number of training points
    nlist = min(FAISS_NLIST, max(1, len(vectors) // 40))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = FAISS_NPROBE
    return index


def save_index(index, path=FAISS_INDEX_PATH) -> None:
    faiss.write_index(index, str(path))


def load_index(path=FAISS_INDEX_PATH):
    """Load the index, or return None if it hasn't been built."""
    if not path.exists():
        return None
    index = faiss.read_index(str(path))
    index.nprobe = FAISS_NPROBE
    return index


def run() -> None:
    """Build the FAISS index from the saved vectors. Run as its own process
    (python -m src.similarity.ann) — never alongside torch, see embed.py."""
    from ..pipeline.config import VECTORS_PATH
    vectors = np.load(VECTORS_PATH)
    save_index(build_index(vectors))
    print(f"wrote {FAISS_INDEX_PATH}  (IVFFlat over {vectors.shape[0]:,} x {vectors.shape[1]})")


if __name__ == "__main__":
    run()
