"""FAISS IVFFlat index — build, save, load."""
import faiss
import numpy as np

from ..pipeline.config import FAISS_INDEX_PATH, FAISS_NLIST, FAISS_NPROBE


def build_index(vectors: np.ndarray):
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
    if not path.exists():
        return None
    index = faiss.read_index(str(path))
    index.nprobe = FAISS_NPROBE
    return index


def run() -> None:
    from ..pipeline.config import VECTORS_PATH
    vectors = np.load(VECTORS_PATH)
    save_index(build_index(vectors))
    print(f"wrote {FAISS_INDEX_PATH}  (IVFFlat over {vectors.shape[0]:,} x {vectors.shape[1]})")


if __name__ == "__main__":
    run()
