"""Build hybrid product vectors (SBERT text + structured features) for similarity search."""
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import (
    CLEAN_PATH, VECTORS_PATH, ID_MAP_PATH,
    SBERT_MODEL, TEXT_WEIGHT, STRUCT_WEIGHT, STRUCTURED_FEATURES,
)


def _l2_normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-9, None)


def build_vectors(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(SBERT_MODEL)
    text = model.encode(
        df["text_blob"].tolist(), batch_size=256,
        show_progress_bar=True, convert_to_numpy=True,
    )
    text = _l2_normalise(text.astype("float32"))
    struct = _l2_normalise(df[STRUCTURED_FEATURES].to_numpy(dtype="float32"))
    hybrid = np.hstack([np.sqrt(TEXT_WEIGHT) * text, np.sqrt(STRUCT_WEIGHT) * struct])
    return _l2_normalise(hybrid)


def run() -> np.ndarray:
    df = pd.read_parquet(CLEAN_PATH)
    print(f"encoding {len(df):,} products with {SBERT_MODEL} ...")
    vectors = build_vectors(df)

    np.save(VECTORS_PATH, vectors)
    display = df[["uniq_id", "product_name", "brand", "child_category",
                  "sales_price", "colour", "primary_image_url"]].copy()
    display["sales_price"] = display["sales_price"].round(0)
    display.to_json(ID_MAP_PATH, orient="records")
    print(f"wrote {VECTORS_PATH}  ({vectors.shape[0]:,} x {vectors.shape[1]})")
    print(f"wrote {ID_MAP_PATH}")
    # NB: the FAISS index is built in a *separate* step (python -m src.similarity.ann)
    # — faiss and torch both link libomp, so building it here would segfault.
    return vectors


if __name__ == "__main__":
    run()
