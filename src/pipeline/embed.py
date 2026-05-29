"""Build the hybrid product vectors used for similarity search.

  text half       SBERT embedding of text_blob (semantic signal)
  structured half scaled numeric + binary features

Each half is L2-normalised, weighted (TEXT_WEIGHT / STRUCT_WEIGHT) and
concatenated. Because the weights sum to 1 the rows come out unit-norm, so
cosine similarity is just a dot product downstream.

Outputs (artifacts/):
  hybrid_vectors.npy   (N x D float32)
  id_map.json          uniq_id + product_name per row (order matches the matrix)

Usage:  python -m src.pipeline.embed     (run after the clean pipeline)
"""
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
    # text half — semantic embedding of the product text
    model = SentenceTransformer(SBERT_MODEL)
    text = model.encode(
        df["text_blob"].tolist(), batch_size=256,
        show_progress_bar=True, convert_to_numpy=True,
    )
    text = _l2_normalise(text.astype("float32"))

    # structured half — continuous + binary features
    struct = _l2_normalise(df[STRUCTURED_FEATURES].to_numpy(dtype="float32"))

    # weighted concat; rows are unit-norm so dot product == cosine
    hybrid = np.hstack([np.sqrt(TEXT_WEIGHT) * text, np.sqrt(STRUCT_WEIGHT) * struct])
    return _l2_normalise(hybrid)


def run() -> np.ndarray:
    df = pd.read_parquet(CLEAN_PATH)
    print(f"encoding {len(df):,} products with {SBERT_MODEL} ...")
    vectors = build_vectors(df)

    np.save(VECTORS_PATH, vectors)
    id_map = df[["uniq_id", "product_name"]].to_dict("records")
    ID_MAP_PATH.write_text(json.dumps(id_map))
    print(f"wrote {VECTORS_PATH}  ({vectors.shape[0]:,} x {vectors.shape[1]})")
    print(f"wrote {ID_MAP_PATH}")
    return vectors


if __name__ == "__main__":
    run()
