"""Feature-engineering step: derive + scale numeric features and label-encode
categoricals. Scaler params and encoder vocabularies are recorded in `meta` so
a query product can be encoded identically later (Phase 3).
"""
import numpy as np
import pandas as pd

# numeric feature -> the column it is scaled from
_SCALE_MAP = {
    "price_norm": "log1p_price",
    "rating_norm": "rating",
    "discount_norm": "discount_pct",
    "rank_norm": "log1p_child_rank",
}

_ENCODE_COLUMNS = ["brand", "child_category", "browsenode"]


def engineer(out: pd.DataFrame, meta: dict) -> pd.DataFrame:
    # log-compress the heavily right-skewed fields
    out["log1p_price"] = np.log1p(out["sales_price"])
    out["log1p_child_rank"] = np.log1p(out["child_rank"])

    # min-max scale numeric features to [0,1]; remember params for reuse
    for dst, src in _SCALE_MAP.items():
        lo, hi = float(out[src].min()), float(out[src].max())
        out[dst] = (out[src] - lo) / (hi - lo) if hi > lo else 0.0
        meta["scalers"][dst] = {"src": src, "min": lo, "max": hi}
    return out


def encode(out: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Label-encode the categoricals (one-hot is impractical: ~6.5k brands)."""
    for col in _ENCODE_COLUMNS:
        codes, uniques = pd.factorize(out[col])
        out[col + "_code"] = codes
        meta["encoders"][col] = [str(u) for u in uniques.tolist()]
    return out
