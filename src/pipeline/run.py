"""Pipeline orchestration: extract -> transform -> fill -> engineer -> load.

The reasoning behind each decision is written up in notes.md. The text-embedding
and image-embedding stages live downstream (embed.py) — this module produces only
the clean, feature-engineered table they consume.
"""
import pandas as pd

from .config import EXCLUDED_COLUMNS
from .extract import load_raw
from .transform import transform, deduplicate
from .impute import impute
from .features import engineer, encode
from .load import save


def run() -> pd.DataFrame:
    df = load_raw()
    out = transform(df)
    out = deduplicate(out)
    out = impute(out)

    meta = {"rows": 0, "scalers": {}, "encoders": {}, "excluded_columns": EXCLUDED_COLUMNS}
    out = engineer(out, meta)
    out = encode(out, meta)
    meta["rows"] = len(out)

    save(out, meta)
    assert out.drop(columns=["primary_image_url"]).notna().all().all(), "unexpected nulls remain"
    print("no nulls remain (except optional primary_image_url) — pipeline OK")
    return out
