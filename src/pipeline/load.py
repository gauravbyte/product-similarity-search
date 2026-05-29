"""Load step: persist the clean table and feature metadata to artifacts/."""
import json
from pathlib import Path

import pandas as pd

from .config import OUT_DIR


def save(out: pd.DataFrame, meta: dict, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(exist_ok=True)
    out.to_parquet(out_dir / "products_clean.parquet", index=False)
    (out_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir / 'products_clean.parquet'}  ({out.shape[0]:,} x {out.shape[1]})")
    print(f"wrote {out_dir / 'feature_meta.json'}")
