"""Extract step: read the raw product catalogue."""
from pathlib import Path

import pandas as pd

from .config import DATA_PATH


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    print(f"loaded {len(df):,} rows x {df.shape[1]} cols")
    return df
