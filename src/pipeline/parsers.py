"""Pure helpers that turn the dataset's messy string / dict fields into clean
scalars or text. No DataFrame state — each takes one raw value and returns one
clean value, which keeps them easy to test and reuse.
"""
import re

import numpy as np
import pandas as pd

from .config import DETAIL_SKIP_KEYS


def parse_price(s: pd.Series) -> pd.Series:
    """'₹1,299' -> 1299.0"""
    cleaned = s.astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_discount(s: pd.Series) -> pd.Series:
    """'40%' -> 40.0"""
    cleaned = s.astype(str).str.rstrip("%").replace({"None": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def first_category(d) -> "str | float":
    """sales_rank_in_child_category {'WomensKurtasKurtis': '#1793'} -> category name"""
    return next(iter(d)) if isinstance(d, dict) else np.nan


def parse_rank(d) -> float:
    """{'WomensKurtasKurtis': '#1,793'} -> 1793"""
    if not isinstance(d, dict):
        return np.nan
    digits = re.sub(r"[^\d]", "", str(next(iter(d.values()))))
    return int(digits) if digits else np.nan


def category_path(d) -> str:
    """parent___child_category__all keys are the hierarchy -> joined text"""
    return " ".join(d.keys()) if isinstance(d, dict) else ""


def details_to_text(d) -> str:
    """flatten the product-detail dict to 'key: value | ...', skipping noise keys"""
    if not isinstance(d, dict):
        return ""
    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in d.items() if k not in DETAIL_SKIP_KEYS]
    return " | ".join(parts)


def first_image_url(s) -> "str | float":
    """pipe-separated URL list -> first URL (kept for the image-embedding stage)"""
    return str(s).split("|")[0] if pd.notna(s) else np.nan
