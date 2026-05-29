"""Data-filling step: replace missing values with sensible defaults."""
import pandas as pd


def impute(out: pd.DataFrame) -> pd.DataFrame:
    # price: category median first, then global median for categories with none
    cat_median = out.groupby("child_category")["sales_price"].transform("median")
    out["sales_price"] = out["sales_price"].fillna(cat_median)
    out["sales_price"] = out["sales_price"].fillna(out["sales_price"].median())

    out["discount_pct"] = out["discount_pct"].fillna(0.0)      # missing = no discount
    out["rating"] = out["rating"].fillna(out["rating"].median())
    out["child_rank"] = out["child_rank"].fillna(out["child_rank"].median())

    out["brand"] = out["brand"].fillna("unknown")              # explicit class
    out["child_category"] = out["child_category"].fillna("unknown")
    out["browsenode"] = out["browsenode"].fillna(-1)
    return out
