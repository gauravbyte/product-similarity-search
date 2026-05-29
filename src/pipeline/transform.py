"""Transform step: whitelist + parse the raw columns into a tidy table, build
the text blob for semantic search, and drop duplicate products.
"""
import pandas as pd

from . import parsers


def build_text_blob(df: pd.DataFrame) -> pd.Series:
    """Single text field the semantic-search embedder will consume."""
    name = df["product_name"].fillna("")
    keywords = df["meta_keywords"].fillna("")
    details = df["product_details__k_v_pairs"].apply(parsers.details_to_text)
    cat_path = df["parent___child_category__all"].apply(parsers.category_path)
    blob = name + " " + keywords + " " + details + " " + cat_path
    return blob.str.replace(r"\s+", " ", regex=True).str.strip()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Whitelist the columns we need and parse them (no imputation yet)."""
    out = pd.DataFrame()

    # identity / display
    out["uniq_id"] = df["uniq_id"]
    out["asin"] = df["asin"]
    out["product_name"] = df["product_name"].fillna("")

    # numeric (parsed from messy strings / dicts)
    out["sales_price"] = parsers.parse_price(df["sales_price"])
    out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    out["discount_pct"] = parsers.parse_discount(df["discount_percentage"])
    out["child_rank"] = df["sales_rank_in_child_category"].apply(parsers.parse_rank)
    out["has_reviews"] = df["no__of_reviews"].notna().astype(int)

    # categorical
    out["brand"] = df["brand"]
    out["child_category"] = df["sales_rank_in_child_category"].apply(parsers.first_category)
    out["browsenode"] = df["browsenode"]
    out["is_prime"] = (df["amazon_prime__y_or_n"] == "Y").astype(int)
    out["is_fba"] = (df["delivery_type"] == "fulfilled_by_amazon").astype(int)
    out["is_best_seller"] = (df["best_seller_tag__y_or_n"] == "Y").astype(int)

    # text + image (image carried for the multimodal bonus)
    out["text_blob"] = build_text_blob(df)
    out["primary_image_url"] = df["medium"].apply(parsers.first_image_url)
    return out


def deduplicate(out: pd.DataFrame) -> pd.DataFrame:
    """Same ASIN re-crawled -> keep one row so results never repeat a product."""
    before = len(out)
    out = out.drop_duplicates(subset="asin", keep="first").reset_index(drop=True)
    print(f"dropped {before - len(out):,} duplicate-ASIN rows -> {len(out):,}")
    return out
