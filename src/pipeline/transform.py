"""Whitelist + parse raw columns, build text blob, drop duplicate ASINs."""
import pandas as pd

from . import parsers


def build_text_blob(df: pd.DataFrame) -> pd.Series:
    name = df["product_name"].fillna("")
    keywords = df["meta_keywords"].fillna("")
    details = df["product_details__k_v_pairs"].apply(parsers.details_to_text)
    cat_path = df["parent___child_category__all"].apply(parsers.category_path)
    blob = name + " " + keywords + " " + details + " " + cat_path
    return blob.str.replace(r"\s+", " ", regex=True).str.strip()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["uniq_id"] = df["uniq_id"]
    out["asin"] = df["asin"]
    out["product_name"] = df["product_name"].fillna("")

    out["sales_price"] = parsers.parse_price(df["sales_price"])
    out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    out["discount_pct"] = parsers.parse_discount(df["discount_percentage"])
    out["child_rank"] = df["sales_rank_in_child_category"].apply(parsers.parse_rank)
    out["has_reviews"] = df["no__of_reviews"].notna().astype(int)

    out["brand"] = df["brand"]
    out["child_category"] = df["sales_rank_in_child_category"].apply(parsers.first_category)
    out["browsenode"] = df["browsenode"]
    out["is_prime"] = (df["amazon_prime__y_or_n"] == "Y").astype(int)
    out["is_fba"] = (df["delivery_type"] == "fulfilled_by_amazon").astype(int)
    out["is_best_seller"] = (df["best_seller_tag__y_or_n"] == "Y").astype(int)

    # colour is 80% missing but useful for NL query filtering — keep it
    out["colour"] = df["colour"].fillna("")

    out["text_blob"] = build_text_blob(df)
    out["primary_image_url"] = df["medium"].apply(parsers.first_image_url)
    return out


def deduplicate(out: pd.DataFrame) -> pd.DataFrame:
    before = len(out)
    out = out.drop_duplicates(subset="asin", keep="first").reset_index(drop=True)
    print(f"dropped {before - len(out):,} duplicate-ASIN rows -> {len(out):,}")
    return out
