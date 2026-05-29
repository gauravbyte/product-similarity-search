"""Paths and constants for the ETL pipeline."""
from pathlib import Path

DATA_PATH = Path(
    "data/marketing_sample_for_amazon_com-amazon_fashion_products"
    "__20200201_20200430__30k_data.ldjson"
)
OUT_DIR = Path("artifacts")

# Columns left behind — no usable signal for similarity (rationale in notes.md).
# We whitelist what we need in transform.py, so this list documents what we drop.
EXCLUDED_COLUMNS = [
    "name_of_author_for_books", "formats___editions",   # book fields, ~0% filled
    "weight",                                            # mostly sentinel 999999999
    "colour",                                            # 80% missing
    "technical_details__k_v_pairs",                      # 96% missing
    "no__of_sellers", "no__of_offers", "left_in_stock",  # marketplace / inventory noise
    "seller_id", "seller_name",                          # seller, not product
    "crawl_timestamp", "product_url",                    # not features
]

# Keys inside product_details__k_v_pairs that are noise for a text embedding.
DETAIL_SKIP_KEYS = {
    "ASIN", "Customer_Reviews", "Amazon_Bestsellers_Rank",
    "Date_first_available_at_Amazon_in", "Item_part_number",
}
