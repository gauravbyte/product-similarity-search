"""Paths and constants shared across the pipeline."""
from pathlib import Path

DATA_PATH = Path(
    "data/marketing_sample_for_amazon_com-amazon_fashion_products"
    "__20200201_20200430__30k_data.ldjson"
)
OUT_DIR = Path("artifacts")

EXCLUDED_COLUMNS = [
    "name_of_author_for_books", "formats___editions",   # book fields, ~0% filled
    "weight",                                            # mostly sentinel 999999999
    # colour — kept for NL query filtering (matched against product_name + colour field)
    "technical_details__k_v_pairs",                      # 96% missing
    "no__of_sellers", "no__of_offers", "left_in_stock",  # marketplace / inventory noise
    "seller_id", "seller_name",                          # seller, not product
    "crawl_timestamp", "product_url",                    # not features
]

DETAIL_SKIP_KEYS = {
    "ASIN", "Customer_Reviews", "Amazon_Bestsellers_Rank",
    "Date_first_available_at_Amazon_in", "Item_part_number",
}

SBERT_MODEL = "all-MiniLM-L6-v2"
TEXT_WEIGHT = 0.7
STRUCT_WEIGHT = 0.3

# label-encoded categoricals excluded — codes have no cosine-valid distance
STRUCTURED_FEATURES = [
    "price_norm", "rating_norm", "discount_norm", "rank_norm",
    "is_prime", "is_fba", "is_best_seller", "has_reviews",
]

CLEAN_PATH = OUT_DIR / "products_clean.parquet"
VECTORS_PATH = OUT_DIR / "hybrid_vectors.npy"
ID_MAP_PATH = OUT_DIR / "id_map.json"

FAISS_INDEX_PATH = OUT_DIR / "faiss.index"
# nlist/nprobe tuned by recall@10 sweep — see notes.md
FAISS_NLIST = 400
FAISS_NPROBE = 50