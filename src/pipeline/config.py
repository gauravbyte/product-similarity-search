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

# --- Embedding / similarity (Phase 3) --------------------------------------
SBERT_MODEL = "all-MiniLM-L6-v2"        # 384-dim, fast on CPU
TEXT_WEIGHT = 0.7                        # text vs structured blend in the hybrid vector
STRUCT_WEIGHT = 0.3

# The structured half of the hybrid vector: continuous + binary features only.
# Categoricals (brand/category) are NOT here — their label codes aren't valid
# cosine distances; their names live in text_blob, so the embedding handles them.
STRUCTURED_FEATURES = [
    "price_norm", "rating_norm", "discount_norm", "rank_norm",
    "is_prime", "is_fba", "is_best_seller", "has_reviews",
]

CLEAN_PATH = OUT_DIR / "products_clean.parquet"
VECTORS_PATH = OUT_DIR / "hybrid_vectors.npy"
ID_MAP_PATH = OUT_DIR / "id_map.json"

FAISS_INDEX_PATH = OUT_DIR / "faiss.index"
# Tuned by sweeping nlist x nprobe (see notes.md): fastest point that still
# holds recall@10 >= 0.99 (~0.99 recall, ~7.7x faster than brute force).
FAISS_NLIST = 400      # Voronoi cells (k-means partitions of the vector space)
FAISS_NPROBE = 50      # cells probed per query: recall/speed knob
