# EDA Thoughts — Amazon Fashion Similarity Search

---

## First look at the data

30k rows, 33 columns. Amazon India fashion products, Feb–Apr 2020. Loaded fine as ldjson.

First thing I noticed — a lot of columns are basically empty. `name_of_author_for_books` has 1 row filled. `formats___editions` has 2. These are obviously book-specific fields that crept into the fashion dataset schema. Drop immediately, no thought needed.

`weight` looked useful but every single row is `999999999` — that's a sentinel for "not provided". Zero real values after cleaning. Drop.

`colour` is only 20% filled. Would've been great for fashion similarity but can't use it reliably across 80% of the catalog. Defer it.

---

## The sparse ones I need to think about

`no__of_reviews` is only 11.5% filled. My instinct was to impute but that felt wrong — a fake review count on 88% of products would just add noise. Better to flag presence: `has_reviews = 1/0`. Keep the actual count only when it exists.

`seller_id`, `seller_name` — 27.9% fill. But more importantly, the seller is a marketplace concept, not a product attribute. Two identical products from different sellers shouldn't be dissimilar. Drop both.

`discount_percentage` — 48.7% fill. Here null doesn't mean unknown, it means the product has no discount. So fill with 0. That's a real signal.

`brand` — 72.9% fill, so 8k+ products have no brand. Can't drop it, brand matters for similarity. Fill nulls with `"unknown"` as a real category.

---

## Price surprised me

Median ₹590 but mean ₹862 — right-skewed heavily (skew ≈ 6.7). Log transform fixes it almost completely.

I expected FBA (Fulfilled by Amazon) products to be pricier. Actually the median prices are nearly identical: FBA ₹579, FBM ₹590.5. Not a useful price signal.

But Prime vs non-Prime is a very different story — non-Prime median ₹399, Prime median ₹699. That's a 75% premium. Prime is actually a much better price-tier proxy than delivery type.

---

## Rating is basically useless as a differentiator

Mean 4.04, most products cluster at discrete values (3.0, 3.5, 4.0, 4.5, 5.0). That's because most products have very few reviews — rating rounds to a single decimal. No correlation with price (r ≈ 0.02), no correlation with discount. It still goes in the feature vector but I don't expect it to pull much weight.

---

## The correlation matrix told me one useful thing

`child_rank` and `parent_rank` correlate at r=0.459. That's strong enough that including both is redundant. Drop `parent_rank`, keep `child_rank` (more specific to the sub-category).

Discount and price have r=-0.198 — makes sense, high-end products discount less. Both are still independently informative, just noting they're not independent.

---

## What text fields do I actually have?

`product_name` — 100% fill, primary signal.
`meta_keywords` — 100% fill, adds style/category terms.
`product_details__k_v_pairs` — 96.1% fill and this is gold. It's a dict with things like fabric type, fit, sleeve length, pattern. Parse it into a flat text blob.
`parent___child_category__all` — 85% fill, gives the full category hierarchy path like "Fashion > Women > Ethnic Wear > Sarees". Treat as text, not a categorical.

Plan: concatenate all of these into one `text_blob` per product → SBERT encode → 384-dim dense vector.

Model choice: `all-MiniLM-L6-v2`. Fast on CPU, good for short product descriptions. If Indian product names become a problem, `paraphrase-multilingual-mpnet-base-v2` is the fallback.

---

## Structured features alongside the text

After the text embedding I'll concatenate a small structured vector:
- `log1p_price`, `rating_norm`, `discount_norm` (fill 0 for no discount)
- `is_prime`, `is_fba`, `is_best_seller` (binary)
- `child_category`, `brand`, `browsenode` (label-encoded)
- `log1p_child_rank`, `has_reviews`

That's ~12 dims. Start with 70% weight on text / 30% on structured. Tune later by inspecting whether similar results feel right qualitatively.

---

## What about duplicates?

Same `asin` can appear with different `uniq_id` — re-crawls. Need to check in phase 2 and deduplicate by keeping the most recent `crawl_timestamp`.

---

## Things I'm deferring consciously

- `colour`: would help but 80% missing kills it for a global model
- `other_items_customers_buy`: co-purchase graph signal, better suited for collaborative filtering phase, not content similarity
- Image embeddings (CLIP/ResNet): image URLs are there but downloading + embedding 30k images is a separate pipeline step

---

## Questions I asked myself and answered

**Should I use TF-IDF or SBERT?**
TF-IDF is faster to build and interpretable, good as a sanity check. But it misses semantic similarity — "kurta" and "kurti" won't match. SBERT handles this. Build TF-IDF as a baseline, SBERT as the real thing.

**FAISS vs Annoy vs HNSW?**
For 30k products even brute force cosine is fast enough (<10ms). But I'll use FAISS IVFFlat anyway — zero API change to scale up later, GPU-upgradeable, and it's what I know best. nlist=100, nprobe=10 as starting point.

**Why cosine over euclidean?**
Embeddings live in high-dimensional space. L2 distance degrades there (curse of dimensionality). Cosine is magnitude-invariant — a cheap version of a product and an expensive version with the same description should still score as similar. SBERT is also explicitly trained for cosine.

---

*Phase 1 done. No data changes yet — all observations. Phase 2 is the cleaning script.*

---
---

# Phase 2 Thoughts — Building the ETL Pipeline

Turning the EDA decisions into code (`src/pipeline/`). Why it looks the way it does:

- **Package, not one script.** One module per stage (`extract → transform → impute → features → load`, wired in `run.py`). Parsers are pure functions in their own file, easy to test. Runs as `python -m src.pipeline`.
- **Whitelist, don't drop.** `transform()` picks the columns I want rather than dropping junk — a new junk column in a future dump is ignored by default.
- **Dedup correction.** EDA said "keep most recent `crawl_timestamp`", but I dropped that column, so `drop_duplicates(keep="first")`. The crawl is one short window, so dup ASINs are near-identical anyway. 30,000 → 29,529.
- **Data filling:** price → category median then global fallback (category-median alone leaves ~850 rows where every product is unpriced); discount → 0; brand/category → `"unknown"`; browsenode → -1; rating/rank → median.
- **Features:** `log1p` price + rank to compress skew; min-max scale to [0,1]; label-encode (not one-hot — brand is ~6.5k unique). I persist scaler params + label vocabularies to `feature_meta.json` so Phase 3 encodes a query product *identically* — otherwise distances are meaningless.
- **Parquet over CSV** — preserves dtypes, columnar, smaller. Costs one dep (`pyarrow`).
- **Multimodal-ready:** `text_blob` feeds the text embedder; `primary_image_url` is carried through untouched so the image stage attaches later with no cleaning changes. Embedding/FAISS is *not* in this phase — ETL only.

*Phase 2 done — `python -m src.pipeline` → `products_clean.parquet` (29,529 × 25) + `feature_meta.json`. Next: `embed.py`.*
