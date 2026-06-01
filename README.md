# Amazon Fashion — Product Similarity Search

Submission for the SAP technical exercise. Given a product's `uniq_id`, return the
most similar products from a 30,000-row Amazon India fashion catalogue, exposed as
a REST API with a small demo UI.

The original task description is preserved in [ASSIGNMENT.md](ASSIGNMENT.md). My
reasoning and design decisions at each step are journaled in [notes.md](docs/notes.md).
The full system design and production-scaling write-up — with diagrams — is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Approach in one paragraph

I treat similarity as a **hybrid vector** problem. Each product becomes one vector =
`0.7 ×` a SBERT text embedding of its name/keywords/details/category + `0.3 ×` a small
block of scaled numeric features (price, rating, discount, rank, prime/FBA/bestseller
flags). Vectors are L2-normalised so cosine similarity is a single dot product, and
`find_similar_products` returns the top-N by cosine. An ETL pipeline produces a clean,
feature-engineered table first; the embedding step builds the vectors from it; the API
serves lookups over the pre-built index.

```
ldjson ──ETL──> products_clean.parquet ──embed──> hybrid_vectors.npy ──> FastAPI / find_similar_products
        (clean,                         (SBERT text +                    (cosine top-N)
         impute,                         structured,
         feature-eng)                    L2-normalised)
```

---

## Architecture

The system as built (MVP) and how it scales to multi-tenant production. Full detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

![Architecture — Three Scaling Approaches](docs/images/architecture-diagram.png)

*Three scaling approaches: A (MVP, ≤10 tenants) → B (shared catalogue, 10–50) → C (per-tenant Qdrant, 50+), with the migration path between them.*

---

## Quick start

```bash
make install          # install dependencies into the active environment
make all              # build artifacts (ETL + embeddings) and run the smoke test
make serve            # start the API + demo UI on http://localhost:8000
```

Then open **http://localhost:8000** for the demo (search a keyword → click a product →
see similar items with images), or **/docs** for the Swagger UI.

> The dataset is not committed. `data_download.sh` fetches it, or drop the `.ldjson`
> into `data/`. Generated artifacts live in `artifacts/` (gitignored, rebuilt by the pipeline).

### Other make targets
```
make find-similar ID=<uniq_id> N=5   # CLI: similar products for one id
make index                           # build the FAISS ANN index (Part 3)
make verify                          # smoke-test the engine (asserts self-excluded, unique, KeyError)
make verify-part4                    # smoke-test semantic_search + nl_query
make benchmark                       # compare embedding approaches (tfidf / tfidf_svd / sbert)
make docker-build / docker-run       # containerised API (multi-stage image)
make clean                           # remove generated artifacts
```

---

## API

| Endpoint | Description |
|----------|-------------|
| `GET /find_similar_products?product_id=&num_similar=` | the spec contract — returns `List[str]` of uniq_ids |
| `GET /similar?product_id=&num_similar=` | same ranking, but full product details incl. image (for the UI) |
| `GET /semantic_search?q=&num_results=` | Part 4 free-text semantic search using query-time text embeddings |
| `POST /nl_query` | Part 4 natural-language query: parse simple filters, then semantic search + filtering |
| `GET /search?q=&limit=` | find products by name keyword (discovery) |
| `GET /products/{product_id}` | one product's details |
| `GET /health` | service + index status |
| `GET /` | self-contained HTML demo (images) |

Error codes: `404` unknown product_id, `422` `num_similar` outside 1–50, `503` if the
index isn't built yet.

---

## Project structure

```
src/
├── pipeline/          ETL — one module per stage
│   ├── config.py        paths, dropped columns, embedding settings
│   ├── extract.py       read the ldjson
│   ├── parsers.py       pure helpers for the messy string/dict fields
│   ├── transform.py     whitelist + parse, build text_blob, dedup ASINs
│   ├── impute.py        data filling
│   ├── features.py      numeric engineering + categorical encoding
│   ├── embed.py         SBERT + structured -> hybrid vectors
│   └── run.py           orchestration  (python -m src.pipeline)
├── similarity/
│   ├── engine.py        find_similar_products (cosine over the hybrid vectors)
│   └── benchmark.py     compare tfidf / tfidf_svd / sbert on label proxies
└── api/
    ├── main.py          FastAPI app
    └── demo.html        the demo UI
scripts/                 verify.py, find_similar.py (CLI)
EDA_Amazon_Marketing_Data.ipynb   exploratory analysis
```

---

## Design decisions (short version)

- **Hybrid text + structured vector.** Product text carries the strongest similarity
  signal; numeric features refine it without dominating. Weighted 0.7/0.3, tunable.
- **SBERT (`all-MiniLM-L6-v2`).** I benchmarked TF-IDF, TF-IDF+SVD and SBERT on
  label-based proxies (`make benchmark`) — they tie on lexical overlap, but SBERT
  generalises to varied wording/synonyms, which production text needs. TF-IDF stays a
  documented fallback.
- **Categoricals via text, not numeric codes.** A `brand_code` of 5 vs 6 isn't "closer";
  brand/category names live in the text blob, so the embedding handles them.
- **FAISS IVFFlat ANN** (Part 3) for sub-linear search — `nlist`/`nprobe` tuned by a
  recall-vs-latency sweep to ~7.7× faster than brute force at 0.99 recall@10. The engine
  falls back to exact brute-force cosine if FAISS isn't present, so `find_similar_products`
  always works; `/health` reports the active backend.
- **AI query layer (Part 4).** `/semantic_search` embeds free-text queries with the same
  text embedding model used for the catalogue. `/nl_query` keeps a small parser boundary
  where an LLM can plug in, and uses local parsing for simple price/brand filters so the
  service remains runnable offline.
- **Serving image.** The runtime loads pre-built vectors/FAISS for product-id similarity;
  Part 4 also includes sentence-transformers for query-time semantic search.
- **Data filling:** price → category-median then global fallback; discount → 0 (no
  discount is a real state); brand/category → `"unknown"`. Full rationale in `docs/notes.md`.

---

## Status & roadmap

| Part | Status |
|------|--------|
| EDA + data understanding | Done |
| ETL pipeline (clean / impute / feature-engineer) | Done |
| Part 1 — `find_similar_products` | Done |
| Part 2 — FastAPI service + Docker | Done |
| Embedding evaluation (3 approaches) | Done |
| Part 3 (bonus) — FAISS ANN index | Done — IVFFlat (tuned nlist=400/nprobe=50), ~7.7× faster at 0.99 recall@10; brute-force fallback |
| Part 4 — semantic search + LLM-ready NL-query endpoint + scaling write-up | Done — `/semantic_search`, `/nl_query`, local parser boundary, architecture doc |
| Multimodal (bonus) — image embeddings | Planned — pipeline carries `primary_image_url` for this |

---

## Tech stack

pandas · numpy · pyarrow · scikit-learn · sentence-transformers (SBERT) · FastAPI · uvicorn · Docker
