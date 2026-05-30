"""FastAPI service for the product similarity search (Part 2).

Core:
  GET /find_similar_products?product_id=&num_similar=   -> List[str]   (the spec contract)
  GET /health

Demo-friendly (return product details incl. image, for a UI):
  GET /                       -> HTML demo page (search -> similar, with images)
  GET /search?q=&limit=       -> products whose name matches a keyword
  GET /products/{product_id}  -> one product's details
  GET /similar?product_id=&num_similar=  -> { query, results } with details

The index loads once at startup and is reused. Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..similarity.engine import SimilarityIndex

MAX_RESULTS = 50
DEMO_PAGE = Path(__file__).parent / "demo.html"

state: dict = {"index": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["index"] = SimilarityIndex()
    except FileNotFoundError:
        state["index"] = None      # endpoints return 503 until artifacts exist
    yield
    state["index"] = None


app = FastAPI(title="Amazon Fashion Similarity Search", version="1.1", lifespan=lifespan)
# open CORS so a static/hosted frontend can call the API freely (demo deploy)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def _require_index() -> SimilarityIndex:
    index = state["index"]
    if index is None:
        raise HTTPException(status_code=503, detail="index not loaded — build artifacts with `make all`")
    return index


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(DEMO_PAGE)


@app.get("/health")
def health() -> dict:
    index = state["index"]
    return {"status": "ok" if index else "unavailable",
            "index_size": len(index.records) if index else 0,
            "backend": index.backend if index else None}


@app.get("/find_similar_products")
def find_similar_products(
    product_id: str = Query(..., description="uniq_id of the query product"),
    num_similar: int = Query(10, ge=1, le=MAX_RESULTS, description="number of results (1-50)"),
) -> List[str]:
    index = _require_index()
    try:
        return index.query(product_id, num_similar)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"product_id not found: {product_id}")


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="keyword to match in product names"),
    limit: int = Query(10, ge=1, le=MAX_RESULTS),
) -> List[dict]:
    return _require_index().search(q, limit)


@app.get("/products/{product_id}")
def get_product(product_id: str) -> dict:
    record = _require_index().record(product_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"product_id not found: {product_id}")
    return record


@app.get("/similar")
def similar(
    product_id: str = Query(..., description="uniq_id of the query product"),
    num_similar: int = Query(10, ge=1, le=MAX_RESULTS),
) -> dict:
    """Like /find_similar_products but returns full product details (for the UI)."""
    index = _require_index()
    try:
        ids = index.query(product_id, num_similar)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"product_id not found: {product_id}")
    return {"query": index.record(product_id), "results": [index.record(i) for i in ids]}
