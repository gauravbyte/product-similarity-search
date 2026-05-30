"""FastAPI service — product similarity search + NL query layer."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # prevent fork+OpenMP segfault on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import MAX_RESULTS, NL_CANDIDATES
from .nlq import parse_query
from ..similarity.engine import SimilarityIndex
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class NLQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    num_results: int = Field(10, ge=1, le=MAX_RESULTS)


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
            "backend": index.backend if index else None,
            "query_parser": "llm+local"}


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


@app.get("/semantic_search")
def semantic_search(
    q: str = Query(..., min_length=1, description="free-text product query"),
    num_results: int = Query(10, ge=1, le=MAX_RESULTS),
) -> dict:
    results = _require_index().semantic_search(q, num_results)
    return {"query": q, "results": results}


@app.post("/nl_query")
def nl_query(request: NLQueryRequest) -> dict:
    index = _require_index()
    parsed, parser = parse_query(request.query)
    search_text = parsed["free_text"] or request.query
    if parsed.get("colour"):
        search_text = f"{parsed['colour']} {search_text}"
    candidates = index.semantic_search(
        search_text,
        max(NL_CANDIDATES, request.num_results * 10),
    )
    results = _apply_filters(candidates, parsed)[:request.num_results]
    return {
        "query": request.query,
        "parsed": parsed,
        "parser": parser,
        "results": results,
    }


@app.get("/products/{product_id}")
def get_product(product_id: str) -> dict:
    record = _require_index().record(product_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"product_id not found: {product_id}")
    return record


def _apply_filters(records: List[dict], parsed: dict) -> List[dict]:
    out = []
    for row in records:
        price = float(row.get("sales_price") or 0)
        if parsed.get("min_price") is not None and price < parsed["min_price"]:
            continue
        if parsed.get("max_price") is not None and price > parsed["max_price"]:
            continue
        if parsed.get("brand") and parsed["brand"].lower() not in str(row.get("brand", "")).lower():
            continue
        if parsed.get("category") and parsed["category"].lower() not in str(row.get("child_category", "")).lower():
            continue
        if parsed.get("colour"):
            c = parsed["colour"].lower()
            name = str(row.get("product_name", "")).lower()
            colour_field = str(row.get("colour", "")).lower()
            if c not in name and c not in colour_field:
                continue
        out.append(row)
    return out


@app.get("/similar")
def similar(
    product_id: str = Query(..., description="uniq_id of the query product"),
    num_similar: int = Query(10, ge=1, le=MAX_RESULTS),
) -> dict:
    index = _require_index()
    try:
        results = index.query_records(product_id, num_similar)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"product_id not found: {product_id}")
    return {"query": index.record(product_id), "results": results}
