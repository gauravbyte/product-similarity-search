# Multi-stage build.
#  stage 1 (builder): full deps incl. torch/SBERT -> generate the artifacts
#  stage 2 (runtime): lean image, no torch -> just serve the API
# The serving path only loads the pre-built vectors and does cosine with numpy,
# so the final image stays small.

# ---- builder: produce artifacts/products_clean.parquet + hybrid_vectors.npy ----
FROM python:3.10-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
# each step is its own process — the FAISS build must not share a process with torch
RUN python -m src.pipeline && python -m src.pipeline.embed && python -m src.similarity.ann

# ---- runtime: serve find_similar_products over HTTP ----
FROM python:3.10-slim
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY --from=builder /app/artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
