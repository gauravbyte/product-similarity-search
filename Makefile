# Amazon fashion similarity search — pipeline tasks
# Override the interpreter with: make verify PY=python3
PY ?= .venv/bin/python

DATA    := data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
CLEAN   := artifacts/products_clean.parquet
VECTORS := artifacts/hybrid_vectors.npy

N  ?= 10        # number of similar products for `find-similar`
ID ?=           # product uniq_id for `find-similar` (blank = demo with first product)
PORT ?= 8000
IMAGE ?= fashion-similarity

.PHONY: help install etl embed verify find-similar benchmark serve docker-build docker-run all clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## install dependencies
	$(PY) -m pip install -r requirements.txt

# --- file rules: artifacts rebuild automatically when their inputs change ---
$(CLEAN): $(DATA) $(wildcard src/pipeline/*.py)
	$(PY) -m src.pipeline

$(VECTORS): $(CLEAN) src/pipeline/embed.py
	$(PY) -m src.pipeline.embed

etl: $(CLEAN)            ## run the ETL pipeline -> products_clean.parquet
embed: $(VECTORS)        ## build hybrid vectors -> hybrid_vectors.npy

verify: $(VECTORS)       ## smoke-test find_similar_products on the real data
	$(PY) -m scripts.verify

find-similar: $(VECTORS) ## similar products: make find-similar ID=<uniq_id> N=5
	$(PY) -m scripts.find_similar --id "$(ID)" --n $(N)

benchmark: $(CLEAN)      ## compare embedding approaches (tfidf / tfidf_svd / sbert)
	$(PY) -m src.similarity.benchmark

serve: $(VECTORS)        ## run the FastAPI service locally on $(PORT)
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port $(PORT)

docker-build:            ## build the API image (multi-stage; builds artifacts inside)
	docker build -t $(IMAGE) .

docker-run:              ## run the built image on $(PORT)
	docker run --rm -p $(PORT):8000 $(IMAGE)

all: verify              ## full pipeline end to end (builds artifacts then verifies)

clean:                   ## remove generated artifacts
	rm -f artifacts/*.parquet artifacts/*.npy artifacts/*.json
