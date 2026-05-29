"""Compare text-embedding approaches for product similarity.

There's no labelled relevance data, so we use two label-based proxies: a good
representation should retrieve neighbours that share the query product's
*category* and *brand*, and sit at a similar *price*. We report those over a
random sample of query products.

Approaches compared (all on the same text_blob, scored by cosine):
  tfidf       TF-IDF, word 1-2 grams           — lexical baseline, sparse
  tfidf_svd   TF-IDF -> TruncatedSVD (LSA)     — latent semantic, dense
  sbert       SBERT all-MiniLM-L6-v2 (384-dim) — the production choice, dense

Usage:
  python -m src.similarity.benchmark                       # defaults
  python -m src.similarity.benchmark --sample 1000 --k 20
  python -m src.similarity.benchmark --approaches tfidf sbert
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from ..pipeline.config import CLEAN_PATH, SBERT_MODEL

DEFAULTS = {"sample": 500, "k": 10, "seed": 42, "tfidf_features": 20000, "svd_components": 200}


# --- build each text representation (L2-normalised rows -> cosine = dot) ----
def build_tfidf(texts, cfg):
    X = TfidfVectorizer(ngram_range=(1, 2), max_features=cfg["tfidf_features"],
                        min_df=2, stop_words="english").fit_transform(texts)
    return normalize(X), True            # sparse


def build_tfidf_svd(texts, cfg):
    X = TfidfVectorizer(ngram_range=(1, 2), max_features=cfg["tfidf_features"],
                        min_df=2, stop_words="english").fit_transform(texts)
    svd = TruncatedSVD(n_components=cfg["svd_components"], random_state=cfg["seed"])
    return normalize(svd.fit_transform(X)).astype("float32"), False   # dense


def build_sbert(texts, cfg):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["model"])
    emb = model.encode(texts, batch_size=256, convert_to_numpy=True)
    return normalize(emb).astype("float32"), False   # dense


BUILDERS = {"tfidf": build_tfidf, "tfidf_svd": build_tfidf_svd, "sbert": build_sbert}


# --- nearest neighbours + proxy metrics ------------------------------------
def _topk(mat, q, k, sparse):
    scores = (mat @ mat[q].T).toarray().ravel() if sparse else mat @ mat[q]
    scores[q] = -np.inf
    return np.argpartition(-scores, k)[:k]


def evaluate(mat, sparse, cat, brand, price, queries, k):
    cat_hits, brand_hits, price_diff = [], [], []
    for q in queries:
        nbrs = _topk(mat, q, k, sparse)
        if cat[q] != "unknown":
            cat_hits.append(np.mean(cat[nbrs] == cat[q]))
        if brand[q] != "unknown":
            brand_hits.append(np.mean(brand[nbrs] == brand[q]))
        price_diff.append(np.median(np.abs(price[nbrs] - price[q])) / max(price[q], 1) * 100)
    return {
        f"category_match@{k}": np.mean(cat_hits),
        f"brand_match@{k}": np.mean(brand_hits),
        "price_diff_%_median": np.median(price_diff),
    }


def run(approaches=None, sample=None, k=None, seed=None,
        tfidf_features=None, svd_components=None, model=None, clean_path=CLEAN_PATH):
    cfg = {
        "sample": sample or DEFAULTS["sample"],
        "k": k or DEFAULTS["k"],
        "seed": seed or DEFAULTS["seed"],
        "tfidf_features": tfidf_features or DEFAULTS["tfidf_features"],
        "svd_components": svd_components or DEFAULTS["svd_components"],
        "model": model or SBERT_MODEL,
    }
    approaches = approaches or list(BUILDERS)

    df = pd.read_parquet(clean_path)
    texts = df["text_blob"].tolist()
    cat, brand, price = (df[c].to_numpy() for c in ("child_category", "brand", "sales_price"))

    rng = np.random.default_rng(cfg["seed"])
    queries = rng.choice(len(df), size=min(cfg["sample"], len(df)), replace=False)

    rows = []
    for name in approaches:
        t = time.time()
        mat, sparse = BUILDERS[name](texts, cfg)
        metrics = evaluate(mat, sparse, cat, brand, price, queries, cfg["k"])
        metrics["build_s"] = round(time.time() - t, 1)
        rows.append({"approach": name, **metrics})
        print(f"  {name:10s} done in {metrics['build_s']}s")

    report = pd.DataFrame(rows).set_index("approach").round(3)
    print(f"\nProxy metrics over {len(queries)} random queries (k={cfg['k']}):")
    print(report)
    print("\nHigher category/brand match = better; lower price-diff = neighbours closer in price.")
    return report


def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark text-embedding approaches.")
    p.add_argument("--approaches", nargs="+", choices=list(BUILDERS), default=list(BUILDERS))
    p.add_argument("--sample", type=int, default=DEFAULTS["sample"], help="query products to evaluate")
    p.add_argument("--k", type=int, default=DEFAULTS["k"], help="neighbours per query")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--tfidf-features", type=int, default=DEFAULTS["tfidf_features"])
    p.add_argument("--svd-components", type=int, default=DEFAULTS["svd_components"])
    p.add_argument("--model", default=SBERT_MODEL, help="SBERT model name")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(approaches=a.approaches, sample=a.sample, k=a.k, seed=a.seed,
        tfidf_features=a.tfidf_features, svd_components=a.svd_components, model=a.model)
