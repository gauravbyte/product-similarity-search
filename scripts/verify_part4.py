"""Smoke-test the Part 4 semantic and natural-language query endpoints."""
from fastapi.testclient import TestClient

from src.api.main import app


def main() -> None:
    with TestClient(app) as client:
        semantic = client.get("/semantic_search", params={"q": "cotton saree", "num_results": 3})
        assert semantic.status_code == 200, semantic.text
        semantic_results = semantic.json()["results"]
        assert len(semantic_results) == 3
        assert all("uniq_id" in row and "score" in row for row in semantic_results)

        assert client.get("/semantic_search", params={"q": "", "num_results": 3}).status_code == 422
        assert client.get("/semantic_search", params={"q": "saree", "num_results": 51}).status_code == 422

        nl = client.post("/nl_query", json={"query": "cotton kurtas under 500", "num_results": 5})
        assert nl.status_code == 200, nl.text
        body = nl.json()
        assert body["parser"] in ("llm", "local")
        assert body["parsed"]["max_price"] == 500.0
        assert all(row["sales_price"] <= 500 for row in body["results"])

    print("PASS - Part 4 semantic_search and nl_query smoke checks passed.")


if __name__ == "__main__":
    main()
