# sap-cxii-tech-ex-01
Technical Exercise for SAP. 

## What is the goal? 
You are tasked with implementing a product similarity search feature using the given dataset. Given a product’s `unique ID` (uniq_id), your goal is to retrieve a list of `similar products` based on `specific attributes`.

## Part 1: Similarity Search Function

Write a function called `find_similar_products` that takes in the following parameters:

```python
def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    pass
```

The function should return a list of num_similar product IDs that are most similar to the given product_id. 
The similarity between products should be determined based on the following attributes: `brand, sales_price, weight, and rating`.
You should consider the following requirements while implementing the function:
 - The function should first retrieve the attributes `(brand, sales_price, weight, rating)` of the product identified by the given `product_id`.
 - The function should then compare these attributes with all other products in the dataset.
 - Calculate a similarity score between the given product and each other product using a suitable similarity measure `(e.g., Euclidean distance, cosine similarity, etc.)`.
 - Sort the products based on their similarity scores in descending order.
 - Return a `list of num_similar product IDs` that are most similar to the given `product_id`.

### Here is the sample code

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Assume df is already loaded with the dataset
## You could write a btter data loader for this as well, and find an alternative to pandas.
df = pd.read_json('data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson', lines=True)

def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    pass

## This is a sample stub you can chose the datastructures you deem important.
def calculate_similarity():
    pass
```

## Note:
  - If multiple products have the same similarity score, the tie-breaker can be based on any attribute (e.g., sales_price or rating).

- ### Assumptions: 
  - You can assume that the given `product_id` will exist in the dataset.
  - You can use the provided DataFrame df to access the dataset.
  - Please write the code for the `find_similar_products` function and any additional helper functions you might need, as included in the sample above.

## Part 2: Microservice Implementation
- `Required`: Write a microservice with an endpoint `GET /find_similar_products` which takes two request params(`product_id`, `num_similar`) and returns the result. Use FastAPI for the web app.
  - You can read about [FastAPI](https://fastapi.tiangolo.com/)
- `Required`: Make the image to be deployable on kubernetes with the sample given for the `Dockerfile`
  - You can read about [Docker](https://docs.docker.com/)
  - `Optional`: You don't have to do it on, but if you have experience you can deploy on your local cluster however you want. [Kubernetes](https://kubernetes.io/)
  
### Sample Code for FastAPI: 
```python
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI()

@app.get("/find_similar_products")
def get_similar_products(product_id: str, num_similar: int) -> List[str]:
    try:
        ## You can implement it differently.
        similar_products = find_similar_products(product_id, num_similar)
        return similar_products
    except Exception as e:
        raise e ## Enhance this error case with different HTTP Error Codes.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Bonus - Part 3: Optimization Problem with Vector Searches (Bonus)
- You are tasked with optimizing the similarity search function to handle large datasets efficiently using vector searches. 
- Implement this using any similarity search alogrithm avaiable on hugging face or even OSS papers for faster nearest neighbor search.
- You should reference the paper and algoritms you use, and you should be able to explain why you chose this implementation.

## Extra How to Help? 
## Amazon Dataset for Apparel: 
- You will be working with the `2020` Amazon data below: 
Ref: [Kaggle](https://www.kaggle.com/datasets/promptcloud/amazon-fashion-products-2020)
- Download and unzip the file on local (if you don't have a kaggle account we have checked in the file under `data`) 
- Unzip and you should have a file in `ldjson` format. 
  - `cd data; unzip archive.zip` 
```bash
.
├── archive.zip
└── marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson

1 directory, 2 files
```
