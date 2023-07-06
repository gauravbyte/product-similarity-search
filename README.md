# sap-cxii-tech-ex-01
Technical Exercise for SAP.

## Amazon Dataset for Apparel: 
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

## What is the goal? 
You are tasked with implementing a product similarity search feature using the given dataset. Given a product’s `unique ID` (uniq_id), your goal is to retrieve a list of `similar products` based on `specific attributes`.

## To Implement

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

## Note:

If multiple products have the same similarity score, the tie-breaker can be based on any attribute (e.g., sales_price or rating).

## Assumptions: 
- You can assume that the given `product_id` will exist in the dataset.
- You can use the provided DataFrame df to access the dataset.
- Please write the code for the `find_similar_products` function and any additional helper functions you might need.


## Bonus:  
- Write a microservice with an endpoint `GET /find_similar_products` which takes two request params `(product_id, num_similar)` and returns the result.
- You can use any python frameworks for the web app `(e.g Flask, FastAPI, Django, etc.)`