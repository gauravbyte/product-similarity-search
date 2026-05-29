#!/bin/bash
curl -L -o ./amazon-fashion-products-2020.zip\
  https://www.kaggle.com/api/v1/datasets/download/promptcloud/amazon-fashion-products-2020

# Unzip the downloaded file
unzip -q ./amazon-fashion-products-2020.zip -d ./data/

# Clean up the zip file
rm ./amazon-fashion-products-2020.zip

echo "Data downloaded and extracted successfully to ./data/"
