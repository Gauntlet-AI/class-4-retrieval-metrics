# Retrieval Metrics Calculator

This repository provides example code for calculating three essential retrieval metrics commonly used to evaluate RAG (Retrieval-Augmented Generation) systems and search quality:

## Metrics Included

### 1. Mean Reciprocal Rank (MRR)

MRR measures the effectiveness of retrieval systems by calculating the reciprocal of the rank of the first relevant document. It focuses on the position of the first correct answer, making it particularly useful for question-answering systems.

### 2. Normalized Discounted Cumulative Gain (NDCG)

NDCG evaluates the quality of ranked results by considering both the relevance and position of retrieved documents. It penalizes highly relevant documents appearing lower in the search results, providing a comprehensive measure of ranking quality.

### 3. Recall@K

Recall@K measures the proportion of relevant documents found in the top K retrieved results. This metric is crucial for understanding how well your system captures all relevant information within a specified number of results.

## Usage

This repository contains sample implementations using synthetic data to demonstrate how to calculate these metrics. You can use these implementations as reference when building monitoring systems for your RAG pipeline.

**Note**: The code provided uses dummy data for demonstration purposes. In a real-world application, you would replace this with your actual retrieval results and relevance judgments.

## Files

- `mrr_calculator.py`: Implementation of Mean Reciprocal Rank calculation
- `ndcg_calculator.py`: Implementation of Normalized Discounted Cumulative Gain
- `recall_calculator.py`: Implementation of Recall@K metric

## Getting Started

1. Review the individual metric implementation files
2. Understand how the calculations work with the provided sample data
3. Adapt the code to work with your RAG pipeline's output format
4. Integrate these metrics into your evaluation pipeline

## Contributing

Feel free to contribute improvements or additional metrics by submitting a pull request.
