import numpy as np
from typing import List, Dict, Union, Any

def calculate_dcg(relevance_scores: List[float], k: int = None) -> float:
    """
    Calculate Discounted Cumulative Gain
    
    Args:
        relevance_scores: List of relevance scores in order of retrieval
        k: Number of results to consider (optional)
    
    Returns:
        float: DCG score
    """
    if k is None:
        k = len(relevance_scores)
    
    relevance_scores = relevance_scores[:k]
    gains = [score / np.log2(idx + 2) for idx, score in enumerate(relevance_scores)]
    return np.sum(gains)

def calculate_ndcg(retrieved_docs: List[Any],
                  relevance_dict: Dict[Any, float],
                  k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain
    
    Args:
        retrieved_docs: List of document IDs in order of retrieval
        relevance_dict: Dictionary mapping document IDs to their relevance scores
        k: Number of results to consider (optional)
    
    Returns:
        float: NDCG score
        
    Example:
        >>> retrieved = ['doc1', 'doc2', 'doc3']
        >>> relevance = {'doc1': 3, 'doc2': 2, 'doc3': 1, 'doc4': 0}
        >>> calculate_ndcg(retrieved, relevance, k=2)
        1.0  # Perfect ranking for top 2 results
    """
    if k is None:
        k = len(retrieved_docs)
    
    # Get relevance scores in order of retrieval
    rel_scores = [relevance_dict.get(doc, 0.0) for doc in retrieved_docs]
    
    # Calculate DCG
    dcg = calculate_dcg(rel_scores, k)
    
    # Calculate ideal DCG (IDCG)
    ideal_scores = sorted([score for score in relevance_dict.values()], reverse=True)
    idcg = calculate_dcg(ideal_scores, k)
    
    # Handle case where IDCG is 0
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def calculate_ndcg_for_queries(query_results: List[List[Any]],
                             query_relevance: List[Dict[Any, float]],
                             k: int = None) -> float:
    """
    Calculate average NDCG across multiple queries
    
    Args:
        query_results: List of retrieval results for each query
        query_relevance: List of relevance dictionaries for each query
        k: Number of results to consider for each query (optional)
    
    Returns:
        float: Average NDCG score across all queries
    """
    if len(query_results) != len(query_relevance):
        raise ValueError("Number of queries must match between results and relevance judgments")
    
    ndcg_scores = []
    for results, relevance in zip(query_results, query_relevance):
        ndcg = calculate_ndcg(results, relevance, k)
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)

# Example usage
if __name__ == "__main__":
    # Example 1: Perfect ranking
    retrieved_perfect = ['doc1', 'doc2', 'doc3']
    relevance_perfect = {'doc1': 3, 'doc2': 2, 'doc3': 1, 'doc4': 0}
    ndcg_perfect = calculate_ndcg(retrieved_perfect, relevance_perfect)
    print(f"Perfect NDCG: {ndcg_perfect:.4f}")  # Should be 1.0

    # Example 2: Imperfect ranking
    retrieved_imperfect = ['doc2', 'doc3', 'doc1']  # Highest relevance doc (doc1) is last
    ndcg_imperfect = calculate_ndcg(retrieved_imperfect, relevance_perfect)
    print(f"Imperfect NDCG: {ndcg_imperfect:.4f}")  # Should be lower than 1.0

    # Example 3: Multiple queries
    query_results = [
        ['doc1', 'doc2', 'doc3'],  # Query 1 results
        ['doc2', 'doc1', 'doc4']   # Query 2 results
    ]
    query_relevance = [
        {'doc1': 3, 'doc2': 2, 'doc3': 1, 'doc4': 0},  # Query 1 relevance
        {'doc1': 2, 'doc2': 3, 'doc3': 0, 'doc4': 1}   # Query 2 relevance
    ]
    avg_ndcg = calculate_ndcg_for_queries(query_results, query_relevance)
    print(f"Average NDCG across queries: {avg_ndcg:.4f}")

    # Example 4: NDCG@K (top K results only)
    ndcg_at_2 = calculate_ndcg(retrieved_imperfect, relevance_perfect, k=2)
    print(f"NDCG@2: {ndcg_at_2:.4f}")