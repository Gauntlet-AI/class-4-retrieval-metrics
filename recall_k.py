from typing import List, Set, Union, Any
import numpy as np

def calculate_recall_at_k(retrieved_docs: List[Any],
                         relevant_docs: Set[Any],
                         k: int) -> float:
    """
    Calculate Recall@K for a single query
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        float: Recall@K score between 0 and 1
        
    Example:
        >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4']
        >>> relevant = {'doc1', 'doc3', 'doc5'}
        >>> calculate_recall_at_k(retrieved, relevant, k=2)
        0.3333  # Only 1 out of 3 relevant docs in top 2
    """
    if not relevant_docs:
        raise ValueError("No relevant documents provided")
    if k <= 0:
        raise ValueError("k must be positive")
        
    # Consider only top k retrieved documents
    retrieved_at_k = set(retrieved_docs[:k])
    
    # Calculate number of relevant documents found in top k
    relevant_retrieved = len(retrieved_at_k.intersection(relevant_docs))
    
    # Calculate recall
    recall = relevant_retrieved / len(relevant_docs)
    
    return recall

def calculate_average_recall_at_k(query_retrieved: List[List[Any]],
                                query_relevant: List[Set[Any]],
                                k: Union[int, List[int]]) -> Union[float, List[float]]:
    """
    Calculate average Recall@K across multiple queries
    
    Args:
        query_retrieved: List of retrieved document lists for each query
        query_relevant: List of relevant document sets for each query
        k: Either a single k value or list of k values
    
    Returns:
        Union[float, List[float]]: Average Recall@K score(s)
        
    Example:
        >>> retrieved = [
            ['doc1', 'doc2', 'doc3'],
            ['doc4', 'doc5', 'doc6']
        ]
        >>> relevant = [
            {'doc1', 'doc3'},
            {'doc4', 'doc6'}
        ]
        >>> calculate_average_recall_at_k(retrieved, relevant, k=[1, 2])
        [0.25, 0.5]  # Average Recall@1 and Recall@2
    """
    if len(query_retrieved) != len(query_relevant):
        raise ValueError("Number of queries must match between retrieved and relevant lists")
    
    # Convert single k to list for consistent processing
    k_values = [k] if isinstance(k, int) else k
    
    # Initialize results for each k
    recalls = {k_val: [] for k_val in k_values}
    
    # Calculate recall for each query and each k
    for retrieved, relevant in zip(query_retrieved, query_relevant):
        for k_val in k_values:
            try:
                recall = calculate_recall_at_k(retrieved, relevant, k_val)
                recalls[k_val].append(recall)
            except ValueError as e:
                print(f"Warning: Skipping query - {str(e)}")
                
    # Calculate average for each k
    avg_recalls = [np.mean(recalls[k_val]) for k_val in k_values]
    
    # Return single float if only one k, otherwise return list
    return avg_recalls[0] if isinstance(k, int) else avg_recalls

# Example usage
if __name__ == "__main__":
    # Example 1: Single query, single k
    retrieved_docs = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant_docs = {'doc1', 'doc3', 'doc5'}
    k = 3
    recall = calculate_recall_at_k(retrieved_docs, relevant_docs, k)
    print(f"Recall@{k}: {recall:.4f}")  # Should be 0.6667 (2/3 relevant docs in top 3)

    # Example 2: Multiple queries, single k
    queries_retrieved = [
        ['doc1', 'doc2', 'doc3', 'doc4'],
        ['doc5', 'doc6', 'doc7', 'doc8']
    ]
    queries_relevant = [
        {'doc1', 'doc3'},
        {'doc5', 'doc7', 'doc8'}
    ]
    k = 2
    avg_recall = calculate_average_recall_at_k(queries_retrieved, queries_relevant, k)
    print(f"Average Recall@{k}: {avg_recall:.4f}")

    # Example 3: Multiple queries, multiple k values
    k_values = [1, 2, 3, 4]
    avg_recalls = calculate_average_recall_at_k(queries_retrieved, queries_relevant, k_values)
    for k_val, recall in zip(k_values, avg_recalls):
        print(f"Average Recall@{k_val}: {recall:.4f}")

    # Example 4: Perfect recall
    perfect_retrieved = ['doc1', 'doc3', 'doc2']
    perfect_relevant = {'doc1', 'doc3'}
    recall_perfect = calculate_recall_at_k(perfect_retrieved, perfect_relevant, k=3)
    print(f"Perfect Recall@3: {recall_perfect:.4f}")  # Should be 1.0